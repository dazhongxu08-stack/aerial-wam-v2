#!/usr/bin/env python3
"""Closed-loop Tello collect for consequence post-train (see tello-consequence-collect.md).

Not ``tello_mac_auto_collect``: this policy looks at the live JPEG (DA3 if a
depth ckpt is given) and Tello yaw/height, then sends hover / yaw / small
forward. Writes ``meta.json`` before flight and ``near_miss`` into events after.

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_consequence_collect.py \\
    --goal-desc '对面墙' --goal-approx-m 4 --goal-bearing-deg 0 --air-s 75
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from experiments.aerial.rl.tello_consequence_policy import (
    NEAR_M,
    SLOW_M,
    YAW_STICK,
    TelloConsequencePolicy,
    altitude_rc,
    decision_log,
    wrap,
)
from experiments.aerial.rl.semantic_keepout import SemanticKeepout

BASE = "http://127.0.0.1:8090"
MIN_BAT = 40
MIN_RSSI = -70
RC_HZ = 8.0
REPO = Path(__file__).resolve().parents[3]


def _req(method: str, path: str, body: Optional[dict] = None, timeout: float = 8.0) -> dict:
    data = None
    headers = {"Connection": "close"}
    if body is not None:
        raw = json.dumps(body).encode()
        data = raw
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def status() -> dict:
    return _req("GET", "/api/status")


def cmd(name: str) -> dict:
    return _req("POST", "/api/cmd", {"cmd": name}, timeout=25.0)


def rc(a: int, b: int, c: int, d: int) -> dict:
    return _req("POST", "/api/rc", {"a": a, "b": b, "c": c, "d": d, "source": "auto"})


def snapshot_rgb() -> np.ndarray:
    req = urllib.request.Request(BASE + "/snapshot.jpg", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        raw = resp.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def patch_meta(run_dir: Path, extra: Dict[str, Any]) -> None:
    path = run_dir / "meta.json"
    if not path.parent.is_dir():
        # Ground console may be on another machine; it returns that host's path.
        print("META_REMOTE", json.dumps({"dir": str(run_dir), **extra}, ensure_ascii=False), flush=True)
        return
    meta: Dict[str, Any] = {}
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(extra)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def last_telemetry(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "telemetry.jsonl"
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            n = fh.tell()
            fh.seek(max(0, n - 8000))
            lines = fh.read().decode("utf-8", "ignore").splitlines()
        if not lines:
            return {}
        return json.loads(lines[-1])
    except (OSError, json.JSONDecodeError):
        return {}


def imu_kwargs(st: dict, run_dir: Optional[Path], last_b: int) -> Dict[str, Any]:
    tel: Dict[str, Any] = {}
    if run_dir is not None:
        row = last_telemetry(run_dir)
        tel = dict(row.get("tello") or {})
        rc_row = row.get("rc") or []
        if isinstance(rc_row, (list, tuple)) and len(rc_row) >= 2:
            last_b = int(rc_row[1])
    def _f(key: str, fallback: Any = None) -> Optional[float]:
        v = tel.get(key)
        if v is None:
            v = st.get(key)
        if v is None:
            v = fallback
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None
    return {
        "agx": _f("agx"),
        "agy": _f("agy"),
        "agz": _f("agz"),
        "pitch_deg": _f("pitch", st.get("pitch")),
        "vgx_dms": _f("vgx"),
        "rc_b": int(last_b),
    }


def append_events(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = run_dir / "events.jsonl"
    if not path.parent.is_dir():
        print("EVENTS_REMOTE", json.dumps({"dir": str(run_dir), "rows": rows}, ensure_ascii=False), flush=True)
        return
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _preflight() -> dict:
    st = status()
    print("preflight", {k: st.get(k) for k in ("ok", "battery", "wifi_rssi_dbm", "allow_fly", "airborne", "temp")})
    if not st.get("ok"):
        raise RuntimeError("Tello link down")
    if int(st.get("battery") or 0) < MIN_BAT:
        raise RuntimeError(f"battery {st.get('battery')}% < {MIN_BAT}%")
    rssi = st.get("wifi_rssi_dbm")
    if rssi is not None and int(rssi) < MIN_RSSI:
        raise RuntimeError(f"rssi {rssi}")
    age = st.get("frame_age_s")
    if age is None or float(age) > 2.0:
        raise RuntimeError(f"stale video frame_age_s={age}")
    return st


def land_if_airborne() -> None:
    st = status()
    if not st.get("airborne"):
        print("already grounded")
        return
    try:
        print("land", cmd("land"))
    except Exception as exc:
        print("land http", exc, file=sys.stderr)
        # ``airborne`` stays set when the land "ok" reply is lost after touchdown.
        st = status()
        if st.get("airborne") and int(st.get("tof") or 0) > 20:
            print("emergency", cmd("emergency"))
        else:
            print("landed (tof)", st.get("tof"))


def _gate(st: dict) -> None:
    if not st.get("ok"):
        raise RuntimeError("link lost")
    bat = int(st.get("battery") or 0)
    if bat and bat < 22:
        raise RuntimeError(f"battery {bat}")
    rssi = st.get("wifi_rssi_dbm")
    if rssi is not None and int(rssi) < MIN_RSSI:
        raise RuntimeError(f"rssi {rssi}")
    h = st.get("height_cm")
    if h is not None and int(h) > 180:
        raise RuntimeError(f"too high {h}cm")


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_optional_models(args: argparse.Namespace):
    models = []
    device = str(args.device)
    if args.ckpts:
        import torch

        from experiments.aerial.rl.consequence_model import load_checkpoint

        if device == "auto":
            device = _auto_device(torch)
        for p in args.ckpts.split(","):
            p = p.strip()
            if p:
                models.append(load_checkpoint(REPO / p, device=device))
    pred = None
    if args.depth_ckpt:
        import torch

        from experiments.aerial.rl.depth_predictor import DepthMinPredictor

        if device == "auto":
            device = _auto_device(torch)
        pred = DepthMinPredictor.from_checkpoint(REPO / args.depth_ckpt, device=device)
        pred.reset()
    return models, pred, device


def _depth56(pred, rgb: np.ndarray):
    if pred is None:
        return None
    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.env.obs import Observation

    rgb224 = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR), np.uint8)
    obs = Observation(rgb=rgb224, state=np.zeros(7, np.float32), collided=False)
    d = pred.predict_depth(obs)
    if d is None:
        return None
    return np.asarray(pool_min(np.asarray(d, np.float32)), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goal-desc", required=True)
    ap.add_argument("--goal-approx-m", type=float, required=True)
    ap.add_argument("--goal-bearing-deg", type=float, required=True, help="relative to nose at takeoff, clockwise+ (right)")
    ap.add_argument("--near-m", type=float, default=NEAR_M, help="policy near threshold; lower only for contact runs")
    ap.add_argument("--slow-m", type=float, default=SLOW_M, help="below this fwd clearance use half stick")
    ap.add_argument("--pre-spin-deg", type=float, default=0.0, help="in-place right yaw after settling, before the policy")
    ap.add_argument("--air-s", type=float, default=75.0, help="airborne seconds (60–120)")
    ap.add_argument("--depth-ckpt", default="", help="optional DepthMinPredictor / DA3 ckpt")
    ap.add_argument("--ckpts", default="", help="optional comma list of consequence ckpts (survival shield)")
    ap.add_argument("--avoid", default="", help="comma list of scene classes to fly around, e.g. red_stool")
    ap.add_argument("--avoid-radius-m", type=float, default=0.5, help="keep-out radius around each avoid object")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    global BASE
    BASE = str(args.base)
    air_s = float(args.air_s)
    if air_s < 60.0 or air_s > 120.0:
        print("air-s must be in 60–120", file=sys.stderr)
        return 2

    models, pred, device = _load_optional_models(args)
    avoid = SemanticKeepout.from_names(args.avoid, radius_m=args.avoid_radius_m) if args.avoid else None
    policy = TelloConsequencePolicy(
        models=models, device=device if device != "auto" else "cpu", near_m=args.near_m, slow_m=args.slow_m,
        avoid=avoid,
    )
    _req("POST", "/api/control_mode", {"mode": "auto"})
    run_dir: Optional[Path] = None
    extra_events: List[Dict[str, Any]] = []
    near_n = 0
    hit_n = 0
    last_b = 0
    last_keepout = ""
    try:
        _preflight()
        rec = _req("POST", "/api/record", {"on": True})
        print("record", rec.get("msg"))
        run_dir = Path(str(rec.get("dir") or ""))
        takeoff_yaw = None
        patch_meta(
            run_dir,
            {
                "goal_desc": args.goal_desc,
                "goal_approx_m": float(args.goal_approx_m),
                "goal_bearing_deg": float(args.goal_bearing_deg),
                "control": "auto",
                "intended_use": "consequence_posttrain",
                "policy": "tello_consequence_policy",
                "depth_ckpt": args.depth_ckpt or None,
                "consequence_ckpts": args.ckpts or None,
                "near_m": float(args.near_m),
                "slow_m": float(args.slow_m),
                "pre_spin_deg": float(args.pre_spin_deg),
                "avoid": args.avoid or None,
                "avoid_radius_m": float(args.avoid_radius_m) if args.avoid else None,
                "near_miss_n": 0,
                "collision_n": 0,
            },
        )
        print("takeoff", cmd("takeoff"))
        t_air0 = time.time()
        dt = 1.0 / RC_HZ
        settled = False
        settle_ok_since: Optional[float] = None
        spun_deg, spin_prev = 0.0, None
        while time.time() - t_air0 < air_s:
            t_loop = time.time()
            st = status()
            _gate(st)
            yaw = math.radians(float(st.get("yaw") or 0.0))
            if takeoff_yaw is None:
                takeoff_yaw = yaw
            h_cm = st.get("height_cm")
            height_m = None if h_cm is None else float(h_cm) / 100.0
            # Climb/hold ~1 m before any yaw or creep — avoids hitting furniture
            # that sits next to the pad right after takeoff.
            if not settled:
                in_band = height_m is not None and 0.85 <= height_m <= 1.45
                if in_band:
                    if settle_ok_since is None:
                        settle_ok_since = time.time()
                    elif time.time() - settle_ok_since >= 2.0:
                        settled = True
                        print("settled at {:.2f} m".format(height_m or 0.0))
                else:
                    settle_ok_since = None
                rc(0, 0, altitude_rc(height_m), 0)
                remain = dt - (time.time() - t_loop)
                if remain > 0:
                    time.sleep(remain)
                continue
            if spun_deg < args.pre_spin_deg:
                if spin_prev is not None:
                    spun_deg += abs(math.degrees(wrap(yaw - spin_prev)))
                spin_prev = yaw
                rc(0, 0, altitude_rc(height_m), YAW_STICK)
                if spun_deg >= args.pre_spin_deg:
                    print("pre-spin done {:.0f} deg".format(spun_deg))
                remain = dt - (time.time() - t_loop)
                if remain > 0:
                    time.sleep(remain)
                continue
            rgb = snapshot_rgb()
            d56 = _depth56(pred, rgb)
            dec = policy.decide(
                rgb=rgb,
                height_m=height_m,
                yaw_rad=yaw,
                takeoff_yaw_rad=float(takeoff_yaw),
                goal_bearing0_rad=math.radians(float(args.goal_bearing_deg)),
                depth56=d56,
                **imu_kwargs(st, run_dir, last_b),
            )
            if dec.collision:
                hit_n += 1
                extra_events.append(
                    {
                        "t_s": round(time.time() - t_air0, 3),
                        "event": "collision",
                        "note": dec.note,
                        "fwd_m": round(dec.fwd_m, 3),
                    }
                )
                print("HIT", decision_log(dec))
                rc(0, 0, 0, 0)
                last_b = 0
                break
            if dec.keepout and not last_keepout:
                print("KEEPOUT", decision_log(dec))
            last_keepout = dec.keepout
            if dec.near_miss:
                near_n += 1
                extra_events.append(
                    {
                        "t_s": round(time.time() - t_air0, 3),
                        "event": "near_miss",
                        "note": dec.note,
                        "fwd_m": round(dec.fwd_m, 3),
                    }
                )
                print("NEAR", decision_log(dec))
            rc(*dec.rc)
            last_b = int(dec.rc[1])
            remain = dt - (time.time() - t_loop)
            if remain > 0:
                time.sleep(remain)
        land_if_airborne()
        time.sleep(3)
        return 0
    except Exception as exc:
        print("ABORT", exc, file=sys.stderr)
        try:
            land_if_airborne()
        except Exception:
            pass
        return 1
    finally:
        try:
            rc(0, 0, 0, 0)
        except Exception:
            pass
        try:
            stop = _req("POST", "/api/record", {"on": False})
            print("saved", stop)
        except Exception as exc:
            print("record stop failed", exc, file=sys.stderr)
        try:
            _req("POST", "/api/control_mode", {"mode": "manual"})
        except Exception:
            pass
        if run_dir is not None and run_dir.is_dir():
            append_events(run_dir, extra_events)
            patch_meta(run_dir, {"near_miss_n": near_n, "collision_n": hit_n})
            print(f"near_miss_n={near_n} collision_n={hit_n} dir={run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
