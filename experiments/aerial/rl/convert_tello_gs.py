#!/usr/bin/env python3
"""Convert Tello ground-station runs into schema-v2 ``episode_*.npz``.

Source: ``artifacts/tello_gs/run_*/`` (JPEG + ``telemetry.jsonl`` + ``events.jsonl``).
Target: ``dataset.episode_arrays`` — RGB 224, proprio4, RC→body-delta actions.
No depth (Tello has none). IMU lin_acc from ``agx/agy/agz`` (0.001 g); gyro absent.

Scene has **one** QR: native JPEGs are scanned; the largest detection stamps
``goal`` (episode-local pose at that frame). ``arrived`` if that quad is large
enough and the episode did not emergency-stop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.env.tello_env import (
    _DEFAULT_MAX_V_MPS,
    _DEFAULT_MAX_YAW_RAD_S,
    rc_to_body_delta,
)
from experiments.aerial.rl.tello_qr import (
    DEFAULT_ARRIVE_AREA_FRAC,
    arrived_from_hit,
    best_hit,
    detect_qr_jpeg,
)

RGB_SIZE = 224
# Tello SDK: ag* are thousandths of g.
_AG_TO_MS2 = 9.81 / 1000.0
_MIN_STEPS = 3
REPO_ROOT = Path(__file__).resolve().parents[3]


def _f(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _jpeg_path(run_dir: Path, frame_seq: int) -> Path:
    return run_dir / "frames" / f"{int(frame_seq):06d}.jpg"


def _resize_rgb(path: Path, size: int = RGB_SIZE) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size, size), resample=Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def attach_fastwam_pose(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dead-reckon x/y in FastWAM (x fwd, y left) from vg*; z from height."""
    x = y = 0.0
    prev_t: Optional[float] = None
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        t_s = float(rec.get("t_s") or 0.0)
        dt = 0.0 if prev_t is None else max(0.0, t_s - prev_t)
        prev_t = t_s
        tel = rec.get("tello") if isinstance(rec.get("tello"), dict) else {}
        pose = rec.get("pose") if isinstance(rec.get("pose"), dict) else {}
        yaw = math.radians(_f(tel.get("yaw")) if _f(tel.get("yaw")) is not None else (_f(pose.get("yaw_deg")) or 0.0))
        vgx = (_f(tel.get("vgx")) or 0.0) * 0.1  # m/s body forward
        vgy_right = (_f(tel.get("vgy")) or 0.0) * 0.1
        fwd, left = vgx, -vgy_right
        c, s = math.cos(yaw), math.sin(yaw)
        vx = c * fwd - s * left
        vy = s * fwd + c * left
        x += vx * dt
        y += vy * dt
        h = _f(tel.get("h"))
        z = (h / 100.0) if h is not None else float(pose.get("z_m") or 0.0)
        rec["fw_pos"] = np.array([x, y, z], dtype=np.float64)
        rec["fw_yaw"] = float(yaw)
        rec["fw_vel"] = np.array([vx, vy, 0.0], dtype=np.float32)
        out.append(rec)
    # vz from height finite difference (vgz is often 0 on EDU).
    for i, rec in enumerate(out):
        if i + 1 < len(out):
            dt = float(out[i + 1]["t_s"]) - float(rec["t_s"])
            if dt > 1e-4:
                rec["fw_vel"][2] = float((out[i + 1]["fw_pos"][2] - rec["fw_pos"][2]) / dt)
        elif i > 0:
            rec["fw_vel"][2] = float(out[i - 1]["fw_vel"][2])
    return out


def unique_frame_last(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    for rec in rows:
        seq = int(rec.get("frame_seq") or -1)
        if seq < 0:
            continue
        if seq not in last:
            order.append(seq)
        last[seq] = rec
    return [last[s] for s in order]


def subsample_hz(rows: Sequence[Dict[str, Any]], hz: float) -> List[Dict[str, Any]]:
    if hz <= 0 or len(rows) < 2:
        return list(rows)
    period = 1.0 / float(hz)
    out: List[Dict[str, Any]] = [rows[0]]
    next_t = float(rows[0]["t_s"]) + period
    for rec in rows[1:]:
        if float(rec["t_s"]) + 1e-9 >= next_t:
            out.append(rec)
            next_t = float(rec["t_s"]) + period
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def _imu_from_tello(tel: Dict[str, Any]) -> Dict[str, Any]:
    ax, ay, az = _f(tel.get("agx")), _f(tel.get("agy")), _f(tel.get("agz"))
    if ax is None and ay is None and az is None:
        return {}
    return {
        "ang_vel": [float("nan"), float("nan"), float("nan")],
        "lin_acc": [
            (ax or 0.0) * _AG_TO_MS2,
            (ay or 0.0) * _AG_TO_MS2,
            (az or 0.0) * _AG_TO_MS2,
        ],
    }


def _had_emergency(events: Sequence[Dict[str, Any]]) -> bool:
    return any(ev.get("event") == "emergency" for ev in events)


def row_to_obs(
    rec: Dict[str, Any],
    rgb: np.ndarray,
    *,
    goal: Optional[np.ndarray] = None,
    qr: Optional[Dict[str, Any]] = None,
) -> Observation:
    tel = rec.get("tello") if isinstance(rec.get("tello"), dict) else {}
    pos = np.asarray(rec["fw_pos"], dtype=np.float64).reshape(3)
    vel = np.asarray(rec["fw_vel"], dtype=np.float32).reshape(3)
    yaw = float(rec["fw_yaw"])
    state = np.array(
        [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], yaw],
        dtype=np.float32,
    )
    info: Dict[str, Any] = {
        "source": "tello_gs",
        "frame_seq": int(rec.get("frame_seq") or -1),
        "rc": list(rec.get("rc") or [0, 0, 0, 0]),
        "control_mode": rec.get("control_mode"),
    }
    if goal is not None:
        info["goal"] = np.asarray(goal, dtype=np.float32).reshape(3)
    if qr is not None:
        info["qr"] = qr
    return Observation(
        rgb=rgb,
        state=state,
        collided=False,
        depth=None,
        imu=_imu_from_tello(tel),
        t=float(rec.get("t_s") or 0.0),
        info=info,
        baro_alt=_f(tel.get("baro")),
        agl_m=float(pos[2]),
    )


def run_to_transitions(
    run_dir: Path,
    *,
    hz: float = 5.0,
    rgb_size: int = RGB_SIZE,
    max_v_mps: float = _DEFAULT_MAX_V_MPS,
    max_yaw_rad_s: float = _DEFAULT_MAX_YAW_RAD_S,
    scan_qr: bool = True,
    min_qr_area_frac: float = DEFAULT_ARRIVE_AREA_FRAC,
) -> List[Transition]:
    run_dir = Path(run_dir)
    rows = _load_jsonl(run_dir / "telemetry.jsonl")
    events = _load_jsonl(run_dir / "events.jsonl")
    if not rows:
        return []
    posed = attach_fastwam_pose(rows)
    air = [r for r in posed if r.get("airborne")]
    framed = unique_frame_last(air)
    keep: List[Dict[str, Any]] = []
    for rec in framed:
        jp = _jpeg_path(run_dir, int(rec.get("frame_seq") or -1))
        if jp.is_file():
            rec = dict(rec)
            rec["_jpeg"] = jp
            keep.append(rec)
    keep = subsample_hz(keep, hz)
    if len(keep) < _MIN_STEPS:
        return []
    crash = _had_emergency(events)
    qr_per_frame = [detect_qr_jpeg(rec["_jpeg"]) if scan_qr else [] for rec in keep]
    picked = best_hit(qr_per_frame)
    goal: Optional[np.ndarray] = None
    qr_meta: Optional[Dict[str, Any]] = None
    arrived = False
    if picked is not None:
        idx, hit = picked
        goal = np.asarray(keep[idx]["fw_pos"], dtype=np.float32).reshape(3)
        arrived = arrived_from_hit(hit, collided=crash, min_area_frac=min_qr_area_frac)
        qr_meta = {
            "data": hit.data,
            "area_frac": hit.area_frac,
            "frame_index": int(idx),
            "frame_seq": int(keep[idx].get("frame_seq") or -1),
            "arrived": bool(arrived),
        }
    rgbs = [_resize_rgb(rec["_jpeg"], rgb_size) for rec in keep]
    obs_list = [
        row_to_obs(
            rec,
            rgb,
            goal=goal,
            qr=(
                {
                    "data": qr_per_frame[i][0].data,
                    "area_frac": qr_per_frame[i][0].area_frac,
                }
                if qr_per_frame[i]
                else None
            ),
        )
        for i, (rec, rgb) in enumerate(zip(keep, rgbs))
    ]
    trans: List[Transition] = []
    n = len(keep)
    for i in range(n):
        rec = keep[i]
        dt = float(keep[i + 1]["t_s"]) - float(rec["t_s"]) if i + 1 < n else (
            float(rec["t_s"]) - float(keep[i - 1]["t_s"]) if i > 0 else 1.0 / max(hz, 1e-6)
        )
        dt = max(dt, 1e-3)
        rc = list(rec.get("rc") or [0, 0, 0, 0])
        while len(rc) < 4:
            rc.append(0)
        action = rc_to_body_delta(
            rc[:4],
            dt,
            max_v_mps=max_v_mps,
            max_yaw_rad_s=max_yaw_rad_s,
        )
        last = i == n - 1
        collided = bool(crash and last)
        nxt = obs_list[i + 1] if i + 1 < n else obs_list[i]
        if collided:
            nxt = Observation(
                rgb=nxt.rgb,
                state=nxt.state,
                collided=True,
                depth=None,
                imu=nxt.imu,
                t=nxt.t,
                info=dict(nxt.info),
                baro_alt=nxt.baro_alt,
                agl_m=nxt.agl_m,
            )
        reward = 0.0
        if goal is not None:
            p0 = np.asarray(obs_list[i].position, dtype=np.float64)
            p1 = np.asarray(nxt.position, dtype=np.float64)
            g = np.asarray(goal, dtype=np.float64)
            reward = float(np.linalg.norm(p0 - g) - np.linalg.norm(p1 - g))
        info: Dict[str, Any] = {"rc": rc[:4], "dt": dt, "run_dir": str(run_dir)}
        if goal is not None:
            info["goal"] = np.asarray(goal, dtype=np.float32)
        if qr_meta is not None:
            info["qr"] = qr_meta
        trans.append(
            Transition(
                obs=obs_list[i],
                action=action,
                reward=reward,
                done=last,
                next_obs=nxt,
                info=info,
            )
        )
    if trans:
        trans[0].info["arrived"] = bool(arrived)
    return trans


def _write_episode(
    out_dir: Path,
    index: int,
    transitions: Sequence[Transition],
    *,
    arrived: bool,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{index:05d}.npz"
    arrays = ds.episode_arrays(transitions)
    arrays["arrived"] = np.asarray(bool(arrived))
    np.savez_compressed(path, **arrays)
    return path


def convert_runs(
    src: Path,
    out_dir: Path,
    *,
    hz: float = 5.0,
    scan_qr: bool = True,
    min_qr_area_frac: float = DEFAULT_ARRIVE_AREA_FRAC,
) -> Dict[str, Any]:
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in src.glob("run_*") if p.is_dir())
    if src.name.startswith("run_") and (src / "telemetry.jsonl").is_file():
        runs = [src]
    entries: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    index = 0
    for run_dir in runs:
        trans = run_to_transitions(
            run_dir, hz=hz, scan_qr=scan_qr, min_qr_area_frac=min_qr_area_frac
        )
        if not trans:
            skipped.append({"run": run_dir.name, "reason": "too_short_or_no_airborne_frames"})
            continue
        arrived = bool(trans[0].info.get("arrived"))
        qr = trans[0].info.get("qr") if isinstance(trans[0].info.get("qr"), dict) else None
        path = _write_episode(out_dir, index, trans, arrived=arrived)
        report = ds.quality_report(trans)
        report["file"] = path.name
        report["run"] = run_dir.name
        report["hard_fail"] = ds.assert_nontrivial(report)
        report["quarantine"] = ds.quarantine_reasons(report)
        reports.append(report)
        entries.append(
            {
                "index": index,
                "file": path.name,
                "run": run_dir.name,
                "steps": report["steps"],
                "path_length_m": report["path_length_m"],
                "has_depth": False,
                "has_goal": bool(trans[0].info.get("goal") is not None),
                "arrived": arrived,
                "qr": qr,
                "hard_fail": report["hard_fail"],
                "quarantine": report["quarantine"],
            }
        )
        index += 1
    meta = {
        "source": str(src),
        "hz": hz,
        "rgb": [RGB_SIZE, RGB_SIZE, 3],
        "action": "rc_to_body_delta (Tello stick → dx,dy,dz,dyaw)",
        "goal": "single scene QR (largest-quad pose; not payment-filtered)",
        "arrive_area_frac": min_qr_area_frac,
        "depth": None,
        "skipped": skipped,
    }
    ds.write_manifest(out_dir, entries, meta=meta)
    if reports:
        ds.write_quality_summary(out_dir, reports)
    return {"episodes": entries, "skipped": skipped, "out_dir": str(out_dir)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tello_gs",
        help="tello_gs root or a single run_* directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tello_gs_npz",
        help="output directory for episode_*.npz",
    )
    parser.add_argument("--hz", type=float, default=5.0, help="resample rate; 0 = every unique frame")
    parser.add_argument(
        "--min-qr-area-frac",
        type=float,
        default=DEFAULT_ARRIVE_AREA_FRAC,
        help="QR quad / frame area for arrived (single scene QR)",
    )
    parser.add_argument("--no-qr", action="store_true", help="skip JPEG QR scan")
    args = parser.parse_args(argv)
    result = convert_runs(
        args.src,
        args.out,
        hz=args.hz,
        scan_qr=not args.no_qr,
        min_qr_area_frac=args.min_qr_area_frac,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["episodes"] else 1


if __name__ == "__main__":
    sys.exit(main())
