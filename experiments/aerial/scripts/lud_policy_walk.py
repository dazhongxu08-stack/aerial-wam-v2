#!/usr/bin/env python3
"""Walk the Lumos dog with the Tello consequence policy, supervised by H12.

Runs on the Orin next to ``h12_lud_bridge.py``. Each tick: camera frame ->
DA3 depth -> TelloConsequencePolicy (+ consequence shield) -> walk command
posted to the bridge as ``{"src": "auto"}``. The bridge only lets auto drive
while H12 is in WALK with centred sticks; any stick, CH6 or CH5 takes over.

There is no odometry: distance is the integral of the accepted forward
command, so ``--dist-m`` is an estimate.

Modes:
  --dry    run the policy and log, send nothing
  --probe  short fixed moves to check the forward / yaw signs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for root in (Path("/home/ubuntu/lumos_sdk"), Path.home() / "lumos_sdk"):
    for p in (root / "python", root / "lcm_typedef" / "python"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

from experiments.aerial.rl.tello_consequence_policy import (  # noqa: E402
    FWD_STICK,
    SLOW_M,
    YAW_STICK,
    TelloConsequencePolicy,
)

IMU_CHANNEL = "lcm_imu_data"
LCM_URL = "udpm://239.255.76.67:7667?ttl=255"
CMD_TTL_S = 0.6


class Camera:
    """Keeps only the newest frame so the policy never sees a stale buffer."""

    def __init__(self, dev: str, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {dev}")
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._at = 0.0
        self._n = 0
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self._run:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame, self._at = bgr, time.time()
                self._n += 1

    def latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            return self._frame, self._at, self._n

    def close(self) -> None:
        self._run = False
        time.sleep(0.1)
        self.cap.release()


class Imu:
    def __init__(self, url: str) -> None:
        import lcm
        from imu_data_lcmt import imu_data_lcmt

        self._type = imu_data_lcmt
        self._lc = lcm.LCM(url)
        self._lc.subscribe(IMU_CHANNEL, self._on)
        self._lock = threading.Lock()
        self.rpy: Optional[Tuple[float, float, float]] = None
        self.at = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _on(self, _ch: str, data: bytes) -> None:
        msg = self._type.decode(data)
        with self._lock:
            self.rpy = tuple(float(v) for v in msg.navRPY)
            self.at = time.time()

    def _loop(self) -> None:
        while True:
            self._lc.handle_timeout(200)

    def yaw(self) -> Tuple[Optional[float], float]:
        with self._lock:
            return (None if self.rpy is None else self.rpy[2]), time.time() - self.at


class Sender:
    """Re-sends the latest command at a fixed rate; goes quiet when it is stale."""

    def __init__(self, base: str, hz: float, enabled: bool) -> None:
        self.url = base.rstrip("/") + "/api/cmd"
        self.period = 1.0 / hz
        self.enabled = enabled
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0)
        self._at = 0.0
        self._run = True
        self.accepted = False
        self.accepted_at = 0.0
        self.last_reply: Dict[str, Any] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, vx: float, vyaw: float) -> None:
        with self._lock:
            self._cmd, self._at = (vx, vyaw), time.time()

    def _post(self, vx: float, vyaw: float) -> Dict[str, Any]:
        body = json.dumps({"src": "auto", "vx": vx, "vyaw": vyaw}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=0.3).read())

    def _loop(self) -> None:
        while self._run:
            t0 = time.time()
            with self._lock:
                (vx, vyaw), at = self._cmd, self._at
            if self.enabled and t0 - at <= CMD_TTL_S:
                try:
                    rep = self._post(vx, vyaw)
                    self.last_reply = rep
                    self.accepted = bool(rep.get("auto_accepted"))
                    if self.accepted:
                        self.accepted_at = time.time()
                except Exception as exc:  # noqa: BLE001
                    self.accepted = False
                    self.last_reply = {"error": str(exc)}
            else:
                self.accepted = False
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def stop(self) -> None:
        if self.enabled:
            for _ in range(5):
                try:
                    self._post(0.0, 0.0)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.1)
        self._run = False


def depth56_from(pred, rgb: np.ndarray) -> Optional[np.ndarray]:
    from PIL import Image

    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.env.obs import Observation

    rgb224 = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR), np.uint8)
    d = pred.predict_depth(Observation(rgb=rgb224, state=np.zeros(7, np.float32), collided=False))
    if d is None:
        return None
    return np.asarray(pool_min(np.asarray(d, np.float32)), np.float32)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth-ckpt", required=True)
    ap.add_argument("--ckpts", default="", help="comma list of consequence ckpts (survival shield)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--base", default="http://127.0.0.1:8092")
    ap.add_argument("--dist-m", type=float, default=10.0)
    ap.add_argument("--max-s", type=float, default=150.0)
    ap.add_argument("--vx-max", type=float, default=0.3, help="dog m/s at full policy forward stick")
    ap.add_argument("--vyaw-max", type=float, default=0.3, help="dog rad/s at full policy yaw stick")
    ap.add_argument("--vx-sign", type=float, default=1.0, help="+1 if bridge vx>0 walks forward")
    ap.add_argument("--yaw-cmd-sign", type=float, default=1.0, help="+1 if bridge vyaw>0 turns left (CCW)")
    ap.add_argument("--imu-yaw-sign", type=float, default=1.0, help="+1 if navRPY yaw grows turning left (CCW)")
    ap.add_argument("--imu-yaw-deg", action="store_true", help="navRPY is in degrees")
    ap.add_argument("--near-m", type=float, default=1.15)
    ap.add_argument("--slow-m", type=float, default=SLOW_M)
    ap.add_argument("--shield", choices=("veto", "argmax"), default="veto")
    ap.add_argument("--surv-min", type=float, default=0.7)
    ap.add_argument("--hz", type=float, default=5.0, help="policy rate (tuned at 5 Hz)")
    ap.add_argument("--send-hz", type=float, default=10.0)
    ap.add_argument("--save-every-s", type=float, default=1.0)
    ap.add_argument("--out", default=str(Path.home() / "lud_walk_runs"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if str(args.device).startswith("cuda"):
        os.environ.setdefault("AERIAL_DA3_FP16", "1")
    run_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "walk.jsonl", "a", buffering=1)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    cam = Camera(args.camera, args.width, args.height)
    imu = Imu(LCM_URL)
    sender = Sender(args.base, args.send_hz, enabled=not args.dry)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def imu_yaw_ccw() -> Optional[float]:
        y, age = imu.yaw()
        if y is None or age > 0.5:
            return None
        if args.imu_yaw_deg:
            y = math.radians(y)
        return args.imu_yaw_sign * y

    def save(tag: str, bgr: np.ndarray) -> str:
        name = f"{tag}.jpg"
        cv2.imwrite(str(run_dir / "frames" / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return name

    t_wait = time.time()
    while (cam.latest()[2] < 30 or imu_yaw_ccw() is None) and time.time() - t_wait < 15:
        time.sleep(0.1)
    if cam.latest()[2] < 30:
        print("camera never warmed up", flush=True)
        return 2
    if imu_yaw_ccw() is None:
        print("no lcm_imu_data on the LCM bus", flush=True)
        return 2
    print(f"run_dir={run_dir} dry={args.dry} probe={args.probe}", flush=True)

    try:
        if args.probe:
            return run_probe(args, cam, sender, imu_yaw_ccw, save, log, stop)
        return run_walk(args, cam, sender, imu_yaw_ccw, save, log, stop)
    finally:
        sender.stop()
        cam.close()
        log.close()
        print(f"done; frames in {run_dir / 'frames'}", flush=True)


def run_probe(args, cam, sender, imu_yaw_ccw, save, log, stop) -> int:
    """Fixed moves: still, forward, still, yaw left, still. Logs yaw and frames."""
    steps = [
        ("still0", 0.0, 0.0, 1.5),
        ("fwd", 0.15, 0.0, 3.0),
        ("still1", 0.0, 0.0, 1.5),
        ("yaw_left", 0.0, 0.25, 2.5),
        ("still2", 0.0, 0.0, 1.5),
    ]
    for name, vx, vyaw, dur in steps:
        t0 = time.time()
        yaw0 = imu_yaw_ccw()
        accepted_ticks = ticks = 0
        while time.time() - t0 < dur and not stop["flag"]:
            sender.set(args.vx_sign * vx, args.yaw_cmd_sign * vyaw)
            time.sleep(0.1)
            ticks += 1
            accepted_ticks += int(sender.accepted)
        yaw1 = imu_yaw_ccw()
        bgr = cam.latest()[0]
        frame = save(f"probe_{name}", bgr) if bgr is not None else ""
        rec = {
            "t": time.time(), "probe": name, "vx": vx, "vyaw": vyaw,
            "yaw0": yaw0, "yaw1": yaw1,
            "dyaw_deg": None if yaw0 is None or yaw1 is None else math.degrees(wrap(yaw1 - yaw0)),
            "accepted_frac": accepted_ticks / max(1, ticks), "frame": frame,
            "bridge": sender.last_reply,
        }
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        if stop["flag"]:
            break
    return 0


def run_walk(args, cam, sender, imu_yaw_ccw, save, log, stop) -> int:
    from experiments.aerial.rl.consequence_model import load_checkpoint
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    pred = DepthMinPredictor.from_checkpoint(str(REPO / args.depth_ckpt), device=args.device)
    pred.reset()
    models = [load_checkpoint(str(REPO / p.strip()), device=args.device) for p in args.ckpts.split(",") if p.strip()]
    policy = TelloConsequencePolicy(
        models=models, device=args.device, near_m=args.near_m, slow_m=args.slow_m,
        surv_min=args.surv_min, shield_mode=args.shield,
    )
    print(f"policy ready: shield models={len(models)} mode={args.shield} surv_min={args.surv_min}", flush=True)

    start_ccw = imu_yaw_ccw()
    assert start_ccw is not None
    dist = 0.0
    last_t = time.time()
    t_start = last_t
    last_save = 0.0
    last_frame_n = -1
    period = 1.0 / args.hz
    reason = "max_s"
    while not stop["flag"]:
        t0 = time.time()
        if t0 - t_start > args.max_s:
            break
        bgr, f_at, f_n = cam.latest()
        ccw = imu_yaw_ccw()
        if bgr is None or t0 - f_at > 0.5 or f_n == last_frame_n or ccw is None:
            sender.set(0.0, 0.0)
            time.sleep(0.05)
            continue
        last_frame_n = f_n
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        d56 = depth56_from(pred, rgb)
        # Policy yaw follows Tello: clockwise positive.
        dec = policy.decide(
            rgb=rgb, height_m=None, yaw_rad=-ccw, takeoff_yaw_rad=-start_ccw,
            goal_bearing0_rad=0.0, depth56=d56,
        )
        _a, b, _c, d = dec.rc
        vx = args.vx_max * float(b) / FWD_STICK
        vyaw = -args.vyaw_max * float(d) / YAW_STICK
        now = time.time()
        if sender.accepted or args.dry:
            dist += max(0.0, vx) * (now - last_t) * math.cos(wrap(ccw - start_ccw))
        last_t = now
        sender.set(args.vx_sign * vx, args.yaw_cmd_sign * vyaw)
        frame = ""
        if now - last_save >= args.save_every_s:
            last_save = now
            frame = save(f"{now - t_start:07.2f}_{dec.family}", bgr)
        rec = {
            "t": now, "el": round(now - t_start, 2), "family": dec.family,
            "fwd": round(dec.fwd_m, 2), "left": round(dec.left_m, 2), "right": round(dec.right_m, 2),
            "bearing_deg": round(math.degrees(dec.bearing_rad), 1),
            "yaw_ccw_deg": round(math.degrees(wrap(ccw - start_ccw)), 1),
            "vx": round(vx, 3), "vyaw": round(vyaw, 3), "accepted": sender.accepted,
            "dist_m": round(dist, 2), "src": dec.depth_source, "note": dec.note, "frame": frame,
        }
        print(
            f"{rec['el']:6.1f}s {dec.family:9s} fwd={dec.fwd_m:4.2f} L={dec.left_m:4.2f} R={dec.right_m:4.2f} "
            f"yaw={rec['yaw_ccw_deg']:+6.1f} vx={vx:+.2f} vyaw={vyaw:+.2f} acc={int(sender.accepted)} "
            f"dist={dist:5.2f} | {dec.note}",
            flush=True,
        )
        log.write(json.dumps(rec) + "\n")
        if dist >= args.dist_m:
            reason = "dist"
            break
        time.sleep(max(0.0, period - (time.time() - t0)))
    else:
        reason = "signal"
    print(f"stop reason={reason} dist_est={dist:.2f} m", flush=True)
    log.write(json.dumps({"t": time.time(), "stop": reason, "dist_m": dist}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
