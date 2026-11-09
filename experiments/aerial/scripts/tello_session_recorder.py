#!/usr/bin/env python3
"""Save a Tello ground-station session: JPEG frames, telemetry, integrated trajectory."""
from __future__ import annotations

import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


def _f(state: Dict[str, Any], key: str) -> Optional[float]:
    raw = state.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class TelloSessionRecorder:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self.active = False
        self.run_id = ""
        self.dir: Optional[Path] = None
        self.n_frames = 0
        self.n_samples = 0
        self.started_s = 0.0
        self._x = 0.0
        self._y = 0.0
        self._tel: Optional[TextIO] = None
        self._evt: Optional[TextIO] = None
        self._csv: Optional[TextIO] = None
        self._csv_w: Optional[csv.writer] = None
        self._last_frame_seq = -1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            dur = (time.time() - self.started_s) if self.active else 0.0
            return {
                "on": self.active,
                "run_id": self.run_id or None,
                "dir": str(self.dir) if self.dir else None,
                "n_frames": self.n_frames,
                "n_samples": self.n_samples,
                "duration_s": round(dur, 2),
            }

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self.active:
                return {"ok": False, "msg": "already recording", **self.snapshot()}
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.root / f"run_{stamp}"
            (run_dir / "frames").mkdir(parents=True, exist_ok=True)
            tel = (run_dir / "telemetry.jsonl").open("w", encoding="utf-8")
            evt = (run_dir / "events.jsonl").open("w", encoding="utf-8")
            traj = (run_dir / "trajectory.csv").open("w", encoding="utf-8", newline="")
            writer = csv.writer(traj)
            writer.writerow(
                [
                    "t_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "yaw_deg",
                    "pitch_deg",
                    "roll_deg",
                    "vgx_dms",
                    "vgy_dms",
                    "vgz_dms",
                    "h_cm",
                    "tof_cm",
                    "baro_m",
                    "frame_seq",
                    "rc_a",
                    "rc_b",
                    "rc_c",
                    "rc_d",
                    "wifi_rssi_dbm",
                ]
            )
            meta = {
                "run_id": stamp,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "vg_units": "Tello SDK vgx/vgy/vgz are dm/s",
                "pose": "x/y = integral of -vg*0.1 (vg is in the yaw=0 world axes, sign-flipped); z from h_cm/100",
            }
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            self.active = True
            self.run_id = stamp
            self.dir = run_dir
            self.n_frames = 0
            self.n_samples = 0
            self.started_s = time.time()
            self._x = 0.0
            self._y = 0.0
            self._tel = tel
            self._evt = evt
            self._csv = traj
            self._csv_w = writer
            self._last_frame_seq = -1
            evt.write(json.dumps({"t_s": 0.0, "event": "record_start"}) + "\n")
            evt.flush()
            return {"ok": True, "msg": f"recording {run_dir}", **self.snapshot()}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self.active:
                return {"ok": False, "msg": "not recording", **self.snapshot()}
            snap = dict(self.snapshot())
            t_s = time.time() - self.started_s
            if self._evt is not None:
                self._evt.write(json.dumps({"t_s": round(t_s, 3), "event": "record_stop"}) + "\n")
            self._close_files_unlocked()
            if self.dir is not None:
                summary = {
                    "run_id": self.run_id,
                    "stopped_at": datetime.now(timezone.utc).isoformat(),
                    "duration_s": round(t_s, 3),
                    "n_frames": self.n_frames,
                    "n_samples": self.n_samples,
                    "dir": str(self.dir),
                }
                (self.dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
            self.active = False
            return {"ok": True, "msg": f"saved {snap.get('dir')}", **snap, "on": False}

    def event(self, name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if not self.active or self._evt is None:
                return
            rec = {"t_s": round(time.time() - self.started_s, 3), "event": name}
            if extra:
                rec.update(extra)
            self._evt.write(json.dumps(rec) + "\n")
            self._evt.flush()

    def save_jpeg(self, jpeg: bytes, frame_seq: int) -> None:
        with self._lock:
            if not self.active or self.dir is None:
                return
            if frame_seq == self._last_frame_seq:
                return
            self._last_frame_seq = frame_seq
            path = self.dir / "frames" / f"{frame_seq:06d}.jpg"
            path.write_bytes(jpeg)
            self.n_frames += 1

    def sample(
        self,
        *,
        tello_state: Dict[str, Any],
        rc: tuple,
        frame_seq: int,
        airborne: bool,
        allow_fly: bool,
        control_mode: str,
        wifi_rssi_dbm: Optional[int],
        wifi_noise_dbm: Optional[int],
        dt: float,
    ) -> None:
        with self._lock:
            if not self.active or self._tel is None or self._csv_w is None:
                return
            t_s = time.time() - self.started_s
            yaw = _f(tello_state, "yaw") or 0.0
            vgx = _f(tello_state, "vgx") or 0.0
            vgy = _f(tello_state, "vgy") or 0.0
            vgz = _f(tello_state, "vgz") or 0.0
            # Tello SDK vg* is dm/s in the yaw=0 world axes, sign-flipped: forward flight at yaw 0 reads
            # vgx < 0 and at yaw 90 reads vgy < 0 (run_20261016_133831).
            vx_e = -vgx * 0.1
            vy_e = -vgy * 0.1
            self._x += vx_e * dt
            self._y += vy_e * dt
            h_cm = _f(tello_state, "h")
            z_m = (h_cm / 100.0) if h_cm is not None else None
            row = {
                "t_s": round(t_s, 3),
                "frame_seq": frame_seq,
                "airborne": airborne,
                "allow_fly": allow_fly,
                "control_mode": control_mode,
                "rc": list(rc),
                "wifi_rssi_dbm": wifi_rssi_dbm,
                "wifi_noise_dbm": wifi_noise_dbm,
                "pose": {
                    "x_m": round(self._x, 4),
                    "y_m": round(self._y, 4),
                    "z_m": None if z_m is None else round(z_m, 4),
                    "yaw_deg": yaw,
                },
                "tello": dict(tello_state),
            }
            self._tel.write(json.dumps(row) + "\n")
            self._csv_w.writerow(
                [
                    round(t_s, 3),
                    round(self._x, 4),
                    round(self._y, 4),
                    "" if z_m is None else round(z_m, 4),
                    yaw,
                    _f(tello_state, "pitch"),
                    _f(tello_state, "roll"),
                    vgx,
                    vgy,
                    vgz,
                    h_cm,
                    _f(tello_state, "tof"),
                    _f(tello_state, "baro"),
                    frame_seq,
                    rc[0],
                    rc[1],
                    rc[2],
                    rc[3],
                    wifi_rssi_dbm,
                ]
            )
            self.n_samples += 1
            if self.n_samples % 20 == 0:
                self._tel.flush()
                if self._csv is not None:
                    self._csv.flush()

    def _close_files_unlocked(self) -> None:
        for fh in (self._tel, self._evt, self._csv):
            if fh is None:
                continue
            try:
                fh.flush()
                fh.close()
            except OSError:
                pass
        self._tel = None
        self._evt = None
        self._csv = None
        self._csv_w = None
