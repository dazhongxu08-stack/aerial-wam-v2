"""Tello H.264 UDP stream → BGR/RGB frames (PyAV path used on 125).

Same approach as ``robomaster-tt-control/tt_control/video_stream.py``:
bind a plain UDP socket, feed packets into ``av.CodecContext`` H.264, publish
the latest BGR frame. Avoids OpenCV's ffmpeg UDP demuxer (SPS/PPS / freeze).
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelloStreamCameraConfig:
    port: int = 11111
    local_ip: str = "0.0.0.0"
    width: int = 960
    height: int = 720
    wam_size: int = 224
    open_timeout_s: float = 15.0
    stale_restart_s: float = 2.0
    # Kept for API compatibility with callers; PyAV path rarely needs respawn.
    restart_cooldown_s: float = 5.0


class TelloStreamCamera:
    """Latest-frame camera via PyAV H.264 over UDP (125-proven path)."""

    def __init__(
        self,
        port: int = 11111,
        local_ip: str = "0.0.0.0",
        width: int = 960,
        height: int = 720,
        wam_size: int = 224,
        open_timeout_s: float = 15.0,
        stale_restart_s: float = 2.0,
        restart_cooldown_s: float = 5.0,
    ) -> None:
        self.config = TelloStreamCameraConfig(
            port=port,
            local_ip=local_ip,
            width=width,
            height=height,
            wam_size=wam_size,
            open_timeout_s=open_timeout_s,
            stale_restart_s=stale_restart_s,
            restart_cooldown_s=restart_cooldown_s,
        )
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._last_bgr: Optional[np.ndarray] = None
        self._last_seq = 0
        self._last_ts = 0.0
        self._consumed_seq = 0
        self._started_at = 0.0
        self._decode_errors = 0
        self._pkt_count = 0
        self._byte_count = 0

    def start_worker(self) -> None:
        """Bind UDP and start decode thread. Caller should ``streamon``, then ``wait_ready``."""
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        sock.bind((str(self.config.local_ip), int(self.config.port)))
        sock.settimeout(1.0)
        self._sock = sock
        self._running = True
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, name="tello-pyav", daemon=True)
        self._thread.start()
        logger.info(
            "Tello PyAV listen %s:%s",
            self.config.local_ip,
            self.config.port,
        )

    def open(self) -> None:
        """Bind + wait for first frame (assumes stream already on or arriving)."""
        self.start_worker()
        self.wait_ready()

    def wait_ready(self) -> None:
        if self._sock is None or not self._running:
            raise RuntimeError("camera not started")
        deadline = time.time() + float(self.config.open_timeout_s)
        while time.time() < deadline:
            with self._lock:
                seq = self._last_seq
                bgr = self._last_bgr
                ts = self._last_ts
            if bgr is not None and seq > 0:
                logger.info(
                    "TelloStreamCamera ready seq=%s shape=%s",
                    seq,
                    bgr.shape,
                )
                return
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError("Tello PyAV decode thread exited")
            time.sleep(0.05)
        raise RuntimeError("Tello stream not ready (PyAV produced no frame)")

    def _loop(self) -> None:
        try:
            import av  # type: ignore
        except ImportError as exc:
            logger.error("PyAV (av) not installed: %s", exc)
            self._running = False
            return

        codec = av.CodecContext.create("h264", "r")
        assert self._sock is not None
        last_log = time.time()
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                now = time.time()
                if now - last_log >= 2.0:
                    logger.info(
                        "Tello PyAV waiting pkts=%s bytes=%s seq=%s err=%s",
                        self._pkt_count,
                        self._byte_count,
                        self._last_seq,
                        self._decode_errors,
                    )
                    last_log = now
                continue
            except OSError:
                break
            if not data:
                continue
            self._pkt_count += 1
            self._byte_count += len(data)
            try:
                packets = codec.parse(data)
            except Exception:  # noqa: BLE001
                self._decode_errors += 1
                continue
            for packet in packets:
                try:
                    frames = codec.decode(packet)
                except Exception:  # noqa: BLE001
                    self._decode_errors += 1
                    continue
                for frame in frames:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                    except Exception:  # noqa: BLE001
                        self._decode_errors += 1
                        continue
                    now = time.time()
                    with self._lock:
                        self._last_bgr = img
                        self._last_seq += 1
                        self._last_ts = now

    def _ensure_fresh(self) -> None:
        """Restart only if the decode thread died or frames stopped for a long time."""
        age = time.time() - self._last_ts if self._last_ts else 999.0
        alive = self._thread is not None and self._thread.is_alive()
        since = time.time() - self._started_at if self._started_at else 999.0
        cooldown = max(float(self.config.restart_cooldown_s), 8.0)
        stale = max(float(self.config.stale_restart_s), 5.0)
        if since < cooldown:
            return
        if alive and (not self._last_ts) and since < float(self.config.open_timeout_s):
            return
        if alive and self._last_ts and age <= stale:
            return
        if alive and age <= stale:
            return
        logger.warning(
            "Tello PyAV stale age=%.2fs alive=%s — restarting listener",
            age,
            alive,
        )
        port = int(self.config.port)
        lip = str(self.config.local_ip)
        self.close()
        self.config.port = port
        self.config.local_ip = lip
        self.start_worker()

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        import cv2  # type: ignore

        with self._lock:
            bgr = None if self._last_bgr is None else self._last_bgr.copy()
            seq = self._last_seq
        if bgr is None:
            self._ensure_fresh()
            with self._lock:
                bgr = None if self._last_bgr is None else self._last_bgr.copy()
                seq = self._last_seq
        if bgr is None:
            raise RuntimeError("no Tello video frame")
        self._consumed_seq = seq
        h, w = int(self.config.wam_size), int(self.config.wam_size)
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb_full, (w, h), interpolation=cv2.INTER_AREA)
        return bgr, np.ascontiguousarray(rgb, dtype=np.uint8)

    def read_bgr_new(self) -> Optional[np.ndarray]:
        """Return a new BGR frame if seq advanced; else None."""
        with self._lock:
            seq = self._last_seq
            bgr = self._last_bgr
            ts = self._last_ts
        if bgr is None:
            age = time.time() - ts if ts else 999.0
            if age > float(self.config.stale_restart_s):
                self._ensure_fresh()
            return None
        if seq == self._consumed_seq:
            age = time.time() - ts if ts else 999.0
            if age > float(self.config.stale_restart_s):
                self._ensure_fresh()
            return None
        self._consumed_seq = seq
        return bgr.copy()

    @property
    def frame_seq(self) -> int:
        with self._lock:
            return int(self._last_seq)

    @property
    def frame_age_s(self) -> float:
        with self._lock:
            ts = self._last_ts
        if not ts:
            return float("inf")
        return max(0.0, time.time() - ts)

    def close(self) -> None:
        self._running = False
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass
        th = self._thread
        self._thread = None
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        with self._lock:
            self._last_bgr = None
            self._last_seq = 0
            self._last_ts = 0.0
        self._consumed_seq = 0


def main(argv: Optional[list] = None) -> int:
    import sys

    _ = list(sys.argv[1:] if argv is None else argv)
    print(
        "TelloStreamCamera now uses in-process PyAV. "
        "Import TelloStreamCamera and call start_worker()/wait_ready().",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
