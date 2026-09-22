"""Tello UDP SDK 3.0 client (adapted from robomaster-tt-control)."""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def guess_local_ip(tello_ip: str = "192.168.10.1") -> str:
    """Pick the local interface IP that can reach the Tello AP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((tello_ip, 8889))
        return str(sock.getsockname()[0])
    except OSError:
        return "0.0.0.0"
    finally:
        sock.close()


class TelloClient:
    def __init__(
        self,
        local_ip: str,
        tello_ip: str = "192.168.10.1",
        cmd_port: int = 8889,
        state_port: int = 8890,
        local_cmd_port: Optional[int] = None,
    ) -> None:
        self.local_ip = local_ip
        self.tello_addr = (tello_ip, cmd_port)
        self.state_port = state_port
        # Local bind port for commands. Defaults to cmd_port (8889). If another
        # process already handshake'd from a different port (e.g. 9000), Tello
        # locks replies there — pass that port until the aircraft is power-cycled.
        bind_cmd = int(cmd_port if local_cmd_port is None else local_cmd_port)

        self._cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd.bind((local_ip, bind_cmd))
        self._cmd.settimeout(5.0)

        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind((local_ip, state_port))
        self._state_sock.settimeout(1.0)

        self._lock = threading.Lock()
        self._running = False
        self._state_thread: Optional[threading.Thread] = None
        self._ka_thread: Optional[threading.Thread] = None
        self.state: dict[str, str] = {}
        self._last_state_ts: float = 0.0
        self._on_state: Optional[Callable[[dict[str, str]], None]] = None

    def start_state_listener(
        self, on_state: Optional[Callable[[dict[str, str]], None]] = None
    ) -> None:
        self._on_state = on_state
        self._running = True
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()
        self._ka_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ka_thread.start()

    def _keepalive_loop(self, interval: float = 5.0) -> None:
        while self._running:
            slept = 0.0
            while slept < interval:
                if not self._running:
                    return
                time.sleep(0.5)
                slept += 0.5
            with self._lock:
                try:
                    self._cmd.sendto(b"command", self.tello_addr)
                except OSError:
                    pass

    def _state_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._state_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            text = data.decode(errors="ignore").strip()
            parsed: dict[str, str] = {}
            for part in text.split(";"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    parsed[k] = v
            if parsed:
                self.state = parsed
                self._last_state_ts = time.time()
                if self._on_state:
                    self._on_state(parsed)

    def _drain_stale(self) -> None:
        self._cmd.settimeout(0.0)
        try:
            while True:
                self._cmd.recvfrom(2048)
        except (BlockingIOError, socket.timeout, OSError):
            pass

    def send(
        self, cmd: str, wait_response: bool = True, timeout: float = 5.0
    ) -> Optional[str]:
        with self._lock:
            logger.info(">>> %s", cmd)
            self._drain_stale()
            self._cmd.settimeout(timeout)
            try:
                self._cmd.sendto(cmd.encode("utf-8"), self.tello_addr)
            except OSError as e:
                logger.error("send failed: %s", e)
                return None
            if not wait_response:
                return None
            try:
                data, _ = self._cmd.recvfrom(2048)
                resp = data.decode(errors="ignore").strip()
                logger.info("<<< %s", resp)
                return resp
            except socket.timeout:
                logger.warning("timeout: %s", cmd)
                return None

    @property
    def state_age_s(self) -> float:
        if not self._last_state_ts:
            return float("inf")
        return time.time() - self._last_state_ts

    def connect(self, retries: int = 4) -> bool:
        for _ in range(max(1, retries)):
            if self.send("command", timeout=2.5) == "ok":
                return True
        return False

    def stream_on(self) -> bool:
        return self.send("streamon", timeout=5.0) == "ok"

    def stream_off(self) -> None:
        self.send("streamoff", timeout=3.0)

    def takeoff(self) -> Optional[str]:
        return self.send("takeoff", timeout=20.0)

    def land(self) -> Optional[str]:
        return self.send("land", timeout=20.0)

    def emergency(self) -> None:
        logger.warning(">>> emergency (bypass lock)")
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.local_ip, 0))
            sock.sendto(b"emergency", self.tello_addr)
        except OSError as e:
            logger.error("emergency send failed: %s", e)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def rc(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> None:
        """a=left/right, b=fwd/back, c=up/down, d=yaw; no ack."""
        a = max(-100, min(100, int(a)))
        b = max(-100, min(100, int(b)))
        c = max(-100, min(100, int(c)))
        d = max(-100, min(100, int(d)))
        self.send(f"rc {a} {b} {c} {d}", wait_response=False)

    def battery(self) -> Optional[int]:
        raw = self.state.get("bat")
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
        resp = self.send("battery?", timeout=3.0)
        try:
            return int(resp) if resp else None
        except ValueError:
            return None

    def height_cm(self) -> Optional[int]:
        raw = self.state.get("h")
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def close(self) -> None:
        self._running = False
        if self._state_thread and self._state_thread.is_alive():
            self._state_thread.join(timeout=1.0)
        if self._ka_thread and self._ka_thread.is_alive():
            self._ka_thread.join(timeout=1.0)
        for s in (self._cmd, self._state_sock):
            try:
                s.close()
            except OSError:
                pass
