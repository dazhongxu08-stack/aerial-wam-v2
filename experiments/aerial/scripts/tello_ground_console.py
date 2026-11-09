#!/usr/bin/env python3
"""Tello ground console: live camera + keyboard RC (Orin).

Default is **ground-only** (no takeoff / no non-zero RC). Pass ``--allow-fly``
to unlock takeoff and stick input.

Example (on Orin, Tello Wi-Fi already up)::

  python -m experiments.aerial.scripts.tello_ground_console --host 127.0.0.1 --port 8090

From Mac (USB SSH)::

  ./experiments/aerial/scripts/mac_tello_ground_console.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("tello_ground_console")

_JPEG_QUALITY = 70
# Frames grabbed per decoded frame: higher drops more to keep latency low, lower records smoother video.
_GRAB_N = int(os.environ.get("TELLO_GRAB_N", "6"))

_wifi_lock = threading.Lock()
_wifi_rssi_dbm: Optional[int] = None
_wifi_noise_dbm: Optional[int] = None
_wifi_poll_started = False


def _mac_tello_rssi() -> Tuple[Optional[int], Optional[int]]:
    """Mac STA RSSI of the current Wi-Fi (Tello AP on en0)."""
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPAirPortDataType"],
            text=True,
            timeout=8,
        )
    except Exception:  # noqa: BLE001
        return None, None
    rssi = noise = None
    for line in out.splitlines():
        if "Signal / Noise:" not in line:
            continue
        nums = []
        for tok in line.replace("dBm", " ").replace("/", " ").split():
            try:
                nums.append(int(float(tok)))
            except ValueError:
                continue
        if len(nums) >= 2:
            rssi, noise = nums[0], nums[1]
        elif nums:
            rssi = nums[0]
        break
    return rssi, noise


def _wifi_poll_loop() -> None:
    global _wifi_rssi_dbm, _wifi_noise_dbm
    while True:
        rssi, noise = _mac_tello_rssi()
        with _wifi_lock:
            _wifi_rssi_dbm, _wifi_noise_dbm = rssi, noise
        time.sleep(2.0)


def _start_wifi_poll() -> None:
    global _wifi_poll_started
    if _wifi_poll_started:
        return
    _wifi_poll_started = True
    threading.Thread(target=_wifi_poll_loop, daemon=True, name="wifi-rssi").start()


def _wifi_status_fields(tello_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with _wifi_lock:
        rssi, noise = _wifi_rssi_dbm, _wifi_noise_dbm
    tello_snr = None
    if tello_state and "wifi" in tello_state:
        try:
            tello_snr = int(float(tello_state["wifi"]))
        except ValueError:
            tello_snr = None
    return {
        "wifi_rssi_dbm": rssi,
        "wifi_noise_dbm": noise,
        "tello_wifi_snr": tello_snr,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _encode_jpeg(bgr: np.ndarray, quality: int = _JPEG_QUALITY) -> bytes:
    import cv2  # type: ignore

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _placeholder_jpeg(text: str = "waiting for camera") -> bytes:
    import cv2  # type: ignore

    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(
        img,
        text,
        (40, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return _encode_jpeg(img, quality=70)


def _rejoin_tello_wifi(
    ssid: str = "TELLO-E9F873",
    iface_hint: str = "",
    ping_ip: str = "192.168.10.1",
) -> Tuple[bool, str]:
    """Best-effort: ensure Orin USB Wi-Fi is on the Tello AP and L3 is alive.

    Tello is a single-STA AP. If H12 (or anything else) joins the same SSID,
    Orin's SDK handshake fails until that client leaves and we rejoin.
    """
    sudo_pass = os.environ.get("ORIN_SUDO_PASS", os.environ.get("WSY_PASS", "cupcake777"))

    def _sudo(argv: list[str], timeout: float = 25.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sudo", "-S", "-p", "", *argv],
            input=sudo_pass + "\n",
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def _ping_ok() -> bool:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "1", ping_ip],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0

    if _ping_ok():
        return True, "already reachable"

    iface = iface_hint.strip()
    if not iface:
        try:
            out = subprocess.check_output(
                ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev"],
                text=True,
                timeout=5,
            )
            for line in out.splitlines():
                parts = line.split(":")
                if len(parts) < 4:
                    continue
                dev, typ, state, conn = parts[0], parts[1], parts[2], parts[3]
                if typ == "wifi" and "connected" in state and (
                    conn.upper().startswith("TELLO") or "TELLO" in conn.upper()
                ):
                    iface = dev
                    break
            if not iface:
                for line in out.splitlines():
                    parts = line.split(":")
                    if len(parts) >= 2 and parts[1] == "wifi" and parts[0].startswith("wlx"):
                        iface = parts[0]
                        break
        except Exception as exc:  # noqa: BLE001
            return False, f"nmcli list failed: {exc}"

    if not iface:
        # last resort known USB Realtek name
        for cand in ("wlx6c1ff783eb62",):
            if Path(f"/sys/class/net/{cand}").exists():
                iface = cand
                break
    if not iface:
        return False, "no wifi iface"

    steps = [
        ["nmcli", "connection", "down", "TELLO-USB"],
        ["nmcli", "device", "wifi", "rescan", "ifname", iface],
    ]
    for argv in steps:
        try:
            _sudo(argv, timeout=15)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)

    up = _sudo(
        ["nmcli", "connection", "up", "TELLO-USB", "ifname", iface],
        timeout=30,
    )
    if up.returncode != 0:
        up = _sudo(
            [
                "nmcli",
                "device",
                "wifi",
                "connect",
                ssid,
                "ifname",
                iface,
                "name",
                "TELLO-USB",
            ],
            timeout=40,
        )
    time.sleep(2.0)
    try:
        _sudo(["ip", "addr", "replace", "192.168.10.2/24", "dev", iface], timeout=10)
    except Exception:  # noqa: BLE001
        pass

    for _ in range(8):
        if _ping_ok():
            return True, f"rejoined {iface}"
        time.sleep(1.0)
    detail = (up.stdout or "") + (up.stderr or "")
    return False, f"ping {ping_ip} failed after rejoin on {iface}: {detail[-200:]}"


def _video_worker_main(port: int) -> int:
    """Subprocess entry: grab Tello UDP frames and write length-prefixed JPEGs to stdout.

    Isolation matters: OpenCV/ffmpeg often wedges after a few H.264 frames; the
    parent can kill this process and respawn a clean capture.
    """
    import cv2  # type: ignore

    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|framedrop;1",
    )
    urls = [
        f"udp://0.0.0.0:{int(port)}?overrun_nonfatal=1&fifo_size=50000000",
        f"udp://@0.0.0.0:{int(port)}",
    ]
    out = sys.stdout.buffer
    last_sig = None
    same = 0
    while True:
        cap = None
        for u in urls:
            c = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
            if not c.isOpened():
                c.release()
                continue
            try:
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
            try:
                c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 300)
            except Exception:  # noqa: BLE001
                pass
            ok, frame = c.read()
            if ok and frame is not None:
                cap = c
                break
            c.release()
        if cap is None:
            time.sleep(0.4)
            continue
        last_sig = None
        same = 0
        while True:
            got = False
            for _ in range(_GRAB_N):
                if not cap.grab():
                    break
                got = True
            frame = None
            if got:
                ok, frame = cap.retrieve()
                if not (ok and frame is not None):
                    frame = None
            if frame is None:
                ok, frame = cap.read()
                if not (ok and frame is not None):
                    frame = None
            if frame is None:
                same += 1
                if same > 40:
                    break
                time.sleep(0.02)
                continue
            # Downscale for UI bandwidth / encode speed.
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            patch = frame[::40, ::40]
            sig = (int(patch[:, :, 0].mean()), int(patch[:, :, 1].mean()), int(patch.sum()) % 1000003)
            if sig == last_sig:
                same += 1
                if same > 25:
                    break  # reopen capture inside worker
                continue
            last_sig = sig
            same = 0
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
            if not ok:
                continue
            payload = buf.tobytes()
            try:
                out.write(struct.pack(">I", len(payload)))
                out.write(payload)
                out.flush()
            except BrokenPipeError:
                cap.release()
                return 0
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)


class _SubprocessVideo:
    """Own OpenCV capture in a child process; kill+respawn on stall."""

    def __init__(self, port: int = 11111) -> None:
        self.port = int(port)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._latest: Optional[bytes] = None
        self._seq = 0
        self._last_ts = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._kill_proc()

    def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

    def _spawn(self) -> subprocess.Popen:
        # Re-exec this file as a video worker (avoid package-import fragility).
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--video-worker",
            str(self.port),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            bufsize=0,
        )

    def _run(self) -> None:
        while not self._stop:
            self._kill_proc()
            try:
                proc = self._spawn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("video worker spawn failed: %s", exc)
                time.sleep(1.0)
                continue
            self._proc = proc
            assert proc.stdout is not None
            logger.info("video worker pid=%s", proc.pid)
            stall_deadline = time.time() + 15.0
            try:
                while not self._stop and proc.poll() is None:
                    # Read 4-byte length with a soft stall watchdog via select-ish polling.
                    hdr = self._read_exact(proc.stdout, 4, stall_deadline)
                    if hdr is None:
                        logger.warning("video worker stall — killing pid=%s", proc.pid)
                        break
                    (nbytes,) = struct.unpack(">I", hdr)
                    if nbytes <= 0 or nbytes > 2_000_000:
                        break
                    stall_deadline = time.time() + 1.2
                    payload = self._read_exact(proc.stdout, nbytes, stall_deadline)
                    if payload is None:
                        logger.warning("video worker frame stall — killing pid=%s", proc.pid)
                        break
                    stall_deadline = time.time() + 1.2
                    with self._lock:
                        self._latest = payload
                        self._seq += 1
                        self._last_ts = time.time()
            finally:
                self._kill_proc()
            time.sleep(0.1)

    @staticmethod
    def _read_exact(stream, n: int, deadline: float) -> Optional[bytes]:
        import select

        buf = bytearray()
        fd = stream.fileno()
        while len(buf) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], min(0.2, remaining))
            if not ready:
                continue
            try:
                chunk = os.read(fd, n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def latest(self) -> Tuple[Optional[bytes], int, float]:
        with self._lock:
            return self._latest, self._seq, self._last_ts



_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Tello Ground Console</title>
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #3fb950;
    --warn: #d29922;
    --danger: #f85149;
    --border: #30363d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    min-height: 100vh;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap;
  }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
  .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--muted);
  }
  .badge.ok { color: var(--accent); border-color: #238636; }
  .badge.lock { color: var(--warn); border-color: #9e6a03; }
  .badge.fly { color: var(--danger); border-color: #da3633; }
  main {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.7fr);
    gap: 16px; padding: 16px; max-width: 1280px; margin: 0 auto;
  }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .video {
    background: #000; border: 1px solid var(--border); border-radius: 6px;
    overflow: hidden; aspect-ratio: 4/3; display: flex; align-items: center;
    justify-content: center;
  }
  .video img { width: 100%; height: 100%; object-fit: contain; background: #000; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 14px 16px;
  }
  .panel h2 { margin: 0 0 10px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .stat { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; font-size: 13px; }
  .stat div span { display: block; color: var(--muted); font-size: 11px; }
  .rc { margin-top: 12px; font-size: 13px; }
  .keys { margin-top: 12px; font-size: 12px; color: var(--muted); line-height: 1.55; }
  .keys kbd {
    display: inline-block; min-width: 1.4em; text-align: center;
    padding: 1px 5px; border: 1px solid var(--border); border-radius: 3px;
    background: #0d1117; color: var(--text); font-size: 11px;
  }
  .btns { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
  button {
    font: inherit; font-size: 12px; color: var(--text); background: #21262d;
    border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; cursor: pointer;
  }
  button:hover { background: #30363d; }
  button.danger { border-color: #da3633; color: #ffa198; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .log {
    margin-top: 12px; height: 120px; overflow: auto; font-size: 11px;
    color: var(--muted); border-top: 1px solid var(--border); padding-top: 8px;
  }
</style>
</head>
<body>
<header>
  <h1>Tello Ground Console</h1>
  <span id="modeBadge" class="badge lock">GROUND LOCK</span>
  <span id="pilotBadge" class="badge">MANUAL</span>
  <span id="linkBadge" class="badge">link…</span>
</header>
<main>
  <section class="video"><img id="cam" alt="tello camera"/></section>
  <aside class="panel">
    <h2>Telemetry</h2>
    <div class="stat">
      <div><span>battery</span><strong id="bat">—</strong></div>
      <div><span>height</span><strong id="h">—</strong></div>
      <div><span>yaw</span><strong id="yaw">—</strong></div>
      <div><span>tof</span><strong id="tof">—</strong></div>
      <div><span>temp</span><strong id="temp">—</strong></div>
      <div><span>airborne</span><strong id="air">—</strong></div>
    </div>
    <div class="rc">RC <strong id="rc">0 0 0 0</strong></div>
    <div class="keys">
      <div><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> 前后左右</div>
      <div><kbd>↑</kbd><kbd>↓</kbd> 升降 · <kbd>Q</kbd><kbd>E</kbd> 偏航</div>
      <div><kbd>Space</kbd> 悬停清零</div>
      <div><kbd>T</kbd> 起飞 · <kbd>L</kbd> 降落 · <kbd>Esc</kbd> 急停</div>
      <div style="margin-top:6px">默认地面锁：起飞与非零 RC 被拒绝。服务端加 <code>--allow-fly</code> 才解锁。</div>
    </div>
    <div class="btns">
      <button id="btnPilot" type="button">切手动 / Orin自动</button>
      <button id="btnHover" type="button">Hover (0)</button>
      <button id="btnLand" type="button">Land</button>
      <button id="btnEmerg" class="danger" type="button">EMERGENCY</button>
    </div>
    <div class="log" id="log"></div>
  </aside>
</main>
<script>
const pressed = new Set();
let allowFly = false;
let controlMode = "manual"; // manual | auto
let tickTimer = null;

function log(msg) {
  const el = document.getElementById("log");
  const t = new Date().toLocaleTimeString();
  el.textContent = `[${t}] ${msg}\\n` + el.textContent;
}

function sticksFromKeys() {
  let a = 0, b = 0, c = 0, d = 0;
  const v = 60;
  if (pressed.has("a") || pressed.has("arrowleft")) a -= v;
  if (pressed.has("d") || pressed.has("arrowright")) a += v;
  if (pressed.has("w")) b += v;
  if (pressed.has("s")) b -= v;
  if (pressed.has("arrowup")) c += v;
  if (pressed.has("arrowdown")) c -= v;
  if (pressed.has("q")) d -= v;
  if (pressed.has("e")) d += v;
  return {a, b, c, d};
}

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) log((j && j.error) || (path + " failed"));
  else if (j.msg) log(j.msg);
  return j;
}

function sendRc() {
  if (controlMode === "auto") return; // Orin owns sticks
  const s = sticksFromKeys();
  document.getElementById("rc").textContent = `${s.a} ${s.b} ${s.c} ${s.d}`;
  api("/api/rc", Object.assign({source: "manual"}, s));
}

function updatePilotBadge() {
  const el = document.getElementById("pilotBadge");
  const btn = document.getElementById("btnPilot");
  if (controlMode === "auto") {
    el.textContent = "ORIN AUTO";
    el.className = "badge fly";
    if (btn) btn.textContent = "当前: Orin自动 → 切手动";
  } else {
    el.textContent = "MANUAL";
    el.className = "badge ok";
    if (btn) btn.textContent = "当前: 手动 → 切Orin自动";
  }
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    allowFly = !!j.allow_fly;
    if (j.control_mode === "auto" || j.control_mode === "manual") {
      controlMode = j.control_mode;
      updatePilotBadge();
    }
    const mode = document.getElementById("modeBadge");
    if (allowFly) {
      mode.textContent = "FLY UNLOCKED";
      mode.className = "badge fly";
    } else {
      mode.textContent = "GROUND LOCK";
      mode.className = "badge lock";
    }
    const link = document.getElementById("linkBadge");
    if (j.ok) { link.textContent = "SDK ok"; link.className = "badge ok"; }
    else { link.textContent = "SDK fail"; link.className = "badge"; }
    document.getElementById("bat").textContent = j.battery != null ? (j.battery + "%") : "—";
    document.getElementById("h").textContent = j.height_cm != null ? (j.height_cm + " cm") : "—";
    document.getElementById("yaw").textContent = j.yaw != null ? (j.yaw + "°") : "—";
    document.getElementById("tof").textContent = j.tof != null ? (j.tof + " cm") : "—";
    document.getElementById("temp").textContent = j.temp != null ? (j.temp + "°C") : "—";
    document.getElementById("air").textContent = j.airborne ? "yes" : "no";
    if (Array.isArray(j.rc)) {
      document.getElementById("rc").textContent = j.rc.join(" ");
    }
  } catch (e) {
    document.getElementById("linkBadge").textContent = "status err";
  }
}

window.addEventListener("keydown", (ev) => {
  const k = ev.key.toLowerCase();
  if ([" ", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(k) || k.length === 1) {
    ev.preventDefault();
  }
  if (k === "t") { api("/api/cmd", {cmd: "takeoff"}); return; }
  if (k === "l") { api("/api/cmd", {cmd: "land"}); return; }
  if (k === "escape") { api("/api/cmd", {cmd: "emergency"}); return; }
  if (k === " " || k === "spacebar") {
    pressed.clear();
    api("/api/rc", {a:0,b:0,c:0,d:0});
    document.getElementById("rc").textContent = "0 0 0 0";
    return;
  }
  pressed.add(k);
  sendRc();
});
window.addEventListener("keyup", (ev) => {
  pressed.delete(ev.key.toLowerCase());
  sendRc();
});
window.addEventListener("blur", () => {
  pressed.clear();
  api("/api/rc", {a:0,b:0,c:0,d:0});
});

document.getElementById("btnPilot").onclick = async () => {
  const next = controlMode === "auto" ? "manual" : "auto";
  const j = await api("/api/control_mode", {mode: next});
  if (j && j.ok && (j.control_mode === "auto" || j.control_mode === "manual")) {
    controlMode = j.control_mode;
    pressed.clear();
    updatePilotBadge();
  }
};
document.getElementById("btnHover").onclick = () => {
  pressed.clear();
  api("/api/rc", {a:0,b:0,c:0,d:0, source: "manual"});
};
document.getElementById("btnLand").onclick = () => api("/api/cmd", {cmd: "land"});
document.getElementById("btnEmerg").onclick = () => api("/api/cmd", {cmd: "emergency"});
updatePilotBadge();

// Poll snapshots (more reliable than MJPEG through SSH tunnel / some browsers).
const cam = document.getElementById("cam");
function refreshCam() {
  cam.src = "/snapshot.jpg?t=" + Date.now();
}
cam.onload = () => { setTimeout(refreshCam, 80); };
cam.onerror = () => { setTimeout(refreshCam, 400); };
refreshCam();

refreshStatus();
setInterval(refreshStatus, 1000);
setInterval(() => { if (pressed.size) sendRc(); }, 100);
log("console ready");
</script>
</body>
</html>
"""


class TelloConsoleState:
    def __init__(
        self,
        *,
        tello_ip: str,
        local_ip: str = "",
        local_cmd_port: int,
        allow_fly: bool,
        stream_port: int,
    ) -> None:
        self.allow_fly = bool(allow_fly)
        # manual = H12/browser sticks; auto = Orin policy owns /api/rc (source=auto)
        self.control_mode = "manual"
        self.airborne = False
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = _placeholder_jpeg("starting…")
        self._stop = False
        self._rc = (0, 0, 0, 0)
        self._last_rc_ts = 0.0
        self._last_tello_rc_ts = 0.0
        # RC watchdog: the last stick is re-sent forever, so a dead controller would fly on blindly.
        # Hover once no new RC arrives for RC_HOVER_S; in auto mode land after RC_LAND_S.
        self._rc_hover_s = float(os.environ.get("TELLO_RC_HOVER_S", "1.5"))
        self._rc_land_s = float(os.environ.get("TELLO_RC_LAND_S", "4.0"))
        self._sdk_busy = False
        self.client = None  # type: ignore
        self._link_ok = False
        self._link_error = "connecting…"
        self._cap = None
        self._stream_port = int(stream_port)
        self._frame_seq = 0
        self._last_frame_ts = 0.0
        self._last_jpeg_sig = None
        self._video_worker = None  # TelloStreamCamera | legacy worker
        self._tello_ip = str(tello_ip)
        self._local_ip_arg = str(local_ip or "")
        self._local_cmd_port = int(local_cmd_port)
        self._reconnect_gate = threading.Lock()
        from experiments.aerial.scripts.tello_session_recorder import TelloSessionRecorder

        rec_root = Path(os.environ.get("TELLO_GS_LOG", str(_repo_root() / "artifacts" / "tello_gs")))
        self._recorder = TelloSessionRecorder(Path(rec_root))
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True, name="tello-record")
        self._record_thread.start()

        # HTTP must come up even if Tello/video is slow or down.
        _start_wifi_poll()
        self._boot_thread = threading.Thread(target=self._boot_tello, daemon=True, name="tello-boot")
        self._boot_thread.start()

    def _resolve_bind_ip(self, tello_ip: str) -> str:
        from experiments.aerial.rl.env.tello_client import guess_local_ip

        if self._local_ip_arg:
            return str(self._local_ip_arg)
        bind_ip = guess_local_ip(tello_ip)
        import subprocess

        try:
            out = subprocess.check_output(
                ["ip", "-4", "-o", "addr", "show", "up"],
                text=True,
                timeout=2,
            )
            for line in out.splitlines():
                parts = line.split()
                if "inet" in parts:
                    idx = parts.index("inet")
                    cand = parts[idx + 1].split("/")[0]
                    if cand.startswith("192.168.10."):
                        return cand
        except Exception:  # noqa: BLE001
            pass
        return bind_ip

    def _boot_tello(self) -> None:
        root = _repo_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from experiments.aerial.rl.env.tello_client import TelloClient

        while not self._stop:
            try:
                with self._lock:
                    self._latest_jpeg = _placeholder_jpeg("connecting to Tello…")
                    self._link_error = "connecting…"
                bind_ip = self._resolve_bind_ip(self._tello_ip)
                logger.info(
                    "Tello console local=%s cmd_port=%s allow_fly=%s",
                    bind_ip,
                    self._local_cmd_port,
                    self.allow_fly,
                )
                client = TelloClient(
                    local_ip=bind_ip,
                    tello_ip=self._tello_ip,
                    local_cmd_port=self._local_cmd_port,
                )
                if not client.connect(retries=4):
                    client.close()
                    with self._lock:
                        self._link_ok = False
                        self._link_error = "SDK handshake failed"
                        self._latest_jpeg = _placeholder_jpeg("Tello SDK fail — retrying")
                    time.sleep(3.0)
                    continue
                client.start_state_listener()
                time.sleep(0.2)
                # Do NOT streamon here — video loop binds PyAV first, then streamon.
                self.client = client
                with self._lock:
                    self._link_ok = True
                    self._link_error = ""
                    self._latest_jpeg = _placeholder_jpeg("waiting for camera…")
                self._jpeg_thread = threading.Thread(target=self._jpeg_file_loop, daemon=True)
                self._jpeg_thread.start()
                skip_cv = os.environ.get("TELLO_SKIP_VIDEO", "").strip().lower() in ("1", "true", "yes")
                if skip_cv:
                    logger.warning("TELLO_SKIP_VIDEO — OpenCV off; using sidecar JPEG if present")
                    try:
                        ok = bool(self.client.stream_on())
                        logger.info("streamon %s", "ok" if ok else "fail")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("streamon: %s", exc)
                else:
                    self._vid_thread = threading.Thread(target=self._video_loop, daemon=True)
                    self._vid_thread.start()
                self._rc_thread = threading.Thread(target=self._rc_keepalive_loop, daemon=True)
                self._rc_thread.start()
                logger.info("Tello link ready")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tello boot error: %s", exc)
                with self._lock:
                    self._link_ok = False
                    self._link_error = str(exc)
                    self._latest_jpeg = _placeholder_jpeg("link error — retrying")
                time.sleep(3.0)

    def _release_cap(self) -> None:
        if self._cap is None:
            return
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass
        self._cap = None

    def _open_capture(self, stream_port: int) -> None:
        import os

        import cv2  # type: ignore

        self._release_cap()
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "fflags;nobuffer|flags;low_delay|framedrop;1",
        )
        # Plain udp://@ worked previously for first frame on this Orin build.
        urls = [
            f"udp://@0.0.0.0:{int(stream_port)}",
            f"udp://0.0.0.0:{int(stream_port)}?overrun_nonfatal=1&fifo_size=50000000",
            f"udp://@0.0.0.0:{int(stream_port)}?overrun_nonfatal=1&fifo_size=50000000",
        ]
        for u in urls:
            if self._stop:
                return
            logger.info("video trying %s", u)
            try:
                cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
            except Exception as exc:  # noqa: BLE001
                logger.warning("video open error %s: %s", u, exc)
                continue
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
            # Kick a fresh keyframe after the socket is listening.
            if self.client is not None:
                try:
                    self.client.stream_off()
                    time.sleep(0.15)
                    self.client.stream_on()
                    logger.info("stream restart after capture open")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("stream restart failed: %s", exc)
            frame = None
            deadline = time.time() + 20.0
            while time.time() < deadline and not self._stop:
                ok, fr = cap.read()
                if ok and fr is not None:
                    frame = fr
                    break
                time.sleep(0.01)
            if frame is None:
                logger.warning("video no frame yet: %s", u)
                cap.release()
                continue
            jpg = _encode_jpeg(frame)
            self._cap = cap
            with self._lock:
                self._latest_jpeg = jpg
                self._frame_seq += 1
                self._last_frame_ts = time.time()
                self._last_jpeg_sig = hash(jpg[:64] + jpg[-64:])
            logger.info("video open %s shape=%s", u, frame.shape)
            return
        logger.warning("video not ready yet — will keep retrying in loop")
        time.sleep(1.0)

    def _read_fresh_frame(self):
        """Drain OpenCV/ffmpeg buffer so we show the latest Tello frame."""
        if self._cap is None:
            return None
        got = False
        for _ in range(8):
            if not self._cap.grab():
                break
            got = True
        if got:
            ok, frame = self._cap.retrieve()
            if ok and frame is not None:
                return frame
        ok, frame = self._cap.read()
        if ok and frame is not None:
            return frame
        return None

    def _video_loop(self) -> None:
        """Decode Tello UDP in a child process (PyAV+OpenCV ffmpeg crash on Mac)."""
        while not self._stop and self.client is None:
            time.sleep(0.3)
        if self._stop:
            return

        worker = _SubprocessVideo(port=self._stream_port)
        worker.start()
        self._video_worker = worker  # type: ignore[assignment]
        try:
            for attempt in range(5):
                try:
                    ok = bool(self.client.stream_on())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("streamon attempt %s error: %s", attempt, exc)
                    ok = False
                if ok:
                    logger.info("streamon ok (attempt %s)", attempt)
                    break
                time.sleep(0.6)
            else:
                logger.warning("streamon not acked — waiting for frames anyway")
            time.sleep(0.3)
            try:
                self.client.stream_on()
            except Exception:  # noqa: BLE001
                pass
            last_seq = -1
            while not self._stop:
                jpeg, seq, ts = worker.latest()
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    with self._lock:
                        self._latest_jpeg = jpeg
                        self._frame_seq = int(seq)
                        self._last_frame_ts = float(ts)
                    self._recorder.save_jpeg(jpeg, int(seq))
                time.sleep(0.02)
        finally:
            worker.stop()
            self._video_worker = None

    def _jpeg_file_loop(self) -> None:
        path = os.environ.get("TELLO_FRAME_JPG", "/tmp/tello_latest.jpg")
        last_mtime = 0.0
        while not self._stop:
            try:
                st = os.stat(path)
                if st.st_mtime != last_mtime and st.st_size > 2000:
                    last_mtime = st.st_mtime
                    with open(path, "rb") as f:
                        jpg = f.read()
                    with self._lock:
                        self._latest_jpeg = jpg
                        self._frame_seq += 1
                        self._last_frame_ts = time.time()
                        seq = self._frame_seq
                    self._recorder.save_jpeg(jpg, seq)
            except OSError:
                pass
            time.sleep(0.04)

    def _rc_keepalive_loop(self) -> None:
        """Re-send last RC ~10 Hz while airborne so sticks don't time out."""
        while not self._stop:
            time.sleep(0.1)
            if self.client is None:
                continue
            with self._lock:
                airborne = self.airborne
                rc = self._rc
                age = time.time() - self._last_rc_ts
            if not (airborne and self.allow_fly) or self._sdk_busy:
                continue
            if age > self._rc_hover_s and rc != (0, 0, 0, 0):
                with self._lock:
                    self._rc = rc = (0, 0, 0, 0)
                logger.warning("rc watchdog: no RC for %.1fs → hover", age)
                self._recorder.event("rc_watchdog_hover", {"age_s": round(age, 2)})
            if self.control_mode == "auto" and age > self._rc_land_s:
                logger.warning("rc watchdog: no RC for %.1fs in auto → land", age)
                self._recorder.event("rc_watchdog_land", {"age_s": round(age, 2)})
                with self._lock:
                    self._last_rc_ts = time.time()
                self.cmd("land")
                continue
            self.client.rc(*rc)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def status(self) -> Dict[str, Any]:
        if self.client is None:
            out = {
                "ok": False,
                "allow_fly": self.allow_fly,
                "control_mode": self.control_mode,
                "airborne": False,
                "battery": None,
                "height_cm": None,
                "yaw": None,
                "tof": None,
                "temp": None,
                "rc": list(self._rc),
                "state_age_s": None,
                "frame_seq": self._frame_seq,
                "frame_age_s": None,
                "error": self._link_error,
            }
            out.update(_wifi_status_fields())
            out["recording"] = self._recorder.snapshot()
            return out
        st = dict(self.client.state)
        bat = self.client.battery()
        h = self.client.height_cm()
        yaw = None
        tof = None
        temp = None
        try:
            if "yaw" in st:
                yaw = int(float(st["yaw"]))
            if "tof" in st:
                tof = int(float(st["tof"]))
            if "temph" in st:
                temp = int(float(st["temph"]))
            elif "templ" in st:
                temp = int(float(st["templ"]))
        except ValueError:
            pass
        out = {
            "ok": self.client.state_age_s < 3.0,
            "allow_fly": self.allow_fly,
            "control_mode": self.control_mode,
            "airborne": self.airborne,
            "battery": bat,
            "height_cm": h,
            "yaw": yaw,
            "tof": tof,
            "temp": temp,
            "rc": list(self._rc),
            "state_age_s": round(self.client.state_age_s, 2),
            "frame_seq": self._frame_seq,
            "frame_age_s": round(time.time() - self._last_frame_ts, 2) if self._last_frame_ts else None,
            "error": self._link_error,
        }
        out.update(_wifi_status_fields(st))
        # Raw SDK strings: vgx/vgy are dm/s in the yaw=0 world axes with the sign flipped.
        out["tello"] = {k: st.get(k) for k in ("vgx", "vgy", "vgz", "agx", "agy", "agz", "pitch", "roll")}
        out["recording"] = self._recorder.snapshot()
        return out

    def set_rc(
        self, a: int, b: int, c: int, d: int, *, source: str = "manual"
    ) -> Tuple[bool, str]:
        if self.client is None:
            return False, self._link_error or "Tello not connected"
        src = "auto" if str(source).strip().lower() == "auto" else "manual"
        mode = self.control_mode
        if mode == "auto" and src != "auto":
            return False, "control_mode=auto: manual RC ignored (H12/browser)"
        if mode == "manual" and src == "auto":
            return False, "control_mode=manual: auto RC ignored"
        a = max(-100, min(100, int(a)))
        b = max(-100, min(100, int(b)))
        c = max(-100, min(100, int(c)))
        d = max(-100, min(100, int(d)))
        if not self.allow_fly and (a, b, c, d) != (0, 0, 0, 0):
            with self._lock:
                self._rc = (0, 0, 0, 0)
            self.client.rc(0, 0, 0, 0)
            return False, "ground lock: non-zero RC blocked (start with --allow-fly)"
        with self._lock:
            self._rc = (a, b, c, d)
            self._last_rc_ts = time.time()
        now = time.time()
        if self.allow_fly and self.airborne:
            self.client.rc(a, b, c, d)
            self._last_tello_rc_ts = now
            return True, f"rc {a} {b} {c} {d} ({src})"
        if now - self._last_tello_rc_ts > 0.4:
            self.client.rc(0, 0, 0, 0)
            self._last_tello_rc_ts = now
        if not self.airborne and (a, b, c, d) != (0, 0, 0, 0):
            return True, "rc armed for after takeoff (currently grounded)"
        return True, "rc 0 0 0 0"

    def set_control_mode(self, mode: Optional[str] = None) -> Tuple[bool, str]:
        """Switch stick ownership. ``mode=None`` toggles manual↔auto."""
        if mode is None:
            mode = "auto" if self.control_mode == "manual" else "manual"
        m = str(mode).strip().lower()
        if m not in ("manual", "auto"):
            return False, f"unknown control_mode {mode!r} (want manual|auto)"
        prev = self.control_mode
        self.control_mode = m
        # Always clear sticks on handoff so neither side leaves a latch
        with self._lock:
            self._rc = (0, 0, 0, 0)
        if self.client is not None:
            try:
                self.client.rc(0, 0, 0, 0)
            except Exception:  # noqa: BLE001
                pass
        logger.warning("control_mode %s → %s", prev, m)
        return True, f"control_mode={m}"

    def cmd(self, name: str) -> Tuple[bool, str]:
        if self.client is None:
            return False, self._link_error or "Tello not connected"
        name = str(name).strip().lower()
        if name == "takeoff":
            if not self.allow_fly:
                return False, "ground lock: takeoff blocked"
            resp = self.client.takeoff()
            ok = resp == "ok"
            if ok:
                with self._lock:
                    self._last_rc_ts = time.time()
                self.airborne = True
            self._recorder.event("takeoff", {"ok": ok, "resp": resp})
            return ok, f"takeoff → {resp}"
        if name == "land":
            resp = self.client.land()
            ok = resp == "ok"
            if ok:
                self.airborne = False
            self._recorder.event("land", {"ok": ok, "resp": resp})
            with self._lock:
                self._rc = (0, 0, 0, 0)
            self.client.rc(0, 0, 0, 0)
            return ok, f"land → {resp}"
        if name == "emergency":
            self.client.emergency()
            self.airborne = False
            self._recorder.event("emergency")
            with self._lock:
                self._rc = (0, 0, 0, 0)
            return True, "emergency sent"
        if name == "streamon":
            ok = bool(self.client.stream_on())
            return ok, f"streamon → {'ok' if ok else 'fail'}"
        if name == "streamoff":
            try:
                self.client.stream_off()
            except Exception as exc:  # noqa: BLE001
                return False, f"streamoff → {exc}"
            return True, "streamoff → ok"
        # Mission-style moves: "up 20" / "down 20" / "forward 30" … (cm, Tello min usually 20)
        move_cmds = ("up", "down", "forward", "back", "left", "right", "cw", "ccw")
        parts = name.split()
        if parts and parts[0] in move_cmds:
            if not self.allow_fly:
                return False, "ground lock: move blocked"
            if len(parts) == 1:
                parts = [parts[0], "20"]
            try:
                cm = int(float(parts[1]))
            except ValueError:
                return False, f"bad distance in {name!r}"
            cm = max(20, min(500, cm))
            sdk = f"{parts[0]} {cm}"
            self._sdk_busy = True
            try:
                resp = self.client.send(sdk, timeout=20.0)
            finally:
                with self._lock:
                    self._last_rc_ts = time.time()
                self._sdk_busy = False
            ok = resp == "ok"
            return ok, f"{sdk} → {resp}"
        return False, f"unknown cmd {name}"

    def set_allow_fly(self, enabled: Optional[bool] = None) -> Tuple[bool, str]:
        """Enable/disable ground lock at runtime. ``enabled=None`` toggles."""
        if enabled is None:
            enabled = not bool(self.allow_fly)
        self.allow_fly = bool(enabled)
        if not self.allow_fly:
            with self._lock:
                self._rc = (0, 0, 0, 0)
            if self.client is not None:
                try:
                    self.client.rc(0, 0, 0, 0)
                except Exception:  # noqa: BLE001
                    pass
            logger.info("GROUND LOCK on (allow_fly=False)")
            return True, "ground lock ON"
        logger.warning("FLY UNLOCKED (allow_fly=True)")
        return True, "fly unlocked"

    def set_recording(self, enabled: Optional[bool] = None) -> Dict[str, Any]:
        if enabled is None:
            enabled = not self._recorder.active
        if enabled:
            return self._recorder.start()
        return self._recorder.stop()

    def _record_loop(self) -> None:
        last_t: Optional[float] = None
        while True:
            time.sleep(0.05)
            if not self._recorder.active:
                last_t = None
                continue
            now = time.time()
            dt = 0.05 if last_t is None else max(1e-3, now - last_t)
            last_t = now
            st = dict(self.client.state) if self.client is not None else {}
            with self._lock:
                rc = self._rc
                seq = self._frame_seq
            wifi = _wifi_status_fields(st)
            try:
                self._recorder.sample(
                    tello_state=st,
                    rc=rc,
                    frame_seq=seq,
                    airborne=self.airborne,
                    allow_fly=self.allow_fly,
                    control_mode=self.control_mode,
                    wifi_rssi_dbm=wifi.get("wifi_rssi_dbm"),
                    wifi_noise_dbm=wifi.get("wifi_noise_dbm"),
                    dt=dt,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("record sample: %s", exc)

    def reconnect(self) -> Tuple[bool, str]:
        """Tear down Tello SDK + video, rejoin Wi-Fi if needed, boot again (async).

        Used by H12 ``重启控制台`` via ``POST /api/reconnect``.
        Restarts Tello SDK/video and rejoins TELLO Wi-Fi (HTTP server stays up).
        """
        if not self._reconnect_gate.acquire(blocking=False):
            return False, "reconnect already in progress"

        def _worker() -> None:
            try:
                logger.info("Tello reconnect: shutting down link/video")
                with self._lock:
                    self._link_ok = False
                    self._link_error = "reconnecting…"
                    self._latest_jpeg = _placeholder_jpeg("reconnecting…")
                    self._rc = (0, 0, 0, 0)
                self.airborne = False
                self._stop = True
                time.sleep(0.9)

                vw = getattr(self, "_video_worker", None)
                if vw is not None:
                    try:
                        if hasattr(vw, "stop"):
                            vw.stop()
                        elif hasattr(vw, "close"):
                            vw.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._video_worker = None

                client = self.client
                self.client = None
                if client is not None:
                    try:
                        client.rc(0, 0, 0, 0)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        client.stream_off()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass

                self._release_cap()
                self._frame_seq = 0
                self._last_frame_ts = 0.0

                # H12 (or other STA) may have stolen the Tello AP — rejoin first.
                with self._lock:
                    self._link_error = "rejoining Wi-Fi…"
                    self._latest_jpeg = _placeholder_jpeg("rejoining Tello Wi-Fi…")
                wifi_ok, wifi_msg = _rejoin_tello_wifi()
                logger.info("Tello reconnect wifi: ok=%s %s", wifi_ok, wifi_msg)
                if not wifi_ok:
                    with self._lock:
                        self._link_ok = False
                        self._link_error = f"wifi rejoin failed: {wifi_msg}"
                        self._latest_jpeg = _placeholder_jpeg("Wi-Fi rejoin failed")
                    # still try boot — AP might recover

                self._stop = False
                logger.info("Tello reconnect: booting again")
                self._boot_tello()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tello reconnect failed: %s", exc)
                with self._lock:
                    self._link_ok = False
                    self._link_error = f"reconnect failed: {exc}"
                    self._latest_jpeg = _placeholder_jpeg("reconnect failed")
            finally:
                try:
                    self._reconnect_gate.release()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=_worker, daemon=True, name="tello-reconnect").start()
        return True, "orin reconnect started"

    def close(self) -> None:
        self._stop = True
        try:
            if self._recorder.active:
                self._recorder.stop()
        except Exception:  # noqa: BLE001
            pass
        vw = getattr(self, "_video_worker", None)
        if vw is not None:
            try:
                if hasattr(vw, "stop"):
                    vw.stop()
                elif hasattr(vw, "close"):
                    vw.close()
            except Exception:  # noqa: BLE001
                pass
            self._video_worker = None
        try:
            if self.client is not None:
                self.client.rc(0, 0, 0, 0)
                if self.airborne:
                    self.client.land()
                self.client.stream_off()
                self.client.close()
        except Exception:  # noqa: BLE001
            pass
        self._release_cap()
        self.client = None


def _make_handler(state: TelloConsoleState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TelloGroundConsole/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            msg = fmt % args
            if "/snapshot.jpg" in msg or "/api/rc" in msg:
                return
            logger.info("%s - %s", self.address_string(), msg)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = _INDEX_HTML.encode("utf-8")
                self._send(200, "text/html; charset=utf-8", body)
                return
            if path == "/api/status":
                payload = json.dumps(state.status()).encode("utf-8")
                self._send(200, "application/json", payload)
                return
            if path == "/snapshot.jpg":
                jpg = state.latest_jpeg()
                if not jpg:
                    self.send_error(503, "no frame")
                    return
                self._send(200, "image/jpeg", jpg)
                return
            if path == "/stream.mjpg":
                self._mjpeg()
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                data = {}
            if path == "/api/rc":
                ok, msg = state.set_rc(
                    int(data.get("a", 0)),
                    int(data.get("b", 0)),
                    int(data.get("c", 0)),
                    int(data.get("d", 0)),
                    source=str(data.get("source", "manual") or "manual"),
                )
                code = 200 if ok else 403
                self._send(code, "application/json", json.dumps({"ok": ok, "msg": msg}).encode())
                return
            if path == "/api/cmd":
                ok, msg = state.cmd(str(data.get("cmd", "")))
                code = 200 if ok else 403
                self._send(code, "application/json", json.dumps({"ok": ok, "msg": msg}).encode())
                return
            if path == "/api/reconnect" or path == "/api/restart_console":
                ok, msg = state.reconnect()
                code = 200 if ok else 409
                self._send(code, "application/json", json.dumps({"ok": ok, "msg": msg}).encode())
                return
            if path == "/api/control_mode" or path == "/api/pilot_mode":
                if "mode" in data:
                    mode = data.get("mode")
                elif "control_mode" in data:
                    mode = data.get("control_mode")
                else:
                    mode = None  # toggle
                ok, msg = state.set_control_mode(mode)
                code = 200 if ok else 400
                self._send(
                    code,
                    "application/json",
                    json.dumps(
                        {
                            "ok": ok,
                            "msg": msg,
                            "control_mode": str(state.control_mode),
                        }
                    ).encode(),
                )
                return
            if path == "/api/record":
                if "on" in data:
                    enabled = bool(data.get("on"))
                elif "enabled" in data:
                    enabled = bool(data.get("enabled"))
                else:
                    enabled = None
                rec = state.set_recording(enabled)
                self._send(200, "application/json", json.dumps(rec).encode())
                return
            if path == "/api/allow_fly":
                if "allow_fly" in data:
                    enabled = bool(data.get("allow_fly"))
                elif "enabled" in data:
                    enabled = bool(data.get("enabled"))
                else:
                    enabled = None  # toggle
                ok, msg = state.set_allow_fly(enabled)
                self._send(
                    200,
                    "application/json",
                    json.dumps(
                        {"ok": ok, "msg": msg, "allow_fly": bool(state.allow_fly)}
                    ).encode(),
                )
                return
            self.send_error(404)

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg(self) -> None:
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = state.latest_jpeg()
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(1.0 / 12.0)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--video-worker":
        return int(_video_worker_main(int(sys.argv[2])))

    p = argparse.ArgumentParser(description="Tello ground console (camera + keyboard)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--tello-ip", default="192.168.10.1")
    p.add_argument(
        "--local-ip",
        default="",
        help="Bind IP on Tello subnet (default: auto; prefer 192.168.10.x over USB bridge)",
    )
    p.add_argument("--local-cmd-port", type=int, default=9000)
    p.add_argument(
        "--allow-fly",
        action="store_true",
        help="Unlock takeoff and non-zero RC (default: ground lock)",
    )
    args = p.parse_args()

    if args.allow_fly:
        logger.warning("FLY UNLOCKED — takeoff and RC enabled")
    else:
        logger.info("GROUND LOCK — camera + telemetry only; takeoff/RC blocked")

    state = TelloConsoleState(
        tello_ip=str(args.tello_ip),
        local_ip=str(args.local_ip or ""),
        local_cmd_port=int(args.local_cmd_port),
        allow_fly=bool(args.allow_fly),
        stream_port=11111,
    )
    handler = _make_handler(state)
    httpd = ThreadingHTTPServer((args.host, int(args.port)), handler)
    httpd.allow_reuse_address = True
    logger.info("open http://%s:%d/  (tunnel from Mac if needed)", args.host, int(args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
