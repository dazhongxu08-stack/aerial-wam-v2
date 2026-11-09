#!/usr/bin/env python3
"""Small page: dispatch a chassis task and watch the front camera.

The browser talks only to this process. Robot calls are bound to the office
address so they are not sent out the phone hotspot.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROBOT = os.environ.get("YHS_ROBOT", "10.229.66.48")
GATEWAY = int(os.environ.get("YHS_GATEWAY_PORT", "9090"))
BIND = os.environ.get("YHS_BIND", "10.229.62.163")
PORT = int(os.environ.get("YHS_CONSOLE_PORT", "8765"))
# Off the office LAN the robot is reached through the bridge, not this address.
FPV_LOCAL = int(os.environ.get("YHS_FPV_PORT", "19100"))
GATEWAY_LOCAL = int(os.environ.get("YHS_GATEWAY_LOCAL", "19090"))
BRIDGE = os.environ.get("YHS_BRIDGE", "cursor-125-public")

_frame_lock = threading.Lock()
_frame = b""
_frame_at = 0.0
_tunnel: subprocess.Popen | None = None
_tunnel_at = 0.0


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>办公室底盘</title>
<style>
  body { margin: 0; font: 16px/1.4 sans-serif; background: #111; color: #eee; }
  main { max-width: 960px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 12px; }
  img { width: 100%; background: #000; border-radius: 8px; min-height: 360px; object-fit: contain; }
  #live { color: #888; font-size: 13px; margin: 8px 0 0; }
  form { display: flex; gap: 8px; margin-top: 12px; }
  input { flex: 1; font: inherit; padding: 10px 12px; border-radius: 8px; border: 1px solid #444; background: #1c1c1c; color: inherit; }
  button { font: inherit; padding: 10px 16px; border: 0; border-radius: 8px; background: #2f6fed; color: white; }
  button.stop { background: #a33; }
  pre { white-space: pre-wrap; color: #bbb; min-height: 3em; }
</style>
<main>
  <h1>第一视角</h1>
  <img id="view" alt="前置相机">
  <p id="live">画面连接中…</p>
  <form id="task">
    <input id="text" name="text" placeholder="任务，例如：向前巡视一圈" autocomplete="off" required>
    <button type="submit">下发</button>
    <button class="stop" type="button" id="stop">停止</button>
  </form>
  <pre id="out">等待任务</pre>
</main>
<script>
const out = document.getElementById("out");
const view = document.getElementById("view");
const live = document.getElementById("live");
let last = {};
let prevUrl = "";
async function pullFrame() {
  try {
    const res = await fetch("/jpg?t=" + Date.now(), {cache: "no-store"});
    if (!res.ok) throw new Error(String(res.status));
    const url = URL.createObjectURL(await res.blob());
    view.src = url;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
    prevUrl = url;
    live.textContent = "画面 " + new Date().toLocaleTimeString();
  } catch (err) {
    live.textContent = "画面中断，正在重连…";
  }
  setTimeout(pullFrame, 200);
}
pullFrame();
document.getElementById("task").onsubmit = async (ev) => {
  ev.preventDefault();
  const text = document.getElementById("text").value.trim();
  out.textContent = "下发中…";
  const res = await fetch("/api/task", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})});
  const body = await res.json();
  last = body;
  out.textContent = JSON.stringify(body, null, 2);
};
document.getElementById("stop").onclick = async () => {
  out.textContent = "停止中…";
  const res = await fetch("/api/stop", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(last.data || last)});
  out.textContent = JSON.stringify(await res.json(), null, 2);
};
</script>
"""


def _socket(host: str, port: int) -> socket.socket:
    sock = socket.socket()
    sock.settimeout(8)
    try:
        sock.bind((BIND, 0))
    except OSError:
        pass
    sock.connect((host, port))
    return sock


def _http(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = b"" if payload is None else json.dumps(payload).encode()
    last = "gateway unreachable"
    for _ in range(4):
        conn = None
        try:
            if _office_bind():
                conn = HTTPConnection(ROBOT, GATEWAY, timeout=20, source_address=(BIND, 0))
            else:
                _ensure_fpv_tunnel()
                conn = HTTPConnection("127.0.0.1", GATEWAY_LOCAL, timeout=20)
            conn.request(method, path, body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw = resp.read()
            try:
                data = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                data = {"raw": raw.decode("utf-8", "replace")}
            return resp.status, data
        except OSError as exc:
            last = str(exc)
            time.sleep(0.4)
        finally:
            if conn is not None:
                conn.close()
    return 503, {"ok": False, "detail": last}


def _find_id(value, key: str) -> str:
    if isinstance(value, dict):
        if value.get(key):
            return str(value[key])
        for item in value.values():
            found = _find_id(item, key)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_id(item, key)
            if found:
                return found
    return ""


def _zero_motion() -> str:
    ask = tempfile.NamedTemporaryFile("w", delete=False)
    ask.write("#!/bin/sh\necho 1\n")
    ask.close()
    os.chmod(ask.name, 0o700)
    remote = r"""
set +u
source /opt/ros/jazzy/setup.bash
source /home/user/work/robot-sensors/install/setup.bash
set -u
export ROS_LOCALHOST_ONLY=1
touch /tmp/relative_search.stop
rm -f /home/user/rt/search_goal.json
if [ -f /tmp/relative_search.pid ]; then
  kill "$(cat /tmp/relative_search.pid)" 2>/dev/null || true
fi
python3 - << 'END'
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
rclpy.init()
node = Node("console_stop")
pub = node.create_publisher(Twist, "/nav/cmd_vel", 10)
zero = Twist()
t0 = time.time()
while time.time() - t0 < 0.6:
    pub.publish(zero)
    rclpy.spin_once(node, timeout_sec=0.05)
node.destroy_node(); rclpy.shutdown()
print("zero")
END
"""
    env = os.environ.copy()
    env["SSH_ASKPASS"] = ask.name
    env["SSH_ASKPASS_REQUIRE"] = "force"
    try:
        if _office_bind():
            proc = subprocess.run(
                ["ssh", "-b", BIND, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=12",
                 "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
                 f"user@{ROBOT}", remote],
                env=env, capture_output=True, text=True, timeout=20,
            )
        else:
            hop = (
                "set +e\n"
                "cat > /tmp/rt_ask << 'ASK'\n#!/bin/sh\necho 1\nASK\n"
                "chmod 700 /tmp/rt_ask\n"
                "export SSH_ASKPASS=/tmp/rt_ask SSH_ASKPASS_REQUIRE=force DISPLAY=\"${DISPLAY:-:0}\"\n"
                f"ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no "
                f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 user@{ROBOT} 'bash -s' << 'REMOTE'\n"
                f"{remote}\nREMOTE\n"
                "rm -f /tmp/rt_ask\n"
            )
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", BRIDGE, "bash", "-s"],
                input=hop, env=env, capture_output=True, text=True, timeout=30,
            )
        return (proc.stdout or proc.stderr or "").strip()[-200:]
    finally:
        os.unlink(ask.name)


def _read_ws_frame(sock: socket.socket) -> bytes:
    header = _read_exact(sock, 2)
    if not header:
        return b""
    first, second = header
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    if second & 0x80:
        _read_exact(sock, 4)
    payload = _read_exact(sock, length) if length else b""
    if opcode == 0x8:
        return b""
    if opcode == 0x9:
        return b"\x00"
    return payload


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def _office_bind() -> bool:
    sock = socket.socket()
    try:
        sock.bind((BIND, 0))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ensure_fpv_tunnel() -> None:
    """Forward the robot camera while this machine is off the office LAN.

    Do not connect to the local port to test it. Each connect opens a
    channel to the camera, and those idle channels stop the picture.
    """
    global _tunnel, _tunnel_at
    if _tunnel is not None and _tunnel.poll() is None:
        return
    if time.time() - _tunnel_at < 3:
        return
    _tunnel_at = time.time()
    _tunnel = subprocess.Popen(
        [
            "ssh", "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=12",
            "-L", f"127.0.0.1:{FPV_LOCAL}:{ROBOT}:9100",
            "-L", f"127.0.0.1:{GATEWAY_LOCAL}:{ROBOT}:{GATEWAY}",
            BRIDGE,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _rgb_loop() -> None:
    global _frame, _frame_at
    while True:
        conn = None
        try:
            if _office_bind():
                conn = HTTPConnection(ROBOT, 9100, timeout=4, source_address=(BIND, 0))
            else:
                _ensure_fpv_tunnel()
                conn = HTTPConnection("127.0.0.1", FPV_LOCAL, timeout=4)
            conn.request("GET", "/jpg")
            resp = conn.getresponse()
            data = resp.read()
            if data.startswith(b"\xff\xd8"):
                with _frame_lock:
                    _frame = data
                    _frame_at = time.time()
            else:
                raise OSError("not a jpeg")
        except Exception:
            with _frame_lock:
                if _frame_at and time.time() - _frame_at > 2.0:
                    _frame = b""
            time.sleep(0.4)
            continue
        finally:
            if conn is not None:
                conn.close()
        time.sleep(0.08)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/jpg"):
            with _frame_lock:
                frame = _frame
            if not frame:
                self._send(503, b"no frame", "text/plain")
                return
            self._send(200, frame, "image/jpeg")
            return
        if self.path.startswith("/fpv.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        frame = _frame
                    if frame:
                        head = (
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode()
                            + b"\r\n\r\n"
                        )
                        self.wfile.write(head + frame + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._send(400, b'{"ok":false,"detail":"bad json"}', "application/json")
            return
        if self.path == "/api/task":
            text = str(req.get("text") or "").strip()
            if not text:
                self._send(400, b'{"ok":false,"detail":"empty"}', "application/json")
                return
            status, data = _http("POST", "/api/agent/speak", {"text": text})
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")
            return
        if self.path == "/api/stop":
            task_id = _find_id(req, "task_id")
            session_id = _find_id(req, "session_id")
            cancel: dict = {}
            if task_id:
                _status, cancel = _http("POST", "/api/agent/task/cancel", {"task_id": task_id, "session_id": session_id})
            motion = _zero_motion()
            body = {"ok": True, "cancel": cancel, "motion": motion}
            self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")
            return
        self._send(404, b'{"ok":false}', "application/json")


def main() -> None:
    threading.Thread(target=_rgb_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
