from __future__ import annotations

import threading
import time
import urllib.request

from experiments.aerial.deploy.orin_camera_stream import _OrinCameraStreamHandler, main
from experiments.aerial.deploy.real_camera import MockCamera


def test_index_and_snapshot_with_mock_camera() -> None:
    cam = MockCamera(wam_size=64)
    cam.open()
    _OrinCameraStreamHandler.frame_fn = cam.read
    _OrinCameraStreamHandler.jpeg_quality = 70
    _OrinCameraStreamHandler.min_frame_interval_s = 0.05

    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _OrinCameraStreamHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3).read()
        assert b"stream.mjpg" in html
        snap = urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot.jpg", timeout=3).read()
        assert snap.startswith(b"\xff\xd8\xff")
    finally:
        httpd.shutdown()
        httpd.server_close()
        cam.close()


def test_main_mock_exits_on_keyboard_interrupt(monkeypatch) -> None:
    cam = MockCamera()
    cam.open()

    class _ImmediateServer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            return None

    monkeypatch.setattr("experiments.aerial.deploy.orin_camera_stream.ThreadingHTTPServer", _ImmediateServer)
    monkeypatch.setattr(
        "experiments.aerial.deploy.orin_camera_stream._open_camera",
        lambda args: (cam, cam.read),
    )
    assert main(["--mock", "--port", "18088"]) == 0
