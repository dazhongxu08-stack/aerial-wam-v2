"""YOLO-World side process. The driving interpreter has no ultralytics."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

REQ = Path(os.environ.get("YHS_YOLO_REQ", "/tmp/yolo_req.json"))
FRAME = Path(os.environ.get("YHS_YOLO_FRAME", "/tmp/yolo_frame.rgb"))
SIGHT = Path(os.environ.get("YHS_YOLO_SIGHT", "/tmp/yolo_sight.json"))
READY = Path(os.environ.get("YHS_YOLO_READY", "/tmp/yolo_ready"))
MODEL = os.environ.get("YHS_YOLO_MODEL", "/home/user/rt/weights/detector/yolov8s-worldv2.pt")


def _load(prompts):
    from ultralytics import YOLO

    model = YOLO(MODEL)
    model.set_classes(list(prompts))
    return model


def _boxes(model, rgb, prompts):
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    result = model.predict(bgr, imgsz=640, conf=0.05, device="cuda", verbose=False)[0]
    dets = []
    if result.boxes is None:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for box, score, kind in zip(xyxy, conf, cls):
        if kind < 0 or kind >= len(prompts):
            continue
        dets.append({
            "x0": float(box[0]), "y0": float(box[1]), "x1": float(box[2]), "y1": float(box[3]),
            "conf": float(score), "prompt": prompts[int(kind)],
        })
    return dets


def _write(stamp, dets) -> None:
    tmp = SIGHT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"stamp": stamp, "dets": dets}), encoding="utf-8")
    os.replace(tmp, SIGHT)


def main() -> None:
    os.chdir("/home/user/rt")
    prompts: list[str] = []
    model = _load(["object"])
    model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=64, conf=0.05, device="cuda", verbose=False)
    READY.write_text("1", encoding="utf-8")
    seen = None
    while True:
        try:
            req = json.loads(REQ.read_text(encoding="utf-8"))
            stamp = req.get("stamp")
            want = list(req.get("prompts") or [])
            width, height = int(req["w"]), int(req["h"])
            raw = FRAME.read_bytes()
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            time.sleep(0.02)
            continue
        if not want or stamp == seen or len(raw) != width * height * 3:
            time.sleep(0.01)
            continue
        if want != prompts:
            model = _load(want)
            prompts = want
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        try:
            dets = _boxes(model, rgb, prompts)
        except Exception as exc:
            _write(stamp, [])
            seen = stamp
            READY.write_text(f"err {exc}", encoding="utf-8")
            continue
        _write(stamp, dets)
        seen = stamp


if __name__ == "__main__":
    main()
