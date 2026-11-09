"""Open-vocabulary detection (YOLO-World) with a colour / attribute check.

YOLO-World proposes boxes for the text prompts; a box only counts as the target
when the requested colour covers enough of its interior. Far targets are often
below YOLO-World's reach, so ``ColorHints`` also proposes saturated blobs of the
requested colour: they may steer the search closer but never confirm a target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import Detection


@dataclass(frozen=True)
class HsvBand:
    """OpenCV HSV (H 0-180); ``h_lo > h_hi`` wraps through red."""

    h_lo: int
    h_hi: int
    s_lo: int = 0
    s_hi: int = 255
    v_lo: int = 0
    v_hi: int = 255

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue = (h >= self.h_lo) | (h <= self.h_hi) if self.h_lo > self.h_hi else (h >= self.h_lo) & (h <= self.h_hi)
        return hue & (s >= self.s_lo) & (s <= self.s_hi) & (v >= self.v_lo) & (v <= self.v_hi)


COLOR_BANDS: Dict[str, Tuple[HsvBand, ...]] = {
    "red": (HsvBand(170, 10, 90, 255, 60),),
    "orange": (HsvBand(11, 22, 110, 255, 80),),
    "yellow": (HsvBand(23, 34, 90, 255, 90),),
    "green": (HsvBand(35, 85, 70, 255, 50),),
    "blue": (HsvBand(95, 130, 80, 255, 50),),
    "purple": (HsvBand(131, 160, 60, 255, 50),),
    "white": (HsvBand(0, 180, 0, 40, 190),),
    "black": (HsvBand(0, 180, 0, 255, 0, 50),),
    "gray": (HsvBand(0, 180, 0, 40, 60, 180),),
}


def color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], bool)
    for b in COLOR_BANDS[color]:
        m |= b.mask(hsv)
    return m


def color_fraction(rgb: np.ndarray, bbox: Sequence[float], color: str, shrink: float = 0.15) -> float:
    """Share of the box interior (``shrink`` trimmed off each side) in the colour band."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if x1 - x0 >= 8 and y1 - y0 >= 8:
        dx, dy = shrink * (x1 - x0), shrink * (y1 - y0)
        x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(w, max(xi0 + 1, int(round(x1)))), min(h, max(yi0 + 1, int(round(y1))))
    crop = rgb[yi0:yi1, xi0:xi1]
    if crop.size == 0:
        return 0.0
    return float(color_mask(crop, color).mean())


class YoloWorld:
    """Thin ultralytics YOLO-World wrapper; prompts can change per mission.

    ``tiles`` = (columns, top fraction, bottom fraction): besides the full frame,
    the band between those image rows (ground ahead, where small objects sit) is
    cut into overlapping tiles that are each run at ``imgsz``, i.e. magnified.
    YOLO-World text prompts need the CLIP weights under ``./weights/clip/``.
    """

    def __init__(self, model_path: str = "yolov8s-worldv2.pt", device: str = "cuda", imgsz: int = 1280,
                 conf: float = 0.05, tiles: Optional[Sequence[float]] = None, tile_imgsz: int = 960,
                 nms_iou: float = 0.5) -> None:
        self.model_path = model_path
        self.model = None
        self.device, self.imgsz, self.conf = device, int(imgsz), float(conf)
        self.tiles = None if not tiles else (int(tiles[0]), float(tiles[1]), float(tiles[2]))
        self.tile_imgsz = int(tile_imgsz)
        self.nms_iou = float(nms_iou)
        self.prompts: List[str] = []

    def set_prompts(self, prompts: Sequence[str]) -> None:
        from ultralytics import YOLO

        # set_classes on a model that has already run on CUDA fails (text tokens stay on CPU): reload.
        self.prompts = list(prompts)
        self.model = YOLO(self.model_path)
        self.model.set_classes(self.prompts)

    def _predict(self, bgr: np.ndarray, imgsz: int, ox: float = 0.0, oy: float = 0.0):
        r = self.model.predict(bgr, imgsz=imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy])
        return list(zip(xyxy, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))

    def __call__(self, rgb: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        out = self._predict(bgr, self.imgsz)
        if self.tiles:
            h, w = bgr.shape[:2]
            n, f0, f1 = self.tiles
            y0, y1 = int(f0 * h), int(f1 * h)
            tw = int(np.ceil(w / (n - 0.15 * (n - 1))))
            for k in range(n):
                x0 = min(w - tw, int(k * 0.85 * tw))
                out += self._predict(np.ascontiguousarray(bgr[y0:y1, x0:x0 + tw]), self.tile_imgsz, x0, y0)
        if not out:
            return []
        boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b, _, _ in out]
        keep = cv2.dnn.NMSBoxes(boxes, [float(c) for _, c, _ in out], self.conf, self.nms_iou)
        keep = np.asarray(keep).reshape(-1)
        return [(tuple(float(v) for v in out[i][0]), float(out[i][1]), self.prompts[out[i][2]]) for i in keep]


    def focus(self, rgb: np.ndarray, center: Tuple[float, float], half: int = 120,
              imgsz: int = 640) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        """Magnified pass on a square crop around ``center`` (pixel u, v), e.g. a candidate being verified."""
        h, w = rgb.shape[:2]
        x0 = int(np.clip(center[0] - half, 0, max(0, w - 2 * half)))
        y0 = int(np.clip(center[1] - half, 0, max(0, h - 2 * half)))
        crop = np.ascontiguousarray(rgb[y0:y0 + 2 * half, x0:x0 + 2 * half, ::-1])
        out = self._predict(crop, imgsz, x0, y0)
        return [(tuple(float(v) for v in b), float(c), self.prompts[k]) for b, c, k in out]


class YoloWorldTS:
    """YOLO-World TorchScript export with the vocabulary baked in (no ultralytics / CLIP at runtime).

    ``classes`` is the export order. ``set_prompts`` cannot change the vocabulary; prompts outside it are
    logged and never detected.
    """

    def __init__(self, model_path: str, classes: Sequence[str], device: str = "cuda", imgsz: int = 640,
                 conf: float = 0.05, nms_iou: float = 0.5) -> None:
        import torch

        self.torch = torch
        self.model = torch.jit.load(model_path, map_location=device).eval()
        self.classes = [str(c) for c in classes]
        self.device, self.imgsz, self.conf, self.nms_iou = device, int(imgsz), float(conf), float(nms_iou)
        self.prompts: List[str] = list(self.classes)
        self.target_idx: List[int] = []

    def set_prompts(self, prompts: Sequence[str]) -> None:
        missing = [p for p in prompts if p not in self.classes]
        if missing:
            import logging

            logging.getLogger("aerial_finder.detector").warning("not in exported vocabulary: %s", missing)

    def set_targets(self, targets: Sequence[str]) -> None:
        """Target classes are also scored on their own channel, so a near class (truck) cannot hide them."""
        self.target_idx = [self.classes.index(t) for t in targets if t in self.classes]

    def _predict(self, rgb: np.ndarray, ox: float = 0.0, oy: float = 0.0):
        h, w = rgb.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        img = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        py, px = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        img[py:py + nh, px:px + nw] = cv2.resize(np.ascontiguousarray(rgb), (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = self.torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        with self.torch.no_grad():
            y = self.model(x)[0].float().cpu().numpy()  # (4 + nc, N), cx cy w h in letterbox pixels
        cls, conf = y[4:].argmax(0), y[4:].max(0)
        cols, confs, clss = [np.flatnonzero(conf >= self.conf)], [conf[conf >= self.conf]], [cls[conf >= self.conf]]
        for k in self.target_idx:
            own = np.flatnonzero((y[4 + k] >= self.conf) & (cls != k))
            cols.append(own), confs.append(y[4 + k, own]), clss.append(np.full(len(own), k))
        keep = np.concatenate(cols)
        if not len(keep):
            return []
        b, conf, cls = y[:4, keep].T, np.concatenate(confs), np.concatenate(clss)
        x0 = (b[:, 0] - b[:, 2] / 2 - px) / s + ox
        y0 = (b[:, 1] - b[:, 3] / 2 - py) / s + oy
        x1 = (b[:, 0] + b[:, 2] / 2 - px) / s + ox
        y1 = (b[:, 1] + b[:, 3] / 2 - py) / s + oy
        return list(zip(np.stack([x0, y0, x1, y1], 1), conf, cls.astype(int)))

    def _nms(self, out):
        res = []
        for k in sorted({int(c) for _, _, c in out}):
            grp = [o for o in out if int(o[2]) == k]
            boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b, _, _ in grp]
            keep = np.asarray(cv2.dnn.NMSBoxes(boxes, [float(c) for _, c, _ in grp], self.conf, self.nms_iou)).reshape(-1)
            res += [(tuple(float(v) for v in grp[i][0]), float(grp[i][1]), self.classes[k]) for i in keep]
        return res

    def __call__(self, rgb: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        h, w = rgb.shape[:2]
        out = self._predict(rgb)
        # 2x centre crop for far, small objects (as tello_find_object).
        cx0, cy0 = w // 4, int(h * 0.2)
        out += [(b / 2 + np.array([cx0, cy0, cx0, cy0]), c, k)
                for b, c, k in self._predict(cv2.resize(rgb[cy0:cy0 + h // 2, cx0:cx0 + w // 2], (w, h)))]
        return self._nms(out)

    def focus(self, rgb: np.ndarray, center: Tuple[float, float], half: int = 160,
              imgsz: int = 640) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        h, w = rgb.shape[:2]
        x0 = int(np.clip(center[0] - half, 0, max(0, w - 2 * half)))
        y0 = int(np.clip(center[1] - half, 0, max(0, h - 2 * half)))
        return self._nms(self._predict(rgb[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0))


class ColorHints:
    """Compact saturated blobs of one colour, as unverified far-range candidates."""

    def __init__(self, work_width: int = 960, min_area_px: int = 6, max_area_frac: float = 0.02,
                 max_aspect: float = 4.0, max_hints: int = 12) -> None:
        self.work_width = int(work_width)
        self.min_area_px = int(min_area_px)
        self.max_area_frac = float(max_area_frac)
        self.max_aspect = float(max_aspect)
        self.max_hints = int(max_hints)

    def __call__(self, rgb: np.ndarray, color: str) -> List[Detection]:
        h, w = rgb.shape[:2]
        s = min(1.0, self.work_width / float(w))
        small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else rgb
        m = color_mask(small, color).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        out: List[Detection] = []
        max_area = self.max_area_frac * m.size
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < self.min_area_px or area > max_area or max(bw, bh) > self.max_aspect * max(1, min(bw, bh)):
                continue
            fill = area / float(bw * bh)
            if fill < 0.3:
                continue
            bbox = (x / s, y / s, (x + bw) / s, (y + bh) / s)
            out.append(Detection(label=f"{color} blob", prompt=color, bbox=bbox, conf=0.0, verified=False,
                                 color_frac=fill, source="color", extra={"area_px": area / (s * s)}))
        out.sort(key=lambda d: -d.extra["area_px"])
        return out[: self.max_hints]


# Common street objects added to the vocabulary so look-alikes get their own class
# (a red barrel otherwise scores as "stool"); boxes of these classes are dropped.
DISTRACTORS = ("barrel", "trash can", "fire hydrant", "traffic cone", "car", "person", "potted plant", "sign",
               "box", "mailbox", "bicycle", "table", "telephone booth", "kiosk")


class TargetDetector:
    """Target prompts + distractor vocabulary + colour check for one mission target."""

    def __init__(self, yolo, *, min_color_frac: float = 0.2, hints: Optional[ColorHints] = None,
                 min_conf: float = 0.05, distractors: Sequence[str] = DISTRACTORS) -> None:
        self.yolo = yolo
        self.min_color_frac = float(min_color_frac)
        self.hints = hints
        self.min_conf = float(min_conf)
        self.distractors = tuple(distractors)
        self.label = ""
        self.color: Optional[str] = None
        self.targets: Tuple[str, ...] = ()

    def configure(self, label: str, prompts: Sequence[str], color: Optional[str]) -> None:
        if color is not None and color not in COLOR_BANDS:
            raise ValueError(f"unsupported colour {color!r}; known {sorted(COLOR_BANDS)}")
        self.label, self.color = label, color
        self.targets = tuple(prompts)
        vocab = list(dict.fromkeys(list(prompts) + [d for d in self.distractors
                                                    if not any(d in p or p in d for p in prompts)]))
        self.yolo.set_prompts(vocab)
        if hasattr(self.yolo, "set_targets"):
            self.yolo.set_targets(self.targets)

    def focus(self, rgb: np.ndarray, center: Tuple[float, float]) -> List[Detection]:
        """Target-class detections from a magnified crop around ``center`` (falls back to the full frame)."""
        f = getattr(self.yolo, "focus", None)
        return self._target_dets(rgb, f(rgb, center) if f is not None else self.yolo(rgb), "focus")

    def __call__(self, rgb: np.ndarray) -> List[Detection]:
        dets = self._target_dets(rgb, self.yolo(rgb), "full")
        if self.hints is not None and self.color:
            boxes = [d.bbox for d in dets]
            for h in self.hints(rgb, self.color):
                if not any(_overlap(h.bbox, b) for b in boxes):
                    dets.append(h)
        return dets

    def _target_dets(self, rgb: np.ndarray, raw, mode: str) -> List[Detection]:
        dets: List[Detection] = []
        for bbox, conf, prompt in raw:
            if conf < self.min_conf or prompt not in self.targets:
                continue
            frac = color_fraction(rgb, bbox, self.color) if self.color else 1.0
            dets.append(Detection(label=self.label, prompt=prompt, bbox=bbox, conf=conf,
                                  verified=frac >= self.min_color_frac, color_frac=frac, extra={"mode": mode}))
        return dets


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
