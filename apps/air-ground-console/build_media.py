"""Build synced FPV mp4 + track JSON from the local real-flight archive.

Sources (not copied whole):
  ~/Desktop/真机数据-无人机与狗-20261022/
    无人机-Tello/tello_gs-20261015-16/run_*
    来自Orin-待拉取/.../aerial_finder_runs/walk_12m_*
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

import av
import cv2

ROOT = Path.home() / "Desktop/真机数据-无人机与狗-20261022"
TELLO = ROOT / "无人机-Tello/tello_gs-20261015-16"
DOG = ROOT / "来自Orin-待拉取/pull_20261022_193148/aerial_finder_runs"
OUT = Path(__file__).resolve().parent / "media"
DOG_NAME = re.compile(
    r"^(?P<t>\d+(?:\.\d+)?)_x(?P<x>[+-]?\d+(?:\.\d+)?)_y(?P<y>[+-]?\d+(?:\.\d+)?)_yaw(?P<yaw>[+-]?\d+(?:\.\d+)?)"
)


def angdiff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= math.pi * 2
    while d < -math.pi:
        d += math.pi * 2
    return d


def encode(path: Path, image_paths: list[Path], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise SystemExit(f"unreadable {image_paths[0]}")
    h, w = first.shape[:2]
    if w % 2 or h % 2:
        raise SystemExit(f"odd size {w}x{h}")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "30", "preset": "veryfast"}
    for i, src in enumerate(image_paths):
        im = first if i == 0 else cv2.imread(str(src))
        if im is None:
            continue
        frame = av.VideoFrame.from_ndarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        if i and i % 200 == 0:
            print(f"  {path.name} {i}/{len(image_paths)}", flush=True)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"wrote {path} {path.stat().st_size / 1e6:.1f} MB {len(image_paths)} frames", flush=True)


def calibrate_yaw(points: list[dict]) -> None:
    """Store yaw radians, clockwise from north, matching the displacement heading."""
    best_sign, best_err = 1, 1e9
    for sign in (1, -1):
        errs = []
        for a, b in zip(points, points[1:]):
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            if math.hypot(dx, dy) < 0.25:
                continue
            course = math.atan2(dx, dy)
            errs.append(abs(angdiff(sign * a["yaw0"], course)))
        err = statistics.median(errs) if errs else 1e9
        if err < best_err:
            best_sign, best_err = sign, err
    print(f"  yaw sign {best_sign} median err {best_err:.2f} rad")
    for p in points:
        p["yaw"] = round(best_sign * p["yaw0"], 3)
        del p["yaw0"]


def tello_clip(run_name: str, clip_id: str, short: str, when: str, place: str, stride: int = 2) -> dict:
    run = TELLO / run_name
    meta = json.loads((run / "meta.json").read_text())
    rows = list(csv.DictReader((run / "trajectory.csv").open()))
    seqs = sorted({int(float(r["frame_seq"])) for r in rows})
    by = {}
    for r in rows:
        by.setdefault(int(float(r["frame_seq"])), r)
    frames = sorted((run / "frames").glob("*.jpg"), key=lambda p: int(p.stem))[::stride]
    yaw0 = math.radians(float(by[int(frames[0].stem)]["yaw_deg"]))
    fwd = (math.cos(yaw0), math.sin(yaw0))
    right = (math.cos(yaw0 + math.pi / 2), math.sin(yaw0 + math.pi / 2))
    origin = by[int(frames[0].stem)]
    x0, y0 = float(origin["x_m"]), float(origin["y_m"])
    t0 = float(origin["t_s"])
    paths, points = [], []
    for f in frames:
        seq = int(f.stem)
        row = by.get(seq)
        if row is None:
            seq = min(seqs, key=lambda s: abs(s - int(f.stem)))
            row = by[seq]
        paths.append(f)
        dx, dy = float(row["x_m"]) - x0, float(row["y_m"]) - y0
        points.append({
            "t": round(float(row["t_s"]) - t0, 2),
            "x": round(dx * right[0] + dy * right[1], 2),
            "y": round(dx * fwd[0] + dy * fwd[1], 2),
            "yaw0": math.radians(float(row["yaw_deg"]) - math.degrees(yaw0)),
            "alt": round(float(row["h_cm"] or 0) / 100.0, 2),
        })
    print(clip_id, "frames", len(paths), flush=True)
    calibrate_yaw(points)
    out = OUT / f"{clip_id}.mp4"
    encode(out, paths, 10)
    goal_m = float(meta.get("goal_approx_m") or 0)
    return {
        "id": clip_id,
        "vehicle": "air",
        "short": short,
        "when": when,
        "place": place,
        "task": meta.get("goal_desc") or "",
        "src": f"media/{out.name}",
        "fps": 10,
        "goal": {"x": 0, "y": round(goal_m, 1), "label": meta.get("goal_desc") or "目标"} if goal_m else None,
        "points": points,
    }


def dog_clip(run_name: str, clip_id: str, short: str, when: str, place: str) -> dict:
    run = DOG / run_name
    meta = json.loads((run / "meta.json").read_text())
    parsed = []
    for f in (run / "frames").rglob("*.jpg"):
        m = DOG_NAME.match(f.name)
        if not m:
            continue
        parsed.append((float(m.group("t")), float(m.group("x")), float(m.group("y")), float(m.group("yaw")), f))
    parsed.sort(key=lambda r: r[0])
    t0 = parsed[0][0]
    paths, points = [], []
    for t, x, y, yaw, f in parsed:
        # Body x is forward, y is to the right. Map x east, y north, start facing east.
        paths.append(f)
        points.append({
            "t": round(t - t0, 2),
            "x": round(x - parsed[0][1], 2),
            "y": round(-(y - parsed[0][2]), 2),
            "yaw0": math.pi / 2 + math.radians(yaw),
            "alt": 0.4,
        })
    print(clip_id, "frames", len(paths), "t", points[0]["t"], points[-1]["t"], flush=True)
    calibrate_yaw(points)
    out = OUT / f"{clip_id}.mp4"
    encode(out, paths, 10)
    gx, gy = meta.get("goal_xy") or [12, 0]
    return {
        "id": clip_id,
        "vehicle": "ground",
        "short": short,
        "when": when,
        "place": place,
        "task": f"正前方 {meta.get('meters', gx)} m",
        "src": f"media/{out.name}",
        "fps": 10,
        "goal": {"x": round(gx - parsed[0][1], 2), "y": round(-(gy - parsed[0][2]), 2), "label": "12 m 到达圈"},
        "points": points,
    }


def main() -> None:
    clips = [
        tello_clip("run_20261016_171900", "tello-1719", "Tello 1719", "2026-10-16 17:19", "户外围挡", stride=2),
        tello_clip("run_20261015_170703", "tello-170703", "Tello 170703", "2026-10-15 17:07", "户外围挡", stride=1),
        dog_clip("walk_12m_20261020_231931", "dog-231931", "Lumos 231931", "2026-10-20 23:19", "仓库通道"),
    ]
    payload = {
        "note": "轨迹来自机载里程计，不是测绘坐标。无人机与机器狗不是同一次、同一场地的记录，起点只是对齐在一张图上。",
        "clips": clips,
    }
    dest = OUT / "tracks.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("wrote", dest, dest.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
