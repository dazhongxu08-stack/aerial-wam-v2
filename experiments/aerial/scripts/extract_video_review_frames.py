#!/usr/bin/env python3
"""Extract start/mid/end frames from dual-view MP4s for visual review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2


def video_info(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 5.0)
    cap.release()
    dur = n / fps if fps > 0 else 0.0
    return n, dur


def extract_frame(video: Path, frame_idx: int, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(str(out), frame)
    return out.is_file()


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/urban_complex_videos_from_125")
    out_root = root / "_review_frames"
    rows = []
    for mp4 in sorted(root.rglob("*dual_view_dashboard.mp4")):
        tag = mp4.stem.replace("_dual_view_dashboard", "")
        batch = mp4.parent.parent.name if mp4.parent.name == "videos" else mp4.parent.name
        frame_key = f"{batch}__{tag}"
        summary = mp4.parent / f"{tag}_trajectory_summary.json"
        meta = {}
        if summary.is_file():
            meta = json.loads(summary.read_text(encoding="utf-8"))
        nframes, dur = video_info(mp4)
        if dur <= 0 and meta.get("duration_seconds"):
            dur = float(meta["duration_seconds"])
        if nframes <= 0 and dur > 0:
            nframes = max(1, int(dur * 5))
        idxs = [0]
        if nframes > 2:
            idxs.extend([nframes // 2, nframes - 1])
        frame_paths = []
        for i, fi in enumerate(idxs):
            fp = out_root / frame_key / f"{i:02d}_f{fi}.jpg"
            if extract_frame(mp4, fi, fp):
                frame_paths.append(str(fp.relative_to(root)))
        rows.append({
            "tag": tag,
            "batch": batch,
            "frame_key": frame_key,
            "mp4": str(mp4.relative_to(root)),
            "dur_s": round(dur, 1),
            "arrived": meta.get("arrived"),
            "prog": round(float(meta.get("progress_ratio", 0)) * 100, 1) if meta else None,
            "d_fin": round(float(meta.get("final_distance_m", 0)), 1) if meta else None,
            "route_idx": meta.get("route_idx"),
            "frames": frame_paths,
        })
    report = out_root / "index.json"
    report.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} videos -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
