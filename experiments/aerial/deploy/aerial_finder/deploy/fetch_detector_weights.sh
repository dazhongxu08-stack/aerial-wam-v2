#!/usr/bin/env bash
# Download open-vocab YOLO-World (+ CLIP) into a weights tree for packaging.
#
#   ./fetch_detector_weights.sh [/path/to/weights]
# Default: <repo>/artifacts/tello_gs/models  and also copies into ./weights layout.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/.." && pwd)
REPO=$(cd "$PKG/../../../.." && pwd)
DEST=${1:-"$REPO/artifacts/tello_gs/models"}
mkdir -p "$DEST" "$DEST/clip"

echo "[fetch] dest=$DEST"
python3 - <<PY
from pathlib import Path
import shutil
import os

dest = Path("$DEST").resolve()
pt = dest / "yolov8s-worldv2.pt"
os.chdir(dest)
if pt.exists() and pt.stat().st_size > 10_000_000:
    print("[fetch] already have", pt, pt.stat().st_size)
else:
    from ultralytics import YOLO
    print("[fetch] downloading yolov8s-worldv2.pt via ultralytics ...")
    YOLO("yolov8s-worldv2.pt")
    # Ultralytics writes next to cwd or caches; collect candidates
    cands = [dest / "yolov8s-worldv2.pt", Path.cwd() / "yolov8s-worldv2.pt"]
    home = Path.home()
    for root in (home / ".config", home / "Library/Application Support", home / ".cache"):
        if not root.exists():
            continue
        for p in root.rglob("yolov8s-worldv2.pt"):
            cands.append(p)
            break
    found = next((p for p in cands if p.exists() and p.stat().st_size > 10_000_000), None)
    if found is None:
        raise SystemExit("download finished but yolov8s-worldv2.pt not found")
    if found.resolve() != pt.resolve():
        shutil.copy2(found, pt)
    print("[fetch] ok", pt, pt.stat().st_size)

# CLIP weights: ultralytics CLIP looks under ./weights/clip when cwd is install root.
# Also stash a copy next to the .pt for pack_release to pick up.
clip_dst = dest / "clip" / "ViT-B-32.pt"
if clip_dst.exists() and clip_dst.stat().st_size > 100_000:
    print("[fetch] already have", clip_dst)
else:
    # Trigger CLIP fetch by constructing YOLO-World text tower once if possible;
    # otherwise leave a marker for pack to warn.
    try:
        import clip  # ultralytics CLIP fork or openai clip
    except Exception:
        clip = None
    # Prefer copying from torch hub / clip cache if present
    cands = list(Path.home().rglob("ViT-B-32.pt"))[:20] if False else []
    for root in (Path.home() / ".cache", Path.home() / ".clip"):
        if root.exists():
            cands.extend(root.rglob("ViT-B-32.pt"))
    found = next((p for p in cands if p.is_file() and p.stat().st_size > 100_000), None)
    if found:
        clip_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, clip_dst)
        print("[fetch] clip", found, "->", clip_dst)
    else:
        print("[fetch] WARN: ViT-B-32.pt not found yet; first Orin run may download CLIP, or install ultralytics CLIP offline")
PY

ls -lh "$DEST/yolov8s-worldv2.pt" "$DEST/clip/ViT-B-32.pt" 2>/dev/null || true
