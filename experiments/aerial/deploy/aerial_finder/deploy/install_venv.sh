#!/usr/bin/env bash
# Create/refresh venv inside a product tree (weights + aerial_finder/ already present).
#
# Product layout:
#   sudo ./install_venv.sh [/opt/aerial_finder]
#
# Legacy (dev copy into DEST):
#   sudo ./install_venv.sh /path/to/aerial_finder_pkg /path/to/weights [/opt/aerial_finder]
#
# Uses --system-site-packages so JetPack torch on Orin is reused.
set -euo pipefail

if [[ $# -ge 2 ]]; then
  SRC=$(readlink -f "$1"); WEIGHTS=$(readlink -f "$2"); DEST=${3:-/opt/aerial_finder}
  mkdir -p "$DEST" "$DEST/runs"
  rsync -a --delete --exclude __pycache__ --exclude tests "$SRC/" "$DEST/aerial_finder/"
  rsync -a "$WEIGHTS/" "$DEST/weights/"
else
  DEST=$(readlink -f "${1:-$(cd "$(dirname "$0")/.." && pwd)}")
fi

mkdir -p "$DEST/runs"
test -d "$DEST/aerial_finder" || { echo "missing $DEST/aerial_finder"; exit 1; }
test -f "$DEST/config.json" || { echo "missing $DEST/config.json"; exit 1; }

REQ="$DEST/requirements.txt"
[[ -f "$REQ" ]] || REQ="$DEST/aerial_finder/requirements.txt"

python3 -m venv --system-site-packages "$DEST/venv"
"$DEST/venv/bin/pip" install -q -U pip
"$DEST/venv/bin/pip" install -q -r "$REQ"
"$DEST/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Paths required by configs/product_dog.json → config.json (open-vocab YOLO-World)
need=(
  weights/da3_ft_head_fp16.pt
  weights/nav/surv_s218.pt
  weights/nav/surv_s219.pt
  weights/nav/surv_s220.pt
  weights/detector/yolov8s-worldv2.pt
  config/sjcam_dog_fisheye.json
)
missing=0
for f in "${need[@]}"; do
  if [[ ! -f "$DEST/$f" ]]; then
    echo "missing $DEST/$f" >&2
    missing=1
  fi
done
if [[ ! -f "$DEST/weights/clip/ViT-B-32.pt" ]]; then
  echo "WARN missing $DEST/weights/clip/ViT-B-32.pt (first run may download CLIP)" >&2
fi
[[ "$missing" -eq 0 ]] || { echo "fill weights/ (see deploy/fetch_detector_weights.sh) then re-run" >&2; exit 1; }

# Import check (no CUDA mission)
PYTHONPATH="$DEST" "$DEST/venv/bin/python" - <<PY
import json, sys
sys.path.insert(0, "$DEST")
from aerial_finder.mission.parser import parse_instruction
from aerial_finder import __main__ as m
cfg = json.load(open("$DEST/config.json"))
spec = parse_instruction("正前方10米找一个红色的凳子")
assert spec.distance_m == 10 and spec.color == "red" and spec.label == "stool"
print("parse_ok", spec.to_dict())
print("root", m._install_root(__import__("pathlib").Path("$DEST/config.json")))
PY

echo "venv ready at $DEST/venv"
echo "run: $DEST/bin/aerial-finder serve --config $DEST/config.json"
echo "  or: systemctl start aerial-finder"
