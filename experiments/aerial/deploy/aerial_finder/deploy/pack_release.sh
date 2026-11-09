#!/usr/bin/env bash
# Build a deployable Orin tarball (code + relative config + weights).
#
# Usage:
#   ./pack_release.sh [--out DIR] [--weights DIR] [--skip-missing-weights] [--version V]
#   ./fetch_detector_weights.sh   # once, to get yolov8s-worldv2.pt
#
# Env overrides: DA3_PT, SURV_DIR, DETECTOR_TS, YOLO_WORLD_PT, CLIP_PT
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PKG_SRC=$(cd "$HERE/.." && pwd)
REPO=$(cd "$PKG_SRC/../../../.." && pwd)
OUT_DIR="${OUT_DIR:-$PWD/dist}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
SKIP_MISSING=0
VERSION="${VERSION:-$(date +%Y%m%d)}"
MODELS="$REPO/artifacts/tello_gs/models"

DA3_PT="${DA3_PT:-$REPO/experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/da3_ft_head_fp16.pt}"
SURV_DIR="${SURV_DIR:-$REPO/experiments/aerial/rl/artifacts/shorew03_krs6_20261013}"

YOLO_WORLD_PT="${YOLO_WORLD_PT:-$MODELS/yolov8s-worldv2.pt}"
CLIP_PT="${CLIP_PT:-$MODELS/clip/ViT-B-32.pt}"

DETECTOR_TS="${DETECTOR_TS:-$HOME/yw_stool.torchscript}"
[[ -f "$DETECTOR_TS" ]] || DETECTOR_TS="$MODELS/yw_stool_cpu.torchscript"
[[ -f "$DETECTOR_TS" ]] || DETECTOR_TS="$HOME/yw_forklift.torchscript"
[[ -f "$DETECTOR_TS" ]] || DETECTOR_TS="$MODELS/yw_forklift_cpu.torchscript"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) mkdir -p "$2"; OUT_DIR=$(cd "$2" && pwd); shift 2 ;;
    --weights) WEIGHTS_DIR=$(cd "$2" && pwd); shift 2 ;;
    --skip-missing-weights) SKIP_MISSING=1; shift ;;
    --version) VERSION=$2; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Auto-fetch open-vocab detector if missing (needs network)
if [[ ! -f "$YOLO_WORLD_PT" ]]; then
  echo "[pack] yolov8s-worldv2.pt missing — running fetch_detector_weights.sh"
  bash "$HERE/fetch_detector_weights.sh" "$MODELS" || true
fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/aerial-finder-pack.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/aerial_finder"
mkdir -p "$ROOT"/{bin,weights/nav,weights/detector,weights/clip,config,runs,deploy}

echo "[pack] staging $ROOT"
rsync -a --delete \
  --exclude '__pycache__' --exclude 'tests' --exclude '.git' \
  --exclude 'deploy/dist' --exclude '*.pyc' \
  "$PKG_SRC/" "$ROOT/aerial_finder/"

install -m 0755 "$HERE/bin/aerial-finder" "$ROOT/bin/aerial-finder"
install -m 0755 "$HERE/install.sh" "$ROOT/deploy/install.sh"
install -m 0755 "$HERE/install_venv.sh" "$ROOT/deploy/install_venv.sh"
install -m 0755 "$HERE/fetch_detector_weights.sh" "$ROOT/deploy/fetch_detector_weights.sh"
install -m 0644 "$HERE/aerial-finder.service" "$ROOT/deploy/aerial-finder.service"
cp "$PKG_SRC/configs/product_dog.json" "$ROOT/config.json"
cp "$PKG_SRC/configs/product_dog_stool_ts.json" "$ROOT/config/product_dog_stool_ts.json"
cp "$PKG_SRC/configs/sjcam_dog_fisheye.json" "$ROOT/config/sjcam_dog_fisheye.json"
cp "$PKG_SRC/requirements.txt" "$ROOT/requirements.txt"
printf '%s\n' "$VERSION" > "$ROOT/VERSION"
cp "$HERE/PACKAGE.md" "$ROOT/PACKAGE.md"

copy_w() {
  local src=$1 dst=$2 required=${3:-1}
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
    echo "[pack] weight $(basename "$dst") <- $src ($(du -h "$src" | awk '{print $1}'))"
  elif [[ "$required" -eq 0 || "$SKIP_MISSING" -eq 1 ]]; then
    echo "[pack] WARN missing $src (optional/skipped)" >&2
  else
    echo "[pack] ERROR missing required weight: $src" >&2
    exit 1
  fi
}

if [[ -n "$WEIGHTS_DIR" ]]; then
  rsync -a "$WEIGHTS_DIR/" "$ROOT/weights/"
  echo "[pack] weights from $WEIGHTS_DIR"
else
  copy_w "$DA3_PT" "$ROOT/weights/da3_ft_head_fp16.pt" 1
  copy_w "$SURV_DIR/c_s218/consequence_best.pt" "$ROOT/weights/nav/surv_s218.pt" 1
  copy_w "$SURV_DIR/c_s219/consequence_best.pt" "$ROOT/weights/nav/surv_s219.pt" 1
  copy_w "$SURV_DIR/c_s220/consequence_best.pt" "$ROOT/weights/nav/surv_s220.pt" 1
  # Open-vocab YOLO-World (default config.json)
  copy_w "$YOLO_WORLD_PT" "$ROOT/weights/detector/yolov8s-worldv2.pt" 1
  copy_w "$CLIP_PT" "$ROOT/weights/clip/ViT-B-32.pt" 0
  # Narrow stool TorchScript (optional alternate config)
  copy_w "$DETECTOR_TS" "$ROOT/weights/detector/model.torchscript" 0
fi

cat > "$ROOT/weights/MANIFEST.txt" <<EOF
# Default config.json uses open-vocab YOLO-World:
detector/yolov8s-worldv2.pt
clip/ViT-B-32.pt          # offline CLIP for text prompts (recommended)
da3_ft_head_fp16.pt
nav/surv_s218.pt
nav/surv_s219.pt
nav/surv_s220.pt
# Optional narrow export (config/product_dog_stool_ts.json):
detector/model.torchscript
EOF

NAME="aerial-finder-${VERSION}-orin-aarch64"
mkdir -p "$OUT_DIR"
TAR="$OUT_DIR/${NAME}.tar.gz"
tar -C "$STAGE" -czf "$TAR" aerial_finder
echo "[pack] wrote $TAR ($(du -h "$TAR" | awk '{print $1}'))"
echo "[pack] default detector: yolo_world (prompts from instruction, e.g. stool/fire extinguisher)"
echo "[pack] install on Orin:"
echo "  sudo tar -C /opt -xzf $(basename "$TAR")"
echo "  sudo /opt/aerial_finder/deploy/install.sh"
echo "  sudo systemctl enable --now aerial-finder"
