#!/usr/bin/env bash
# Build /opt/aerial_orin product tree on the Orin (no training checkout at runtime).
# Does not start finder serve or dog walk.
set -euo pipefail

ROOT="${ROOT:-$HOME/aerial-wam-v2}"
DEST="${DEST:-/opt/aerial_orin}"
STAGE="${STAGE:-$HOME/aerial_orin_stage}"
PY="${PY:-python3}"

AF="$ROOT/experiments/aerial/deploy/aerial_finder"
DA3="$ROOT/experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/da3_ft_head_fp16.pt"
SURV="$ROOT/experiments/aerial/rl/artifacts/shorew03_krs6_20261013"
PROG="$ROOT/experiments/aerial/rl/artifacts/env_bh_20261015"

die() { echo "ERROR: $*" >&2; exit 1; }
need() { [[ -e "$1" ]] || die "missing $1"; }

need "$ROOT"
need "$AF/__main__.py"
need "$DA3"
need "$HOME/h12_lud_bridge.py"
need "$HOME/lumos_sdk"

rm -rf "$STAGE"
mkdir -p "$STAGE"/{bin,config,weights/{nav,detector,clip,parser},runs,experiments/aerial/rl,experiments/aerial/scripts,experiments/aerial/rl/env}

echo "[pack] aerial_finder package"
rsync -a --delete --exclude '__pycache__' --exclude tests --exclude deploy/dist \
  --exclude '*.pyc' --exclude weights \
  "$AF/" "$STAGE/aerial_finder/"

echo "[pack] vendor experiments.aerial.rl (runtime .py only)"
rsync -a --include '*/' --include '*.py' --exclude '__pycache__' --exclude tests \
  --exclude '*.pyc' --exclude artifacts \
  "$ROOT/experiments/aerial/rl/" "$STAGE/experiments/aerial/rl/"
install -m 0644 "$ROOT/experiments/__init__.py" "$STAGE/experiments/__init__.py"
rsync -a --include '*.py' --exclude '*' "$ROOT/experiments/aerial/" "$STAGE/experiments/aerial/"
install -m 0644 "$ROOT/experiments/aerial/scripts/tello_consequence_collect.py" \
  "$STAGE/experiments/aerial/scripts/tello_consequence_collect.py"
touch "$STAGE/experiments/aerial/scripts/__init__.py"

echo "[pack] lumos_sdk + H12"
rsync -a --delete --exclude '__pycache__' --exclude '.git' \
  "$HOME/lumos_sdk/" "$STAGE/lumos_sdk/"
install -m 0644 "$HOME/h12_lud_bridge.py" "$STAGE/bin/h12_lud_bridge.py"
# Prefer product SDK path over leftover home checkouts.
python3 - <<'PY'
from pathlib import Path
p = Path.home() / "aerial_orin_stage/bin/h12_lud_bridge.py"
t = p.read_text()
needle = "roots = ["
insert = 'roots = [\n    Path(os.environ.get("AERIAL_ORIN_HOME", "/opt/aerial_orin")) / "lumos_sdk",\n'
if "AERIAL_ORIN_HOME" not in t:
    if "import os" not in t.split("THIS =")[0]:
        t = t.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport os\n", 1)
    t = t.replace(needle, insert, 1)
    p.write_text(t)
print("h12 roots patched")
PY

echo "[pack] weights (neutral names)"
cp -f "$DA3" "$STAGE/weights/m_d.bin"
cp -f "$SURV/c_s218/consequence_best.pt" "$STAGE/weights/m_n0.bin"
cp -f "$SURV/c_s219/consequence_best.pt" "$STAGE/weights/m_n1.bin"
cp -f "$SURV/c_s220/consequence_best.pt" "$STAGE/weights/m_n2.bin"
cp -f "$PROG/b_s218/consequence_best.pt" "$STAGE/weights/m_g0.bin"
cp -f "$PROG/b_s219/consequence_best.pt" "$STAGE/weights/m_g1.bin"
cp -f "$PROG/b_s220/consequence_best.pt" "$STAGE/weights/m_g2.bin"
cp -f "$AF/weights/detector/yolov8s-worldv2.pt" "$STAGE/weights/m_o.bin"
rsync -a "$AF/weights/parser/Qwen2.5-0.5B-Instruct/" "$STAGE/weights/m_p/"
# CLIP filename is required by the text encoder cache; keep off the product tree.
mkdir -p "$HOME/.cache/clip"
cp -f "$ROOT/weights/clip/ViT-B-32.pt" "$HOME/.cache/clip/ViT-B-32.pt"
cp -f "$AF/configs/sjcam_dog_fisheye.json" "$STAGE/config/sjcam_dog_fisheye.json"
cp -f "$AF/configs/product_dog_f35.json" "$STAGE/config.json"
date +%Y%m%d > "$STAGE/VERSION"

cat > "$STAGE/cuda_env.sh" <<'EOF'
# Product CUDA search path (torch wheel + extracted CUDA 12.6).
export LD_LIBRARY_PATH="/home/nvidia/.local/lib/python3.10/site-packages/torch/lib:/home/nvidia/cuda-runtime/usr/local/cuda-12.6/lib64:/home/nvidia/cuda-runtime/usr/local/cuda-12.6/targets/aarch64-linux/lib:/home/nvidia/cuda-runtime/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="/home/nvidia/.local/bin:${PATH:-}"
export AERIAL_DA3_FP16=1
export AERIAL_ORIN_HOME="${AERIAL_ORIN_HOME:-/opt/aerial_orin}"
export AERIAL_FINDER_HOME="${AERIAL_FINDER_HOME:-$AERIAL_ORIN_HOME}"
export PYTHONPATH="${AERIAL_ORIN_HOME}${PYTHONPATH:+:$PYTHONPATH}"
EOF

cat > "$STAGE/bin/rt-task" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cuda_env.sh"
export AERIAL_ORIN_HOME="$ROOT" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$ROOT/bin/rt_crypt.pyc" ]]; then
  python3 "$ROOT/bin/rt_crypt.pyc" decrypt >/dev/null
  export AERIAL_FINDER_HOME="${RT_RAM:-/dev/shm/rtw}"
  trap 'python3 "$ROOT/bin/rt_crypt.pyc" wipe >/dev/null 2>&1 || true' EXIT
else
  export AERIAL_FINDER_HOME="$ROOT"
fi
cd "$ROOT"
CFG="${AERIAL_FINDER_CONFIG:-$ROOT/config.json}"
if [[ $# -eq 0 ]]; then
  echo "refusing auto serve; pass: parse '...'  or  serve --config $CFG" >&2
  exit 2
fi
python3 -m aerial_finder "$@"
EOF

cat > "$STAGE/bin/rt-link" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cuda_env.sh"
export AERIAL_ORIN_HOME="$ROOT"
exec python3 "$ROOT/bin/h12_lud_bridge.pyc" --bind 0.0.0.0 --port 8092 "$@"
EOF

cat > "$STAGE/bin/rt-air" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cuda_env.sh"
export AERIAL_ORIN_HOME="$ROOT" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec python3 "$ROOT/experiments/aerial/scripts/tello_consequence_collect.pyc" \
  --base "${TELLO_HUB:-http://192.168.54.2:8090}" \
  --device cuda \
  --depth-ckpt "$ROOT/weights/m_d.bin" \
  --ckpts "$ROOT/weights/m_n0.bin,$ROOT/weights/m_n1.bin,$ROOT/weights/m_n2.bin" \
  "$@"
EOF

cat > "$STAGE/bin/rt-api" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cuda_env.sh"
export AERIAL_ORIN_HOME="$ROOT" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$ROOT/bin/rt_crypt.pyc" ]]; then
  python3 "$ROOT/bin/rt_crypt.pyc" decrypt >/dev/null
  export AERIAL_FINDER_HOME="${RT_RAM:-/dev/shm/rtw}"
  trap 'python3 "$ROOT/bin/rt_crypt.pyc" wipe >/dev/null 2>&1 || true' EXIT
else
  export AERIAL_FINDER_HOME="$ROOT"
fi
cd "$ROOT"
CFG="${AERIAL_FINDER_CONFIG:-$ROOT/config.json}"
if [[ $# -eq 0 ]]; then
  echo "usage: rt-api parse '...' | health | serve" >&2
  exit 2
fi
python3 -m aerial_finder.iface --config "$CFG" "$@"
EOF
chmod 0755 "$STAGE/bin/rt-task" "$STAGE/bin/rt-link" "$STAGE/bin/rt-air" "$STAGE/bin/rt-api"

echo "[pack] compile bytecode"
"$PY" -m compileall -q -x 'tests' "$STAGE/aerial_finder" "$STAGE/experiments" || true

echo "[pack] install $DEST (sudo)"
echo nvidia | sudo -S mkdir -p "$DEST"
echo nvidia | sudo -S rsync -a --delete "$STAGE/" "$DEST/"
echo nvidia | sudo -S chown -R nvidia:nvidia "$DEST"
echo nvidia | sudo -S chmod 0755 "$DEST/bin/"*

HERE=$(cd "$(dirname "$0")" && pwd)
if [[ -x "$HERE/freeze_product.sh" ]]; then
  echo "[pack] freeze plaintext"
  bash "$HERE/freeze_product.sh" "$DEST"
fi
echo "[pack] installed $(du -sh "$DEST" | awk '{print $1}') -> $DEST"
echo "[pack] entries: $DEST/bin/{rt-task,rt-link,rt-air}"
