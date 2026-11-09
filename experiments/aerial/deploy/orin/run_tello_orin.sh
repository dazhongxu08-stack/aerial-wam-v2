#!/usr/bin/env bash
# Tello policy on Orin 112: JPEG from Mac hub :8090. No takeoff unless MODE=fly.
# Usage: MODE=bench ./run_tello_orin.sh     # import + one JPEG if hub up
#        MODE=fly  ./run_tello_orin.sh      # only when operator said 飞/采
set -euo pipefail
ROOT="${ROOT:-$HOME/aerial-wam-v2}"
PY="${PY:-python3}"
# CUDA 12.6 userspace extracted beside JetPack (see ~/cuda_env.sh)
[[ -f "$HOME/cuda_env.sh" ]] && source "$HOME/cuda_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AERIAL_DA3_FP16=1
BASE="${TELLO_HUB:-http://192.168.54.2:8090}"
DEPTH="$ROOT/experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/da3_ft_head_fp16.pt"
SURV="$ROOT/experiments/aerial/rl/artifacts/shorew03_krs6_20261013"
MODE="${MODE:-bench}"
cd "$ROOT"

if [[ "$MODE" == "bench" ]]; then
  exec "$PY" - <<PY
import os, urllib.request
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
p = DepthMinPredictor.from_checkpoint("$DEPTH", device="cuda")
print("da3_ok", type(p._model).__name__)
from experiments.aerial.rl.consequence_model import load_checkpoint
ck = "$SURV/c_s218/consequence_best.pt"
try:
    load_checkpoint(ck, device="cuda")
    print("krs6_ok", ck)
except Exception as e:
    print("krs6_skip", e)
base = os.environ.get("TELLO_HUB", "$BASE")
try:
    urllib.request.urlopen(base + "/api/status", timeout=2).read()
    print("hub_ok", base)
except Exception as e:
    print("hub_skip", e)
print("bench_done_no_takeoff")
PY
fi

echo "refusing auto takeoff; set MODE=fly only after 飞/采"
exit 2
