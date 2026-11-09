#!/usr/bin/env bash
# Lumos finder on Orin 112. Default: import + config check, no walk, no takeoff.
# Needs /dev/video0 + H12 bridge :8092 for a real mission (not started here).
set -euo pipefail
ROOT="${ROOT:-$HOME/aerial-wam-v2}"
PY="${PY:-python3}"
CFG="$ROOT/experiments/aerial/deploy/aerial_finder/configs/lumos_dog_f35.json"
[[ -f "$HOME/cuda_env.sh" ]] && source "$HOME/cuda_env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AERIAL_DA3_FP16=1
cd "$ROOT"
MODE="${MODE:-bench}"

if [[ "$MODE" != "serve" ]]; then
  exec "$PY" - <<PY
import json, os
from pathlib import Path
cfg = json.loads(Path("$CFG").read_text())
missing = []
for k in ("out_dir",):
    Path(cfg[k]).mkdir(parents=True, exist_ok=True)
for p in [cfg["depth"]["path"], *cfg["nav"]["surv_ckpts"], *cfg["nav"]["prog_ckpts"], cfg["detector"]["model"]]:
    if not Path(p).is_file():
        missing.append(p)
print("camera", cfg["backend"]["camera"], "exists", Path(cfg["backend"]["camera"]).exists())
print("missing_weights", missing)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
DepthMinPredictor.from_checkpoint(cfg["depth"]["path"], device="cuda")
print("da3_ok")
print("bench_done_no_walk")
PY
fi

echo "MODE=serve would start HTTP 8088; not launching walk from this script"
exit 2
