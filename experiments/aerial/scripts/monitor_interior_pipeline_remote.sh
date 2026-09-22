#!/usr/bin/env bash
# Local wrapper: sync monitor to 125 and keep it running (SSH retry).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SSH_HOST="${SSH_HOST:-cursor-125-public}"
STAMP="${STAMP:-20260916_night}"

upload() {
  tar czf - -C "$ROOT" \
    experiments/aerial/scripts/monitor_interior_pipeline_resume.sh \
    experiments/aerial/scripts/run_interior_pipeline_from_checkpoint.sh \
    experiments/aerial/scripts/run_interior_full_pipeline.sh \
    experiments/aerial/scripts/wam_phase2_v11_interior_terminal_eval.sh \
    experiments/aerial/scripts/wam_phase2_v11_interior_ft_sr_eval.sh \
    experiments/aerial/rl/backfill_polyline_subgoals.py \
    | ssh -o BatchMode=yes -o ConnectTimeout=40 "$SSH_HOST" \
      "cd ~/aerial-wam-v2 && tar xzf - 2>/dev/null; chmod +x experiments/aerial/scripts/monitor_interior_pipeline_resume.sh experiments/aerial/scripts/run_interior_pipeline_from_checkpoint.sh"
}

start_remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=40 "$SSH_HOST" bash -s <<REMOTE
set -euo pipefail
cd ~/aerial-wam-v2
pkill -f monitor_interior_pipeline_resume.sh 2>/dev/null || true
sleep 1
# seed checkpoint state from artifacts already on disk
export STAMP=${STAMP}
STATE=artifacts/pipeline_checkpoint_\${STAMP}.json
python3 - <<'PY' "\$STATE"
import json, os, datetime
from pathlib import Path
stamp = os.environ["STAMP"]
root = Path.home() / "aerial-wam-v2"
done = []
if (root / f"artifacts/review_all_interior_routes_20260917.json").is_file():
    done.append("step0_review")
if (root / f"artifacts/wam_phase2_v11_interior_terminal_eval_{stamp}/eval_all.json").is_file():
    done.append("step1a_terminal")
poly = root / f"experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_{stamp}_poly_curated"
if poly.is_dir() and list(poly.glob("episode_*.npz")):
    done.append("step1b_poly_backfill")
if (root / f"experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_{stamp}_poly/v4_ac_latest.pt").is_file():
    done.append("step2_poly_ft")
if (root / f"experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_{stamp}_focus134/v4_ac_latest.pt").is_file():
    done.append("step2b_focus_ft")
json.dump({"done": sorted(set(done)), "seeded": datetime.datetime.now().astimezone().isoformat()}, open(os.environ["STATE"], "w"), indent=2)
print("seeded", done)
PY
nohup bash experiments/aerial/scripts/monitor_interior_pipeline_resume.sh \
  >> artifacts/monitor_interior_pipeline_${STAMP}.nohup.log 2>&1 &
echo monitor_pid=\$!
REMOTE
}

upload || true
start_remote
