#!/usr/bin/env bash
# Progress monitor for overnight PathExpert collect on 125 (cursor-125-public).
set -euo pipefail

SSH_HOST="${SSH_HOST:-cursor-125-public}"
OUT_REL="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night"
CUR_REL="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night_curated"
LOG_REL="artifacts/collect_urban_complex_path_expert_20260916_night_until_targets.log"
TARGET_U="${TARGET_U:-20}"
TARGET_A="${TARGET_A:-8}"

fetch_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_HOST" bash -s <<'REMOTE'
OUT=~/aerial-wam-v2/experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night
CUR=~/aerial-wam-v2/experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night_curated
LOG=~/aerial-wam-v2/artifacts/collect_urban_complex_path_expert_20260916_night_until_targets.log
PUSH_LOG=~/aerial-wam-v2/artifacts/collect_urban_complex_path_expert_20260916_night_inland_push.log
npz=$(ls -1 "$OUT"/episode_*.npz 2>/dev/null | wc -l | tr -d ' ')
running=$(pgrep -af collect_path_expert_dataset | head -1 | sed 's/.*route-indices/route-indices/' | cut -c1-80)
pass=$(grep -E "PASS |TARGETS MET|until-targets|CYCLE " "$LOG" "$PUSH_LOG" 2>/dev/null | tail -1 | sed 's/.*=== /=== /' | cut -c1-100)
last=$(grep "wrote episode" "$LOG" "$PUSH_LOG" 2>/dev/null | tail -1 | sed 's/.*wrote //')
cur_u=0 cur_a=0 cur_n=0
if test -f "$CUR/manifest.json"; then
  read -r cur_n cur_u cur_a < <(python3 - <<'PY'
import json
d=json.load(open("/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night_curated/manifest.json"))
e=d.get("episodes",[])
print(len(e), sum(1 for x in e if x.get("usable")), sum(1 for x in e if x.get("arrived")))
PY
)
fi
echo "NPZ=${npz} CUR=${cur_n}/${cur_u}u/${cur_a}a RUN=${running:-DONE} PASS=${pass:-?} LAST=${last:-none}"
REMOTE
}

notify_mac() {
  local msg="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"PathExpert 补采 125\"" 2>/dev/null || true
  fi
  echo "AGENT_LOOP_TICK_overnight_pe {\"prompt\":\"追进度 overnight PathExpert 补采\",\"status\":\"${msg//\"/\\\"}\"}"
}

main_once() {
  local line
  line="$(fetch_status 2>/dev/null || echo "SSH_FAIL")"
  notify_mac "$line"
}

main_once
