# V12 mainline stack (source from eval scripts; do not execute directly).
# V11 + terminal goal pin/creep + near-miss retry.
ROOT_V12="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_V12/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

WAM_PHASE2_V12_STACK=(
  "${WAM_PHASE2_V11_STACK[@]}"
  --terminal-pin-rem-m 20
  --terminal-creep-rem-m 15
  --near-miss-retry-m 5
  --near-miss-retries 1
)
