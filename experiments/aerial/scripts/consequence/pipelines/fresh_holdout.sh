#!/bin/bash
# Fresh held-out interior routes (interior_west_northeast) -> strict alt-band check on TRAIN routes -> frozen method 3x on fresh.
# Pre-declared rule for the band return (train only): adopt 0.05 unless it arrives less than 0.0 or collides >2 more; else 0.0.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V8=$ART/consequence_v8_20261007
CK8=$V8/m_full_s0/consequence_best.pt,$V8/m_full_s1/consequence_best.pt,$V8/m_full_s2/consequence_best.pt
FR=$ART/interior_od_fresh_20261007; FRESH_ANN=$FR/fresh_holdout.json
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
T=$ART/consequence_tune_band_20261007; mkdir -p $T $ART/logs
exec >>$FR/run.log 2>&1
echo "FRESH START $(date -Is)"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }

# 1) validate fresh candidates in AirSim; gate = >=12 blocked held-out routes, else draw more candidates and re-validate
SEED=20261007; NC=30; t=0
while :; do
  wait_airsim || { bash /home/yao/aerial_airsim_persistent/recover_renderer.sh >/dev/null 2>&1; sleep 120; }
  timeout 7200 $PY -u experiments/aerial/scripts/validate_od_routes_airsim.py --candidates $FR/candidates.json \
    --out-dir $FR --ann-prefix $FR/fresh --min-train-blocked 0 --min-holdout-blocked 12 >> $FR/validate.log 2>&1
  rc=$?
  NB=$($PY -c "import json;print(json.load(open('$FR/AUTO_REVIEW.json'))['summary']['holdout']['blocked'])" 2>/dev/null || echo 0)
  echo "validate attempt $t exit $rc blocked=$NB $(date -Is)"
  [ "$NB" -ge 12 ] && break
  t=$((t + 1)); [ $t -ge 4 ] && { echo "FRESH GATE FAIL after $t attempts; stopping before any flight"; touch /tmp/airsim_watchdog.stop; exit 1; }
  SEED=$((SEED + 1)); NC=$((NC + 15))
  $PY experiments/aerial/scripts/generate_interior_od_routes.py --regions $FR/regions_fresh.json --train-n 0 \
    --holdout-regions interior_west_northeast --holdout-n $NC --seed $SEED --out $FR/candidates.json
done
NF=$($PY -c "import json;print(len(json.load(open('$FRESH_ANN'))['episodes']))")
echo "fresh annotation routes=$NF $(date -Is)"

fly() {  # out_dir expected_n extra-args...
  local D=$1 N=$2; shift 2; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK8 --out-dir $D "$@" \
      >> $ART/logs/consequence_fresh_20261007.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
RULE="--policy model --alt-fix --alt-band --replan-every 1 --surv-min 0.95"

# 2) band return on train routes only
for RET in 0.0 0.05; do
  for S in 0 1; do
    fly $T/ret${RET}_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR $RULE --alt-band-return $RET --seed $S
  done
done
WIN=$($PY - <<EOF
import json, glob
r = {}
for ret in ["0.0", "0.05"]:
    eps = [e for d in sorted(glob.glob("$T/ret" + ret + "_seed?")) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"] if "arrived" in e]
    arr = sum(e["arrived"] for e in eps); col = sum(e["severe_collision"] for e in eps)
    prog = sum(e["progress_ratio"] for e in eps) / max(1, len(eps))
    r[ret] = (arr, col)
    print(f"TRAIN ret={ret}: flights={len(eps)} arrived={arr} collisions={col} mean_progress={prog:.3f}", flush=True)
ok = r["0.05"][0] >= r["0.0"][0] and r["0.05"][1] <= r["0.0"][1] + 2
print("WINNER", "0.05" if ok else "0.0")
EOF
)
echo "$WIN"
RET=$(echo "$WIN" | sed -n 's/^WINNER //p'); OTHER=0.05; [ "$RET" = "0.05" ] && OTHER=0.0
echo "FROZEN: v8 + alt-band return $RET + replan 1 + surv 0.95 $(date -Is)"

# 3) frozen method on fresh held-out, then baselines (comparisons only, nothing chosen from them)
for S in 0 1 2; do fly $ART/consequence_mpc_fresh_v8_ret${RET}_s${S}_20261007 $NF --annotation $FRESH_ANN $RULE --alt-band-return $RET --seed $S; done
echo "FRESH_FROZEN_DONE $(date -Is)"
for S in 0 1 2; do fly $ART/consequence_mpc_fresh_goalline_s${S}_20261007 $NF --annotation $FRESH_ANN --policy goal_line --seed $S; done
for S in 0 1 2; do fly $ART/consequence_mpc_fresh_v8_ret${OTHER}_s${S}_20261007 $NF --annotation $FRESH_ANN $RULE --alt-band-return $OTHER --seed $S; done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "v8_ret$RET=consequence_mpc_fresh_v8_ret${RET}_s0_20261007,consequence_mpc_fresh_v8_ret${RET}_s1_20261007,consequence_mpc_fresh_v8_ret${RET}_s2_20261007" \
  "v8_ret$OTHER=consequence_mpc_fresh_v8_ret${OTHER}_s0_20261007,consequence_mpc_fresh_v8_ret${OTHER}_s1_20261007,consequence_mpc_fresh_v8_ret${OTHER}_s2_20261007" \
  "goal_line=consequence_mpc_fresh_goalline_s0_20261007,consequence_mpc_fresh_goalline_s1_20261007,consequence_mpc_fresh_goalline_s2_20261007"
touch /tmp/airsim_watchdog.stop
echo "FRESH_END $(date -Is)"
