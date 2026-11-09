#!/bin/bash
# Clean evaluation set + retrain variance.
# 1) New held-out interior routes (interior_east + interior_west_northeast): drop candidates crossing a measured
#    invisible volume or duplicating an existing route, validate in AirSim, drop routes whose eval-style reset
#    touches geometry. Gate: >= 20 blocked clean routes in the pool; otherwise draw more and repeat.
# 2) Meanwhile retrain mem8 (v8 data, 1 s history) and control (v8 data + long fail states, no history) with
#    seeds 10-12 and 20-22, so each recipe has three ensembles like v8 (v8, reseed a, reseed b).
# 3) Fly all nine ensembles with the fov rule on the clean pool (seed 0) and on the clean east routes (seed 0).
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007; LONGB=$ART/consequence_fail_long_20261010
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
OUT=$ART/clean_eval_20261012; CR=$OUT/routes; M=$OUT/models
EAST_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_holdout.json
FRESH_ANN=$ART/interior_od_fresh_20261007/fresh_holdout.json
EAST_KEEP=od_holdout_00,od_holdout_01,od_holdout_03,od_holdout_04,od_holdout_05,od_holdout_07,od_holdout_08,od_holdout_11,od_holdout_12,od_holdout_17,od_holdout_19,od_holdout_22
POOL=$CR/clean_pool.json
mkdir -p $CR $M $ART/logs; exec >>$OUT/run.log 2>&1
echo "CLEAN START $(date -Is)"

train() {  # out seed extra-args...
  local D=$1 S=$2; shift 2
  [ -f $D/consequence_best.pt ] && [ -f $D/train_history.json ] && return 0
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $D --val-routes $VAL --input-depth da3 \
    --seed $S --steps 10000 --eval-every 500 "$@" > $D.log 2>&1
  echo "train $(basename $D) exit $? $(date -Is)"
}
(
  for S in 10 11 12 20 21 22; do
    train $M/mem8_s$S $S --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL --frame-offsets 0,2,5 --hist-actions 5
    train $M/ctrl_s$S $S --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL,$LONGB
  done
  echo "TRAIN_DONE $(date -Is)"
) &
TRAIN_PID=$!

rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
gate() { $PY -c "import json;print(int(json.load(open('$CR/AUTO_REVIEW_CLEAN.json'))['summary']['gate_pass']))" 2>/dev/null || echo 0; }

SEED=20261012; R=0
while [ "$(gate)" != 1 ]; do
  [ $R -ge 8 ] && { echo "CLEAN ROUTE GATE FAIL after $R rounds $(date -Is)"; break; }
  D=$CR/round$R; mkdir -p $D
  $PY experiments/aerial/scripts/generate_interior_od_routes.py --regions $ART/interior_od_fresh_20261007/regions_fresh.json \
    --train-n 0 --holdout-regions interior_east,interior_west_northeast --holdout-n 30,30 --seed $SEED \
    --out $D/candidates_raw.json > $D/generate.log 2>&1
  $PY experiments/aerial/scripts/consequence/clean_routes.py prefilter --cand $D/candidates_raw.json --out $D/candidates.json \
    --review $D/PREFILTER.json --existing $EAST_ANN,$FRESH_ANN,$POOL,$CR/seen.json --seen $CR/seen.json
  wait_airsim || { bash /home/yao/aerial_airsim_persistent/recover_renderer.sh >/dev/null 2>&1; sleep 120; }
  timeout 7200 $PY -u experiments/aerial/scripts/validate_od_routes_airsim.py --candidates $D/candidates.json \
    --out-dir $D --ann-prefix $D/val --min-train-blocked 0 --min-holdout-blocked 0 > $D/validate.log 2>&1
  echo "round $R validate exit $? $(date -Is)"
  wait_airsim || { bash /home/yao/aerial_airsim_persistent/recover_renderer.sh >/dev/null 2>&1; sleep 120; }
  timeout 3600 $PY -u experiments/aerial/scripts/consequence/clean_routes.py spawncheck --ann $D/val_holdout.json \
    --pool $POOL --review $CR/AUTO_REVIEW_CLEAN.json > $D/spawncheck.log 2>&1
  echo "round $R $(tail -1 $D/spawncheck.log) $(date -Is)"
  R=$((R + 1)); SEED=$((SEED + 1))
done
NP=$($PY -c "import json;print(len(json.load(open('$POOL'))['episodes']))" 2>/dev/null || echo 0)
echo "CLEAN ROUTES pool=$NP gate=$(gate) $(date -Is)"
wait $TRAIN_PID

RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --fov-max-deg 35"
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
fly() {  # ckpts out_dir expected_n extra-args...
  local CK=$1 D=$2 N=$3; shift 3; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    timeout 10800 $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --out-dir $D $RULE "$@" \
      >> $ART/logs/consequence_clean_eval_20261012.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
ck() { local P=$1; shift; local s=""; for S in "$@"; do s="$s,$P$S/consequence_best.pt"; done; echo ${s#,}; }
declare -A ENS
ENS[v8]=$(ck $ART/consequence_v8_20261007/m_full_s 0 1 2)
ENS[v8_rsA]=$(ck $ART/consequence_v8_reseed_20261010/m_s 10 11 12)
ENS[v8_rsB]=$(ck $ART/consequence_v8_reseed_20261010/m_s 20 21 22)
ENS[mem8]=$(ck $ART/consequence_mem8_20261009/m_s 0 1 2)
ENS[mem8_rsA]=$(ck $M/mem8_s 10 11 12)
ENS[mem8_rsB]=$(ck $M/mem8_s 20 21 22)
ENS[ctrl]=$(ck $ART/consequence_memlong_20261010/c_s 0 1 2)
ENS[ctrl_rsA]=$(ck $M/ctrl_s 10 11 12)
ENS[ctrl_rsB]=$(ck $M/ctrl_s 20 21 22)
for K in v8 v8_rsA v8_rsB mem8 mem8_rsA mem8_rsB ctrl ctrl_rsA ctrl_rsB; do
  fly ${ENS[$K]} $OUT/fly_pool_${K}_s0 $NP --annotation $POOL --use-spawn-z --seed 0
done
echo "POOL FLIGHTS DONE $(date -Is)"
for K in mem8_rsA mem8_rsB ctrl_rsA ctrl_rsB; do
  fly ${ENS[$K]} $OUT/fly_east_${K}_s0 12 --annotation $EAST_ANN --routes $EAST_KEEP --seed 0
done
touch /tmp/airsim_watchdog.stop
echo "CLEAN_END $(date -Is)"
