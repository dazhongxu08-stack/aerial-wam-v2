#!/bin/bash
# Retrain variance: v8's exact recipe and data with two new seed triples (10-12, 20-22), each flown as a
# 3-model ensemble with the fov rule on train (2 seeds x 16) + east (3 seeds). Not a promotion run: the
# question is whether a fresh retrain reproduces v8+fov (east arrived-coll 30 after artifact removal).
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
M=$ART/consequence_v8_reseed_20261010
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
mkdir -p $M $ART/logs; exec >>$M/run.log 2>&1
echo "RESEED START $(date -Is)"
for S in 10 11 12 20 21 22; do
  [ -f $M/m_s$S/consequence_best.pt ] && [ -f $M/m_s$S/train_history.json ] && continue
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $M/m_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $M/train_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
$PY - <<EOF
import json
for name, d, seeds in [("v8", "$ART/consequence_v8_20261007/m_full_s", (0, 1, 2)), ("reseed", "$M/m_s", (10, 11, 12, 20, 21, 22))]:
    for s in seeds:
        h = json.load(open(f"{d}{s}/train_history.json"))["history"]
        b = min(h, key=lambda r: r["val"]["hazard"] * 2 + r["val"]["cones"] + r["val"]["dgoal"])
        print(f"VAL {name} s{s} best_step={b['step']} hazard={b['val']['hazard']:.4f} cones={b['val']['cones']:.4f} dgoal={b['val']['dgoal']:.4f}")
EOF
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --fov-max-deg 35"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
fly() {  # ckpts out_dir expected_n extra-args...
  local CK=$1 D=$2 N=$3; shift 3; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    timeout 7200 $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --out-dir $D $RULE "$@" \
      >> $ART/logs/consequence_v8_reseed_20261010.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
for R in a b; do
  if [ $R = a ]; then SS="10 11 12"; else SS="20 21 22"; fi
  CK=$(for S in $SS; do printf "%s," $M/m_s$S/consequence_best.pt; done); CK=${CK%,}
  for S in 0 1; do fly $CK $M/${R}_train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
  for S in 0 1 2; do fly $CK $ART/consequence_mpc_reseed_${R}_band_s${S}_20261010 18 --seed $S; done
  echo "REPLICA $R DONE $(date -Is)"
done
touch /tmp/airsim_watchdog.stop
$PY experiments/aerial/scripts/consequence/sim_artifact_summary.py \
  --replays $ART/collision_probe_20261010/RESULTS.json,$ART/collision_probe_hist_20261010/RESULTS.json \
  "v8fov_east=consequence_mpc_fov_band_s?_20261009" "rsA_east=consequence_mpc_reseed_a_band_s?_20261010" \
  "rsB_east=consequence_mpc_reseed_b_band_s?_20261010" "v8fov_train=consequence_fov_20261009/train_seed?" \
  "rsA_train=consequence_v8_reseed_20261010/a_train_seed?" "rsB_train=consequence_v8_reseed_20261010/b_train_seed?"
echo "RESEED_END $(date -Is)"
