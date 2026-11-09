#!/bin/bash
# v9 data with branch states on the val routes held out and counted in checkpoint selection -> gate -> train + east -> fresh if promoted.
# Pre-declared promotion (same as v9): (arrived - collisions) over train(2 seeds x 16) + east(3 seeds) must exceed v8's 25.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007; FAIL2=$ART/consequence_fail2_20261008
V9R=$ART/consequence_v9r_20261008
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK=$V9R/m_full_s0/consequence_best.pt,$V9R/m_full_s1/consequence_best.pt,$V9R/m_full_s2/consequence_best.pt
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
FRESH_ANN=$ART/interior_od_fresh_20261007/fresh_holdout.json
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95"
mkdir -p $V9R $ART/logs; exec >>$V9R/run.log 2>&1
echo "V9R START $(date -Is)"
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }

for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V9R/m_full_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL,$FAIL2 --val-routes $VAL --branch-val --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V9R/train_full_s$S.log 2>&1
  echo "train s$S exit $? best=$($PY -c "import torch;print(torch.load('$V9R/m_full_s$S/consequence_best.pt',map_location='cpu',weights_only=False).get('step'))" 2>/dev/null) $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V9R/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V9R/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V9R/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v9r east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
cat > $V9R/sets.json <<EOF
{"east_old": {"models": {"v9r": ["$V9R/rows_east_old_s0.json", "$V9R/rows_east_old_s1.json", "$V9R/rows_east_old_s2.json"]}},
 "east_new": {"models": {"v9r": ["$V9R/rows_east_new_s0.json", "$V9R/rows_east_new_s1.json", "$V9R/rows_east_new_s2.json"]}}}
EOF
$PY experiments/aerial/scripts/consequence/audit_v8_v9.py $V9R/sets.json

rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
fly() {  # out_dir expected_n extra-args...
  local D=$1 N=$2; shift 2; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --out-dir $D $RULE "$@" \
      >> $ART/logs/consequence_mpc_v9r_20261008.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
for S in 0 1; do fly $V9R/train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
for S in 0 1 2; do fly $ART/consequence_mpc_v9r_band_s${S}_20261008 18 --seed $S; done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "v9r_band=consequence_mpc_v9r_band_s0_20261008,consequence_mpc_v9r_band_s1_20261008,consequence_mpc_v9r_band_s2_20261008" \
  "v8_band=consequence_mpc_v8_band_s0_20261007,consequence_mpc_v8_band_s1_20261007,consequence_mpc_v8_band_s2_20261007"
PROMOTE=$($PY - <<EOF
import json, glob
def score(pat, skip=()):
    eps = [e for d in sorted(glob.glob(pat)) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"]
           if "arrived" in e and e["route_id"] not in skip]
    return sum(e["arrived"] for e in eps), sum(e["severe_collision"] for e in eps), len(eps)
skip = ("od_holdout_06", "od_holdout_13")
t9 = score("$V9R/train_seed?"); t8 = score("$ART/consequence_tune_band_20261007/ret0.0_seed?")
e9 = score("$ART/consequence_mpc_v9r_band_s?_20261008", skip); e8 = score("$ART/consequence_mpc_v8_band_s?_20261007", skip)
print(f"TRAIN v9r arr/coll/n {t9}  v8 {t8}")
print(f"EAST  v9r arr/coll/n {e9}  v8 {e8}")
s9 = t9[0] - t9[1] + e9[0] - e9[1]; s8 = t8[0] - t8[1] + e8[0] - e8[1]
print(f"SCORE v9r {s9} v8 {s8}")
print("PROMOTE", "yes" if s9 > s8 else "no")
EOF
)
echo "$PROMOTE"
if echo "$PROMOTE" | grep -q "PROMOTE yes"; then
  echo "v9r promoted -> fresh held-out (second use of this set) $(date -Is)"
  for S in 0 1 2; do fly $ART/consequence_mpc_fresh_v9r_s${S}_20261008 27 --annotation $FRESH_ANN --use-spawn-z --seed $S; done
  echo "V9R_FRESH_DONE $(date -Is)"
fi
echo "V9R_END $(date -Is)"
