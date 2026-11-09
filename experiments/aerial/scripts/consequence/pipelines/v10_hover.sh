#!/bin/bash
# After v9r: branch failure states with hover + spin-in-place candidates added (no backward motion) -> v10 -> train + east -> fresh if promoted.
# Pre-declared promotion: (arrived - collisions) over train(2 seeds x 16) + east(3 seeds) must exceed v8's 25, and v9r's if v9r was promoted.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007; FAIL2=$ART/consequence_fail2_20261008
FAIL3=$ART/consequence_fail3_hover_20261009
V8=$ART/consequence_v8_20261007; V9R=$ART/consequence_v9r_20261008; V10=$ART/consequence_v10_20261009
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK8=$V8/m_full_s0/consequence_best.pt,$V8/m_full_s1/consequence_best.pt,$V8/m_full_s2/consequence_best.pt
CK=$V10/m_full_s0/consequence_best.pt,$V10/m_full_s1/consequence_best.pt,$V10/m_full_s2/consequence_best.pt
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
FRESH_ANN=$ART/interior_od_fresh_20261007/fresh_holdout.json
EXTRA=hover,spin
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --extra-families $EXTRA"
TARGET=250
mkdir -p $V10 $ART/logs; exec >>$V10/run.log 2>&1
echo "V10 WAIT for v9r $(date -Is)"
while ! grep -q "V9R_END" $V9R/run.log 2>/dev/null; do sleep 60; done
echo "V10 START $(date -Is)"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_states() { $PY -c "import json;print(len(json.load(open('$FAIL3/AUTO_REVIEW.json'))['probes']))" 2>/dev/null || echo 0; }
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }

tries=0
while [ "$(n_states)" -lt $TARGET ] && [ $tries -lt 15 ]; do
  wait_airsim || { sleep 300; tries=$((tries + 1)); continue; }
  timeout 43200 $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK8 --out-dir $FAIL3 --n-states $TARGET \
    --branch-mode fail --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --stuck-s 40 --max-steps 500 \
    --lowclear-region any --lowclear-m 4.0 --fail-lowclear-n 2 --branch-extra-families $EXTRA --seed 71 \
    >> $ART/logs/consequence_fail3_hover_20261009.log 2>&1
  echo "collect attempt $tries exit $? states=$(n_states) $(date -Is)"
  tries=$((tries + 1))
done
N=$(n_states)
echo "collect done states=$N target=$TARGET $(date -Is)"
[ "$N" -lt $TARGET ] && echo "WARNING: states below target ($N < $TARGET); continuing with what was collected"
$PY -c "
import json, collections, numpy as np
r=json.load(open('$FAIL3/AUTO_REVIEW.json')); p=r['probes']
fam=collections.defaultdict(list)
for x in p:
    for f,c in zip(x['families'], x['collided']): fam[f].append(c)
print('fail3 states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'cand_coll_rate %.2f' % np.mean([np.mean(x['collided']) for x in p]),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']),
      collections.Counter(x['why'].split('_')[0] for x in p))
print('collision rate by family', {k: round(float(np.mean(v)), 3) for k, v in sorted(fam.items())})"
for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V10/m_full_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL,$FAIL2,$FAIL3 --val-routes $VAL --branch-val --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V10/train_full_s$S.log 2>&1
  echo "train s$S exit $? best=$($PY -c "import torch;print(torch.load('$V10/m_full_s$S/consequence_best.pt',map_location='cpu',weights_only=False).get('step'))" 2>/dev/null) $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V10/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V10/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V10/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v10 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done

fly() {  # out_dir expected_n extra-args...
  local D=$1 N=$2; shift 2; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --out-dir $D $RULE "$@" \
      >> $ART/logs/consequence_mpc_v10_20261009.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
for S in 0 1; do fly $V10/train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
for S in 0 1 2; do fly $ART/consequence_mpc_v10_band_s${S}_20261009 18 --seed $S; done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "v10_band=consequence_mpc_v10_band_s0_20261009,consequence_mpc_v10_band_s1_20261009,consequence_mpc_v10_band_s2_20261009" \
  "v8_band=consequence_mpc_v8_band_s0_20261007,consequence_mpc_v8_band_s1_20261007,consequence_mpc_v8_band_s2_20261007"
PROMOTE=$($PY - <<EOF
import json, glob
def score(pat, skip=()):
    eps = [e for d in sorted(glob.glob(pat)) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"]
           if "arrived" in e and e["route_id"] not in skip]
    return sum(e["arrived"] for e in eps), sum(e["severe_collision"] for e in eps), len(eps)
skip = ("od_holdout_06", "od_holdout_13")
t = score("$V10/train_seed?"); e = score("$ART/consequence_mpc_v10_band_s?_20261009", skip)
t8 = score("$ART/consequence_tune_band_20261007/ret0.0_seed?"); e8 = score("$ART/consequence_mpc_v8_band_s?_20261007", skip)
s = t[0] - t[1] + e[0] - e[1]; bar = t8[0] - t8[1] + e8[0] - e8[1]
print(f"TRAIN v10 arr/coll/n {t}  v8 {t8}")
print(f"EAST  v10 arr/coll/n {e}  v8 {e8}")
if "PROMOTE yes" in open("$V9R/run.log").read():
    t9 = score("$V9R/train_seed?"); e9 = score("$ART/consequence_mpc_v9r_band_s?_20261008", skip)
    bar = max(bar, t9[0] - t9[1] + e9[0] - e9[1]); print(f"v9r promoted; bar includes v9r {t9} {e9}")
print(f"SCORE v10 {s} bar {bar}")
print("PROMOTE", "yes" if s > bar else "no")
EOF
)
echo "$PROMOTE"
if echo "$PROMOTE" | grep -q "PROMOTE yes"; then
  echo "v10 promoted -> fresh held-out (another use of this set) $(date -Is)"
  for S in 0 1 2; do fly $ART/consequence_mpc_fresh_v10_s${S}_20261009 27 --annotation $FRESH_ANN --use-spawn-z --seed $S; done
  echo "V10_FRESH_DONE $(date -Is)"
fi
touch /tmp/airsim_watchdog.stop
echo "V10_END $(date -Is)"
