#!/bin/bash
# After mem8: collect 250 fail-mode branch states that keep 21 history frames (v8 policy, train interior routes),
# then a 4 s history model (depth t, t-2, t-5, t-10, t-20 + last 20 actions) on v8's data + these states.
# Old probes keep 2 or 6 frames; their missing past is clamped to the oldest frame, as at a flight start.
# Control = no-history model on the same data, flown after, to separate history from the extra data.
# Pre-declared promotion: (arrived - collisions) over train(2 seeds x 16) + east(3 seeds) must exceed the best promoted
# so far (v8 with the current best rule; mem8 if promoted). Fresh held-out only if promoted.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007
LONGB=$ART/consequence_fail_long_20261010
V8=$ART/consequence_v8_20261007; ESC=$ART/consequence_escape_20261009; FOV=$ART/consequence_fov_20261009
MEM8=$ART/consequence_mem8_20261009
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
M=$ART/consequence_memlong_20261010
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
FRESH_ANN=$ART/interior_od_fresh_20261007/fresh_holdout.json
CK8=$V8/m_full_s0/consequence_best.pt,$V8/m_full_s1/consequence_best.pt,$V8/m_full_s2/consequence_best.pt
TARGET=250
mkdir -p $M $ART/logs; exec >>$M/run.log 2>&1
echo "MEMLONG WAIT for mem8 $(date -Is)"
while ! grep -q "MEM8_END" $MEM8/run.log 2>/dev/null; do sleep 60; done
echo "MEMLONG START $(date -Is)"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_states() { $PY -c "import json;print(len(json.load(open('$LONGB/AUTO_REVIEW.json'))['probes']))" 2>/dev/null || echo 0; }
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }

tries=0
while [ "$(n_states)" -lt $TARGET ] && [ $tries -lt 15 ]; do
  wait_airsim || { sleep 300; tries=$((tries + 1)); continue; }
  timeout 43200 $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK8 --out-dir $LONGB --n-states $TARGET \
    --branch-mode fail --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --stuck-s 40 --max-steps 500 \
    --lowclear-region any --lowclear-m 4.0 --fail-lowclear-n 2 --hist-frames 21 --seed 81 \
    >> $ART/logs/consequence_fail_long_20261010.log 2>&1
  echo "collect attempt $tries exit $? states=$(n_states) $(date -Is)"
  tries=$((tries + 1))
done
N=$(n_states)
echo "collect done states=$N target=$TARGET $(date -Is)"
[ "$N" -lt $TARGET ] && echo "WARNING: states below target ($N < $TARGET); continuing with what was collected"
$PY -c "
import json, collections, numpy as np
r=json.load(open('$LONGB/AUTO_REVIEW.json')); p=r['probes']
print('long states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'cand_coll_rate %.2f' % np.mean([np.mean(x['collided']) for x in p]),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']),
      collections.Counter(x['why'].split('_')[0] for x in p), 'routes', len({x['route_id'] for x in p}))"

BR=$BRANCH,$ONP1,$LOW,$FAIL,$LONGB
for S in 0 1 2; do
  if ! { [ -f $M/m_s$S/consequence_best.pt ] && [ -f $M/m_s$S/train_history.json ]; }; then
    $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $M/m_s$S --branch-dirs $BR \
      --val-routes $VAL --input-depth da3 --seed $S --frame-offsets 0,2,5,10,20 --hist-actions 20 \
      --steps 10000 --eval-every 500 > $M/train_s$S.log 2>&1
    echo "train long s$S exit $? $(date -Is)"
  fi
  if ! { [ -f $M/c_s$S/consequence_best.pt ] && [ -f $M/c_s$S/train_history.json ]; }; then
    $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $M/c_s$S --branch-dirs $BR \
      --val-routes $VAL --input-depth da3 --seed $S --steps 10000 --eval-every 500 > $M/train_c_s$S.log 2>&1
    echo "train control s$S exit $? $(date -Is)"
  fi
done
$PY - <<EOF
import json
for name, d in [("v8", "$V8/m_full_s"), ("mem8", "$MEM8/m_s"), ("control", "$M/c_s"), ("long", "$M/m_s")]:
    for s in range(3):
        h = json.load(open(f"{d}{s}/train_history.json"))["history"]
        b = min(h, key=lambda r: r["val"]["hazard"] * 2 + r["val"]["cones"] + r["val"]["dgoal"])
        print(f"VAL {name} s{s} best_step={b['step']} hazard={b['val']['hazard']:.4f} cones={b['val']['cones']:.4f} dgoal={b['val']['dgoal']:.4f} collide_abs_err={b['val']['collide_abs_err']:.4f}")
EOF
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  for K in m c; do
    F=""
    for S in 0 1 2; do
      $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $M/${K}_s$S/consequence_best.pt \
        --select filter --surv-min 0.7 --out $M/rows_east_${P}_${K}_s$S.json > /dev/null 2>&1
      F="$F $M/rows_east_${P}_${K}_s$S.json"
    done
    echo "== ENSEMBLE ${K} east_$P (m=long, c=control)"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
  done
done
echo "MEMLONG OFFLINE DONE $(date -Is)"

if grep -q "PROMOTE yes" $ESC/run.log; then
  EXTRA="--escape-after-s 8 --escape-s 4"; BASE_T="$ESC/train_seed?"; BASE_E="$ART/consequence_mpc_escape_band_s?_20261009"
else
  EXTRA=""; BASE_T="$FOV/train_seed?"; BASE_E="$ART/consequence_mpc_fov_band_s?_20261009"
fi
M8T=""; M8E=""
grep -q "PROMOTE yes" $MEM8/run.log && { M8T="$MEM8/train_seed?"; M8E="$ART/consequence_mpc_mem8_band_s?_20261009"; }
echo "rule extra='$EXTRA' base $BASE_T $BASE_E mem8 '$M8T'"
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --fov-max-deg 35 $EXTRA"
CKL=$M/m_s0/consequence_best.pt,$M/m_s1/consequence_best.pt,$M/m_s2/consequence_best.pt
CKC=$M/c_s0/consequence_best.pt,$M/c_s1/consequence_best.pt,$M/c_s2/consequence_best.pt
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
fly() {  # ckpts out_dir expected_n extra-args...
  local CK=$1 D=$2 N=$3; shift 3; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    timeout 7200 $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --out-dir $D $RULE "$@" \
      >> $ART/logs/consequence_mpc_memlong_20261010.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
score_py() {  # name train_glob east_glob bar_pairs...
$PY - "$@" <<'EOF'
import json, glob, sys
def score(pat, skip=()):
    eps = [e for d in sorted(glob.glob(pat)) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"]
           if "arrived" in e and e["route_id"] not in skip]
    return sum(e["arrived"] for e in eps), sum(e["severe_collision"] for e in eps), sum(e["stuck"] for e in eps), len(eps)
skip = ("od_holdout_06", "od_holdout_13")
name, tp, ep, rest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
t, e = score(tp), score(ep, skip)
s = t[0] - t[1] + e[0] - e[1]
bars = []
for i in range(0, len(rest), 2):
    tb, eb = score(rest[i]), score(rest[i + 1], skip)
    bars.append(tb[0] - tb[1] + eb[0] - eb[1])
    print(f"BAR {rest[i]} train {tb} east {eb} score {bars[-1]}")
print(f"TRAIN {name} arr/coll/stuck/n {t}")
print(f"EAST  {name} arr/coll/stuck/n {e}")
print(f"SCORE {name} {s} bar {max(bars)}")
print("PROMOTE", "yes" if s > max(bars) else "no")
EOF
}
for S in 0 1; do fly $CKL $M/train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
for S in 0 1 2; do fly $CKL $ART/consequence_mpc_memlong_band_s${S}_20261010 18 --seed $S; done
PROMOTE=$(score_py long "$M/train_seed?" "$ART/consequence_mpc_memlong_band_s?_20261010" "$BASE_T" "$BASE_E" ${M8T:+"$M8T" "$M8E"})
echo "$PROMOTE"
if echo "$PROMOTE" | grep -q "PROMOTE yes"; then
  echo "long promoted -> fresh held-out (another use of this set) $(date -Is)"
  for S in 0 1 2; do fly $CKL $ART/consequence_mpc_fresh_memlong_s${S}_20261010 27 --annotation $FRESH_ANN --use-spawn-z --seed $S; done
  echo "MEMLONG_FRESH_DONE $(date -Is)"
fi
echo "control flights (attribution only, not a promotion candidate) $(date -Is)"
for S in 0 1; do fly $CKC $M/ctrl_train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
for S in 0 1 2; do fly $CKC $ART/consequence_mpc_memlong_ctrl_band_s${S}_20261010 18 --seed $S; done
score_py control "$M/ctrl_train_seed?" "$ART/consequence_mpc_memlong_ctrl_band_s?_20261010" "$M/train_seed?" "$ART/consequence_mpc_memlong_band_s?_20261010" | grep -v PROMOTE
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "long=consequence_mpc_memlong_band_s0_20261010,consequence_mpc_memlong_band_s1_20261010,consequence_mpc_memlong_band_s2_20261010" \
  "control=consequence_mpc_memlong_ctrl_band_s0_20261010,consequence_mpc_memlong_ctrl_band_s1_20261010,consequence_mpc_memlong_ctrl_band_s2_20261010" \
  "v8_band=consequence_mpc_v8_band_s0_20261007,consequence_mpc_v8_band_s1_20261007,consequence_mpc_v8_band_s2_20261007"
touch /tmp/airsim_watchdog.stop
echo "MEMLONG_END $(date -Is)"
