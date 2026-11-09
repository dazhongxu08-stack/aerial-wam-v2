#!/bin/bash
# After the escape run: history-input consequence model (depth frames t, t-2, t-5 + last 5 actions) on v8's data
# (branch probes keep 6 frames, so offsets stop at 5). Flown with the current best rule (fov, plus escape if promoted).
# Pre-declared promotion: (arrived - collisions) over train(2 seeds x 16) + east(3 seeds) must exceed that best rule
# flown with v8. Fresh held-out only if promoted.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007
ESC=$ART/consequence_escape_20261009; FOV=$ART/consequence_fov_20261009
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
M=$ART/consequence_mem8_20261009
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
FRESH_ANN=$ART/interior_od_fresh_20261007/fresh_holdout.json
mkdir -p $M $ART/logs; exec >>$M/run.log 2>&1
echo "MEM8 WAIT for escape $(date -Is)"
while ! grep -q "ESCAPE_END" $ESC/run.log 2>/dev/null; do sleep 60; done
echo "MEM8 START $(date -Is)"
for S in 0 1 2; do
  [ -f $M/m_s$S/consequence_best.pt ] && [ -f $M/m_s$S/train_history.json ] && continue
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $M/m_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL --val-routes $VAL --input-depth da3 --seed $S \
    --frame-offsets 0,2,5 --hist-actions 5 --steps 10000 --eval-every 500 > $M/train_s$S.log 2>&1
  echo "train mem8 s$S exit $? $(date -Is)"
done
$PY - <<EOF
import json
for name, d in [("v8", "$ART/consequence_v8_20261007/m_full_s"), ("mem8", "$M/m_s")]:
    for s in range(3):
        h = json.load(open(f"{d}{s}/train_history.json"))["history"]
        b = min(h, key=lambda r: r["val"]["hazard"] * 2 + r["val"]["cones"] + r["val"]["dgoal"])
        print(f"VAL {name} s{s} best_step={b['step']} hazard={b['val']['hazard']:.4f} cones={b['val']['cones']:.4f} dgoal={b['val']['dgoal']:.4f} collide_abs_err={b['val']['collide_abs_err']:.4f}")
EOF
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $M/m_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $M/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $M/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE mem8 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
echo "MEM8 OFFLINE DONE $(date -Is)"

if grep -q "PROMOTE yes" $ESC/run.log; then
  EXTRA="--escape-after-s 8 --escape-s 4"; BASE_T="$ESC/train_seed?"; BASE_E="$ART/consequence_mpc_escape_band_s?_20261009"
else
  EXTRA=""; BASE_T="$FOV/train_seed?"; BASE_E="$ART/consequence_mpc_fov_band_s?_20261009"
fi
echo "rule extra='$EXTRA' base $BASE_T $BASE_E"
CK=$M/m_s0/consequence_best.pt,$M/m_s1/consequence_best.pt,$M/m_s2/consequence_best.pt
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --fov-max-deg 35 $EXTRA"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
fly() {  # out_dir expected_n extra-args...
  local D=$1 N=$2; shift 2; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    timeout 7200 $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --out-dir $D $RULE "$@" \
      >> $ART/logs/consequence_mpc_mem8_20261009.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
for S in 0 1; do fly $M/train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
for S in 0 1 2; do fly $ART/consequence_mpc_mem8_band_s${S}_20261009 18 --seed $S; done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "mem8=consequence_mpc_mem8_band_s0_20261009,consequence_mpc_mem8_band_s1_20261009,consequence_mpc_mem8_band_s2_20261009" \
  "v8_band=consequence_mpc_v8_band_s0_20261007,consequence_mpc_v8_band_s1_20261007,consequence_mpc_v8_band_s2_20261007"
PROMOTE=$($PY - <<EOF
import json, glob
def score(pat, skip=()):
    eps = [e for d in sorted(glob.glob(pat)) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"]
           if "arrived" in e and e["route_id"] not in skip]
    return sum(e["arrived"] for e in eps), sum(e["severe_collision"] for e in eps), sum(e["stuck"] for e in eps), len(eps)
skip = ("od_holdout_06", "od_holdout_13")
t = score("$M/train_seed?"); e = score("$ART/consequence_mpc_mem8_band_s?_20261009", skip)
tb = score("$BASE_T"); eb = score("$BASE_E", skip)
s = t[0] - t[1] + e[0] - e[1]; bar = tb[0] - tb[1] + eb[0] - eb[1]
print(f"TRAIN mem8 arr/coll/stuck/n {t}  base {tb}")
print(f"EAST  mem8 arr/coll/stuck/n {e}  base {eb}")
print(f"SCORE mem8 {s} bar {bar}")
print("PROMOTE", "yes" if s > bar else "no")
EOF
)
echo "$PROMOTE"
if echo "$PROMOTE" | grep -q "PROMOTE yes"; then
  echo "mem8 promoted -> fresh held-out (another use of this set) $(date -Is)"
  for S in 0 1 2; do fly $ART/consequence_mpc_fresh_mem8_s${S}_20261009 27 --annotation $FRESH_ANN --use-spawn-z --seed $S; done
  echo "MEM8_FRESH_DONE $(date -Is)"
fi
touch /tmp/airsim_watchdog.stop
echo "MEM8_END $(date -Is)"
