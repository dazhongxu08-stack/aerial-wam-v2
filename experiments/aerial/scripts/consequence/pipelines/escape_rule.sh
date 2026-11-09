#!/bin/bash
# Current best = v8 + visible-motion rule (fov 35). Add the escape mode (after 8 s without 1 m of goal progress,
# 4 s of choosing the survivor with the largest predicted forward clearance). Settings fixed before any flight.
# Pre-declared promotion: (arrived - collisions) over train(2 seeds x 16) + east(3 seeds) must exceed fov.
# Fresh held-out only if promoted.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; M=$ART/consequence_v8_20261007; FOV=$ART/consequence_fov_20261009
OUT=$ART/consequence_escape_20261009
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
FRESH_ANN=$ART/interior_od_fresh_20261007/fresh_holdout.json
mkdir -p $OUT; exec >>$OUT/run.log 2>&1
echo "ESCAPE START $(date -Is)"
CK=$M/m_full_s0/consequence_best.pt,$M/m_full_s1/consequence_best.pt,$M/m_full_s2/consequence_best.pt
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --fov-max-deg 35 --escape-after-s 8 --escape-s 4"
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
      >> $ART/logs/consequence_mpc_escape_20261009.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
for S in 0 1; do fly $OUT/train_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --seed $S; done
for S in 0 1 2; do fly $ART/consequence_mpc_escape_band_s${S}_20261009 18 --seed $S; done
PROMOTE=$($PY - <<EOF
import json, glob
def score(pat, skip=()):
    eps = [e for d in sorted(glob.glob(pat)) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"]
           if "arrived" in e and e["route_id"] not in skip]
    return sum(e["arrived"] for e in eps), sum(e["severe_collision"] for e in eps), sum(e["stuck"] for e in eps), len(eps)
skip = ("od_holdout_06", "od_holdout_13")
t = score("$OUT/train_seed?"); e = score("$ART/consequence_mpc_escape_band_s?_20261009", skip)
tb = score("$FOV/train_seed?"); eb = score("$ART/consequence_mpc_fov_band_s?_20261009", skip)
s = t[0] - t[1] + e[0] - e[1]; bar = tb[0] - tb[1] + eb[0] - eb[1]
print(f"TRAIN escape arr/coll/stuck/n {t}  fov {tb}")
print(f"EAST  escape arr/coll/stuck/n {e}  fov {eb}")
print(f"SCORE escape {s} bar {bar}")
print("PROMOTE", "yes" if s > bar else "no")
EOF
)
echo "$PROMOTE"
if echo "$PROMOTE" | grep -q "PROMOTE yes"; then
  echo "escape promoted -> fresh held-out (another use of this set) $(date -Is)"
  for S in 0 1 2; do fly $ART/consequence_mpc_fresh_escape_s${S}_20261009 27 --annotation $FRESH_ANN --use-spawn-z --seed $S; done
  echo "ESCAPE_FRESH_DONE $(date -Is)"
fi
touch /tmp/airsim_watchdog.stop
echo "ESCAPE_END $(date -Is)"
