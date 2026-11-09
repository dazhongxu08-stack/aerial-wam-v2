#!/bin/bash
# Score v8 and v9 checkpoints on the fail2 branch states (v8 out-of-sample, v9 in-sample) and both east probe sets, then audit.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; OUT=$ART/consequence_audit_v8_v9_20261008; mkdir -p $OUT
exec >>$OUT/run.log 2>&1
echo "AUDIT START $(date -Is)"
for M in v8:consequence_v8_20261007 v9:consequence_v9_20261008; do
  N=${M%%:*}; D=$ART/${M#*:}
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $ART/consequence_fail2_20261008 \
      --ckpt $D/m_full_s$S/consequence_best.pt --select filter --surv-min 0.95 --out $OUT/rows_fail2_${N}_s$S.json > /dev/null 2>&1
    echo "scored fail2 $N s$S exit $? $(date -Is)"
  done
done
F=$ART/consequence_fail2_20261008/AUTO_REVIEW.json
cat > $OUT/sets.json <<EOF
{"fail2": {"why_file": "$F", "models": {
   "v8": ["$OUT/rows_fail2_v8_s0.json", "$OUT/rows_fail2_v8_s1.json", "$OUT/rows_fail2_v8_s2.json"],
   "v9(in-sample)": ["$OUT/rows_fail2_v9_s0.json", "$OUT/rows_fail2_v9_s1.json", "$OUT/rows_fail2_v9_s2.json"]}},
 "east_old": {"models": {
   "v8": ["$ART/consequence_v8_20261007/rows_east_old_s0.json", "$ART/consequence_v8_20261007/rows_east_old_s1.json", "$ART/consequence_v8_20261007/rows_east_old_s2.json"],
   "v9": ["$ART/consequence_v9_20261008/rows_east_old_s0.json", "$ART/consequence_v9_20261008/rows_east_old_s1.json", "$ART/consequence_v9_20261008/rows_east_old_s2.json"]}},
 "east_new": {"models": {
   "v8": ["$ART/consequence_v8_20261007/rows_east_new_s0.json", "$ART/consequence_v8_20261007/rows_east_new_s1.json", "$ART/consequence_v8_20261007/rows_east_new_s2.json"],
   "v9": ["$ART/consequence_v9_20261008/rows_east_new_s0.json", "$ART/consequence_v9_20261008/rows_east_new_s1.json", "$ART/consequence_v9_20261008/rows_east_new_s2.json"]}}}
EOF
$PY experiments/aerial/scripts/consequence/audit_v8_v9.py $OUT/sets.json > $OUT/AUDIT.txt 2>&1
echo "audit exit $? $(date -Is)"
echo "AUDIT_END $(date -Is)"
