#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V3=$ART/consequence_v3_20261005
exec >>$V3/ens_eval.log 2>&1
echo "START $(date -Is)"
while ! grep -q EXPAND_END $V3/east_expand.log 2>/dev/null || ! grep -q ENS_TRAIN_END $V3/ens_full.log 2>/dev/null; do sleep 60; done
$PY -c "
import json
p=json.load(open('$ART/consequence_probe_holdout_corridor_20261005/AUTO_REVIEW.json'))['probes']
print('east_new probes', len(p))"
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for M in m_full m_full_s1 m_full_s2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V3/$M/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V3/rows_east_${P}_$M.json > /dev/null 2>&1
    echo "== single $M east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $V3/rows_east_${P}_$M.json | head -1
    F="$F $V3/rows_east_${P}_$M.json"
  done
  echo "== ENSEMBLE east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
echo "ENS_EVAL_END $(date -Is)"
