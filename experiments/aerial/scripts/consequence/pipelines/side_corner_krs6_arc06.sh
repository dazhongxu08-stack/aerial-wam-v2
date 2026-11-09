#!/bin/bash
# Continue the three krs6 survival seeds on the straight labels plus the held 0.6 m arc.
# New directory only. Does not write into shorew03_krs6. Does not fly.
cd /home/yao/aerial-wam-v2 || exit 1
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts
DATA=$ART/consequence_data_20261004
SIDE=artifacts/side_corner_branches_20261030,artifacts/side_corner_clear_20261102,artifacts/side_corner_arc06_20261102
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
M=$ART/side_corner_krs6_arc06_20261102
WATER=$ART/water_map_measured_20261012.npz
case "$M" in
  *shorew03_krs6*) echo "refusing to write the frozen krs6 dir"; exit 1 ;;
esac
mkdir -p "$M"
exec >>"$M/run.log" 2>&1
echo "ARC06 TRAIN START $(date -Is)"
for S in 218 219 220; do
  if [ -f "$M/c_s$S/consequence_best.pt" ] && [ -f "$M/c_s$S/train_history.json" ]; then
    echo "skip c_s$S already done $(date -Is)"
    continue
  fi
  $PY -u -m experiments.aerial.rl.train_consequence \
    --data-dir "$DATA" \
    --out-dir "$M/c_s$S" \
    --init-ckpt "$ART/shorew03_krs6_20261013/c_s$S/consequence_best.pt" \
    --branch-dirs "$SIDE" \
    --val-routes "$VAL" \
    --input-depth da3 \
    --seed "$S" \
    --steps 10000 --eval-every 500 --ckpt-every 2000 \
    --water-raster "$WATER" --min-inland-m 80 --shore-weight 0.3 \
    > "$M/train_s$S.log" 2>&1
  echo "train c_s$S exit $? $(date -Is)"
done
echo "ARC06 TRAIN END $(date -Is)"
