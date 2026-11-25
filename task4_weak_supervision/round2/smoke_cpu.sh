#!/bin/bash
# End-to-end CPU smoke test of every round 2 script and arm, on a few images.
# Writes only to $OUT (banks, runs); results.csv is never touched (SMOKE runs are not recorded).
set -eu
cd "$(dirname "$0")"
. ../../shared/config.sh   # paths and TASK4_PYTHON (env, then ../../.env, then server defaults)
PY=$TASK4_PYTHON
OUT=${OUT:-/tmp/task4r2_smoke}
export TASK4R2_HDD=$OUT OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
rm -rf "$OUT"; mkdir -p "$OUT/logs"
for d in Brain Liver Retina_RESC Retina_OCT2017 Chest Histopathology; do
  $PY build_normal_bank.py --dataset $d --device cpu --limit 4 --bank_size 300 > $OUT/logs/bank_$d.log 2>&1 &
done; wait
grep -h '"bank_size"' $OUT/logs/bank_*.log
S="--device cpu --workers 0 --max_per_source 4 --max_steps 6 --epochs 2 --val_limit 3 --eval_limit 8 --log_every 2"
pids=()
for arm in mvfa mil g pu pu_m100 g_m100 g_m10 g_m25; do
  $PY train2.py --arm $arm --fold Liver --seed 0 $S --save_maps > $OUT/logs/train_$arm.log 2>&1 & pids+=($!)
done
$PY train2.py --arm g --fold RESC_OCT --seed 1 $S > $OUT/logs/train_g_RESC_OCT.log 2>&1 & pids+=($!)
$PY train2.py --arm pu --fold Chest --seed 0 --pi 0.05 --tag pi5 $S > $OUT/logs/train_pu_Chest.log 2>&1 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
for f in $OUT/logs/train_*.log; do echo "== $f"; grep -E "epoch .* done|eval_dataset|Error|Traceback" $f | tail -6; done
ls $OUT/runs/*/
exit $fail
