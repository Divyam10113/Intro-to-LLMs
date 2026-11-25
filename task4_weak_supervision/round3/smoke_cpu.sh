#!/bin/bash
# Round 3 CPU smoke test of the six round 3 arms, end to end on a few images (fallback while
# the GPUs are busy; the plan's GPU smoke test is ./run_all.sh smoke, run by go.sh).
# Writes only to $OUT; round 2's normal banks are read (diagnostic only); results.csv is never
# touched (SMOKE runs are not recorded). GPUs are hidden: torch's Adam.step opens a CUDA context
# even for CPU tensors, which fails when other users fill the GPUs (R3-10).
set -eu
cd "$(dirname "$0")"
. ../../shared/config.sh   # paths and TASK4_PYTHON (env, then ../../.env, then server defaults)
PY=$TASK4_PYTHON
OUT=${OUT:-/tmp/task4r3_smoke}
export CUDA_VISIBLE_DEVICES="" TASK4R3_HDD=$OUT TASK4_RUNS=$OUT/runs OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
rm -rf "$OUT"; mkdir -p "$OUT/logs"
S="--device cpu --workers 0 --max_per_source 4 --max_steps 6 --epochs 2 --val_limit 3 --eval_limit 8 --log_every 2"
pids=()
for arm in mvfa mil pu_m100 mil_m100 mil_m10 mil_m25; do
  x=""; [ $arm = mil_m10 ] && x=" --save_maps"
  $PY train2.py --arm $arm --fold Liver --seed 3 $S$x > $OUT/logs/train_$arm.log 2>&1 & pids+=($!)
done
$PY train2.py --arm mil_m25 --fold RESC_OCT --seed 4 $S > $OUT/logs/train_mil_m25_RESC_OCT.log 2>&1 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
for f in $OUT/logs/train_*.log; do echo "== $f"; grep -E "epoch .* done|eval_dataset|Error|Traceback" $f | tail -6; done
ls $OUT/runs/*/
exit $fail
