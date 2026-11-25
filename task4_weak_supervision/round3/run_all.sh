#!/bin/bash
# Round 3 GPU runs (plan Plans/TASK4_ROUND3_PLAN.md §3-§4). Round 2's queue, unchanged:
# one shared dynamic queue over both GPUs, 6 jobs per GPU, each job retried once.
#   ./run_all.sh list r3   # print the 58 round 3 jobs in plan order (minus cuts in logs/cuts.txt)
#   ./run_all.sh smoke     # GPU smoke test: each round 3 arm, 50 steps, Brain fold (not recorded)
#   ./run_all.sh timing    # first 201 steps of each arm at full concurrency (12 jobs; not recorded)
#   ./run_all.sh r3        # the round 3 runs, 20 epochs, results -> results/results.csv
#   ./run_all.sh banks     # only if round 2's normal banks are missing (writes .../r2/banks)
# Normally started through ./go.sh, which waits for free GPUs and does smoke -> timing -> r3.
set -u
cd "$(dirname "$0")"
. ../../shared/config.sh   # paths and TASK4_PYTHON (env, then ../../.env, then server defaults)
PY=$TASK4_PYTHON
GPU_IDS=(${GPUS:-0 1})
LOG=logs; mkdir -p $LOG
JOBS_PER_GPU=${JOBS_PER_GPU:-6}
EPOCHS=${EPOCHS:-20}
export TASK4R3_HDD=${TASK4R3_HDD:-/mnt/HDD1/divyam/task4/r3}
export TASK4_WORKERS=${WORKERS:-3} TASK4_SLOTS=$JOBS_PER_GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
SLOTS=(); for g in "${GPU_IDS[@]}"; do for _ in $(seq "$JOBS_PER_GPU"); do SLOTS+=("$g"); done; done

run_jobs() {
  local jobs=("$@") q="$LOG/.queue.$$"
  echo 0 > "$q"
  echo "== ${#jobs[@]} jobs on ${#SLOTS[@]} slots $(date '+%F %T')"
  for s in "${!SLOTS[@]}"; do
    ( g=${SLOTS[$s]}
      while true; do
        i=$(flock "$q.lock" bash -c "i=\$(cat '$q'); echo \$((i + 1)) > '$q'; echo \$i")
        [ "$i" -ge "${#jobs[@]}" ] && break
        cmd="${jobs[$i]} --device cuda:$g"
        name=$(echo "${jobs[$i]}" | tr ' ' '_' | tr -cd '[:alnum:]_.-' | cut -c1-120)
        echo "[$(date '+%F %T') gpu $g slot $s] start $i: $cmd"
        if $PY $cmd > "$LOG/$name.log" 2>&1; then echo "[$(date '+%F %T') gpu $g slot $s] done  $i"
        else echo "[$(date '+%F %T') gpu $g slot $s] FAILED $i (retrying once)"
             if $PY $cmd > "$LOG/$name.retry.log" 2>&1; then echo "[$(date '+%F %T')] done on retry $i"
             else echo "[$(date '+%F %T')] FAILED TWICE $i: $cmd"; fi
        fi
      done ) &
  done
  wait; rm -f "$q" "$q.lock"; echo "== phase finished $(date '+%F %T')"
}
queue() { local a=(); while IFS= read -r l; do a+=("$l"); done; run_jobs "${a[@]}"; }

CUTS_FILE=${CUTS_FILE:-$LOG/cuts.txt}   # written by gpu_checks.py timing
cut_on() { [ -f "$CUTS_FILE" ] && grep -qx "$1" "$CUTS_FILE"; }

r3_jobs() {   # plan §3 order: blocks 1-7; within a block longest fold first (Brain, Liver, RESC_OCT); Liver s3 saves maps
  block() { local arm=$1; for f in Brain Liver RESC_OCT; do for s in 3 4 5; do
    x=""; [ "$f" = Liver ] && [ $s = 3 ] && x=" --save_maps"
    echo "train2.py --arm $arm --fold $f --seed $s --epochs $EPOCHS$x"; done; done; }
  block mvfa; block mil; block pu_m100                                   # Tier 1, blocks 1-3 (never cut)
  cut_on brain_extra || for s in 6 7; do for a in mvfa mil; do           # Tier 1, block 4 (cut b)
    echo "train2.py --arm $a --fold Brain --seed $s --epochs $EPOCHS"; done; done
  block mil_m100; block mil_m10                                          # Tier 2, blocks 5-6 (never cut)
  cut_on mil_m25 || block mil_m25                                        # Tier 2, block 7 (cut a)
}

skip_done() {  # resume after an interruption: a run with rows in results/results.csv is not run again
  while IFS= read -r j; do
    rid=$(echo "$j" | sed -E 's/.*--arm ([^ ]+) --fold ([^ ]+) --seed ([0-9]+).*/\1_\2_s\3/')
    if [ -f results/results.csv ] && grep -q "^$rid," results/results.csv; then echo "skip (done): $rid" >&2; else echo "$j"; fi
  done; }

R3_ARMS="mvfa mil pu_m100 mil_m100 mil_m10 mil_m25"
smoke_jobs() { for a in $R3_ARMS; do x=""; [ $a = mil_m10 ] && x=" --save_maps"
  echo "train2.py --arm $a --fold Brain --seed 3 --epochs 1 --max_steps 50 --val_limit 8 --eval_limit 16 --log_every 10$x"; done; }
timing_jobs() { for r in 1 2; do for a in $R3_ARMS; do
  echo "train2.py --arm $a --fold Brain --seed $((2 + r)) --tag timing$r --epochs 1 --max_steps 201 --val_limit 4 --no_eval --no_diag --log_every 50"; done; done; }

case "${1:-}" in
  list)   r3_jobs ;;
  r3)     r3_jobs | skip_done | queue ;;
  smoke)  export TASK4_RUNS=$TASK4R3_HDD/smoke_runs; smoke_jobs | queue ;;
  timing) export TASK4_RUNS=$TASK4R3_HDD/timing_runs; timing_jobs | queue ;;
  banks)  for d in Brain Liver Retina_RESC Retina_OCT2017 Chest Histopathology; do echo "build_normal_bank.py --dataset $d"; done | queue ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
