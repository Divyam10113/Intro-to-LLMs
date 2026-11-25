#!/bin/bash
# Follow-up to Task 4 round 3 (task4fu/README.md): MIL+10 % vs MVFA on the Brain fold, fresh seeds 8-12,
# 20 epochs, round 3's code unchanged. Round 3's queue: one shared dynamic queue over both GPUs, each job
# retried once. 5 slots per GPU, so the 10 runs all start at once, 5 per GPU.
#   ./run_all.sh list    # print the 10 jobs
#   ./run_all.sh smoke   # GPU smoke test: both arms, 50 steps, Brain fold, seed 8 (not recorded)
#   ./run_all.sh fu      # the 10 runs, results -> results/results.csv (finished runs are skipped)
# Normally started through ./go.sh.
set -u
cd "$(dirname "$0")"
PY=/home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/python
GPU_IDS=(${GPUS:-0 1})
LOG=logs; mkdir -p $LOG
JOBS_PER_GPU=${JOBS_PER_GPU:-5}
EPOCHS=${EPOCHS:-20}
export TASK4R3_HDD=${TASK4R3_HDD:-/mnt/HDD1/divyam/task4/fu}
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
    sleep 2   # stagger the starts a little (less CLIP-loading contention); 10 slots, 10 jobs: 5 per GPU
  done
  wait; rm -f "$q" "$q.lock"; echo "== phase finished $(date '+%F %T')"
}
queue() { local a=(); while IFS= read -r l; do a+=("$l"); done; run_jobs "${a[@]}"; }

fu_jobs() {   # seed by seed, MVFA then MIL+10 %, so an interrupted booking leaves complete pairs first
  for s in 8 9 10 11 12; do for a in mvfa mil_m10; do
    echo "train2.py --arm $a --fold Brain --seed $s --epochs $EPOCHS"; done; done; }

skip_done() {  # resume after an interruption: a run with rows in results/results.csv is not run again
  while IFS= read -r j; do
    rid=$(echo "$j" | sed -E 's/.*--arm ([^ ]+) --fold ([^ ]+) --seed ([0-9]+).*/\1_\2_s\3/')
    if [ -f results/results.csv ] && grep -q "^$rid," results/results.csv; then echo "skip (done): $rid" >&2; else echo "$j"; fi
  done; }

smoke_jobs() { for a in mvfa mil_m10; do
  echo "train2.py --arm $a --fold Brain --seed 8 --epochs 1 --max_steps 50 --val_limit 8 --eval_limit 16 --log_every 10"; done; }

case "${1:-}" in
  list)  fu_jobs ;;
  fu)    fu_jobs | skip_done | queue ;;
  smoke) export TASK4_RUNS=$TASK4R3_HDD/smoke_runs; smoke_jobs | queue ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
