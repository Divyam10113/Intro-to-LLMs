#!/bin/bash
# Round 2 GPU runs: one shared dynamic queue over both GPUs (round 1's scheme,
# 6 jobs per GPU from round 1's benchmark, B14).
#   ./run_all.sh banks     # 6 normal banks (+ precomputed gates), ~1-2 min
#   ./run_all.sh main      # the 65 default runs (MVFA, MIL, PU, G, PU-pi5, PU+100%, G+100%)
#   ./run_all.sh confirm ARM   # seeds 3-4 for MVFA and ARM on the 3 masked folds (only if ARM qualifies)
#   ./run_all.sh budget    # G+10 %, G+25 % on Brain, Liver seed 0 (only if time is left)
set -u
cd "$(dirname "$0")"
PY=/home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/python
GPU_IDS=(${GPUS:-0 1})
LOG=logs; mkdir -p $LOG
JOBS_PER_GPU=${JOBS_PER_GPU:-6}
export TASK4_WORKERS=${WORKERS:-3} TASK4_SLOTS=$JOBS_PER_GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
SLOTS=(); for g in "${GPU_IDS[@]}"; do for _ in $(seq "$JOBS_PER_GPU"); do SLOTS+=("$g"); done; done

run_jobs() {
  local jobs=("$@") q="$LOG/.queue.$$"
  echo 0 > "$q"
  echo "== ${#jobs[@]} jobs on ${#SLOTS[@]} slots $(date +%T)"
  for s in "${!SLOTS[@]}"; do
    ( g=${SLOTS[$s]}
      while true; do
        i=$(flock "$q.lock" bash -c "i=\$(cat '$q'); echo \$((i + 1)) > '$q'; echo \$i")
        [ "$i" -ge "${#jobs[@]}" ] && break
        cmd="${jobs[$i]} --device cuda:$g"
        name=$(echo "${jobs[$i]}" | tr ' ' '_' | tr -cd '[:alnum:]_.-' | cut -c1-120)
        echo "[$(date +%T) gpu $g slot $s] start $i: $cmd"
        if $PY $cmd > "$LOG/$name.log" 2>&1; then echo "[$(date +%T) gpu $g slot $s] done  $i"
        else echo "[$(date +%T) gpu $g slot $s] FAILED $i (retrying once)"
             if $PY $cmd > "$LOG/$name.retry.log" 2>&1; then echo "[$(date +%T)] done on retry $i"
             else echo "[$(date +%T)] FAILED TWICE $i: $cmd"; fi
        fi
      done ) &
  done
  wait; rm -f "$q" "$q.lock"; echo "== phase finished $(date +%T)"
}
queue() { local a=(); while IFS= read -r l; do a+=("$l"); done; run_jobs "${a[@]}"; }

main_jobs() {   # longest first (batch-1 arms on 5-source folds), Liver s0 saves maps for the heat maps
  for s in 0 1 2; do for f in Brain Liver RESC_OCT; do for arm in mvfa g mil g_m100 pu_m100 pu; do
    x=""; [ "$f" = Liver ] && [ $s = 0 ] && x=" --save_maps"
    echo "train2.py --arm $arm --fold $f --seed $s$x"; done; done
    [ $s = 0 ] && for f in Chest HIS; do for arm in mvfa g mil pu; do echo "train2.py --arm $arm --fold $f --seed 0"; done; done
  done
  for f in Brain Liver RESC_OCT; do echo "train2.py --arm pu --fold $f --seed 0 --pi 0.05 --tag pi5"; done
}

case "${1:-}" in
  banks) for d in Brain Liver Retina_RESC Retina_OCT2017 Chest Histopathology; do echo "build_normal_bank.py --dataset $d"; done | queue ;;
  main)  main_jobs | queue ;;
  confirm) arm=$2; for s in 3 4; do for f in Brain Liver RESC_OCT; do for a in mvfa $arm; do
             echo "train2.py --arm $a --fold $f --seed $s"; done; done; done | queue ;;
  budget) for f in Brain Liver; do for a in g_m10 g_m25; do echo "train2.py --arm $a --fold $f --seed 0"; done; done | queue ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
