#!/bin/bash
# Task 4 GPU runs. Every phase puts its jobs in ONE shared queue served by
# JOBS_PER_GPU worker slots on each GPU in $GPUS: a free slot takes the next
# job immediately, so both cards stay busy until the queue is empty.
#
# Jobs per GPU: k jobs on one card make EACH job slower; it only pays off if
# one job leaves the GPU partly idle. `bench` measures total throughput for
# k = 1, 2, 4, 6, 8 and saves its choice to logs/jobs_per_gpu (bench_summary.py);
# later phases use it. Without a bench result: 1 job per GPU.
#
#   ./run_all.sh bench       # ~5-8 min: total updates/s, per-job s/update, GPU util, VRAM for each k
#   ./run_all.sh phase0      # set_pi (CPU), timing dry run (alone), R0 evals, pi estimator report
#   ./run_all.sh main        # liver + maskfree + secondary in one queue (none depends on the early stop)
#   python analyze.py        # -> early-stop check on Liver (needs the liver runs of `main`)
#   ./run_all.sh rest        # Brain + RESC M-h / M-h-ctrl, seeds 0-2   (or: rest_early_stop -> seed 0 only)
#   python analyze.py        # Step 5 verdicts -> results/analysis.md
#   ./run_all.sh lam05       # ONLY if analyze.py says the lambda = 0.5 exception applies
#   ./run_all.sh m2s         # optional: M at 2S on Liver seed 0
# Single phases still exist: liver, maskfree, secondary.
#
# Env: GPUS="0 1" (default)  JOBS_PER_GPU=k overrides the bench choice
#      WORKERS=4 data-loader workers per job   DRY_RUN=1 prints the schedule only
set -u
cd "$(dirname "$0")"
PY=/home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/python
GPU_IDS=(${GPUS:-0 1})
LOG=logs; mkdir -p $LOG
JOBS_PER_GPU=${JOBS_PER_GPU:-$(cat $LOG/jobs_per_gpu 2>/dev/null || echo 1)}
export TASK4_WORKERS=${WORKERS:-4}
export TASK4_SLOTS=$JOBS_PER_GPU             # recorded in each run's notes (gpu_minutes is then shared time)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} # many processes share 32 cores: no CPU oversubscription
export MKL_NUM_THREADS=$OMP_NUM_THREADS
SLOTS=()
for g in "${GPU_IDS[@]}"; do for _ in $(seq "$JOBS_PER_GPU"); do SLOTS+=("$g"); done; done

run_jobs() {  # each arg is one command line (without the device flag); dynamic shared queue
  local jobs=("$@") q="$LOG/.queue.$$"
  echo 0 > "$q"
  echo "== ${#jobs[@]} jobs on ${#SLOTS[@]} slots (GPUs: ${GPU_IDS[*]}, $JOBS_PER_GPU per GPU) $(date +%T)"
  for s in "${!SLOTS[@]}"; do
    (
      g=${SLOTS[$s]}
      while true; do
        i=$(flock "$q.lock" bash -c "i=\$(cat '$q'); echo \$((i + 1)) > '$q'; echo \$i")
        [ "$i" -ge "${#jobs[@]}" ] && break
        cmd="${jobs[$i]} --device cuda:$g"
        name=$(echo "${jobs[$i]}" | tr ' ' '_' | tr -cd '[:alnum:]_.-' | cut -c1-120)
        echo "[$(date +%T) gpu $g slot $s] start $i: $cmd"
        if [ "${DRY_RUN:-0}" = 1 ]; then sleep 0.$((RANDOM % 5 + 1)); continue; fi
        if $PY $cmd > "$LOG/$name.log" 2>&1; then echo "[$(date +%T) gpu $g slot $s] done  $i"
        else echo "[$(date +%T) gpu $g slot $s] FAILED $i: $cmd (see $LOG/$name.log)"; fi
      done
    ) &
  done
  wait
  rm -f "$q" "$q.lock"
  echo "== phase finished $(date +%T)"
}

T=(Liver Brain Retina_RESC)
liver_jobs()     { for s in 0 1 2; do for c in mh mh_ctrl; do
                     extra=""; [ $s = 0 ] && extra=" --save_maps"
                     echo "train.py --config $c --target Liver --seed $s$extra"; done; done; }
maskfree_jobs()  { echo "train.py --config m --target Liver --seed 0 --save_maps"
                   echo "train.py --config b0 --target Liver --seed 0 --save_maps"
                   for t in Brain Retina_RESC; do echo "train.py --config m --target $t --seed 0"
                                                  echo "train.py --config b0 --target $t --seed 0"; done
                   echo "train.py --config m --target Liver --seed 1"
                   echo "train.py --config m --target Liver --seed 2"; }
secondary_jobs() { echo "train.py --config mh --target Liver --seed 0 --pi_override 0.30 --tag pi0.30"
                   echo "train.py --config m --target Liver --seed 0 --pi_override 0.05 --tag pi0.05"
                   for t in "${T[@]}"; do echo "train.py --config m --target $t --seed 0 --det_init r0 --tag r0det"; done; }
rest_jobs()      { for t in Brain Retina_RESC; do for s in $1; do for c in mh mh_ctrl; do
                     echo "train.py --config $c --target $t --seed $s"; done; done; done; }
queue() { local a=(); while IFS= read -r l; do a+=("$l"); done; run_jobs "${a[@]}"; }

case "${1:-}" in
  bench)
    # Total training throughput vs k concurrent jobs on ONE GPU (400 updates each, no eval);
    # util / VRAM sampled every 2 s while the k jobs run.
    g=${GPU_IDS[0]}; out=$LOG/bench.tsv; : > $out
    for k in 1 2 4 6 8; do
      pids=()
      for j in $(seq $k); do
        $PY train.py --config mh --target Liver --seed $j --steps 400 --no_eval --log_every 0 \
          --tag bench${k}_$j --device cuda:$g > $LOG/bench_${k}_$j.log 2>&1 &
        pids+=($!)
      done
      ( while true; do nvidia-smi -i $g --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done ) > $LOG/bench_smi_$k.csv &
      smi_pid=$!
      wait "${pids[@]}"; kill $smi_pid 2>/dev/null
      $PY bench_summary.py row $k $LOG/bench_smi_$k.csv >> $out && echo "k=$k: $(tail -1 $out)"
    done
    $PY bench_summary.py choose $out $LOG/jobs_per_gpu ;;
  phase0)
    $PY set_pi.py | tee $LOG/set_pi.log                     # pi the runs use (DECISIONS D1, D2), CPU
    # Step 2.3: dry run ALONE on one GPU, so seconds_per_update is an unshared timing
    (SLOTS=("${GPU_IDS[0]}"); run_jobs "train.py --config mh --target Liver --seed 0 --steps 50 --tag dryrun --no_eval")
    jobs=("evaluate.py --target Liver --save_maps /mnt/HDD1/divyam/task4/runs/R0_Liver/maps.npz"
          "evaluate.py --target Brain" "evaluate.py --target Retina_RESC")
    for t in "${T[@]}"; do jobs+=("estimate_pi.py --target $t --branch seg" "estimate_pi.py --target $t --branch det"
                                  "validate_pi.py --target $t"); done
    run_jobs "${jobs[@]}" ;;   # estimator outputs are report-only (DECISIONS D6, issues log B11)
  main)      { liver_jobs; maskfree_jobs; secondary_jobs; } | queue ;;   # liver first: early-stop data soonest
  liver)     liver_jobs | queue ;;
  maskfree)  maskfree_jobs | queue ;;
  secondary) secondary_jobs | queue ;;
  rest)            rest_jobs "0 1 2" | queue ;;
  rest_early_stop) rest_jobs "0" | queue ;;
  lam05) run_jobs "train.py --config mh --target Liver --seed 0 --lam 0.5 --tag lam0.5" ;;
  m2s)   run_jobs "train.py --config m --target Liver --seed 0 --steps 4000 --tag 2S" ;;
  *) sed -n 2,23p "$0"; exit 1 ;;
esac
