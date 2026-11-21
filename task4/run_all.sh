#!/bin/bash
# Task 4 GPU runs, in the plan's order. Each phase spreads its jobs over the
# GPUs listed in $GPUS (default "0 1") and waits for them all.
#
#   ./run_all.sh phase0      # set_pi (CPU), timing dry run, R0 evals, pi estimator report  (~20 min)
#   ./run_all.sh liver       # M-h + M-h-ctrl, seeds 0-2, Liver                          (~25 min on 2 GPUs)
#   python analyze.py        # -> early-stop check (Step 5) on Liver
#   ./run_all.sh rest        # Brain + RESC, seeds 0-2       (or: ./run_all.sh rest_early_stop -> seed 0 only)
#   ./run_all.sh maskfree    # M (3 targets s0 + Liver s1,s2) and B0 (3 targets s0), fully mask-free (DECISIONS D3)
#   ./run_all.sh secondary   # sensitivity: M-h pi=0.30, M pi=0.05 (Liver s0); plan's M with R0 det adapters (3 targets)
#   ./run_all.sh lam05       # ONLY if M-h <= M-h-ctrl on Liver seed 0 (the one allowed exception)
#   ./run_all.sh m2s         # optional: M at 2S on Liver seed 0
#
# GPUS="1" ./run_all.sh liver   -> use only cuda:1
set -u
cd "$(dirname "$0")"
PY=/home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/python
GPUS=(${GPUS:-0 1})
LOG=logs; mkdir -p $LOG

run_jobs() {  # each arg is one command line (without the device flag)
  local jobs=("$@") n=${#GPUS[@]}
  for g in $(seq 0 $((n - 1))); do
    (
      for i in $(seq $g $n $((${#jobs[@]} - 1))); do
        cmd="${jobs[$i]} --device cuda:${GPUS[$g]}"
        name=$(echo "${jobs[$i]}" | tr ' ' '_' | tr -cd '[:alnum:]_.-' | cut -c1-120)
        echo "[gpu ${GPUS[$g]}] $cmd"
        $PY $cmd > "$LOG/$name.log" 2>&1 || echo "FAILED: $cmd (see $LOG/$name.log)"
      done
    ) &
  done
  wait
}

T=(Liver Brain Retina_RESC)
case "${1:-}" in
  phase0)
    # pi used by the runs (DECISIONS D1, D2): from masks / fixed, CPU only
    $PY set_pi.py | tee $LOG/set_pi.log
    # Step 2.3: 50-step dry run to time one update (check seconds_per_update in its run.json)
    run_jobs "estimate_pi.py --target Liver --branch seg"
    run_jobs "train.py --config mh --target Liver --seed 0 --steps 50 --tag dryrun --no_eval"
    jobs=("evaluate.py --target Liver --save_maps /mnt/HDD1/divyam/task4/runs/R0_Liver/maps.npz" "evaluate.py --target Brain" "evaluate.py --target Retina_RESC")
    for t in Brain Retina_RESC; do jobs+=("estimate_pi.py --target $t --branch seg"); done
    for t in "${T[@]}"; do jobs+=("estimate_pi.py --target $t --branch det" "validate_pi.py --target $t"); done
    run_jobs "${jobs[@]}" ;;   # estimator outputs are report-only (DECISIONS D6, issues log B11)
  liver)
    jobs=()
    for s in 0 1 2; do for c in mh mh_ctrl; do
      extra=""; [ $s = 0 ] && extra="--save_maps"
      jobs+=("train.py --config $c --target Liver --seed $s $extra"); done; done
    run_jobs "${jobs[@]}" ;;
  rest|rest_early_stop)
    seeds="0 1 2"; [ "$1" = rest_early_stop ] && seeds="0"
    jobs=()
    for t in Brain Retina_RESC; do for s in $seeds; do for c in mh mh_ctrl; do
      jobs+=("train.py --config $c --target $t --seed $s"); done; done; done
    run_jobs "${jobs[@]}" ;;
  maskfree)
    jobs=("train.py --config m --target Liver --seed 0 --save_maps" "train.py --config b0 --target Liver --seed 0 --save_maps")
    for t in Brain Retina_RESC; do jobs+=("train.py --config m --target $t --seed 0" "train.py --config b0 --target $t --seed 0"); done
    jobs+=("train.py --config m --target Liver --seed 1" "train.py --config m --target Liver --seed 2")
    run_jobs "${jobs[@]}" ;;
  secondary)
    jobs=("train.py --config mh --target Liver --seed 0 --pi_override 0.30 --tag pi0.30"
          "train.py --config m --target Liver --seed 0 --pi_override 0.05 --tag pi0.05")
    for t in "${T[@]}"; do jobs+=("train.py --config m --target $t --seed 0 --det_init r0 --tag r0det"); done
    run_jobs "${jobs[@]}" ;;
  lam05) run_jobs "train.py --config mh --target Liver --seed 0 --lam 0.5 --tag lam0.5" ;;
  m2s)   run_jobs "train.py --config m --target Liver --seed 0 --steps 4000 --tag 2S" ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac
