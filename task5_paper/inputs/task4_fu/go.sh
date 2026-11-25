#!/bin/bash
# Follow-up, one command (task4fu/README.md):
#   tmux new -d -s fu ./go.sh        # start detached; reattach with: tmux attach -t fu
#   python status.py                 # progress at any time, no Claude needed
#
#   1. tests (CPU), and the frozen rule (RULES_FU.md, analyze_fu.py) must match report/FROZEN.sha256
#   2. waits until every GPU in $GPUS has >= $MIN_FREE_MB free and <= $MAX_UTIL % load
#      (3 checks in a row, $POLL s apart). FORCE=1 skips the wait.
#   3. GPU smoke test: both arms, 50 steps, Brain fold, seed 8 (stops on failure; never recorded)
#   4. ./run_all.sh fu: the 10 runs (5 per GPU, all at once, about 1-1.5 h); finished runs are skipped on a restart
#   5. analyze_fu.py -> results/analysis_fu.md (the outcome is printed in logs/stage.txt)
# Restart after an interruption: tmux new -d -s fu './go.sh launch'
set -u
cd "$(dirname "$0")"
PY=/home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/python
export GPUS=${GPUS:-0 1}
MIN_FREE_MB=${MIN_FREE_MB:-45000} MAX_UTIL=${MAX_UTIL:-20} POLL=${POLL:-300}
HDD=${TASK4R3_HDD:-/mnt/HDD1/divyam/task4/fu}
mkdir -p logs
stage() { echo "$(date '+%F %T') $*" | tee -a logs/stage.txt; }

frozen_ok() { [ -f report/FROZEN.sha256 ] && sha256sum -c --quiet report/FROZEN.sha256; }

gpus_free() {
  local ok=1
  for g in $GPUS; do
    read -r free util < <(nvidia-smi -i "$g" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    echo "  gpu $g: ${free} MiB free, ${util} % load"
    { [ "$free" -ge "$MIN_FREE_MB" ] && [ "$util" -le "$MAX_UTIL" ]; } || ok=0
  done
  [ $ok = 1 ]
}

wait_for_gpus() {
  [ "${FORCE:-0}" = 1 ] && { stage "FORCE=1: not waiting for free GPUs"; return; }
  stage "waiting for free GPUs ($GPUS; need >= $MIN_FREE_MB MiB free and <= $MAX_UTIL % load, 3 checks in a row)"
  local n=0
  while [ $n -lt 3 ]; do
    if gpus_free; then n=$((n + 1)); else n=0; fi
    [ $n -lt 3 ] && sleep $([ $n = 0 ] && echo "$POLL" || echo 60)
  done
  stage "GPUs free"
}

launch() {
  frozen_ok || { stage "STOPPED: rules not frozen or changed (report/FROZEN.sha256)"; exit 1; }
  stage "follow-up runs started ($(./run_all.sh list | wc -l) runs in the plan)"
  ./run_all.sh fu >> logs/fu_console.log 2>&1
  stage "follow-up runs finished; $(grep -c 'FAILED TWICE' logs/fu_console.log) failed twice"
  if $PY analyze_fu.py > logs/analyze_fu.log 2>&1; then stage "$(tail -1 logs/analyze_fu.log)"
  else stage "analyze_fu.py failed (logs/analyze_fu.log)"; fi
  stage "DONE. Results: results/analysis_fu.md. Next: Claude writes report/FU_REPORT.md (task4fu/STATE.md)"
}

case "${1:-all}" in
  launch) launch; exit ;;
  all) ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="" $PY -m pytest -q -p no:anyio tests/ > logs/pytest_prelaunch.log 2>&1 || { stage "STOPPED: tests fail (logs/pytest_prelaunch.log)"; exit 1; }
stage "tests pass ($(tail -1 logs/pytest_prelaunch.log))"
frozen_ok || { stage "STOPPED: rules not frozen or changed (report/FROZEN.sha256)"; exit 1; }
wait_for_gpus

stage "GPU smoke test (2 arms x 50 steps)"
rows() { [ -f results/results.csv ] && wc -l < results/results.csv || echo 0; }
n0=$(rows)
rm -rf "$HDD/smoke_runs"
./run_all.sh smoke > logs/smoke_console.log 2>&1
ok=1
for a in mvfa mil_m10; do
  [ -f "$HDD/smoke_runs/${a}_Brain_s8/run.json" ] || ok=0
done
grep -q "FAILED" logs/smoke_console.log && ok=0
[ "$(rows)" = "$n0" ] || ok=0                     # smoke runs must never add rows to results.csv
[ $ok = 1 ] || { stage "STOPPED: smoke test FAILED (logs/smoke_console.log, logs/train2.py_*max_steps*.log)"; exit 1; }
stage "smoke test PASS (both arms ran 50 steps and wrote run.json; nothing recorded in results.csv)"

launch
