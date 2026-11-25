#!/bin/bash
# Round 3, one command (plan §4 Session A steps 3-6, then the runs):
#   tmux new -d -s r3 ./go.sh        # start detached; reattach with: tmux attach -t r3
#   python status.py                 # progress at any time, no Claude needed
#
#   1. waits until every GPU in $GPUS has >= $MIN_FREE_MB free and <= $MAX_UTIL % load
#      (3 checks in a row, $POLL s apart). FORCE=1 skips the wait.
#   2. GPU smoke test: each arm 50 steps on Brain    -> gpu_checks.py smoke (stops on FAIL)
#   3. timing check: 201 steps x 12 jobs              -> gpu_checks.py timing (projection, cuts)
#   4. STOP_AT_G1=1 stops here for a human "go"; then run: tmux new -d -s r3 './go.sh launch'
#   5. ./run_all.sh r3 (58 runs, 20 epochs; finished runs are skipped on a restart)
#   6. analyze3.py -> results/analysis.md
# Restart after an interruption (booking ended, reboot): tmux new -d -s r3 './go.sh launch'
set -u
cd "$(dirname "$0")"
. ../../shared/config.sh   # paths and TASK4_PYTHON (env, then ../../.env, then server defaults)
PY=$TASK4_PYTHON
export GPUS=${GPUS:-0 1}
MIN_FREE_MB=${MIN_FREE_MB:-45000} MAX_UTIL=${MAX_UTIL:-20} POLL=${POLL:-300}
mkdir -p logs
stage() { echo "$(date '+%F %T') $*" | tee -a logs/stage.txt; }

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

mem_sampler() {   # our training processes' GPU memory, every 20 s, during the timing check
  echo "time,pid,used_mib" > logs/gpu_mem_timing.csv
  while true; do
    mine=" $(pgrep -u "$(id -un)" -f train2.py | tr '\n' ' ') "
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' |
      while IFS=, read -r pid mem; do [[ "$mine" == *" $pid "* ]] && echo "$(date +%T),$pid,$mem"; done >> logs/gpu_mem_timing.csv
    sleep 20
  done
}

launch() {
  [ -f report/FROZEN.sha256 ] && (cd . && sha256sum -c --quiet report/FROZEN.sha256) || { stage "STOPPED: rules not frozen or changed (report/FROZEN.sha256)"; exit 1; }
  stage "r3 runs started ($(./run_all.sh list r3 | wc -l) runs in the plan; cuts: $(tr '\n' ' ' < logs/cuts.txt 2>/dev/null))"
  ./run_all.sh r3 >> logs/r3_console.log 2>&1
  stage "r3 runs finished; $(grep -c 'FAILED TWICE' logs/r3_console.log) failed twice"
  $PY analyze3.py > logs/analyze3.log 2>&1 && stage "analyze3.py done -> results/analysis.md" || stage "analyze3.py failed (logs/analyze3.log)"
  stage "DONE. Next: Session B (plan §4): read task4r3/STATE.md"
}

case "${1:-all}" in
  launch) launch; exit ;;
  all) ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac

$PY -m pytest -q -p no:anyio tests/ > logs/pytest_prelaunch.log 2>&1 || { stage "STOPPED: tests fail (logs/pytest_prelaunch.log)"; exit 1; }
stage "tests pass ($(tail -1 logs/pytest_prelaunch.log))"
wait_for_gpus

stage "GPU smoke test"
./run_all.sh smoke > logs/smoke_console.log 2>&1
$PY gpu_checks.py smoke > logs/gpu_checks_smoke.log 2>&1 || { stage "STOPPED: smoke test FAILED (report/gpu_checks.md, logs/)"; exit 1; }
stage "smoke test PASS"

stage "timing check (12 jobs x 201 steps)"
mem_sampler & SAMPLER=$!
./run_all.sh timing > logs/timing_console.log 2>&1
kill $SAMPLER 2>/dev/null
$PY gpu_checks.py timing > logs/gpu_checks_timing.log 2>&1 || { stage "STOPPED: timing check failed (logs/gpu_checks_timing.log)"; exit 1; }
stage "timing: $(grep -o 'Decision:.*' report/gpu_checks.md | tail -1)"

if [ "${STOP_AT_G1:-0}" = 1 ]; then
  stage "STOPPED at gate G1 as asked (STOP_AT_G1=1). Review report/gpu_checks.md, then: tmux new -d -s r3 './go.sh launch'"
  exit 0
fi
launch
