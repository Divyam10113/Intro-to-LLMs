#!/bin/bash
# End-to-end CPU smoke test of every script. Writes only to a scratch dir.
set -e
S=${SMOKE_DIR:-/tmp/task4_smoke}
export TASK4_RESULTS=$S/results TASK4_RUNS=$S/runs
PY=/home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/python
cd "$(dirname "$0")"
rm -rf $S; mkdir -p $S
T=$(date +%s)
$PY set_pi.py                                                   # pi used by runs (DECISIONS D1, D2)
$PY estimate_pi.py --target Liver --branch seg --device cpu      # report-only estimator (D6)
echo "pi done in $(( $(date +%s) - T ))s"
for c in mh mh_ctrl m b0; do                                     # m, b0: fully mask-free (D3)
  $PY train.py --config $c --target Liver --seed 0 --steps 6 --device cpu --workers 2 --eval_limit 16 --log_every 1 --save_maps
done
# secondary configs (D1-D3)
$PY train.py --config mh --target Liver --seed 0 --steps 4 --device cpu --workers 2 --eval_limit 16 --pi_override 0.30 --tag pi0.30
$PY train.py --config m --target Liver --seed 0 --steps 4 --device cpu --workers 2 --eval_limit 16 --det_init r0 --tag r0det
$PY evaluate.py --target Liver --device cpu --limit 16 --ckpt $S/runs/m_Liver_s0/seg_adapters.pth   # loads det_adapters.pth too
$PY evaluate.py --target Liver --device cpu --limit 16
ls -R $S | head -40
echo "SMOKE OK in $(( $(date +%s) - T ))s"
