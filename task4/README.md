# Task 4: PU loss for MVFA (recommended tier)

Code for `TASK4_PLAN_RECOMMENDED.md`, with the deviations in
[`report/DECISIONS.md`](report/DECISIONS.md) (fixed before any GPU run; read it first). MVFA-AD is checked out at `../MVFA-AD`
(commit `a30bf7b`); data, checkpoints and run outputs live on the HDD at
`/mnt/HDD1/divyam/task4/` (`data/extracted`, `ckpt/zero-shot`, `clip`, `runs`).

```bash
source /home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/activate
cd /home/f20241442/divyam-backup-imported/proj/task4
```

## Setup from scratch (already done on this machine)
`MVFA-AD/` and `.venv-task4/` are not in this git repo; recreate them with:
```bash
cd /home/f20241442/divyam-backup-imported/proj
git clone https://github.com/MediaBrain-SJTU/MVFA-AD.git && git -C MVFA-AD checkout a30bf7bb2c51772a36e7be58d3c072be480620fe
ln -sfn /mnt/HDD1/divyam/task4/clip MVFA-AD/CLIP/ckpt                         # CLIP ViT-L-14-336px.pt
rm -rf MVFA-AD/data && ln -sfn /mnt/HDD1/divyam/task4/data/extracted MVFA-AD/data   # replaces an empty, unused data/__init__.py
/home/f20241442/divyam-backup-imported/my_env/bin/python -m venv --system-site-packages .venv-task4
.venv-task4/bin/pip install kornia ftfy regex scikit-image gdown pytest
./download_task4.sh   # data + zero-shot checkpoints to the HDD (copies via the SSD: the HDD rejects chmod)
```
Then extract the archives on the HDD (`tar --no-same-owner --no-same-permissions -xzf <f> -C data/extracted`,
`unzip zero-shot.zip` in `ckpt/`). MVFA-AD's code itself is unmodified.

## Before the GPUs (CPU only)
```bash
python make_splits.py --check          # leakage check (splits already written)
python -m pytest -q -p no:anyio tests/ # 27 pre-run checks (~2 min)
./smoke_cpu.sh                         # every script end to end on CPU, into /tmp/task4_smoke
```

## GPU runs (in the plan's order)
```bash
./run_all.sh bench       # optional, ~5-8 min: chooses jobs per GPU (see below)
./run_all.sh phase0      # set_pi, dry-run timing (alone), R0 evals (+AUPRO), estimator report
./run_all.sh main        # liver + maskfree + secondary in one queue (Liver runs first)
python analyze.py        # early-stop check -> tells you which next command to run
./run_all.sh rest        # or: ./run_all.sh rest_early_stop
python analyze.py        # Step 5 verdicts -> results/analysis.md
python make_paper_inputs.py   # Step 6 tables + Liver heat maps -> report/paper_inputs/
```
**Status (2026-11-22): all GPU runs done** (bench, phase0, main, rest, m2s; ≈ 20 min on 2 GPUs).
Results: [`report/TASK4_REPORT.md`](report/TASK4_REPORT.md).
Both GPUs serve one shared queue: whenever a run finishes, its slot takes the
next one, so neither card idles while work is left. `GPUS="1" ./run_all.sh ...`
restricts to one card. Logs go to `logs/`; each run writes
`runs/<run_id>/{seg_adapters.pth, loss.csv, run.json}` and one row in
`results/results.csv` (appends are file-locked, so concurrent runs are safe).

**Jobs per GPU.** Running k jobs on one card makes each job slower; it only
helps if a single job leaves the GPU partly idle (training is batch-1, as in
MVFA, so it may not reach 100% util). `bench` runs k = 1, 2, 4, 6, 8 concurrent
400-update jobs on one GPU, records total updates/s, per-job s/update, median
util and peak VRAM to `logs/bench.tsv`, and saves the smallest k within 5% of
the best total throughput to `logs/jobs_per_gpu` (k = 1 if one job already
saturates the card). Later phases read it; `JOBS_PER_GPU=k` overrides it; with
no bench result the default is 1. With k > 1, `gpu_minutes` is wall time on a
shared card (noted per run). Results do not depend on k (each run's data order is
fixed by its seed).

## Files
| file | what |
|---|---|
| `pu.py` | PU risk, top-k MIL, π estimator (+ the plan's literal formula, kept to document why it was changed) |
| `metrics.py` | image / pixel AUROC, AUPRO (FPR ≤ 0.3, per-image accumulation) |
| `common.py` | paths, MVFA-identical data loading, CPU-safe MVFA forward, model building/freezing |
| `make_splits.py` | ≤ 500 + 500 per source, 10% π hold-out, leakage check (path + MD5) |
| `schedule.py` | deterministic data order per seed; M-h-ctrl = M-h's mask steps only |
| `set_pi.py` | the π the runs use: `pi.json` (M-h, from masked sources' masks, D1), `pi_maskfree.json` (M, 0.30, D2) |
| `estimate_pi.py` | the plan's estimator on R0 (report only, D6) → `pi_estimated_{seg,det}.json` |
| `validate_pi.py` | estimator vs ground truth on masked sources, every level / branch (issues log B11) |
| `train.py` | configs `mh`, `mh_ctrl`, `m`, `b0` (`--det_init`, `--pi_override`, `--lam`, `--tag`); last step = final model; evaluates at the end |
| `evaluate.py` | R0 / checkpoint evaluation |
| `analyze.py` | Step 5 pre-registered rules |
| `report/ISSUES_LOG.md` | the issues log Task 4 requires |
| `report/DECISIONS.md` | every deviation from the plan, with reasons, fixed before GPU runs |
| `smoke_cpu.sh` | end-to-end CPU test of every script and config |
| `make_paper_inputs.py` | Step 6 hand-off: tables (a), (b), Liver heat-map figure, copies to `report/paper_inputs/` |
| `export_transcript.py` | exports the chat with the LLM to `report/transcript/TRANSCRIPT.md` (e-mail redacted) |
| `report/TASK4_REPORT.md` | the Step 6 report |
