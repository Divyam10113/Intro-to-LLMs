# Task 4, round 3 (final): confirm the positives at 20 epochs, then extend them

Plan: [`../Plans/TASK4_ROUND3_PLAN.md`](../Plans/TASK4_ROUND3_PLAN.md). Rules: [`RULES.md`](RULES.md), frozen before any
run. Decisions: [`report/ISSUES_LOG.md`](report/ISSUES_LOG.md). Where things stand: [`STATE.md`](STATE.md).

Round 2's code (`../task4r2/`) is copied unchanged, with three exceptions:
- the three new arms `mil_m100`, `mil_m10` and `mil_m25` (plan §3);
- nested mask-budget sets (R3-2);
- run outputs go to `/mnt/HDD1/divyam/task4/r3/` (R3-1).

```bash
cd /home/f20241442/divyam-backup-imported/proj/task4r3
tmux new -d -s r3 ./go.sh            # waits for free GPUs -> GPU smoke -> timing check -> 58 runs -> analyze3.py
python status.py                     # progress at any time (stage, runs done, running jobs, failures, GPUs)
tmux attach -t r3                    # watch (detach: Ctrl-b d)
tmux new -d -s r3 './go.sh launch'   # restart after an interruption; finished runs are skipped
```

| file | what |
|---|---|
| `go.sh` | the one command: waits for free GPUs, then runs the smoke test, the timing check, the runs and the analysis. `STOP_AT_G1=1` stops before the runs; `FORCE=1` skips the GPU wait; `GPUS="0"` uses one GPU |
| `run_all.sh` | round 2's queue (6 jobs per GPU, retry once). Modes: `list r3`, `smoke`, `timing`, `r3` (skips finished runs), `banks` |
| `gpu_checks.py` | smoke-test pass criteria; timing projection and the plan's cut rule (R3-4). Writes `report/gpu_checks.md` and `logs/cuts.txt` |
| `train2.py` | round 2's trainer plus the arms `mil_m100`, `mil_m10` and `mil_m25` |
| `common2.py` | as in round 2; outputs go to `r3/`, and round 2's normal banks are read for the in-mask diagnostic |
| `evaluate_target.py` | the only code that reads a target test set (unchanged) |
| `analyze3.py`, `RULES.md` | plan §5 rules, frozen (`report/FROZEN.sha256`). Writes `results/analysis.md` and `results/per_seed.md` |
| `make_figures.py` | Liver heat maps (seed-3 runs) and the label-budget curve, written to `report/` |
| `status.py` | progress without Claude |
| `make_paper_inputs3.py` | builds `report/paper_inputs/task4_r3/` (tables .md/.csv, figures, numbers) from `results/` |
| `tests/` | round 2's 41 checks plus 39 for round 3 (`test_r3.py`) |
| `smoke_cpu.sh` | CPU end-to-end check of the six arms on a few images (used while the GPUs were busy) |
| `results/results.csv` | one row per (run, evaluated dataset), append-only |

Runs: MVFA, MIL and PU+100 % (Tier 1), then MIL+100 %, MIL+10 % and MIL+25 % (Tier 2). Each runs on the
Brain, Liver and RESC+OCT17 folds, seeds 3–5, at 20 epochs. MVFA and MIL also run on Brain with seeds 6–7.
58 runs in total; run on 2026-11-23, 11:06–16:44, with no failures. Results: `results/analysis.md`; report: `report/TASK4R3_REPORT.md`; verdict: `report/VERDICT.md`; paper claims: `report/PAPER_CLAIMS.md`.
