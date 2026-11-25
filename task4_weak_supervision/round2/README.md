# Task 4, round 2: gated MIL (#1) and refined PU (#3) against MVFA, from scratch

Plan: [`../Plans/TASK4_ROUND2_PLAN.md`](../Plans/TASK4_ROUND2_PLAN.md). Task 3 sources: `../Plans/stage3_evolution_A1.json`,
`../Plans/stage4_ranking_A1.json`. Every choice the plan left open is in [`report/ISSUES_LOG.md`](report/ISSUES_LOG.md)
(G1 = spec check, G2 = build and checks). Round 1's code (`../task4/`) is reused unchanged: MVFA forward, data
loading, MVFA's losses, PU risk, metrics.

```bash
source /home/f20241442/divyam-backup-imported/proj/.venv-task4/bin/activate
cd /home/f20241442/divyam-backup-imported/proj/task4r2
python make_valid.py                       # source-validation lists, minus 11 valid/ files identical to test files
python -m pytest -q -p no:anyio tests/     # 41 pre-run checks
./smoke_cpu.sh                             # end to end on a few images (or: run train2.py with the smoke flags on a GPU)
./run_all.sh banks                         # 6 normal banks + precomputed gates (~1.5 min)
./run_all.sh main                          # 65 runs, 12 slots on 2 GPUs (~1 h 50 min)
python analyze2.py                         # §6 rules -> results/analysis.md
./run_all.sh confirm <arm>                 # only if analyze2.py says an arm qualifies
python make_figures.py                     # Liver heat maps -> report/liver_heatmaps_r2.png
```

| file | what |
|---|---|
| `common2.py` | folds (RESC+OCT17 held out together), data lists, frozen layer-24 features, coreset, gate |
| `make_valid.py` | `splits/valid.json`, `splits/valid_excluded.json` |
| `build_normal_bank.py` | idea #1's bank per dataset (20k coreset of frozen layer-24 normal patches) + gates for every image |
| `train2.py` | arms `mvfa`, `mil`, `g`, `pu`, `pu_m100`, `g_m100` (+ `g_m10`, `g_m25`); 5 epochs; source-validation checkpoint; then target evaluation and Task 3's in-mask diagnostic |
| `evaluate_target.py` | the only code that reads a target test set |
| `analyze2.py` | §6 decision rules, mechanically |
| `make_figures.py` | Liver heat maps (round 1's rules) |
| `results/results.csv` | one row per (run, evaluated dataset), append-only |
| `report/` | issues log, report (`TASK4R2_REPORT.md`), verdict (`VERDICT.md`), heat maps, transcript |

**Status (2026-11-22): done.** 69 runs (65 main + 4 budget), 15:29–17:12, no failures; no arm qualified for confirmation seeds. Verdict: no improvement over MIL or MVFA; submit as an honest negative / mixed-result paper.

Run outputs (adapters, loss curves, run records, Liver maps) are on the HDD at `/mnt/HDD1/divyam/task4/r2/`.
