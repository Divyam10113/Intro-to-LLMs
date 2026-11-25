# Task 4 follow-up: does MIL+10 % beat MVFA on Brain? (one pre-registered test)

Round 3 (`../task4r3/`, closed) found, **post hoc**, that MIL with masks for 10 % of the abnormal images beat the re-run MVFA on Brain AUPRO on all 3 seeds (and above the MVFA paper's pixel AUROC). This folder tests that once, on fresh seeds, under a rule frozen before any run: [`RULES_FU.md`](RULES_FU.md) (hashes in `report/FROZEN.sha256`).

**Runs:** `mvfa` and `mil_m10`, Brain fold, seeds 8–12, 20 epochs: 10 runs, all at once (5 per GPU), about 1–1.5 h on 2 GPUs.
**Rule:** CONFIRMED if MIL+10 % has higher Brain AUPRO on 5/5 seeds (p = 0.031) and higher mean pixel AUROC.

```bash
cd /home/f20241442/divyam-backup-imported/proj/task4fu
tmux new -d -s fu ./go.sh            # tests -> waits for free GPUs -> smoke test -> 10 runs -> analyze_fu.py
python status.py                     # progress at any time
tmux attach -t fu                    # watch (detach: Ctrl-b d)
tmux new -d -s fu './go.sh launch'   # restart after an interruption; finished runs are skipped
```

| file | what |
|---|---|
| `go.sh` | the one command (as round 3's, without the timing check: 10 runs cannot exceed any budget) |
| `run_all.sh` | round 3's queue; modes `list`, `smoke`, `fu` |
| `RULES_FU.md`, `analyze_fu.py` | the frozen rule and its analysis; writes `results/analysis_fu.md` |
| `train2.py`, `evaluate_target.py`, `build_normal_bank.py`, `make_valid.py`, `splits/` | round 3's files, byte-identical (a test checks) |
| `common2.py` | round 3's, with the default output folder changed to `/mnt/HDD1/divyam/task4/fu` |
| `tests/` | round 2's 41 checks + `test_fu.py` (queue, outputs, seeded 10 % sets on seeds 8–12, same image order, every rule outcome, frozen hashes) |
| `status.py` | progress without Claude |
| `report/ISSUES_LOG.md` | decisions and the teammate's words, verbatim |

Outputs: runs in `/mnt/HDD1/divyam/task4/fu/runs/`, results in `results/results.csv` (append-only). Round 3's folders are never written.
