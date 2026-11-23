# Round 3 state

**Resume prompt (new Claude session):** *"Read task4r3/STATE.md and continue from the next step. Do not redo finished steps."*

## Done (Session A, 2026-11-22, no GPU)

- [x] **Set up (plan §4.1).** `task4r2/` copied to `task4r3/`. Outputs go to `/mnt/HDD1/divyam/task4/r3/` and results to `task4r3/results/`. Round 2's banks are reused read-only (R3-1).
- [x] **Code (§4.2).**
  - The three arms are added exactly as plan §3 shows.
  - Budget sets are now nested (R3-2: the plan asked for 25 % ⊃ 10 %, which round 2's code did not give).
  - `run_all.sh` has the modes `r3`, `list r3`, `smoke` and `timing`.
  - `analyze3.py` and `RULES.md` were written and **frozen** (`report/FROZEN.sha256`, R3-9).
- [x] **Tests (§4.3).** 80 pass on CPU: round 2's 41 plus 39 new. The CPU smoke test of all six arms passes (R3-10).
- [x] **GPU smoke test (§4.3) and timing check (§4.4)**, 2026-11-23, after the teammate freed the GPUs. Smoke test: PASS (all six arms). The timing check's cut was a startup artifact; the re-measured slowdown on real epochs is 1.11× → no cut (R3-11, `report/gpu_checks.md`).
- [~] **Runs: 58-run queue started 2026-11-23 11:06:36** in tmux session `r3` (`./go.sh launch`). About 6.2 h projected, so done ≈ 17:15. Check with `python status.py`. When it finishes, `go.sh` runs `analyze3.py`, and then it is Session B's turn.
- [~] **Gate G1.**
  - G1a (code, tests, CPU smoke, frozen rules) is ready for the teammate to look at.
  - G1b (GPU smoke and timing) is checked by pre-set criteria inside `go.sh`; review `report/gpu_checks.md` afterwards.
  - Teammate's instruction (verbatim): *"prepare for this, so that whenever we get the GPUs free, we can directly run"*.

## Next command (morning, after the current runs are paused; R3-3b)

The current runs must **free** their GPU memory. Stopping them works. Suspending them (Ctrl-Z / `kill -STOP`) does not, because a suspended process keeps its ~96 GB. Check with `nvidia-smi`: both GPUs should show almost no memory used. `go.sh` waits until at least 45 GB per GPU is free.

```bash
cd /home/f20241442/divyam-backup-imported/proj/task4r3
tmux new -d -s r3 ./go.sh
```

- Check progress with `python status.py`; the stage history is in `logs/stage.txt`.
- **To review the GPU checks before the runs start:** use `STOP_AT_G1=1 ./go.sh` inside tmux. When it stops, run `tmux new -d -s r3 './go.sh launch'`.
- **If the run is interrupted** (booking over, reboot): `tmux new -d -s r3 './go.sh launch'`. Finished runs are skipped.
- **If only one GPU is free:** `GPUS="0" tmux new -d -s r3 ./go.sh`. This takes about twice as long, and the cut rule still uses compute, not wall-clock.

## Session B (when `status.py` says all runs are done and `results/analysis.md` exists)

Kickoff prompt from plan Appendix: *"Read task4r3/STATE.md and Plans/TASK4_ROUND3_PLAN.md §4-§6. Run analyze3.py,
then write TASK4R3_REPORT.md (every run), VERDICT.md, PAPER_CLAIMS.md and report/paper_inputs/ as the plan says.
Apply §5's rules exactly as frozen; say plainly what did not hold. Stop at gate G2."*

Also:
- run `python make_figures.py` (Liver heat maps plus the budget curve);
- add G2 and G3 to `report/ISSUES_LOG.md`, including any cut from `report/gpu_checks.md`;
- refresh the transcript.
