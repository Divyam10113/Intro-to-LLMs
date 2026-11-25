# Round 3 state: TASK 4 CLOSED (2026-11-23)

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
- [x] **Runs: 58-run queue started 2026-11-23 11:06:36** in tmux session `r3` (`./go.sh launch`). About 6.2 h projected, so done ≈ 17:15. Check with `python status.py`. When it finishes, `go.sh` runs `analyze3.py`, and then it is Session B's turn.
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

- [x] **Runs done**, 16:44:31: 58/58, 0 failures. `analyze3.py` ran (`results/analysis.md`).
- [x] **Session B** (2026-11-23): `report/TASK4R3_REPORT.md` (every run), `report/VERDICT.md`, `report/PAPER_CLAIMS.md`, `report/paper_inputs/task4_r3/` (`make_paper_inputs3.py`), figures (`make_figures.py`), ISSUES_LOG G2 and G3, transcript.
- [x] **Gate G2** (2026-11-23): the teammate's reply is logged verbatim (ISSUES_LOG, "Gate G2"). **Task 4 is closed.** Share zip: `task4_final_share.zip` in the project folder.
  Open points for the teammate: `PAPER_PLAN.md` was not supplied (G3-2); where H4's "matches" goes (G3-3).

## Session B (done; kept for reference)

Kickoff prompt from plan Appendix: *"Read task4r3/STATE.md and Plans/TASK4_ROUND3_PLAN.md §4-§6. Run analyze3.py,
then write TASK4R3_REPORT.md (every run), VERDICT.md, PAPER_CLAIMS.md and report/paper_inputs/ as the plan says.
Apply §5's rules exactly as frozen; say plainly what did not hold. Stop at gate G2."*

Also:
- run `python make_figures.py` (Liver heat maps plus the budget curve);
- add G2 and G3 to `report/ISSUES_LOG.md`, including any cut from `report/gpu_checks.md`;
- refresh the transcript.

**After closing (2026-11-23):** a separate, pre-registered follow-up on Brain (MIL+10 % vs MVFA, seeds 8–12) is set up in `../task4fu/` (its own STATE.md). Round 3's files are not changed by it.
