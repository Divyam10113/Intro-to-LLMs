# Follow-up state

**Resume prompt (new Claude session):** *"Read task4fu/STATE.md and continue from the next step. Do not redo finished steps."*

- [x] **Set up (2026-11-23, no GPU).** Code copied from round 3 (identical except the output folder, FU-1). Rule written and **frozen** (`RULES_FU.md`, `analyze_fu.py`, FU-2). 59 tests pass on CPU. A CPU end-to-end check of both arms on seed 8 passed (FU-3).
- [x] **Run (2026-11-24, 20:37–21:38):** 10/10 runs, 0 failures. Launched with `FORCE=1` at the teammate's request (FU-5). Originally planned as: free both GPUs (stop, not suspend, other jobs), then free both GPUs (stop, not suspend, other jobs), then
  ```bash
  cd /home/f20241442/divyam-backup-imported/proj/task4fu
  tmux new -d -s fu ./go.sh
  ```
  `python status.py` shows progress. Expected about 1–1.5 h after the GPUs are free. If interrupted: `tmux new -d -s fu './go.sh launch'`.
- [x] **Result: FU-H CONFIRMED** (AUPRO 5/5, p = 0.031, mean pix Δ +5.35). **Write-up done (2026-11-24):** `report/FU_REPORT.md` (every run, the outcome exactly as the frozen rule gives it); update `../task4r3/report/PAPER_CLAIMS.md` with a short note (headline if CONFIRMED, else the "what did not hold" table / "holds on most seeds"); log the teammate's words; commit; refresh the share zip if asked.

**Follow-up closed.** Report: `report/FU_REPORT.md`. Only open item: refresh the share zip if asked.
