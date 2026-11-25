# Task 5 — the paper (to be added)

This folder is a placeholder. Task 5 (writing the paper) will be committed here later.

Expected inputs, from Task 4:
- `task4_weak_supervision/round3/report/PAPER_CLAIMS.md` (the claims the paper may make);
- `task4_weak_supervision/round3/report/paper_inputs/task4_r3/` (tables, figures, numbers) and
  `task4_weak_supervision/round1/report/paper_inputs/`;
- `task4_weak_supervision/followup/report/FU_REPORT.md` and `followup/results/analysis_fu.md` (the confirmed result).

A Task 5 paper pipeline already exists on the GitHub branch `task4-task5` (commit 7cee4fe, folder
`task5_paper/`). It has not been merged here. When it is, its `README.md` replaces this file
(resolve the add/add conflict in favour of the branch's version).

When Task 5 lands, follow "Adding a task" in ../README.md: replace the `task5` / `task5-dry-run`
targets in ../Makefile, add a `task5` stage to ../pipeline/smoke.py, and add any copied inputs to
HANDOFFS in ../pipeline/sync_handoffs.py (then `make sync`).
