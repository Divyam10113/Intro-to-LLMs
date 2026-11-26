# Task 4: shareable package (rounds 1–3 and the follow-up, final)

Task 4 is **closed** (2026-11-23, gate G2 in `task4r3/report/ISSUES_LOG.md`).

**Follow-up (2026-11-24): CONFIRMED.** In a pre-registered Brain test on fresh seeds 8–12, MIL with 10 % masks beat full-mask MVFA on AUPRO on 5/5 seeds (p = 0.031). Means: AUPRO 72.08 vs 57.38, pixel AUROC 92.36 vs 87.00. This holds on Brain only; image AUROC is not better. See `task4fu/report/FU_REPORT.md`.

## Read in this order

0. **`task4fu/report/FU_REPORT.md`**: the follow-up, with every run, the frozen rule's outcome and its limits. The follow-up section at the end of `PAPER_CLAIMS.md` gives the sentence the paper may use.
1. **`task4r3/report/VERDICT.md`**: the final verdict. Each hypothesis's outcome under the frozen rules, the strongest and weakest numbers, and the workshop checklist.
2. **`task4r3/report/PAPER_CLAIMS.md`**: what the paper can claim (headline H1), the "what did not hold" table, title options and a comparison with the MVFA paper.
3. **`task4r3/report/TASK4R3_REPORT.md`**: the full round 3 report, with every run in the appendix.
4. **`task4r3/report/paper_inputs/task4_r3/`**: the tables (.md and .csv) and figures (Liver heat maps, label-budget curve) for the paper.
5. Earlier rounds:
   - round 2: `task4r2/report/TASK4R2_REPORT.md` and `VERDICT.md`;
   - round 1: `task4/report/TASK4_REPORT.md`, with its paper inputs in `task4/report/paper_inputs/`.

## Contents

| Folder | What |
|---|---|
| `Plans/` | The plans for rounds 2 and 3, plus Task 3's A1 idea files that round 2 used |
| `TASK4_PLAN_RECOMMENDED.md` | Round 1's plan |
| `task4/` | Round 1: code, tests, splits, results, report, paper inputs, transcript |
| `task4r2/` | Round 2: code, tests, splits, logs, results, report, transcript |
| `task4r3/` | Round 3: code (round 2's plus 3 arm names), 80 tests, frozen rules (`RULES.md`, `analyze3.py`, `report/FROZEN.sha256`), launcher (`go.sh`), logs, results, reports, paper inputs, and the full transcript of rounds 1–3 (`report/transcript/TRANSCRIPT.md`) |
| `task4fu/` | Follow-up: round 3's code unchanged (except the output folder), 59 tests, frozen rule (`RULES_FU.md`, `analyze_fu.py`, `report/FROZEN.sha256`), launcher (`go.sh`), logs, results (`results/results.csv`, `analysis_fu.md`), report, issues log, and the full transcript of rounds 1–3 plus the follow-up (`report/transcript/TRANSCRIPT.md`) |
| `download_task4.sh` | Downloads the MVFA data, checkpoints and CLIP weights (not included) |

**Not included:** datasets, CLIP weights, MVFA checkpoints, and the trained adapters and maps. The adapters and maps are on the server at `/mnt/HDD1/divyam/task4/{r2,r3,fu}/runs/`: 3.5 GB for round 3, about 50 MB per run.

The relative links in the plans (`../task4_share/...`) resolve inside this folder.
