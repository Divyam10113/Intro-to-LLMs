# Intro to LLMs: an AI-augmented research pipeline on MVFA (CVPR 2024)

| Folder | Task | What it contains |
|---|---|---|
| `literature_review/` (+ top-level scripts) | Task 1 | QUAL-SG literature-review pipeline, runs v5–v6 and the Task 3 hand-off (`final_ref_set.json`) |
| branch `task2` | Task 2 | Limitation-extraction pipeline and results |
| branch `task3_kevalina` | Task 3 | Research-problem and idea pipeline (ResearchAgent + HypER + HypoEvolve), run 3 |
| `task4_codification/` | Task 4 | Codification of Task 3's A1 ideas. Round 1 (`task4/`), round 2 (`task4r2/`), round 3 (`task4r3/`), pre-registered follow-up (`task4fu/`): code, tests, frozen rules, splits, results, reports and paper inputs. Start with `SHARE_README.md` |
| `task5_paper/` | Task 5 | The 4+1 paper and its pipeline: results builder, verification scripts, PaperOrchestra fork, final LaTeX source and PDF (`runs/final/latex_writeup/`). Start with `README.md`; `bash check_all.sh` rebuilds and checks everything |
| `plans/` | 4–5 | The plans the Task 4 rounds and the paper were run from |

Not included: datasets, CLIP weights, MVFA checkpoints and trained adapters (`task4_codification/download_task4.sh` fetches the public ones), API keys, and raw chat transcripts.
