# Task 5 (final paper) inputs, v2: with Task 4 round 1

Everything the paper needs, **including Task 4 round 1's outputs**. Built on 2026-11-22. Sources: the team repository `Divyam10113/Intro-to-LLMs` at the latest commits (`Vismay` @ `8a312f7` = Task 1 v6; `task2` @ `efbdb3e`; `task3_kevalina` @ `9cc58d7` = Task 3 run 3), Task 4 round 1's output (`task4_share.zip`, run 2026-11-22), and our plans.

**This replaces `Task5_inputs.zip` (v1, 2026-11-21), which had no Task 4 results.** Task 4 **round 2** is not in here yet; `task4_r2/` is reserved for it and `plans/PAPER_PLAN.md` §11 says how to swap it in.

**Format reminder: 4+1.** 4 pages of content (including the title, abstract, figures and tables) plus 1 page of references only.

## Start here
1. `brief/Task 5 - Research paper.md`: the instructor's description of the task.
2. `plans/PAPER_PLAN.md` (**v2**): the paper plan: what the paper is now (§0), page budget, section map, framing rule, the 7 checks, PaperOrchestra fork, phases, and how to update with round 2 (§11).
3. `plans/TASK5_STAGE_A_PLAN.md`: step-by-step detail for Phases 0–2 (banner at the top says what changed).
4. `task4_r1/report/TASK4_REPORT.md`: Task 4 round 1's report. Then `DECISIONS.md` and `ISSUES_LOG.md` in the same folder.

## What round 1 found (one paragraph; rebuild every number with the script)
Adding a PU loss on MVFA's three mask-less datasets does **not** consistently improve localisation beyond plain fine-tuning (AUPRO up in 6 of 9 pairs; +5.2 on Brain, nothing on Liver, noisy on RESC). The mask-free PU model fails (retention 0.58; chance level on Liver) because **π cannot be estimated from MVFA's patch scores**. Plain top-k MIL with no masks keeps 93–99 % of the released model's localisation (single seed, not pre-registered). Hence the paper's framing is an honest **analysis paper** (`PAPER_PLAN.md` §0.3, §5.1).

## Files and the paper sections they feed
| File | What it is | Feeds |
|---|---|---|
| `tools/build_results.py` | Builds `results.json` (the only source of numbers) from any round's folder. **Tested:** reproduces every mean ± std in the round 1 report | §5 Experiments, all number checks |
| `task4_r1/results/results.csv`, `paper_numbers.csv`, `pi*.json` | Round 1's 35 runs, the published MVFA numbers, π files | `results.json` |
| `task4_r1/report/TASK4_REPORT.md` | Round 1's report: idea, issues, results tables (a) and (b), rules applied, limitations | §4 Methodology, §5 Experiments |
| `task4_r1/report/DECISIONS.md` | Deviations from the plan fixed before any GPU run (D1–D6) | §4 Methodology, limitations |
| `task4_r1/report/ISSUES_LOG.md` | 29 issues with the feedback given to the LLM | Contributions (code audit), AI-use note |
| `task4_r1/report/paper_inputs/` | Tables as CSV and Markdown, Liver heat maps (PNG and PDF, with the image list), `analysis.md` | Table 1, Table 2, Figure 2 |
| `task4_r1/report/transcript/TRANSCRIPT.md` | The full conversation with Claude Code that wrote and ran Task 4 | AI-use record |
| `task4_r1/` (code: `pu.py`, `train.py`, `evaluate.py`, `metrics.py`, `tests/`, …) | The code that produced the results | Checking the maths (PU risk, AUPRO) |
| `plans/TASK4_PLAN_RECOMMENDED.md` | The plan round 1 ran | §3 Problem formulation, §4 Methodology, §5 setup |
| `plans/TASK4_ROUND2_PLAN.md` | The next round (not run yet) | §11 of the paper plan |
| `task1/final_ref_set.json` | 594 papers with checked DOIs, abstracts, keywords, topics, relevance 1–5, origin (`cites_mvfa`…) | §2 Related work; the **citation pool** (relevance ≥ 3, then by hand). IQE-CLIP and MADPOT are in it |
| `task1/final_ref_set_excluded.json`, `RUN_REPORT.md`, `Task1_README.md` | Excluded papers with reasons; how Task 1 v6 was made | AI-use note |
| `task2/final_limitations.json` | 39 MVFA limitations (the `full_nim` run Task 3 used) | §1(i): L3, L5, L18 |
| `task2/research_gaps_full_nim.md`, `task2_research_gaps.md`, `TASK2_DOCUMENTATION.md` | Readable versions and how Task 2 worked | Background, AI-use note |
| `task3/run3/explorer_data.json` | All of run 3; `problems` → `"A1"` = the problem statement | §1(i), §3 |
| `task3/run3/gaps_verified.json`, `stage0_gaps.json` | Limitation groups; A1's group = L3, L4, L5, L18 | §1(i) |
| `task3/run3/stage0b_code_verification.json` | MVFA code evidence E1–E20 and code-only gaps C1–C4 (E7, E8, E16 for the background; C2 for the R0 caveat) | §1(i), §5 limitations |
| `task3/run3/stage4_ranking_A1.json`, `stage3_evolution_A1.json` | A1's top 5; full text of `A1-g1-o6` (round 1) and `A1-g1-o2` (round 2) | §1(ii), §4 |
| `task3/run1/…` | The refinement idea `A1-g2-o6` (patch-level PU) | §4 |
| `task3/run3/stage1_problems.json`, `stage2_*.json`, `rejudge_*.json`, `Task3_README.md`, `reports/*.pdf` | Problem versions, MVFA-citing papers, independent checks, Task 3 documentation | Background, §2, AI-use note |
| `references/extra_refs.bib` | 4 references Task 1 never searched for: Elkan & Noto 2008, nnPU (Kiryo 2017), MPE (Ramaswamy 2016), attention MIL (Ilse 2018). Identifiers checked on 21 Nov | §2.3, §3, §4 |

**Already in Task 1's pool, so cite them from there:** MVFA (`10.1109/cvpr52733.2024.01081`), BMAD (`10.1109/cvprw63382.2024.00408`), CLIP (`10.48550/arxiv.2103.00020`), MVTec AD / AUPRO (`10.1007/s11263-020-01400-4`), WinCLIP, APRIL-GAN, AnomalyCLIP, AdaCLIP, VCP-CLIP, IQE-CLIP (`10.1109/tip.2026.3697648`), MADPOT (`10.48550/arxiv.2507.06733`).

## Known caveats in these inputs
- `task4_r1/results/results.csv` has a `pi_json` column with paths from the GPU server (`/home/f20241442/…`). The paper never uses it.
- The report's per-target mean of the paired differences averages the rounded per-pair values, so it can differ from `results.json` by 0.01. Number checks must allow ±0.01.
- Round 1's sensitivity runs (π = 0.30 for M-h, π = 0.05 for M) were chosen after seeing mask statistics; they are labelled secondary and must stay labelled.
- The Task 4 code needs the data, CLIP weights and MVFA's repository on the GPU server to rerun; none of them is in this folder. The results do not need them.

## Not included
- **PaperOrchestra** (`github.com/google-research/paper-orchestra`): the agent clones it at a pinned commit when the run starts (`plans/PAPER_PLAN.md` §2).
- **API keys** (Gemini; optionally Semantic Scholar): environment variables only, never in files.
- **Task 4 round 2** (`task4_r2/PLACEHOLDER.md` says what to put there).
- **Human inputs** (authors and affiliation, Gemini key, template, workshop target): all optional with defaults (`plans/PAPER_PLAN.md` §2). The run asks no questions; the human reviews once at the end (Gate 1).
