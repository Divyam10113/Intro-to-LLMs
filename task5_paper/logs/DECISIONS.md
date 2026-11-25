# Paper run v2 decisions — 2026-11-22

- Fresh source of truth: supplied `Task5_inputs_v2.zip` and `PAPER_PLAN.md` v2. Previous v1 shortlist and logs are archived under `paper/archive/v1_attempt/` and are not used.
- Authors: use the plan default, “Anonymous authors”.
- Template: PaperOrchestra CVPR 2025; no alternate template was specified.
- Workshop: none specified; paper will not claim workshop readiness.
- Gemini: key supplied in chat, used only in the generation process environment if needed. Its value is not persisted to files, logs, or generated artifacts.
- Task 4: Round 1 is the current evidence base. Framing rule §5.1 selects analysis paper F-C. Round 2 is absent; if it arrives, it will be added later as a separate protocol per §11.
- PaperOrchestra source: public repository pinned at `d9bae997e793a015ed38bb1605be9eb58c00284f` (detached HEAD).
- PaperOrchestra F1: implement a curated citation-pool path from the supplied Task 1 data; do not use its literature search for final references.
- PaperOrchestra F2–F4: force the seven-section outline, include the 4+1 budget in writing/refinement instructions, and lock numbers/citation keys and the no-improvement claim in prompts and post-generation checks.
- Model: the default Pro preview hit the account's free-tier daily quota, so use Gemini 3.5 Flash on `v1beta` after a successful minimal generation check.
- Figure choice: use supplied Figure 2 heatmaps and a locally drawn Figure 1 method diagram. The dummy pilot ran without plotting; the full generation follows the plan's plotting setting, with a check to retain the prescribed figure set and page budget.
- Plotting in the full run: invoke PaperOrchestra's plotting workflow, but register the plan's preselected assets from `figures/info.json` instead of asking the image model to regenerate evidence figures.
- Pilot result: PaperOrchestra pipeline completed after repairing missing CVPR package and bibliography escaping; dummy output passed the 4+1 page and citation checks, then was deleted.
- Output: do not commit or push. The final PDF and source remain in `paper/` for the single end review.

## Paper inputs status

- `paper/inputs/results.json` built from the v2 Round 1 folder using the supplied script.
- Result values will be cited only from `results.json`; the Report and CSV tables are verification sources.
- Remaining defaults/limitations are tracked in `paper/OPEN_ITEMS.md`.

## Final update decisions (2026-11-25)

- **F-1 No PaperOrchestra re-run.** The round-1 draft needed heavy manual compression (P-17) and factual fixes (P-18), and its automated review never ran (P-16, quota). Revising the checked draft directly keeps tighter control of numbers and costs no API quota. Raw materials are regenerated so the pipeline can be re-run later with a fresh key. Disclosed in the paper's AI-use paragraph; agreed with the human in chat.
- **F-2 Framing (PAPER_PLAN §5.2).** Round 3 + follow-up = partial win, so a **mixed paper**: headline H1 (MIL keeps 0.969 of MVFA's pixel-AUROC margin, confirmed) and FU-H (10 % masks + MIL beats full-mask MVFA on Brain, 5/5 pre-registered seeds); H4 ("matches") in Results; H2, H3, H5, round 2's ideas #1/#3 and round 1's PU in one "what did not hold" table (Table 2). No claim of beating MVFA overall or the published MVFA numbers (context only).
- **F-3 Rounds never pooled.** Table 1 is round 3 only; rounds 1-2 enter only through Table 2 and labelled sentences.
- **F-4 Figures.** Figure 1 = label-budget curve redrawn from `results.json` at column width (Task 4's figure was unreadable at column width). Figure 2 = round 3 Liver heat maps (image set fixed by round 1's rule before any map was viewed), added once page 4 had room; captioned "illustrative only". No method diagram (space).
- **F-5 In-mask diagnostic.** Quote per-fold means with each run counted once (round 3's convention, its G3-1): round 2 gated MIL 0.466 vs MIL 0.493 (analysis.md's pooled 0.400 / 0.426 double-counts RESC+OCT17 runs). Same conclusion either way.
- **F-6 PatchCore** added to the citation pool from Task 1's set (relevance 3) to cite the normal-bank memory behind gated MIL.
- **F-7 Authors** stay "Anonymous authors" (plan default) until the human supplies them.

- **F-8 No AI-use statement in the submitted text (2026-11-25, human's instruction).** The human asked that the paper not mention LLM use for the course submission; they will discuss further publication with the professor. All LLM, agent and course-task mentions were removed from the manuscript (study design reworded as "candidate refinements ranked in advance"). The full AI-use record stays in `logs/` (GENERATION_LOG, this file, ISSUES_LOG) and in Task 4's transcripts, for that discussion. Noted to the human: the Task 5 brief asks for an LLM-in-the-loop process, and most venues require disclosure of LLM use.
