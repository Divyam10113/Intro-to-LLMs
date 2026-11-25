# Generation Log — Paper v2

- 2026-11-22: Fresh v2 input extraction; prior v1 work archived and excluded.
- 2026-11-22: Built `paper/inputs/results.json` from supplied `task4_r1/` using the supplied `tools/build_results.py`.
- 2026-11-22: Cloned PaperOrchestra and pinned detached HEAD to `d9bae997e793a015ed38bb1605be9eb58c00284f`.
- 2026-11-22: Installed filtered Python dependencies into `paper/.venv`; generated raw materials and figures using Agg backend.
- 2026-11-22: Installed official Tectonic 0.17.0 in `paper/tools/tectonic-bin/`; built its TeX bundle and compiled a five-page dummy document through the elevated network path.
- 2026-11-22: Added number, citation, and 4+1 page checkers; dummy number/citation/page checks pass. Negative fixtures and the PaperOrchestra Gemini pilot are next.
- 2026-11-22: Dummy pilot first launch stopped before API use because Windows cp1252 could not encode a CLI status emoji; retrying with UTF-8 process output. No key was written to disk or logs.
- 2026-11-22: Pro preview returned 404 on the stable endpoint; the beta endpoint confirmed model availability but reported the account's free-tier daily Pro quota exhausted. A minimal Gemini 3.5 Flash generation succeeded. The PaperOrchestra client now uses `v1beta`; pilot will run on `gemini-3.5-flash`. No cost was reported by the API smoke test.
- 2026-11-22: Saved dummy draft reached refinement but compilation first failed because the model omitted `\usepackage{cvpr}`; deterministic package restoration was added. Second compile passed the package stage and found a raw ampersand in generated BibTeX. The metadata exporter now escapes bare ampersands; resumed the saved pilot without re-running outline or section drafting.
- 2026-11-22: Dummy pilot completed with Gemini 3.5 Flash (writer/reflection), one refinement pass and one formatting pass. Final dummy PDF compiled to 5 pages (4 content + references only); all 20 cited keys resolved to the curated pool. Visual page overview reviewed. Dummy run inputs and outputs were deleted after QA; notes remain here.
- 2026-11-22: First full-run attempt stopped before any model call because the Markdown reader used cp1252 for UTF-8 inputs; patched it to read UTF-8 and will restart.
- 2026-11-22: Second full-run attempt generated the outline and registered the two supplied figures, then the Literature agent hit another cp1252 reader before its model call. Added explicit UTF-8 to that reader and configured the restart to reuse the saved outline.
- 2026-11-22: Resumed full run from the saved outline; literature and section writing completed. Compile then found supplied figure files were not copied into the refinement workdir. Added deterministic figure copying and will resume refinement from `raw_draft_paper.tex`.
- 2026-11-22: The saved full draft compiled after restoring figures; automated peer review hit the Gemini free-tier cap (20 requests/day). Stopped its retries. Initial PDF had 6 content pages, so manually compressed the manuscript while preserving required result tables and figures. The compact source now passes number and citation checks; final compile/page check is pending.
- PaperOrchestra pilot is being retried with `gemini-3.5-flash` after confirming the prior model/quota issue. The first attempt ended before outline generation.

## Final update: rounds 2, 3 and the follow-up (2026-11-25)

- Inputs added: `inputs/task4_r2/` (round 2), `inputs/task4_r3/` (round 3), `inputs/task4_fu/` (pre-registered Brain follow-up), from `task4_final_share.zip`; PaperOrchestra fork, citation pool, refs.bib and round-1 raw materials from `paper_project_bundle.zip` (round-1 materials kept in `inputs/raw_materials_round1/`).
- `tools/build_results.py` extended with a reader for the train2.py logger (rounds 2, 3, follow-up): per-arm stats over main seeds, paired differences and one-sided sign tests, retention vs the re-run MVFA and vs R0, Holm tiers, in-mask diagnostic per fold, best-epoch counts. Round 1's output is identical to the teammate's `results.json`. One command: `build_all.sh`.
- PaperOrchestra raw materials regenerated for the final framing by `scripts/prepare_raw_materials_final.py` (idea.md, experiment_log.md, related_work_outline.md, citation registry, figures + info.json); every number read from `results.json`.
- **No PaperOrchestra re-run** (decision F-1). The manuscript `runs/final/latex_writeup/final_paper.tex` was revised from the round-1 PaperOrchestra draft by an LLM agent (Claude Code, Opus 5.5), section by section, with every number and citation checked by script.
- Compiled locally with MiKTeX pdflatex + bibtex (the teammate used Tectonic): 5 pages, 4 content + references only.
- Full pipeline: `check_all.sh` (rebuild numbers, reconcile with reports, raw materials, compile, then numbers, citations, claim audit and 4+1 pages).
