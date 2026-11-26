# Issue Log — Paper v2

### P-01: v1 task inputs lacked results and used a different protocol
- Phase / step: Phase 0
- Type: source mismatch
- How found: compared supplied v2 manifest and plan with the prior v1 bundle.
- Evidence: v2 includes `task4_r1/`, a checked report, decisions, code, and results; the v1 folder did not.
- Feedback given to the model: start fresh from the v2 bundle and ignore older local material.
- Fix: moved v1-generated logs and candidate CSV into `paper/archive/v1_attempt/`; extracted v2 inputs to `paper/inputs/`.
- Outcome: all further work uses v2 sources.

### P-02: PaperOrchestra may cite outside the allowed pool by default
- Phase / step: Phase 1, F1
- Type: source-control gap
- How found: inspected the pinned literature-review agent; it uses Google Search and Semantic Scholar title enrichment.
- Evidence: `methods/agents/literature_review_agent.py` calls Google Search and `s2_title_search`.
- Feedback given to the model: restrict the actual citation registry to the supplied pool and its approved extra references.
- Fix: implement a curated citation-pool path and verify all final citation keys against `refs.bib`.
- Outcome: pending final generation checks.

### P-03: PaperOrchestra template defaults omit a separate Problem Formulation section
- Phase / step: Phase 1, F2
- Type: format mismatch
- How found: inspected `templates/cvpr2025/template.tex`; it has five top-level sections and a placeholder appendix.
- Evidence: v2 plan requires seven main sections, including separate Problem Formulation and a references-only fifth page.
- Feedback given to the model: force the seven-section structure and remove appendix content.
- Fix: patch the outline and use a paper-specific CVPR template copy.
- Outcome: pending compilation and page verification.

### P-04: Writing/refinement prompts permit overlength and numerical edits
- Phase / step: Phase 1, F3/F4
- Type: prompt gap
- How found: inspected the pinned writing and content-refinement prompts.
- Evidence: prompts ask for detailed results and permit edits without a 4+1 budget or immutable citation/numerical-source rules.
- Feedback given to the model: add the page budgets and prohibit changing reported values, citation keys, or the analysis framing.
- Fix: patch prompts and run the numeric/citation/page checks.
- Outcome: pending final generation checks.

### P-05: The supplied results builder omitted secondary `pi*.json` files
- Phase / step: Phase 3
- Type: input completeness
- How found: compared the builder's glob with the round 1 artifact list.
- Evidence: round 1 has multiple named `pi*.json` records, while the supplied script initially selected only three fixed filenames.
- Feedback given to the model: include all per-run π validation artifacts.
- Fix: extended the local builder and rebuilt `results.json`; protocol metadata required by the narrative is recorded in the same JSON.
- Outcome: report reconciliation passes for all 111 checked values (±0.01).

### P-06: Python and TeX toolchain differed from the plan's environment
- Phase / step: Phase 2
- Type: environment
- How found: the environment has Python 3.13, Matplotlib initially selected Tk in a headless run, and no TeX executable was installed.
- Evidence: native Tk initialization failed; Tectonic was not initially present.
- Fix: installed local `paper/.venv` using system-site packages and trimmed dependencies; figure generation uses `MPLBACKEND=Agg`. Downloaded official Tectonic 0.17.0 into `paper/tools/tectonic-bin/`; the initial TeX bundle is cached by the elevated compile process.
- Outcome: Python imports and raw-material figure generation pass. Tectonic compilation succeeds through the elevated network path; local sandbox and elevated processes do not share the Tectonic cache.

### P-07: The curated citation pool is a source registry, not a required citation count
- Phase / step: Phase 1
- Type: page budget
- How found: the upstream prompt required citing 90% of supplied papers, conflicting with the plan's “as many as the text needs” rule.
- Evidence: 20 approved candidate references are available, but related work has a 0.50-page budget.
- Fix: cap the prompt's minimum citations at 10 and retain the citation registry for validation.
- Outcome: pending pilot and final page checks.

### P-08: Windows console encoding blocked the first pilot launch
- Phase / step: Phase 2, dummy pilot
- Type: environment
- How found: CLI startup failed while printing its first status line.
- Evidence: Python cp1252 raised `UnicodeEncodeError` for an emoji before any writer agent ran.
- Fix: rerun with `PYTHONIOENCODING=utf-8` in the process environment.
- Outcome: fixed; UTF-8 CLI output reached the Gemini API. The pilot then exposed the separate model/API issue logged as P-09.

### P-09: PaperOrchestra's default preview model was unavailable on the stable endpoint
- Phase / step: Phase 2, dummy pilot
- Type: model/API compatibility
- How found: first API-backed pilot call returned 404 before outline generation.
- Evidence: `gemini-3.1-pro-preview` was called over API `v1`; the API reported the model unsupported there. Testing on `v1beta` returned a daily free-tier quota limit for Pro.
- Fix: set the SDK API version to `v1beta`; use Google's current stable `gemini-3.5-flash` for drafting and review. A minimal generation check succeeded.
- Outcome: proceeding with this model; disclose it in the run log.

### P-10: Generated BibTeX contains an unescaped ampersand
- Phase / step: Phase 2, dummy pilot
- Type: LaTeX generation
- How found: Tectonic stopped while parsing the generated `.bbl` for an APRIL-GAN title containing `&`.
- Evidence: `initial_draft.bbl` reported `Misplaced alignment tab character &` on the title's reference entry.
- Fix: escape bare ampersands in metadata-derived and supplied BibTeX before compilation.
- Outcome: fixed in the exporter; the saved pilot compiled and passed the 4+1 page check after its BibTeX was escaped.

### P-11: Plotting agent could replace the plan's fixed figures
- Phase / step: Phase 4
- Type: evidence preservation
- How found: the plotting path generates new visuals from a model outline, while the plan provides a fixed Figure 1 method diagram and Figure 2 preselected heat maps.
- Evidence: `paper_writer_with_plotting.py` runs image generation in parallel and replaces the figure metadata with generated outputs.
- Fix: when `figures/info.json` registers existing supplied assets, the agent now passes those assets to the writer instead of regenerating figures.
- Outcome: pending full generation; expected output retains the two prescribed figures.

### P-12: Full pipeline retries would repeat paid drafting calls after local failures
- Phase / step: Phase 4
- Type: run control
- How found: inspected `paper_writer_with_plotting.py`; its outer retry loop reruns Outline, Literature, and Section Writing after any later exception.
- Evidence: a LaTeX error after Section Writing would repeat every generation call up to three times.
- Fix: limit the outer pipeline to one pass; retain the internal up-to-three reflection rounds and resume local fixes from saved outputs.
- Outcome: pending full generation.

### P-13: Full run stopped on Windows default Markdown decoding
- Phase / step: Phase 4
- Type: environment / text encoding
- How found: the first full-run attempt failed while reading UTF-8 `idea.md` with the system cp1252 default, before the first Gemini call.
- Evidence: `UnicodeDecodeError` in `utils/common_utils.py::load_md_file`.
- Fix: open Markdown inputs with explicit UTF-8.
- Outcome: pending restart; no full-run model calls were made on the failed attempt.

### P-14: Literature agent also decoded UTF-8 inputs with the Windows default
- Phase / step: Phase 4
- Type: environment / text encoding
- How found: full run reached its Literature agent after producing the outline and registering the two figures.
- Evidence: `_load_context_files` raised `UnicodeDecodeError` on `idea.md` before literature synthesis.
- Fix: set explicit UTF-8 on the Literature agent's Markdown and outline readers; the outline agent also writes JSON with UTF-8. Reuse the already-generated outline on restart.
- Outcome: pending restart; no literature-generation call was made in the failed attempt.

### P-15: Plotting pipeline copied figure metadata but not figure assets
- Phase / step: Phase 4, refinement compile
- Type: local file handling
- How found: the generated `.tex` referenced the supplied figures, but Tectonic could not find them in `content_refinement_workdir/figures`.
- Evidence: both source figures were present under `raw_materials/figures`; the plotting pipeline created an empty `latex_writeup/figures` and copied returned assets to a different nested path.
- Fix: copy supplied figures to the LaTeX figures folder at pipeline setup; resume the saved draft after copying its figures into the refinement directory.
- Outcome: pending resumed compile; no additional writing calls are needed.

### P-16: Gemini free-tier quota was exhausted during automated peer review
- Phase / step: Phase 5, independent review
- Type: external service quota
- How found: the initial full-paper PDF compiled; the reviewer then received HTTP 429.
- Evidence: API reported a 20 requests/day free-tier limit and a reset in about 20 hours.
- Fix: stop retries; finish content and layout checks locally and perform a separate Codex review against source artifacts.
- Outcome: manual independent review replaces the unavailable Gemini review; recorded in the run log.

### P-17: PaperOrchestra draft exceeded the 4+1 content budget
- Phase / step: Phase 5, page check
- Type: length / layout
- How found: compiled saved first draft before final review.
- Evidence: draft PDF had 6 content pages plus 1 references page.
- Fix: manually condense prose and preserve the two required tables and two figures; recompile and recheck the exact 4+1 count.
- Outcome: pending final PDF check.

### P-18: Generated prose misstated the six-dataset leave-one-modality-out setup
- Phase / step: Phase 5, methodology verification
- Type: factual accuracy
- How found: compared the generated Methodology paragraph with the plan and `results.json` protocol.
- Evidence: generated draft described three datasets as source and three as target, rather than six source-pool datasets with one held out per split.
- Fix: rewrite the Dataset paragraph to state the six-dataset pool, the 3/3 mask split, and the three held-out target modalities.
- Outcome: corrected in the compact final source.

### P-19: Results builder had no reader for rounds 2-3
- Type: input format. How found: round 2+ `results.csv` uses train2.py's columns (arm, fold, eval_dataset), not round 1's (config, target).
- Fix: `build_round_runs()` in `tools/build_results.py`, format auto-detected. Round 1 output unchanged.
- Outcome: `verify_results_all.py` reconciles 443 values with the round 2, round 3 and follow-up reports; round 1 still 111/111.

### P-20: Retention computed from rounded means differed by 0.001
- How found: MIL+10 % Brain retention 1.127 vs the frozen analysis's 1.126.
- Fix: compute retention from unrounded seed means, as analyze2/3.py do. Outcome: all retention values match.

### P-21: Round 2 best-epoch count included the budget runs
- How found: 39/61 vs the report's 37/57. Fix: `--best-epoch-exclude r2=g_m10,g_m25`. Outcome: 37/57 and 54/57 reproduced.

### P-22: check_numbers.py flagged LaTeX layout constants and could pass a sign flip
- How found: it flagged `\cmidrule{4-5}`, a 2.5 pt column separation and the "-50" of the retention formula; a fixture with -14.70 in place of +14.70 passed because such values exist elsewhere in results.json.
- Fix: the check strips layout commands and allows negated structural constants only; new `scripts/audit_claims.py` ties each quantitative claim to its exact results.json field and regenerates Table 1 cell by cell (`logs/CLAIM_AUDIT.md`).
- Outcome: number check PASS; claim audit 42/42 PASS.

### P-23: Equation (1) had level weights the code does not use
- How found: maths check against `task4_r3/train2.py` (`mil_loss` summed over levels) and round 1's `mvfa_mask_loss` (levels summed equally).
- Fix: Eq. (1) sums the four levels with equal weight, and the text says so. Gate (1 - max cosine, layer 24), nnPU risk (pu.py) and the mask-budget objective (build_epoch / mask_budget_set) checked: they match.

### P-24: Tectonic not available locally
- Fix: compile with MiKTeX pdflatex + bibtex; `scripts/check_pages_pdf.py` checks the 4+1 layout on the PDF and the LaTeX log.

### P-25: Both tables were too wide for their columns
- How found: overfull hbox warnings (94 pt, 18 pt). Fix: smaller font and column separation, shorter labels. Outcome: no overfull boxes.

### P-26: Independent review found framing overstatements (no numeric errors)
- How found: different-model review (logs/INDEPENDENT_REVIEW.md). Main points: cross-round "consistency" mixed baselines; "top-ranked" vs ranks 1 and 3; PU failure scope (pi fixed at 0.01, single-seed pi = 0.05 run).
- Fix: 33 text edits, new Table 3 (image AUROC from results.json), heat-map figure narrowed to keep 4+1; claim audit extended to 55 checks.
- Outcome: all checks pass.
