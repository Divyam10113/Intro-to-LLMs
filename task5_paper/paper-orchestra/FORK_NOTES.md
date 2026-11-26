# PaperOrchestra fork used for the Task 5 paper

- Upstream: https://github.com/google-research/paper-orchestra (Apache-2.0, see `LICENSE`), pinned at commit
  `d9bae997e793a015ed38bb1605be9eb58c00284f` (2026-11-15).
- Our changes against that commit are in `FORK_PATCHES.diff` (13 modified files, plus `requirements-local.txt`).
  They implement the plan's fork items F1-F4 and the fixes logged in `../logs/ISSUES_LOG.md` (P-02 to P-15):
  citations only from the curated pool, a forced 7-section outline, the 4+1 page budget in the prompts, locked
  numbers and citation keys, UTF-8 file reading on Windows, Gemini `v1beta` / model selection, deterministic
  LaTeX package and BibTeX fixes, supplied figures passed through instead of regenerated, and a single outer pass.
- Inputs for a re-run: `../inputs/raw_materials/` (built by `../scripts/prepare_raw_materials_final.py`).
  A run needs `GEMINI_API_KEY` in the environment; no key is stored in this repository.
