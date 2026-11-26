# Verification Log

## Results rebuild — 2026-11-22

- Source: supplied `paper/inputs/tools/build_results.py`, round folder `paper/inputs/task4_r1/`.
- Output: `paper/inputs/results.json`.
- Check: rebuilt M-h and control means/std, mask-free means, R0 values, paired deltas, sign test, retention, and π records against `task4_r1/report/TASK4_REPORT.md` and the shipped `report/paper_inputs/*.csv` tables.
- Outcome: values agree to the report's stated ±0.01 tolerance. `mh - mh_ctrl` AUPRO is positive in 6/9 pairs; one-sided sign-test p = 0.2539. M's mean seed-0 retention across targets is 0.583; B0's is 0.966. These figures support the plan's analysis framing, not an improvement claim.
- Remaining: per-section claim map, citation audit, math audit, 4+1 page check, and independent review.
# Verification log — Paper v2

| Check | Claim | Source | Result | Fix / evidence |
|---|---|---|---|---|
| Results reconciliation | Report table values match rebuilt results | `results.json` vs Task 4 Round 1 report tables | PASS | 111 cells checked at ±0.01 by `verify_results.py` |
| Dummy numeric validator | approved number passes; unapproved number rejected | dummy LaTeX + `results.json` | PASS | Validated below; negative fixture follows |
| Dummy citation validator | pooled citation resolves | dummy LaTeX, pool, bibliography | PASS | One cited key resolves |
| Dummy page validator | four content pages + references only page 5 | dummy CVPR document | PASS | Compiled with Tectonic 0.17.0 |
| Dummy pilot page/citation checks | 4 content pages, references-only page 5, citations pooled | PaperOrchestra pilot source | PASS | 20 cited keys; visual overview reviewed; run outputs deleted after QA |
| Full draft numeric claims | All body numeric literals match `results.json` or documented structural constants | compact `final_paper.tex` and results | PASS | `check_numbers.py` |
| Full draft citations | Every citation resolves to an approved pooled key | compact `final_paper.tex`, bibliography, citation pool | PASS | 20 cited keys; `check_citations.py` |
| Full draft independent review | Claims and framing checked against plan and Task 4 artifacts | compact manuscript, `results.json`, Round 1 report | In progress | Gemini quota blocked automated review; Codex manual review underway |

## Final manuscript (2026-11-25): `runs/final/latex_writeup/final_paper.tex`

| Check | Claim / target | Source | Result | Notes |
|---|---|---|---|---|
| Results reconciliation, round 1 | results.json = report tables | task4_r1 report | PASS | 111 values (`verify_results.py`) |
| Results reconciliation, rounds 2-3 + follow-up | results.json = report tables | task4_r2 analysis.md; task4_r3 paper tables; task4_fu analysis_fu.md | PASS | 443 values (`verify_results_all.py`) |
| 1 Numbers | every numeric literal is in results.json or structural | final_paper.tex | PASS | `check_numbers.py` (layout commands stripped, P-22) |
| 1b Claim audit | each quantitative claim = its exact field; Table 1 regenerated | final_paper.tex vs results.json | PASS 42/42 | `audit_claims.py`, see `CLAIM_AUDIT.md` |
| 2 Citations | every cite key in refs.bib and in the pool | references.bib, citation_pool.json | PASS | 21 keys |
| 3 Maths | Eq. 1 MIL, Eq. 2 budget, gate, Eq. 3 nnPU match the code | task4_r3/train2.py, common2.py, task4_r1/pu.py, train.py | PASS after fix | P-23 |
| 4 Claims | no "beats MVFA" overall; published numbers context only; post-hoc labelled; limitations stated | PAPER_CLAIMS.md, VERDICT.md, FU_REPORT.md | PASS | Wording follows the frozen-rule outcomes |
| 5 Abstract/conclusion | written from the finished paper; checks 1, 2, 4 re-run | final_paper.tex | PASS | |
| 6 Format | 4 content pages + references-only page 5; no undefined refs or overfull boxes | final_paper.pdf | PASS | `check_pages_pdf.py`; pages viewed as images |
| 7 Independent review (different model) | reviewer prompt on unsupported claims | Claude Sonnet reviewer vs results.json, reports, rules, code | PASS after fixes | 18 findings (4 major), all addressed: `INDEPENDENT_REVIEW.md` |
