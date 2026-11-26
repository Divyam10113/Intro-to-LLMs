# Task 5: the final paper (4+1)

**Final PDF:** [`runs/final/latex_writeup/final_paper.pdf`](runs/final/latex_writeup/final_paper.pdf); source: `final_paper.tex` next to it.
*Image Labels Are Nearly Enough: A Pre-Registered Study of Weakly Supervised Localisation for Zero-Shot Medical Anomaly Detection.*

Built from Task 4 rounds 1-3 and the pre-registered follow-up. Framing: a mixed paper (PAPER_PLAN §5.2). The two headline results are both confirmed under frozen rules:
- plain MIL with image labels only keeps 0.969 of MVFA's pixel-AUROC margin;
- 10 % masks plus MIL beat full-mask MVFA on brain MRI (5/5 pre-registered seeds).

Everything that did not hold is in Table 2.

## Rebuild and check everything
```bash
bash check_all.sh
```
The script runs these steps in order:
1. rebuild `inputs/results.json` from all Task 4 rounds (`build_all.sh`);
2. reconcile it with every report table (554 values);
3. regenerate PaperOrchestra's raw materials and figures;
4. compile with MiKTeX;
5. check the numbers, the citations, the claim audit (42 claims) and the 4+1 page layout.

## Where things are
| Path | What |
|---|---|
| `inputs/results.json` | The only source of numbers (rounds `r1`, `r2`, `r3`, `fu`; never pooled) |
| `inputs/task4_r1`, `task4_r2`, `task4_r3`, `task4_fu` | Task 4 outputs per round (reports, results, code) |
| `inputs/citation_pool.json`, `refs.bib` | The only allowed citations (Task 1 pool + extra refs + PatchCore) |
| `inputs/raw_materials/` | PaperOrchestra inputs for the final paper (`raw_materials_round1/` = what round 1 ran on) |
| `paper-orchestra/` | The patched PaperOrchestra fork (F1-F4); re-run needs a new Gemini key |
| `tools/build_results.py` | Builds `results.json`; now reads rounds 2, 3 and the follow-up too |
| `scripts/` | Checks: `verify_results*.py`, `check_numbers.py`, `check_citations.py`, `audit_claims.py`, `check_pages_pdf.py`; materials: `prepare_raw_materials_final.py` |
| `logs/` | Generation log, decisions, issues (P-01 to P-25), verification log, claim audit, page budget |
| `OPEN_ITEMS.md` | What the human still has to do at the final review |
| `runs/round1/` | The teammate's round-1 PaperOrchestra run (kept for the AI-use record) |
