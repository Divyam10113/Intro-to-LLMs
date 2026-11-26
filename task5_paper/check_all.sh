#!/bin/bash
# Rebuild the numbers, compile the final paper and run every check (run from anywhere; needs MiKTeX + Python 3.11).
set -e
cd "$(dirname "$0")"
bash build_all.sh
python -E -P scripts/verify_results.py            # round 1 vs its report (111 values)
python -E -P scripts/verify_results_all.py        # rounds 2, 3, follow-up vs their reports (443 values)
python -E -P scripts/prepare_raw_materials_final.py
cp inputs/raw_materials/figures/fig1_budget.pdf inputs/raw_materials/figures/fig2_liver_heatmaps_r3.pdf runs/final/latex_writeup/figures/
( cd runs/final/latex_writeup
  pdflatex -interaction=nonstopmode final_paper.tex >/dev/null; bibtex final_paper >/dev/null
  pdflatex -interaction=nonstopmode final_paper.tex >/dev/null; pdflatex -interaction=nonstopmode final_paper.tex >/dev/null )
W=runs/final/latex_writeup
python -E -P scripts/check_numbers.py $W/final_paper.tex inputs/results.json
python -E -P scripts/check_citations.py $W/final_paper.tex $W/references.bib inputs/citation_pool.json
python -E -P scripts/audit_claims.py
python -E -P scripts/check_pages_pdf.py $W/final_paper.pdf
