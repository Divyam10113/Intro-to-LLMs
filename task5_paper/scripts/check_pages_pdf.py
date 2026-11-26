"""4+1 page check on an already-compiled PDF (MiKTeX/pdflatex build; check_pages.py needs Tectonic).

Passes when: exactly 5 pages; no References heading on pages 1-4; page 5 has the References heading and no body
text (Conclusion / disclosure); no appendix; the LaTeX log has no undefined references/citations or overfull boxes.
    python -E -P scripts/check_pages_pdf.py runs/final/latex_writeup/final_paper.pdf
"""
import re
import sys
from pathlib import Path

import pymupdf

pdf = Path(sys.argv[1])
doc = pymupdf.open(pdf)
pages = [p.get_text() for p in doc]
errs = []
if len(doc) != 5:
    errs.append(f"expected 5 pages, found {len(doc)}")
for i, t in enumerate(pages[:4], 1):
    if re.search(r"(?im)^\s*(references|bibliography)\s*$", t):
        errs.append(f"References heading on content page {i}")
if len(pages) >= 5:
    if not re.search(r"(?im)^\s*references\s*$", pages[4]):
        errs.append("page 5 has no References heading")
    if re.search(r"Conclusion|AI-use disclosure", pages[4]):
        errs.append("body text on page 5")
if "appendix" in "\n".join(pages).lower():
    errs.append("appendix text found")
log = pdf.with_suffix(".log")
if log.exists() and re.search(r"Undefined (control sequence|references)|Citation .* undefined|Overfull",
                              log.read_text(errors="ignore"), re.I):
    errs.append("undefined reference/citation or overfull box in the LaTeX log")
if errs:
    sys.exit("FAIL: " + "; ".join(errs))
print(f"PASS: {pdf.name}: 4 content pages + references-only page 5; clean LaTeX log.")
