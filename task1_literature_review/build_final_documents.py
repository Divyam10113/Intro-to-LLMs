#!/usr/bin/env python3
"""Regenerate the reference lists of the two documents in ``final documents/``.

The documents themselves (prose, structure, preambles) are hand-written and live
in ``final documents/<run>/``.  Only the bibliography block, delimited by

    % >>> BIBLIOGRAPHY ... >>>
    % <<< END BIBLIOGRAPHY <<<

is generated, straight from the citation keys the gap-inputs bundles already
verified (``DELIVERABLES/<run>/gap_inputs/item3_citation_key.csv``), so every
``[n]`` in the rendered text maps to the right paper.

    python3 build_final_documents.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # Paper implementation/
REPO = ROOT.parent
DELIV = ROOT / "DELIVERABLES"
FINAL = REPO / "final documents"

START = "% >>> BIBLIOGRAPHY"
END = "% <<< END BIBLIOGRAPHY <<<"

DOCUMENTS = [
    ("pipeline", FINAL / "pipeline" / "survey.tex", 60),
    ("prompted", FINAL / "prompted" / "literature_review.tex", 60),
]

_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(text: str) -> str:
    return "".join(_ESCAPE.get(ch, ch) for ch in (text or ""))


def read_key(run: str) -> list[dict[str, str]]:
    path = DELIV / run / "gap_inputs" / "item3_citation_key.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["review_number"]))
    return rows


def bibitems(rows: list[dict[str, str]], declared: int) -> list[str]:
    # the reference list starts on a page of its own, so the review body keeps to
    # its page budget and the citations occupy the extra pages
    out = [r"\clearpage", r"\begin{thebibliography}{%d}" % declared]
    for r in rows:
        title = latex_escape(r["title"]).rstrip(".")
        link = (r.get("public_link") or "").strip()
        entry = f"\\bibitem{{ref{int(r['review_number'])}}} {title}."
        if link:
            entry += f" \\url{{{link}}}"
        out.append(entry)
    out.append(r"\end{thebibliography}")
    return out


def inject(path: Path, lines: list[str]) -> int:
    text = path.read_text(encoding="utf-8")
    head_end = text.index(START)
    head_end = text.index("\n", head_end) + 1
    tail = text.index(END)
    body = "\n".join(lines) + "\n\n"
    path.write_text(text[:head_end] + body + text[tail:], encoding="utf-8")
    return len(lines) - 3          # \clearpage, \begin{thebibliography}, \end{thebibliography}


def main() -> None:
    for run, doc, declared in DOCUMENTS:
        rows = read_key(run)
        n = inject(doc, bibitems(rows, declared))
        numbers = [int(r["review_number"]) for r in rows]
        missing = sorted(set(range(1, declared + 1)) - set(numbers))
        print(f"[{run}] {doc.relative_to(REPO)}: {n} references"
              + (f" (missing numbers: {missing})" if missing else ""))


if __name__ == "__main__":
    main()
