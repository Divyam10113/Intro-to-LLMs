"""Short (2-page) literature review generator.

The full QUAL-SG pipeline produces a multi-section survey.  This module turns the
same outputs into a **concise 2-page literature review** (~1 100 words), the
deliverable requested for this project.  Changes approved for run v5:

* change 13 - when the survey exists (step 5), the review is written by
  condensing it, so the two documents agree and the review inherits the
  survey's structure and checked claims; otherwise it is written directly from
  the references (``direct`` mode below);
* change 1 - no "at least three citations per paragraph" rule; a citation only
  where a claim needs support, at most three per claim;
* change 2 - themes come from the survey outline (or are chosen by the model),
  never from a hard-coded topic template;
* changes 7 and 9 - compare rather than list; evidence-backed open challenges;
* change 8 - full abstracts (safety cap ``config.REVIEW_ABSTRACT_CHARS``);
* change 11 - claim check and one revision pass;
* change 3 - bibliography from publisher metadata.

The same generator still serves the two baselines (``naive_rag``, ``fully_llm``).

Outputs (per source): ``literature_review_2page.{md,tex,json}`` in the run folder.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import baselines, bibliography, claim_check, common, config
from .step5_generate import CHALLENGE_INSTRUCTION, COMPARE_INSTRUCTION, build_numbered_references

LOGGER = common.get_logger("short_review")

TARGET_WORDS = 1100
CITATION_RULE = ("Cite a paper only where a claim needs support; use at most three citations for any single "
                 "claim, written exactly as `ref [n]` with numbers from the list above.")
JSON_SHAPE = """{
  "title": "TITLE OF THE REVIEW",
  "abstract": "2-3 sentence summary of the review",
  "sections": [{"heading": "...", "content": "..."}],
  "takeaway": "one-sentence concluding takeaway"
}"""


def outline_themes() -> List[str]:
    """Section titles of the survey outline, when one exists (change 2)."""
    if not config.CANONICAL_OUTLINE.exists():
        return []
    obj = json.loads(common.clean_json_string(config.CANONICAL_OUTLINE.read_text(encoding="utf-8")))
    return [str(s.get("section_title", "")).strip() for s in obj.get("outline", []) if s.get("section_title")]


def survey_is_complete() -> bool:
    if not config.CANONICAL_CONTENT.exists():
        return False
    subs = common.read_json(config.CANONICAL_CONTENT).get("subsections", [])
    return bool(subs) and all(str(s.get("content", "")).strip() and not s.get("error") for s in subs)


def build_condense_prompt(topic: common.Topic) -> str:
    """Change 13: condense the finished survey into the 2-page review."""
    survey = common.read_json(config.CANONICAL_CONTENT)
    blocks: Dict[str, List[str]] = {}
    for sub in survey.get("subsections", []):
        blocks.setdefault(sub["section_title"], []).append(f"### {sub['subsection_title']}\n{sub['content']}")
    text = "\n\n".join(f"## {sec}\n" + "\n\n".join(parts) for sec, parts in blocks.items())
    n = len(blocks)
    return f"""You are condensing a survey on "{topic.title}" into a 2-page literature review of about {TARGET_WORDS} words, no more.

THE SURVEY:
{text}

Instructions:
1. Keep the survey's structure: one paragraph per section ({n} in total), headed by the section title (you may shorten it to 8 words or fewer).
2. Keep the survey's main points and comparisons and their `ref [n]` citations exactly as written; do not add claims or references that are not in the survey.
3. {COMPARE_INSTRUCTION}
4. {CITATION_RULE}
5. In the last paragraph (open challenges): {CHALLENGE_INSTRUCTION}
6. Formal academic tone, no bullet lists, no filler.
7. Return STRICT JSON exactly in this shape:
{JSON_SHAPE}
Do not include commentary, markdown fences or any text outside the JSON object."""


def build_prompt(topic: common.Topic, refs: List[Dict[str, Any]], source: str) -> str:
    """Direct mode: write the review from the references themselves."""
    if refs:
        ref_block = "\n".join(
            f"- ref [{r['refNo']}]: {r['title']}. {common.truncate(r.get('abstract', ''), config.REVIEW_ABSTRACT_CHARS)}"
            for r in refs
        )
        ref_rule = (f"Ground every claim in the provided references; do not invent references. {CITATION_RULE}")
    else:
        ref_block = "(no references are provided - write from general knowledge and do NOT use bracketed citations)"
        ref_rule = "No external references are available: do not fabricate bracketed citations."
    themes = outline_themes()
    if themes:
        theme_rule = ("Write one thematic paragraph per theme, in this order: " + " -> ".join(themes) + ".")
    else:
        theme_rule = ("Choose 4 to 6 themes that together tell a coherent story of the references above; the last "
                      "theme is always open challenges. Write one thematic paragraph per theme.")

    return f"""You are an academic expert writing a short literature review on:
"{topic.title}"

Length: about {TARGET_WORDS} words total (this must fit on ~2 pages), no more.

Available references:
{ref_block}

{theme_rule}

Rules:
1. Formal academic tone, precise and dense; no bullet lists, no filler.
2. Each thematic paragraph must have a short bold-free heading (Title Case, <= 8 words).
3. {ref_rule}
4. {COMPARE_INSTRUCTION}
5. In the open-challenges paragraph: {CHALLENGE_INSTRUCTION}
6. Return STRICT JSON exactly in this shape:
{JSON_SHAPE}
Do not include commentary, markdown fences or any text outside the JSON object."""


def numbered_refs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs = []
    for i, r in enumerate(rows, start=1):
        title = str(r.get("title", "")).strip()
        if title:
            refs.append({"refNo": i, "title": title, "abstract": str(r.get("abstract", "")),
                         "doi": common.norm_doi(r.get("doi"))})
    return refs


def load_refs(source: str, topic: common.Topic) -> List[Dict[str, Any]]:
    if source == "fully_llm":
        return []
    if source == "qual_sg":
        return build_numbered_references(config.GENERATION_MAX_REFS)
    if source == "naive_rag":
        return numbered_refs(baselines.naive_rank(topic, topic.max_refs, config.INTERMEDIATES_DIR / "_naive_refs.csv"))
    raise ValueError(f"unknown source {source!r}")


def render_markdown(obj: Dict[str, Any], bib: Dict[str, Dict[str, Any]], source: str) -> str:
    lines = [f"# {obj.get('title', '')}", ""]
    if obj.get("abstract"):
        lines += [f"*{obj['abstract']}*", ""]
    for sec in obj.get("sections", []):
        lines += [f"## {sec.get('heading', '').strip()}", "", str(sec.get("content", "")).strip(), ""]
    if obj.get("takeaway"):
        lines += [f"**Takeaway.** {obj['takeaway'].strip()}", ""]
    if bib:
        lines += ["## References", ""]
        for key in sorted(bib, key=int):
            lines.append(f"{key}. {bibliography.format_text(bib[key])}")
    lines += ["", f"<!-- generated by qual_sg.short_review | source={source} -->"]
    return "\n".join(lines)


def render_tex(obj: Dict[str, Any], bib: Dict[str, Dict[str, Any]], topic: common.Topic) -> str:
    esc = bibliography.tex_escape

    def body(text: str) -> str:
        parts, pos = [], 0
        for m in claim_check.CITE_RE.finditer(text):
            parts += [esc(text[pos:m.start()]), "\\cite{ref%s}" % m.group(1)]
            pos = m.end()
        return "".join(parts + [esc(text[pos:])])

    out = ["% Short literature review generated by qual_sg.short_review", r"\documentclass[11pt]{article}",
           r"\usepackage[utf8]{inputenc}", r"\usepackage[margin=1in]{geometry}", r"\usepackage{url}",
           r"\begin{document}", f"\\title{{{esc(obj.get('title', topic.title))}}}", r"\date{}", r"\maketitle"]
    if obj.get("abstract"):
        out += [r"\begin{abstract}", esc(str(obj["abstract"])), r"\end{abstract}"]
    for sec in obj.get("sections", []):
        out += [f"\\section*{{{esc(str(sec.get('heading', '')))}}}", body(str(sec.get("content", ""))), ""]
    if obj.get("takeaway"):
        out += [r"\paragraph{Takeaway.} " + esc(str(obj["takeaway"])), ""]
    if bib:
        out.append(f"\\begin{{thebibliography}}{{{len(bib)}}}")
        for key in sorted(bib, key=int):
            out.append(f"\\bibitem{{ref{key}}} {bibliography.format_tex(bib[key])}")
        out.append(r"\end{thebibliography}")
    out.append(r"\end{document}")
    return "\n".join(out)


def generate(source: str, outdir: Path, force: bool = False) -> Optional[Dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "literature_review_2page.json"
    if json_path.exists() and not force:
        LOGGER.info("%s already exists (set QUAL_SG_FORCE=1 to redo) - skipping", json_path)
        return json.loads(json_path.read_text(encoding="utf-8"))

    topic = common.load_topic()
    refs = load_refs(source, topic)
    mode = "condense_survey" if source == "qual_sg" and survey_is_complete() else "direct"
    LOGGER.info("short review [%s]: %d reference(s), mode=%s", source, len(refs), mode)
    writer = common.make_llm_client("writer")
    prompt = build_condense_prompt(topic) if mode == "condense_survey" else build_prompt(topic, refs, source)
    t0 = time.time()
    last_err = None
    messages = [{"role": "user", "content": prompt}]
    obj: Dict[str, Any] = {}
    res = None
    for attempt in range(1, 4):
        try:
            res = writer.chat(messages, tag=f"short-review-{source}")
            obj = json.loads(common.clean_json_string(res.text))
            if not obj.get("sections"):
                raise ValueError("no 'sections' in output")
            markers = claim_check.CITE_RE.findall(" ".join(str(s.get("content", "")) for s in obj["sections"]))
            if refs and len(markers) < 3:
                raise ValueError(f"only {len(markers)} in-text citations in the whole review")
            break
        except common.PendingLLM:
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            LOGGER.warning("attempt %d failed: %s", attempt, exc)
            messages = messages + [{"role": "user", "content": (
                "Your previous answer was not usable (" + str(exc)[:120] + "). Return ONLY valid JSON matching "
                "the schema above, citing references where claims need support.")}]
    else:
        raise RuntimeError(f"short review generation failed: {last_err}")

    check: Dict[str, Any] = {}
    if config.CLAIM_CHECK and refs:
        refs_by_no = {int(r["refNo"]): r for r in refs}
        blocks = [{"heading": str(s.get("heading", "")), "content": str(s.get("content", ""))} for s in obj["sections"]]
        check = claim_check.check_and_revise(blocks, refs_by_no, common.make_llm_client("eval"), writer,
                                             tag=f"review-{source}")
        obj["sections"] = check["blocks"]

    bib = bibliography.load_or_build(refs) if refs else {}
    words = sum(len(str(s.get("content", "")).split()) for s in obj["sections"]) + len(str(obj.get("abstract", "")).split())
    bundle = {
        "source": source,
        "mode": mode,
        "topic": topic.title,
        "title": obj.get("title", topic.title),
        "abstract": obj.get("abstract", ""),
        "sections": obj["sections"],
        "takeaway": obj.get("takeaway", ""),
        "references": [{k: r[k] for k in ("refNo", "title", "doi")} for r in refs],
        "word_count": words,
        "models": {"writer": getattr(res, "model", ""), **(check.get("models") or {})},
        "claim_check": {k: v for k, v in check.items() if k not in ("blocks",)} if check else None,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    common.write_json(json_path, bundle)
    (outdir / "literature_review_2page.md").write_text(render_markdown(obj, bib, source), encoding="utf-8")
    (outdir / "literature_review_2page.tex").write_text(render_tex(obj, bib, topic), encoding="utf-8")
    LOGGER.info("short review [%s]: %d words, %d sections, mode=%s -> %s", source, words, len(obj["sections"]), mode, outdir)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="2-page literature review generator")
    parser.add_argument("--source", choices=["qual_sg", "naive_rag", "fully_llm", "all"], default="qual_sg")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config.ensure_dirs()
    sources = ["qual_sg", "naive_rag", "fully_llm"] if args.source == "all" else [args.source]
    for src in sources:
        outdir = config.OUT_DIR if src == "qual_sg" else (config.OUT_DIR / "baselines" / src)
        generate(src, outdir, force=args.force)


if __name__ == "__main__":
    main()
