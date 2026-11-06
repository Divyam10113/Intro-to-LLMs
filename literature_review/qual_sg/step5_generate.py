"""Step 5 - section-wise survey generation (QUAL-SG / SurveyGen, paper §2.5).

Given the outline produced by :mod:`qual_sg.step4_outline` and the ranked
reference pool from Step 3, this stage writes the survey one *subsection* at a
time.  The prompt starts from the one shipped with the authors' code
(``Paper implementation/survey_generation.py``, ``build_subsection_prompt``) and
carries the changes approved for run v5:

* change 4 - each subsection sees only the ``config.REFS_PER_SUBSECTION``
  references closest to it (embedding similarity), keeping global ``ref [n]``
  numbers, instead of every selected abstract;
* change 6 - a word range derived from ``config.SURVEY_TARGET_WORDS`` replaces
  "a minimum of 300 words";
* change 7 - paragraphs are organised around points of comparison, not lists;
* change 9 - open-challenge subsections must point to concrete evidence;
* change 11 - every subsection goes through a claim check and one revision
  pass (:mod:`qual_sg.claim_check`);
* change 3 - the bibliography is built from publisher metadata
  (:mod:`qual_sg.bibliography`).

Artefacts
---------
* ``config.CANONICAL_SECTIONS_DIR/{big:02d}_{sub:02d}_{slug}.json`` - one file
  per subsection (final text, draft, references given, claim-check verdicts,
  models); this is also the *resume* unit;
* ``config.CANONICAL_CONTENT`` - the bundled survey ``{title, subsections: [...]}``;
* ``config.CANONICAL_TEX`` / ``survey.md`` - LaTeX and Markdown renderings.

With the file backend a subsection whose LLM calls are still pending is skipped
and reported; re-running the step after the answers arrive completes it.

Running:  ``python3 -m qual_sg.step5_generate``  (from ``Paper implementation/``)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import bibliography, claim_check, common, config

LOGGER_NAME = "step5_generate"

#: Attempts per subsection before accepting a low-quality/parse-failed answer.
MAX_SUBSECTION_ATTEMPTS = 3

#: In-text citation marker produced by the prompt, e.g. ``ref [3]``.
CITE_RE = re.compile(r"\bref\s*\[\s*(\d+)\s*\]")

_LATEX_SPECIALS = "&%$#_{}"
CHALLENGE_RE = re.compile(r"challenge|open problem|future|outlook|limitation", re.I)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def load_outline() -> Dict[str, Any]:
    """Load :data:`config.CANONICAL_OUTLINE`, tolerating markdown fences."""
    with open(config.CANONICAL_OUTLINE, "r", encoding="utf-8") as fh:
        return json.loads(common.clean_json_string(fh.read()))


def build_numbered_references(max_refs: int) -> List[Dict[str, Any]]:
    """Number the selected references ``1..N`` with ``N = min(len(selected), max_refs)``.

    Each record keeps title, abstract, DOI (for the bibliography) and its
    ``refNo`` - the number the model is told to cite as ``ref [n]``.
    """
    df = pd.read_csv(config.CANONICAL_RANKED_CSV)
    selected = df[df["tag"] == "selected"]

    refs: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        if not title or not abstract or abstract.lower() == "nan":
            continue
        refs.append(
            {
                "title": title,
                "abstract": abstract,
                "doi": common.norm_doi(str(row.get("doi", "")) or ""),
            }
        )
        if max_refs > 0 and len(refs) >= max_refs:
            break

    for i, ref in enumerate(refs, start=1):
        ref["refNo"] = i
    return refs


def stringify_outline_for_prompt(outline_obj: Dict[str, Any]) -> str:
    """Flatten the outline into the ``Survey Title / Sections`` text block."""
    title = str(outline_obj.get("title", "")).strip()
    lines = [f"Survey Title: {title}", "Sections:"]
    for sec in outline_obj.get("outline", []):
        s_title = str(sec.get("section_title", "")).strip()
        s_desc = str(sec.get("description", "")).strip()
        lines.append(f"- {s_title}: {s_desc}")
        subs = sec.get("subsections", {}) or {}
        if isinstance(subs, dict):
            for sub_title, sub_desc in subs.items():
                lines.append(f"    • {sub_title}: {str(sub_desc).strip()}")
    return "\n".join(lines)


def format_references_for_prompt(refs: List[Dict[str, Any]]) -> str:
    """Render the numbered reference list (format of the authors' script)."""
    return "\n".join(
        [f"- ref [{r['refNo']}]: Title: {r['title']}\n  Abstract: {r['abstract']}" for r in refs]
    )


def word_range(n_subsections: int) -> Tuple[int, int]:
    """Per-subsection word range from the survey budget (change 6)."""
    per = config.SURVEY_TARGET_WORDS / max(1, n_subsections)
    return int(round(per * 0.8 / 10) * 10), int(round(per * 1.15 / 10) * 10)


COMPARE_INSTRUCTION = (
    "Organise each paragraph around one point of comparison: the supervision a method needs, how it "
    "adapts the model, or where it works and fails. State the point, then support it with one to three "
    "papers. Do not write sentences that only list papers one after another."
)
CHALLENGE_INSTRUCTION = (
    "This subsection belongs to the open-challenges part of the survey. Each open challenge must point to "
    "concrete evidence in the cited papers: a result that differs sharply between methods or datasets, a "
    "setting no cited method covers, or a limitation the authors state. Name the papers involved. Do not "
    "include challenges that would apply to any survey in this area."
)


def build_subsection_prompt(
    topic: str,
    outline_text: str,
    subsection_title: str,
    refs_prompt: str,
    words: Tuple[int, int],
    is_challenges: bool,
) -> str:
    """The Step-5 subsection prompt (authors' template plus approved changes 6, 7, 9)."""
    lo, hi = words
    extra = f"\n    6. {CHALLENGE_INSTRUCTION}" if is_challenges else ""
    last = 7 if is_challenges else 6
    return f"""
You are an academic expert in the field of "{topic}" with deep expertise in survey writing. Your task is to write a subsection of a survey, based on the following information:

The overall structure of the survey is as follows:
  - {outline_text}

You are now asked to write the following subsection:
- Subsection: {subsection_title}

The following papers, the ones most relevant to this subsection, including their titles and abstracts, are provided for reference:
- {refs_prompt}

Instructions for generating subsection content:
    1. Carefully analyze the provided references and identify those highly relevant to the subsection topic as the basis for your generation.
    2. Ensure the content directly addresses the specific topic of the subsection; it should be between {lo} and {hi} words.
    3. Ensure that all claims are supported by the provided references. Cite each reference using in-text citations in the format ref [n], with the reference numbers exactly as given above; cite only the references listed above.
    4. {COMPARE_INSTRUCTION}
    5. Maintain alignment with the parent section and overall survey topic, using a formal academic tone.{extra}
    {last}. Your output should strictly be a JSON object with the following two fields.
    {{
      "content": "the content of the subsection",
      "references": [
        {{
          "refNo": "reference number",
          "title": "title of the reference"
        }}
      ]
    }}

Now, please generate the content for the given subsection: {subsection_title}. Please ensure the output is formatted according to the requirements mentioned above. Do not include any explanation, commentary, or preamble in your response.
""".strip()


def iter_subsections(outline_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Enumerate ``(section, subsection)`` jobs in stable document order."""
    jobs: List[Dict[str, Any]] = []
    sections = outline_obj.get("outline", []) or []
    total_sections = len(sections)
    for big_idx, sec in enumerate(sections, start=1):
        if not isinstance(sec, dict):
            continue
        subs = sec.get("subsections") or {}
        sub_titles = list(subs.keys()) if isinstance(subs, dict) else []
        sec_title = str(sec.get("section_title", ""))
        for sub_idx, sub_title in enumerate(sub_titles, start=1):
            jobs.append(
                {
                    "big_idx": big_idx,
                    "total_sections": total_sections,
                    "section_title": sec_title,
                    "section_description": str(sec.get("description", "")),
                    "sub_idx": sub_idx,
                    "total_subs": len(sub_titles),
                    "subsection_title": sub_title,
                    "subsection_description": str(subs.get(sub_title, "")),
                    "is_challenges": big_idx == total_sections or bool(CHALLENGE_RE.search(sec_title)),
                }
            )
    return jobs


def assign_references(jobs: List[Dict[str, Any]], refs: List[Dict[str, Any]], k: int) -> Dict[int, List[int]]:
    """The ``k`` references closest to each subsection (change 4), as ``refNo`` lists.

    Similarity is the cosine between the subsection's title + description (and
    its section) and each reference's title + abstract, with the pipeline's
    embedding model; TF-IDF is the fallback when the model cannot be loaded.
    """
    job_texts = [f"{j['section_title']}. {j['subsection_title']}. {j['subsection_description']}" for j in jobs]
    ref_texts = [f"{r['title']}. {common.truncate(r['abstract'], 2000)}" for r in refs]
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(config.EMBED_MODEL_NAME)
        jv = model.encode(job_texts, normalize_embeddings=True)
        rv = model.encode(ref_texts, normalize_embeddings=True)
        sims = np.asarray(jv) @ np.asarray(rv).T
    except Exception:  # noqa: BLE001 - offline / no torch: lexical similarity instead
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        vec = TfidfVectorizer(stop_words="english").fit(job_texts + ref_texts)
        jm, rm = vec.transform(job_texts), vec.transform(ref_texts)
        sims = (jm @ rm.T).toarray()
    out: Dict[int, List[int]] = {}
    for i in range(len(jobs)):
        top = np.argsort(-sims[i])[:k]
        out[i] = sorted(int(refs[j]["refNo"]) for j in top)
    return out


def subsection_path(job: Dict[str, Any]) -> Any:
    """Resume/artefact path for one subsection: ``{big:02d}_{sub:02d}_{slug}.json``."""
    slug = common.sanitize_filename(str(job["subsection_title"]))
    return config.CANONICAL_SECTIONS_DIR / f"{job['big_idx']:02d}_{job['sub_idx']:02d}_{slug}.json"


# --------------------------------------------------------------------------- #
# Per-subsection generation
# --------------------------------------------------------------------------- #
def generate_subsection(
    client: Any,
    prompt: str,
    words: Tuple[int, int],
    logger,
) -> Tuple[Dict[str, Any], bool]:
    """Generate one subsection, retrying on bad JSON or a length far outside the range.

    Returns ``(entry, failed)``; ``entry`` has ``{content, references, word_count,
    length_flag, model}``.  :class:`common.PendingLLM` propagates (file backend).
    """
    lo, hi = words
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    last_obj: Optional[Dict[str, Any]] = None
    last_error: str = ""

    for attempt in range(1, MAX_SUBSECTION_ATTEMPTS + 1):
        result = client.chat(messages, json_mode=True, tag="subsection")
        try:
            obj = json.loads(common.clean_json_string(result.text or ""))
            if not isinstance(obj, dict):
                raise ValueError("JSON root must be an object")
            content = obj.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("'content' must be a non-empty string")
            references = obj.get("references", [])
            if not isinstance(references, list):
                references = []
        except Exception as exc:  # noqa: BLE001 - malformed answer, retry
            last_error = f"attempt {attempt}: {exc}"
            logger.warning("subsection JSON invalid on attempt %d/%d: %s", attempt, MAX_SUBSECTION_ATTEMPTS, exc)
            messages = messages + [{"role": "user", "content": (
                "Return only valid JSON matching the requested schema: "
                '{"content": "...", "references": [...]}. No markdown fences, no commentary.')}]
            continue

        n_words = len(content.split())
        flag = "short" if n_words < 0.6 * lo else ("long" if n_words > 1.5 * hi else "")
        last_obj = {"content": content, "references": references, "word_count": n_words,
                    "length_flag": flag, "model": result.model}
        if not flag:
            return last_obj, False
        logger.warning("subsection %s on attempt %d/%d: %d words (target %d-%d)",
                       flag, attempt, MAX_SUBSECTION_ATTEMPTS, n_words, lo, hi)
        last_error = f"attempt {attempt}: {n_words} words (target {lo}-{hi})"
        messages = messages + [{"role": "user", "content": (
            f"The content field has {n_words} words; the target is {lo}-{hi} words. Rewrite it to fit the "
            "range and return only valid JSON with the same fields.")}]

    if last_obj is not None:
        logger.warning("accepted subsection outside the length range after %d attempts: %s",
                       MAX_SUBSECTION_ATTEMPTS, last_error)
        return last_obj, False
    logger.error("subsection failed after %d attempts: %s", MAX_SUBSECTION_ATTEMPTS, last_error)
    return {"content": "", "references": [], "error": last_error}, True


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _tex_escape(text: str) -> str:
    """Escape LaTeX specials in headings/titles."""
    out: List[str] = []
    for ch in str(text):
        if ch in _LATEX_SPECIALS:
            out.append("\\" + ch)
        elif ch == "\\":
            out.append("\\textbackslash{}")
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _cites_to_latex(content: str) -> str:
    """``ref [3]`` -> ``\\cite{ref3}``; everything else stays escaped plain text."""
    text = content or ""
    parts: List[str] = []
    pos = 0
    for match in CITE_RE.finditer(text):
        parts.append(_tex_escape(text[pos:match.start()]))
        parts.append("\\cite{ref%s}" % match.group(1))
        pos = match.end()
    parts.append(_tex_escape(text[pos:]))
    return "".join(parts)


def _ordered_entries(outline_obj: Dict[str, Any], entries: List[Dict[str, Any]]):
    lookup = {(str(e.get("section_title", "")), str(e.get("subsection_title", ""))): e for e in entries}
    for sec in outline_obj.get("outline", []) or []:
        sec_title = str(sec.get("section_title", ""))
        subs = sec.get("subsections") or {}
        yield sec_title, [(str(t), lookup.get((sec_title, str(t)), {})) for t in (subs if isinstance(subs, dict) else {})]


def build_tex(title: str, outline_obj: Dict[str, Any], entries: List[Dict[str, Any]],
              bib: Dict[str, Dict[str, Any]]) -> str:
    """Render the survey as LaTeX (sections, subsections, metadata bibliography)."""
    lines: List[str] = [f"\\section*{{{_tex_escape(title)}}}", ""]
    for sec_title, subs in _ordered_entries(outline_obj, entries):
        lines += [f"\\section{{{_tex_escape(sec_title)}}}", ""]
        for sub_title, entry in subs:
            content = entry.get("content", "") or ""
            if content.strip():
                lines += [f"\\subsection{{{_tex_escape(sub_title)}}}", _cites_to_latex(content), ""]
    lines.append(f"\\begin{{thebibliography}}{{{len(bib)}}}")
    for key in sorted(bib, key=int):
        lines.append(f"\\bibitem{{ref{key}}} {bibliography.format_tex(bib[key])}")
    lines += ["\\end{thebibliography}", ""]
    return "\n".join(lines)


def build_markdown(title: str, outline_obj: Dict[str, Any], entries: List[Dict[str, Any]],
                   bib: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for sec_title, subs in _ordered_entries(outline_obj, entries):
        lines += [f"## {sec_title}", ""]
        for sub_title, entry in subs:
            if (entry.get("content") or "").strip():
                lines += [f"### {sub_title}", "", entry["content"].strip(), ""]
    lines += ["## References", ""]
    for key in sorted(bib, key=int):
        lines.append(f"{key}. {bibliography.format_text(bib[key])}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    config.ensure_dirs()
    logger = common.get_logger(LOGGER_NAME)

    if not config.CANONICAL_OUTLINE.exists():
        logger.error("outline not found: %s (run step4 first)", config.CANONICAL_OUTLINE)
        return 1
    if not config.CANONICAL_RANKED_CSV.exists():
        logger.error("ranked CSV not found: %s (run step3 first)", config.CANONICAL_RANKED_CSV)
        return 1

    outline_obj = load_outline()
    topic = common.load_topic()
    refs = build_numbered_references(config.GENERATION_MAX_REFS)
    if not refs:
        logger.error("no numbered references available in %s", config.CANONICAL_RANKED_CSV)
        return 1
    refs_by_no = {int(r["refNo"]): r for r in refs}

    outline_text = stringify_outline_for_prompt(outline_obj)
    jobs = iter_subsections(outline_obj)
    if not jobs:
        logger.error("outline %s contains no subsections", config.CANONICAL_OUTLINE)
        return 1
    words = word_range(len(jobs))
    assignment = assign_references(jobs, refs, config.REFS_PER_SUBSECTION)
    common.write_json(config.INTERMEDIATES_DIR / "step5_reference_assignment.json", [
        {"subsection": j["subsection_title"], "section": j["section_title"], "refs": assignment[i],
         "is_challenges": j["is_challenges"]} for i, j in enumerate(jobs)])

    force = bool(os.environ.get("QUAL_SG_FORCE"))
    logger.info(
        "step5: topic=%r, %d section(s), %d subsection(s), %d reference(s), %d per subsection, "
        "words %d-%d each, claim check=%s, force=%s",
        topic.title, len(outline_obj.get("outline", []) or []), len(jobs), len(refs),
        config.REFS_PER_SUBSECTION, words[0], words[1], config.CLAIM_CHECK, force,
    )

    writer = common.make_llm_client("writer")
    critic = common.make_llm_client("eval")
    n_failures = 0
    pending: List[str] = []
    entries: List[Dict[str, Any]] = []

    for i, job in enumerate(jobs):
        path = subsection_path(job)
        label = f"[{i + 1}/{len(jobs)}] {job['subsection_title']!r}"
        if path.exists() and not force:
            entries.append(common.read_json(path))
            logger.info("%s resumed %s", label, path.name)
            continue

        given = [refs_by_no[n] for n in assignment[i]]
        prompt = build_subsection_prompt(
            topic.title, outline_text, str(job["subsection_title"]),
            format_references_for_prompt(given), words, job["is_challenges"],
        )
        try:
            result, failed = generate_subsection(writer, prompt, words, logger)
            entry = {
                "section_title": job["section_title"],
                "subsection_title": job["subsection_title"],
                "refs_given": assignment[i],
                "is_challenges": job["is_challenges"],
                "word_range": list(words),
                "draft_content": result.get("content", ""),
                "content": result.get("content", ""),
                "references": result.get("references", []),
                "length_flag": result.get("length_flag", ""),
                "models": {"writer": result.get("model", "")},
            }
            if failed:
                n_failures += 1
                entry["error"] = result.get("error", "generation failed")
                logger.error("%s failed: %s", label, entry["error"])
                entries.append(entry)
                continue
            if config.CLAIM_CHECK:
                check = claim_check.check_and_revise(
                    [{"heading": job["subsection_title"], "content": entry["content"]}],
                    refs_by_no, critic, writer, tag=f"s{job['big_idx']:02d}{job['sub_idx']:02d}",
                )
                entry["content"] = check["blocks"][0]["content"]
                entry["claim_check"] = {k: v for k, v in check.items() if k not in ("blocks", "draft_blocks")}
                entry["models"].update(check["models"])
            entry["word_count"] = len(entry["content"].split())
            entry["cited_refs"] = sorted({int(n) for n in CITE_RE.findall(entry["content"])})
            entry["cited_outside_given"] = [n for n in entry["cited_refs"] if n not in assignment[i]]
            common.write_json(path, entry)
            entries.append(entry)
            logger.info("%s wrote %s (%d words, %d cited refs, revised=%s)", label, path.name,
                        entry["word_count"], len(entry["cited_refs"]),
                        (entry.get("claim_check") or {}).get("revised"))
        except common.PendingLLM as exc:
            pending.append(job["subsection_title"])
            logger.info("%s waiting for an LLM answer (%s)", label, exc)

    if pending:
        raise common.PendingLLM(f"{len(pending)} subsection(s) waiting for LLM answers")

    title = outline_obj.get("title", topic.title)
    common.write_json(config.CANONICAL_CONTENT, {"title": title, "subsections": entries})
    bib = bibliography.load_or_build(refs)
    config.CANONICAL_TEX.write_text(build_tex(title, outline_obj, entries, bib), encoding="utf-8")
    (config.OUT_DIR / "survey.md").write_text(build_markdown(title, outline_obj, entries, bib), encoding="utf-8")
    logger.info("wrote %s, %s and survey.md (%d subsections, failures=%d)",
                config.CANONICAL_CONTENT, config.CANONICAL_TEX, len(entries), n_failures)
    logger.info("writer: %s | critic: %s", writer.summary(), critic.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
