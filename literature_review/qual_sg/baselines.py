"""Paper baselines for Task 1 / Task 2 (SurveyGen §3.1).

The main pipeline (``qual_sg.step1..6``) *is* QUAL-SG.  To reproduce the paper's
comparison (Table 3) this module re-runs the generation stages under the two
baselines:

* ``fully_llm``  - Task 1 "Fully-LLMGen": outline + content generated from the
  topic alone, no retrieved references.
* ``naive_rag``  - Task 2 "Naive-RAG": the *same* candidate pool, but ranked by
  Eq. (1) topic/abstract embedding similarity only (no LLM relevance judge, no
  academic-impact terms, no diversity term), then the identical outline /
  section prompts as QUAL-SG.

Outputs are written under ``baselines/<mode>/`` so the QUAL-SG artefacts are
never touched::

    baselines/fully_llm/survey_outline.json
    baselines/fully_llm/survey_content.json
    baselines/fully_llm/sections/*.json
    baselines/naive_rag/paper_after_reranking.csv
    baselines/naive_rag/survey_outline.json
    ...

Usage (LLM backend must be configured; ``QUAL_SG_BACKEND=harness`` or a Groq key)::

    python3 -m qual_sg.baselines --mode naive_rag
    python3 -m qual_sg.baselines --mode fully_llm
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import common, config

LOGGER = common.get_logger("baselines")

BASELINE_DIR = config.OUT_DIR / "baselines"


# --------------------------------------------------------------------------- #
# Naive-RAG ranking (Eq. 1 only)
# --------------------------------------------------------------------------- #
def naive_rank(topic: common.Topic, k: int, out_csv: Path) -> List[Dict[str, Any]]:
    """Rank the candidate pool by cos(topic, title+abstract) and keep top-``k``."""
    records = common.read_json(config.CANONICAL_REF_SET)
    work = [r for r in records if isinstance(r.get("abstract"), str) and r["abstract"].strip()]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL_NAME)
    docs = [f"{r.get('title','')}. {r.get('abstract','')}" for r in work]
    q = model.encode([topic.query], normalize_embeddings=True)
    d = model.encode(docs, normalize_embeddings=True, batch_size=64)
    sims = d @ q[0]
    order = np.argsort(-sims)[:k]
    rows = []
    for rank, i in enumerate(order, start=1):
        r = work[int(i)]
        rows.append(
            {
                "title": r.get("title", ""),
                "doi": common.norm_doi(r.get("doi")),
                "abstract": r.get("abstract", ""),
                "origin": r.get("origin", ""),
                "co_cited_count": r.get("co_cited_count", 0),
                "similarity_eq1": round(float(sims[int(i)]), 6),
                "rank_similarity": rank,
                "tag": "selected",
            }
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOGGER.info("naive_rag: ranked %d candidates, kept top %d -> %s", len(work), len(rows), out_csv)
    return rows


# --------------------------------------------------------------------------- #
# Fully-LLM (Task 1) prompts - topic only, no references
# --------------------------------------------------------------------------- #
OUTLINE_NO_REFS_PROMPT = """You are an academic expert in the field of "{topic}" with deep expertise in survey writing.
Your task is to generate a well-structured outline for a survey on this topic, using only your own knowledge (no external papers are provided).

Step 1: Identify 3 to 7 major thematic sections that define the overall scope and objectives of the survey. For each section, provide an academically styled title and a brief description summarizing its focus and relevance.
Step 2: For each major section, list several subsections representing more specific research areas or concepts.
Step 3: Return STRICT JSON in exactly this shape:
{{
  "title": "TITLE OF THE SURVEY",
  "outline": [
    {{"section_title": "...", "description": "...", "subsections": {{"subsection title": "content"}}}}
  ]
}}
Do not include any commentary, explanation or markdown fences - return only the JSON object."""

SECTION_NO_REFS_PROMPT = """You are an academic expert in the field of "{topic}" with deep expertise in survey writing.
The overall structure of the survey is:
{outline_text}

Write the following subsection:
- Subsection: {subsection_title}

Instructions:
1. Ensure the content directly addresses the specific topic of the subsection; write at least 300 words.
2. Use a formal academic tone with logically structured arguments and scholarly language.
3. You have NO access to external references: do not fabricate bracketed citations. Describing the state of the art in general terms is expected.
4. Return STRICT JSON: {{"content": "the content of the subsection", "references": []}}
Do not include any explanation, preface or markdown fences."""


def run_fully_llm(topic: common.Topic) -> None:
    outdir = BASELINE_DIR / "fully_llm"
    (outdir / "sections").mkdir(parents=True, exist_ok=True)
    client = common.make_llm_client("writer")

    obj, res = client.chat_json(
        [{"role": "user", "content": OUTLINE_NO_REFS_PROMPT.format(topic=topic.title)}],
        tag="baseline-outline",
    )
    common.write_json(outdir / "survey_outline.json", obj)
    sections = obj.get("outline", []) or []
    LOGGER.info("fully_llm outline: %d sections (tokens=%s)", len(sections), res.usage.get("total_tokens"))

    # Compact outline text for the section prompts.
    lines = []
    for sec in sections:
        lines.append(f"- {sec.get('section_title','')}: {sec.get('description','')}")
        for sub, desc in (sec.get("subsections") or {}).items():
            lines.append(f"    * {sub}: {desc}")
    outline_text = "\n".join(lines)

    bundle = []
    for bi, sec in enumerate(sections, start=1):
        for si, sub in enumerate((sec.get("subsections") or {}).keys(), start=1):
            prompt = SECTION_NO_REFS_PROMPT.format(
                topic=topic.title, outline_text=outline_text, subsection_title=sub
            )
            try:
                entry, _ = client.chat_json([{"role": "user", "content": prompt}], tag="baseline-section")
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("fully_llm subsection failed (%s): %s", sub, exc)
                entry = {"content": "", "references": []}
            row = {
                "section_title": sec.get("section_title", ""),
                "subsection_title": sub,
                "content": entry.get("content", ""),
                "references": entry.get("references", []),
            }
            bundle.append(row)
            common.write_json(outdir / "sections" / f"{bi:02d}_{si:02d}_{common.sanitize_filename(sub)}.json", row)
            LOGGER.info("fully_llm [%d] %s (%d words)", len(bundle), sub[:60], len(str(row['content']).split()))
    common.write_json(outdir / "survey_content.json", {"title": obj.get("title", topic.title), "subsections": bundle})


# --------------------------------------------------------------------------- #
# Naive-RAG generation: reuse the QUAL-SG generation prompts on a naive ranking
# --------------------------------------------------------------------------- #
def run_naive_rag(topic: common.Topic) -> None:
    outdir = BASELINE_DIR / "naive_rag"
    outdir.mkdir(parents=True, exist_ok=True)
    ranked_csv = outdir / "paper_after_reranking.csv"
    naive_rank(topic, topic.max_refs, ranked_csv)

    # Point the shared generation steps at this baseline's inputs/outputs.
    config.CANONICAL_RANKED_CSV = ranked_csv
    config.CANONICAL_OUTLINE = outdir / "survey_outline.json"
    config.CANONICAL_CONTENT = outdir / "survey_content.json"
    config.CANONICAL_TEX = outdir / "literature_review.tex"
    config.CANONICAL_SECTIONS_DIR = outdir / "sections"

    step4 = importlib.import_module("qual_sg.step4_outline")
    step5 = importlib.import_module("qual_sg.step5_generate")
    step4.main()
    step5.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="SurveyGen Task-1/Task-2 baselines")
    parser.add_argument("--mode", choices=["fully_llm", "naive_rag", "both"], default="naive_rag")
    args = parser.parse_args()

    config.ensure_dirs()
    topic = common.load_topic()
    started = time.time()
    if args.mode in ("naive_rag", "both"):
        run_naive_rag(topic)
    if args.mode in ("fully_llm", "both"):
        run_fully_llm(topic)
    LOGGER.info("baselines '%s' done in %.1fs", args.mode, time.time() - started)


if __name__ == "__main__":
    main()
