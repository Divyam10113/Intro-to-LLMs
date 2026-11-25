"""Step 4 - survey outline generation (QUAL-SG / SurveyGen, paper Section 2.5).

The outline stage is the first *generative* stage of the pipeline.  Following
SurveyGen (Bao et al., EMNLP 2025, arXiv:2508.17647) we prompt the LLM with

    topic  +  the titles/abstracts of the highest-ranked references (Step 3)

and ask for a three-part plan: 3-7 major thematic sections (Step 1 of the
prompt), several subsections per section (Step 2), rendered as a single JSON
document (Step 3).  The prompt is the *verbatim* prompt shipped with the
authors' code (``Paper implementation/outline_generation.py``) so that the
reproduction stays faithful; only the reference cut-off
(:data:`config.OUTLINE_MAX_REFS`) and the model provider differ.

Robustness
----------
* the raw model answer is always persisted to :data:`config.STEP4_RAW` for
  auditing, even when it is not valid JSON;
* up to three attempts are made, each failure appending a terse
  ``Return only valid JSON`` repair turn to the conversation;
* the parsed object is validated (root ``title`` string, ``outline`` list of
  ``{section_title, description, subsections}`` records with dict subsections)
  before it is written to :data:`config.CANONICAL_OUTLINE`.

Running:  ``python3 -m qual_sg.step4_outline``  (from ``Paper implementation/``)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import common, config

LOGGER_NAME = "step4_outline"

#: Number of LLM attempts before giving up on a valid outline document.
MAX_OUTLINE_ATTEMPTS = 3

#: Terse repair instruction appended after a failed parse/validation attempt.
REPAIR_MESSAGE = (
    "Your previous answer was not valid JSON. Return only valid JSON, with no "
    "markdown fences, commentary or preamble."
)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def load_selected_refs(max_refs: int) -> List[Dict[str, str]]:
    """Return the first ``max_refs`` selected papers that have an abstract.

    Rows are read from :data:`config.CANONICAL_RANKED_CSV` (Step 3 output) and
    filtered on ``tag == 'selected'``.  Papers whose abstract is missing/``nan``
    are skipped, exactly as the authors' script does, because the outline prompt
    conditions the section plan on the abstracts.
    """
    df = pd.read_csv(config.CANONICAL_RANKED_CSV)
    selected = df[df["tag"] == "selected"]

    refs: List[Dict[str, str]] = []
    for _, row in selected.iterrows():
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        if not abstract or abstract.lower() == "nan":
            continue
        refs.append({"title": title, "abstract": abstract})
        if len(refs) >= max_refs:
            break
    return refs


# --------------------------------------------------------------------------- #
# Prompt (verbatim from outline_generation.py)
# --------------------------------------------------------------------------- #
def build_outline_prompt(topic: str, refs: List[Dict[str, str]]) -> str:
    """Build the Step-4 outline prompt.

    The template below is copied verbatim from the authors'
    ``outline_generation.py``: a role framing, the reference list, the three
    planning instructions and the JSON example.  ``{topic}`` is the *title* of
    the topic row (the reference script uses only the title).
    """
    references_list = ""
    for r in refs:
        references_list += f"\n- Title: {r['title']}\n  Abstract: {r['abstract']}\n"

    return f"""
You are an academic expert in the field of "{topic}" with deep expertise in survey writing. Your task is to generate a well-structured outline for a survey on this topic. 

The following highly relevant papers, including their titles and abstracts, are provided for reference:
{references_list}

Please follow the instructions below:
Step 1: Based on the given topic, identify {config.SURVEY_SECTIONS} major thematic sections that define the overall scope and objectives of the survey. For each section, provide a academically styled title, along with a brief description summarizing its focus and relevance.

Step 2: For the major thematic sections, list subsections representing more specific research areas, concepts, or points to be covered, with {config.SURVEY_SUBSECTIONS} subsections in total across all sections. The survey is about {config.SURVEY_TARGET_WORDS} words long (about 10 pages), so size the outline so that every subsection can be covered properly within that length. Subsections should be conceptually related to their parent section and serve to further structure the survey. The last section covers open challenges: plan its subsections around specific gaps visible in the provided papers (a result that differs sharply between methods or datasets, a setting no paper covers, or a limitation the authors state), not generic statements that would fit any survey in this area.

Step 3: Your output should be in JSON format and must include the survey title, a structured outline with section titles, descriptions, and subsections. Ensure the structure is logically coherent, well-aligned with the topic, and suitable for developing a full-length academic survey. 

The output example should follow the format below:
{{
  "title": "TITLE OF THE SURVEY",
  "outline": [
    {{
      "section_title": "SECTION TITLE",
      "description": "A brief description summarizing the focus and scope of this section.",
      "subsections": {{
            "subsection title1": "content",
            "...": "...",
            "subsection title n": "content"
      }}
    }},
    {{
      "section_title": "SECTION TITLE 2",
      "description": "Another major theme description.",
      "subsections": {{
            "subsection title 1": "content",
            "...": "...",
            "subsection title n": "content"
      }}
    }}
  ]
}}

Now, based on the given topic {topic}, please generate the outline by following the steps above. Do not include any additional commentary, explanation, or formatting instructions, only return the structured JSON output as specified.
"""


# --------------------------------------------------------------------------- #
# Output validation
# --------------------------------------------------------------------------- #
def validate_outline(obj: Any) -> Dict[str, Any]:
    """Validate the parsed outline and return it, raising ``ValueError`` if bad.

    Required shape (the JSON example of the prompt)::

        {"title": str,
         "outline": [{"section_title": str, "description": str,
                      "subsections": {str: str}}, ...]}
    """
    if not isinstance(obj, dict):
        raise ValueError(f"outline root must be a JSON object, got {type(obj).__name__}")
    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("outline root is missing a non-empty 'title' string")
    outline = obj.get("outline")
    if not isinstance(outline, list) or not outline:
        raise ValueError("outline root is missing a non-empty 'outline' list")
    for i, sec in enumerate(outline, start=1):
        if not isinstance(sec, dict):
            raise ValueError(f"outline[{i}] must be an object")
        if not isinstance(sec.get("section_title"), str) or not sec["section_title"].strip():
            raise ValueError(f"outline[{i}].section_title must be a non-empty string")
        if not isinstance(sec.get("description"), str):
            raise ValueError(f"outline[{i}].description must be a string")
        subs = sec.get("subsections")
        if not isinstance(subs, dict) or not subs:
            raise ValueError(f"outline[{i}].subsections must be a non-empty object")
        for sub_title in subs:
            if not isinstance(sub_title, str) or not sub_title.strip():
                raise ValueError(f"outline[{i}] has a subsection with an empty title")
    return obj


def count_sections(outline: Dict[str, Any]) -> Tuple[int, int]:
    """Return ``(n_sections, n_subsections)`` for logging."""
    sections = outline.get("outline", []) or []
    n_subs = sum(len(sec.get("subsections") or {}) for sec in sections if isinstance(sec, dict))
    return len(sections), n_subs


# --------------------------------------------------------------------------- #
# LLM call with repair retries
# --------------------------------------------------------------------------- #
def generate_outline(client: common.LLMClient, prompt: str, logger) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call the model until a valid outline is parsed; persist the raw answer.

    Returns ``(outline_object, usage_totals)``.  The raw model text of the last
    attempt is written to :data:`config.STEP4_RAW`.
    """
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error: Exception = ValueError("no attempt executed")

    for attempt in range(1, MAX_OUTLINE_ATTEMPTS + 1):
        result = client.chat(messages, json_mode=True, tag="outline")
        raw_text = result.text or ""
        config.STEP4_RAW.parent.mkdir(parents=True, exist_ok=True)
        config.STEP4_RAW.write_text(raw_text, encoding="utf-8")

        for key in usage_totals:
            usage_totals[key] += int(result.usage.get(key, 0) or 0)

        try:
            parsed = json.loads(common.clean_json_string(raw_text))
            outline = validate_outline(parsed)
            logger.info(
                "outline accepted on attempt %d/%d (model=%s, cached=%s)",
                attempt, MAX_OUTLINE_ATTEMPTS, result.model, result.cached,
            )
            return outline, usage_totals
        except Exception as exc:  # noqa: BLE001 - retry on any malformed answer
            last_error = exc
            logger.warning(
                "outline attempt %d/%d failed: %s", attempt, MAX_OUTLINE_ATTEMPTS, exc
            )
            if attempt < MAX_OUTLINE_ATTEMPTS:
                messages = messages + [{"role": "user", "content": REPAIR_MESSAGE}]

    raise RuntimeError(f"could not obtain a valid outline after {MAX_OUTLINE_ATTEMPTS} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    config.ensure_dirs()
    logger = common.get_logger(LOGGER_NAME)

    if not config.CANONICAL_RANKED_CSV.exists():
        logger.error("ranked CSV not found: %s (run step3 first)", config.CANONICAL_RANKED_CSV)
        return 1

    topic = common.load_topic()
    refs = load_selected_refs(config.OUTLINE_MAX_REFS)
    if not refs:
        logger.error("no selected references with an abstract in %s", config.CANONICAL_RANKED_CSV)
        return 1

    logger.info(
        "step4: topic=%r, %d reference(s) (max %d)",
        topic.title, len(refs), config.OUTLINE_MAX_REFS,
    )

    prompt = build_outline_prompt(topic.title, refs)
    client = common.make_llm_client("outline")

    try:
        outline, usage = generate_outline(client, prompt, logger)
    except common.PendingLLM:
        raise  # file backend: the outline prompt is waiting for its answer
    except Exception as exc:  # noqa: BLE001 - report instead of crashing the pipeline
        logger.error("outline generation failed: %s", exc)
        return 1

    n_sections, n_subsections = count_sections(outline)
    common.write_json(config.CANONICAL_OUTLINE, outline)
    logger.info(
        "wrote %s | %d sections, %d subsections | title=%r",
        config.CANONICAL_OUTLINE, n_sections, n_subsections, outline.get("title", ""),
    )
    logger.info(
        "token usage: prompt=%d completion=%d total=%d",
        usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
    )
    logger.info("client summary: %s", client.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
