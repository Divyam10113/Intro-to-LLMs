"""Claim check and one revision pass for generated review text (approved change 11).

A critic (the ``eval`` stage model, chosen to differ from the writer) reads every
sentence that cites ``ref [n]`` together with the abstracts of the cited papers
and labels it supported / overstated / not supported.  If anything is flagged,
the writer gets one pass to fix only the flagged sentences.  Verdicts, the draft
and the revision are all returned so they can be stored with the output.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from . import common

CITE_RE = re.compile(r"\bref\s*\[\s*(\d+)\s*\]")
VERDICTS = ("supported", "overstated", "not_supported")

CRITIC_PROMPT = """You are checking the citations in a passage from a scientific literature review.

For every sentence that cites one or more references (written as `ref [n]`), decide whether the abstracts of the cited papers support it:
- "supported": the claim follows from the cited abstract(s);
- "overstated": the abstracts support only a weaker or narrower version of the claim;
- "not_supported": the abstracts do not support the claim, or the claim is credited to the wrong paper.
Judge only against the abstracts given below, not outside knowledge. Skip sentences without citations.

PASSAGE:
{passage}

CITED REFERENCES:
{refs}

Return only JSON, no commentary:
{{"sentences": [{{"sentence": "<the sentence, verbatim>", "refs": [<n>, ...], "verdict": "supported|overstated|not_supported", "reason": "<one short sentence; empty when supported>"}}]}}"""

REVISE_PROMPT = """Revise this passage from a scientific literature review.

A checker compared each cited sentence with the abstracts of the papers it cites and flagged the sentences listed below. Fix only those sentences: narrow an overstated claim to what the abstract supports, and re-attribute or remove a claim the cited paper does not support. Leave every other sentence unchanged. Keep the `ref [n]` citation format, cite only the references listed below, and keep the headings and roughly the same length.

PASSAGE (JSON):
{blocks}

FLAGGED SENTENCES:
{flagged}

REFERENCES AVAILABLE:
{refs}

Return only JSON, no commentary:
{{"blocks": [{{"heading": "<same heading>", "content": "<revised text>"}}]}}"""


def cited_numbers(blocks: Sequence[Dict[str, str]]) -> List[int]:
    nums = {int(n) for b in blocks for n in CITE_RE.findall(str(b.get("content", "")))}
    return sorted(nums)


def _refs_block(refs_by_no: Dict[int, Dict[str, Any]], nums: Sequence[int]) -> str:
    lines = []
    for n in nums:
        ref = refs_by_no.get(n)
        if ref:
            lines.append(f"- ref [{n}]: {ref['title']}\n  Abstract: {ref.get('abstract', '')}")
    return "\n".join(lines)


def _passage(blocks: Sequence[Dict[str, str]]) -> str:
    return "\n\n".join(f"## {b.get('heading', '')}\n{b.get('content', '')}".strip() for b in blocks)


def check_and_revise(
    blocks: List[Dict[str, str]],
    refs_by_no: Dict[int, Dict[str, Any]],
    critic: Any,
    writer: Any,
    tag: str,
) -> Dict[str, Any]:
    """Return ``{blocks, draft_blocks, verdicts, counts, revised, models}``.

    Raises :class:`common.PendingLLM` (file backend) until both calls are answered.
    """
    nums = cited_numbers(blocks)
    unknown = [n for n in nums if n not in refs_by_no]
    refs_text = _refs_block(refs_by_no, nums)
    obj, res = critic.chat_json(
        [{"role": "user", "content": CRITIC_PROMPT.format(passage=_passage(blocks), refs=refs_text)}],
        tag=f"claim-check-{tag}",
    )
    verdicts = [s for s in (obj.get("sentences") or []) if isinstance(s, dict)]
    for s in verdicts:
        s["verdict"] = str(s.get("verdict", "")).strip().lower().replace(" ", "_")
    counts = {v: sum(1 for s in verdicts if s["verdict"] == v) for v in VERDICTS}
    flagged = [s for s in verdicts if s["verdict"] in ("overstated", "not_supported")]
    result: Dict[str, Any] = {
        "draft_blocks": blocks, "blocks": blocks, "verdicts": verdicts, "counts": counts,
        "cited_refs": nums, "citations_to_unknown_refs": unknown, "revised": False,
        "models": {"critic": res.model},
    }
    if not flagged:
        return result
    flagged_text = "\n".join(
        f"- \"{s.get('sentence', '')}\" -> {s['verdict']}: {s.get('reason', '')}" for s in flagged
    )
    obj2, res2 = writer.chat_json(
        [{"role": "user", "content": REVISE_PROMPT.format(
            blocks=json.dumps([{"heading": b.get("heading", ""), "content": b.get("content", "")} for b in blocks],
                              ensure_ascii=False, indent=1),
            flagged=flagged_text, refs=refs_text)}],
        tag=f"claim-revise-{tag}",
    )
    revised = [b for b in (obj2.get("blocks") or []) if isinstance(b, dict) and str(b.get("content", "")).strip()]
    if len(revised) == len(blocks):
        result["blocks"] = [{"heading": blocks[i].get("heading", ""), "content": str(revised[i]["content"]).strip()}
                            for i in range(len(blocks))]
        result["revised"] = True
        result["models"]["reviser"] = res2.model
    else:
        result["revision_rejected"] = f"expected {len(blocks)} block(s), got {len(revised)}"
    return result
