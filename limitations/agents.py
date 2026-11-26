"""Agent prompts and calls.

Worker agents (arXiv:2601.11578): Extractor, Analyzer, Reviewer, Citation.
Aspect taxonomy (LimitGen, arXiv:2507.02694) + one domain aspect for medical AD.
Meta agents: Judge (rubric scores, one regeneration round max), Verifier (ours),
Master (merge, provenance, ranking), Gap-status (ours) and Problem formulation.
"""
import json
import re
from difflib import SequenceMatcher

from common import chat, chat_json

ASPECTS = """\
- Methodology: inappropriate or oversimplified method choices, unstated assumptions that may not hold, data quality/preprocessing issues.
- Experimental Design: insufficient or weak baselines, limited or inappropriate datasets, evaluation protocol flaws (e.g. leakage between "seen" and "unseen" domains), missing ablations.
- Results & Analysis: insufficient metrics, no failure-case or error analysis, no variance/significance reporting, over-claimed conclusions.
- Literature: missing or mischaracterised related work, contributions not positioned against relevant methods.
- Clinical & Deployment (domain-specific): annotation cost, modality/anatomy coverage, clinical granularity of outputs, interpretability, compute/latency, robustness to acquisition shift."""

ITEM_SCHEMA = """\
Return a JSON list. Each item:
{
  "aspect": one of ["Methodology","Experimental Design","Results & Analysis","Literature","Clinical & Deployment"],
  "limitation": "one precise sentence naming the limitation, specific to THIS paper",
  "explanation": "2-4 sentences: why it is a limitation and what it undermines (validity, generality, reliability, clinical use)",
  "evidence": [{"source": "BASE" or a passage id like "C0123", "quote": "verbatim text copied from that source, <= 40 words"}],
  "suggested_fix": "one concrete experiment or method change that would address it"
}
Every item needs at least one evidence quote copied verbatim (no paraphrase) from the given sources.
Avoid generic statements ("more datasets would help") unless you tie them to a specific design choice of the paper."""

CHAIN = ("Work in a 'chain of limitations': (1) locate the relevant part of the paper, (2) state the potential "
         "gap, (3) argue why it is a limitation using the text, (4) re-check the claim against the text before "
         "keeping it.")


def _paper_block(paper):
    return f"=== INPUT PAPER (full text) ===\n{paper}\n=== END INPUT PAPER ==="


# -----------------------------------------------------------------------------
# Worker agents
# -----------------------------------------------------------------------------
def extractor(paper, feedback=None):
    sys = ("You are the Extractor Agent, an expert in scientific literature analysis. Extract every limitation, "
           "caveat, failure case or future-work statement that the AUTHORS THEMSELVES state anywhere in the paper "
           "(including discussion, experiments, conclusion, supplementary). Do not infer anything the authors did "
           "not say. If the authors state very few, return only those. " + CHAIN)
    return _work("extractor", sys, _paper_block(paper), feedback)


def analyzer(paper, feedback=None):
    sys = ("You are the Analyzer Agent, a methodological auditor. Identify IMPLICIT limitations the authors did not "
           "state: weaknesses evident from the method's design, assumptions, training/evaluation protocol, data "
           "and statistical analysis. Cover each aspect below where the paper gives you grounds:\n" + ASPECTS +
           "\n" + CHAIN)
    return _work("analyzer", sys, _paper_block(paper), feedback)


def reviewer(paper, feedback=None):
    sys = ("You are the Reviewer Agent: an expert reviewer for a top computer-vision / medical-imaging venue. "
           "Write the weaknesses a rigorous peer reviewer would raise: reproducibility (missing details, "
           "hyper-parameters, compute), fairness and strength of baselines, evaluation protocol, statistical "
           "reliability, clarity of claims vs. evidence, and ethical/clinical considerations. "
           "Use these aspects:\n" + ASPECTS + "\n" + CHAIN)
    return _work("reviewer", sys, _paper_block(paper), feedback)


def citation(paper, passages, feedback=None):
    sys = ("You are the Citation Agent. You see the input paper and passages from papers it cites and papers that "
           "cite it (later work). Identify limitations of the INPUT PAPER that become visible only in the light of "
           "this literature: weaknesses that later papers point out, capabilities/settings later work handles "
           "that the input paper cannot, stronger baselines or evaluation practices it does not use, and "
           "assumptions that related work contradicts. Each item MUST cite at least one passage id (C....) as "
           "evidence and may also quote the input paper (source BASE). Do not credit the input paper with "
           "anything the passages say about other methods. Aspects:\n" + ASPECTS + "\n" + CHAIN)
    lit = "\n\n".join(f"[{p['cid']}] {p['title']} ({p['year']}) - {p['section']}\n{p['text']}" for p in passages)
    user = f"{_paper_block(paper)}\n\n=== LITERATURE PASSAGES ===\n{lit}\n=== END PASSAGES ==="
    return _work("citation", sys, user, feedback)


def _work(name, system, user, feedback):
    system = system + "\n\n" + ITEM_SCHEMA
    if feedback:
        user += ("\n\n=== FEEDBACK ON YOUR PREVIOUS ATTEMPT (from the Judge) ===\n" + feedback +
                 "\nRegenerate the full list, fixing these issues.")
    items = chat_json(system, user, temperature=0.3, tag=name + ("-regen" if feedback else ""))
    items = items if isinstance(items, list) else items.get("limitations", [])
    for i, it in enumerate(items):
        it["id"] = f"{name[:3].upper()}{i + 1}"
        it["agent"] = name
    return items


# -----------------------------------------------------------------------------
# Judge (reference-free rubric, arXiv:2601.11578 sec. IV-E; F/S/I from LimitGen)
# -----------------------------------------------------------------------------
JUDGE_DIMS = ["depth", "originality", "actionability", "topic_coverage", "faithfulness", "soundness", "importance"]


def judge(paper, agent_name, items):
    sys = ("You are the Judge Agent, a strict quality auditor of generated research limitations. Score the "
           f"{agent_name} agent's list on each dimension from 1-10: depth, originality (not generic), actionability, "
           "topic_coverage (breadth over the relevant aspects), faithfulness (accurate about the paper, no "
           "misinformation), soundness (logically valid), importance (affects the main findings). Then give "
           "specific, actionable feedback: which items are wrong or generic, and what important limitations are "
           'missing. Return JSON: {"scores": {dimension: int}, "feedback": "..."}')
    user = f"{_paper_block(paper)}\n\n=== {agent_name.upper()} OUTPUT ===\n{json.dumps(items, indent=1)}"
    out = chat_json(sys, user, temperature=0, tag=f"judge-{agent_name}")
    scores = {d: float(out.get("scores", {}).get(d, 0)) for d in JUDGE_DIMS}
    return {"scores": scores, "mean": sum(scores.values()) / len(scores), "feedback": out.get("feedback", "")}


# -----------------------------------------------------------------------------
# Verifier (ours): deterministic quote check + LLM support check
# -----------------------------------------------------------------------------
def _toks(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def quote_found(quote, source_text):
    """True if the quote appears in the source (exact after normalisation, or a contiguous run
    covering >= 70% of its tokens - tolerates PDF line-break / LaTeX artefacts)."""
    q, s = _toks(quote), _toks(source_text)
    if not q:
        return False
    if " ".join(q) in " ".join(s):
        return True
    m = SequenceMatcher(None, s, q, autojunk=False).find_longest_match(0, len(s), 0, len(q))
    return m.size >= 0.7 * len(q)


def check_quotes(items, base_text, chunk_by_id):
    for it in items:
        ok = []
        for ev in it.get("evidence", []):
            src = str(ev.get("source", ""))
            text = base_text if src.upper().startswith("BASE") else chunk_by_id.get(src, {}).get("text", "")
            ev["quote_verified"] = bool(text) and quote_found(ev.get("quote", ""), text)
            ok.append(ev["quote_verified"])
        it["quotes_verified"] = f"{sum(ok)}/{len(ok)}"
        it["has_verified_evidence"] = any(ok)
    return items


def verify_support(paper, items, chunk_by_id):
    """LLM entailment check: does the (verified) evidence actually support the claim?"""
    if not items:
        return items
    packed = []
    for it in items:
        ev = [{"source": e["source"], "quote": e.get("quote", ""),
               "context": (chunk_by_id.get(e["source"], {}).get("text", "")[:1200]
                           if not str(e["source"]).upper().startswith("BASE") else None)}
              for e in it.get("evidence", []) if e.get("quote_verified")]
        packed.append({"id": it["id"], "agent": it["agent"], "limitation": it["limitation"],
                       "explanation": it.get("explanation", ""), "evidence": ev})
    sys = ("You are the Verifier. For each claimed limitation of the INPUT PAPER, decide whether it is (a) factually "
           "accurate about the input paper and (b) supported by its evidence. Mark 'supported', 'partial' "
           "(plausible but evidence is weak/indirect or claim overstated) or 'unsupported' (wrong about the paper, "
           "or evidence does not back it, or it attributes another paper's property to the input paper). "
           "For items from agent 'extractor' also decide 'author_stated': true only if the AUTHORS THEMSELVES "
           "explicitly acknowledge this limitation/caveat/future work in the text; false if it is the agent's own "
           "inference from the paper (even if correct). "
           'Return JSON: [{"id": "...", "verdict": "...", "author_stated": true/false/null, "reason": "one sentence"}]')
    user = f"{_paper_block(paper)}\n\n=== CLAIMS ===\n{json.dumps(packed, indent=1)}"
    verdicts = {v["id"]: v for v in chat_json(sys, user, temperature=0, tag="verifier") if isinstance(v, dict)}
    for it in items:
        v = verdicts.get(it["id"], {})
        it["verdict"], it["verdict_reason"] = v.get("verdict", "unsupported"), v.get("reason", "not returned")
        if it["agent"] == "extractor":
            it["author_stated"] = v.get("author_stated") is True
    return items


# -----------------------------------------------------------------------------
# Master: merge, provenance, rank
# -----------------------------------------------------------------------------
PROVENANCE = {"extractor": "Author-stated", "analyzer": "Inferred", "reviewer": "Peer-review", "citation": "Literature"}


def _provenance(it):
    # Extractor items the Verifier found to be inferences, not author statements, are relabelled
    if it["agent"] == "extractor" and it.get("author_stated") is False:
        return "Inferred"
    return PROVENANCE[it["agent"]]


def master(paper, items):
    slim = [{"id": it["id"], "provenance": _provenance(it), "aspect": it["aspect"],
             "limitation": it["limitation"], "explanation": it.get("explanation", ""),
             "suggested_fix": it.get("suggested_fix", ""), "verdict": it.get("verdict")} for it in items]
    sys = ("You are the Master Agent. Consolidate the validated limitations from all worker agents into one "
           "de-duplicated list. Cluster items that describe the same underlying issue and merge each cluster into "
           "a single, most complete statement. Keep every source id in 'merged_from' and the union of provenance "
           "tags. Do not invent new limitations. Rate importance 1-5 (5 = undermines the paper's main claims; "
           "1 = trivial). Order by importance. Return JSON list: "
           '[{"limitation": "...", "explanation": "...", "aspect": "...", "provenance": ["Author-stated", ...], '
           '"merged_from": ["ANA1", ...], "importance": int, "suggested_fix": "..."}]')
    user = f"{_paper_block(paper)}\n\n=== VALIDATED LIMITATIONS ===\n{json.dumps(slim, indent=1)}"
    merged = chat_json(sys, user, temperature=0, tag="master")
    by_id = {it["id"]: it for it in items}
    for k, m in enumerate(merged, 1):
        m["lid"] = f"L{k}"
        m["evidence"] = [e for i in m.get("merged_from", []) for e in by_id.get(i, {}).get("evidence", [])
                         if e.get("quote_verified")]
    return merged


# -----------------------------------------------------------------------------
# Gap status (ours): is the limitation already addressed by later work?
# -----------------------------------------------------------------------------
def gap_status(lim, passages, review_excerpt):
    sys = ("You decide whether a limitation of a 2024 paper (MVFA, CLIP adaptation for zero/few-shot medical anomaly "
           "detection) has since been addressed by later work. Use only the passages and the literature-review "
           "excerpt given. Status: 'open' (no passage addresses it), 'partially_addressed' (some work tackles part "
           "of it or only in a narrower setting), 'addressed' (later work clearly resolves it). Name the "
           "addressing passages by id and state what remains unresolved. "
           'Return JSON: {"status": "...", "addressed_by": ["C...."], "remaining_gap": "..."}')
    lit = "\n\n".join(f"[{p['cid']}] {p['title']} ({p['year']}) - {p['section']}\n{p['text'][:1500]}"
                      for p in passages)
    user = (f"LIMITATION: {lim['limitation']}\nEXPLANATION: {lim.get('explanation', '')}\n\n"
            f"=== LATER-WORK PASSAGES ===\n{lit}\n\n=== LITERATURE REVIEW EXCERPT (Task 1) ===\n{review_excerpt}")
    return chat_json(sys, user, temperature=0, tag=f"status-{lim['lid']}")


# -----------------------------------------------------------------------------
# Research problem formulation
# -----------------------------------------------------------------------------
def formulate_problem(paper_abstract, limitations, review_gaps):
    sys = ("You are a senior researcher writing the 'research gaps and problem statement' section of a project "
           "proposal that continues the INPUT PAPER. Using ONLY the limitations given (with their status in later "
           "literature), write:\n"
           "1. '## Research gaps and limitations' - a bulleted list of the 12-15 MOST important limitations "
           "(importance 5 and 4 first; merge near-duplicates). Each bullet: bold short title, one or two sentences, "
           "then tags in brackets [Lk | aspect | provenance | status].\n"
           "2. '## Research problem' - ONE paragraph (180-260 words) defining a METHODOLOGICAL research problem that "
           "continues the input paper: a new capability or method to build, not an audit or re-evaluation of the "
           "input paper. Choose 3-5 limitations that are 'open' or 'partially_addressed', important, and causally "
           "related, so that one method could address them together; do not try to cover everything. State the "
           "setting, what is assumed/available at train and test time, what the method must achieve, why existing "
           "work (name it) does not solve it, and how success would be measured (datasets/protocol/metrics); "
           "evaluation-protocol limitations belong only in that last part. End with a line "
           "'Addresses: L.., L..' listing the chosen ids.\n"
           "Write in precise academic English. Return Markdown only.")
    user = (f"INPUT PAPER ABSTRACT:\n{paper_abstract}\n\nLIMITATIONS (JSON):\n{json.dumps(limitations, indent=1)}"
            f"\n\nDOMAIN-LEVEL OPEN PROBLEMS FROM THE TASK-1 LITERATURE REVIEW:\n{review_gaps}")
    return chat(sys, user, temperature=0.4, tag="problem")
