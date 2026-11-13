"""
Stage 0 — Gap ingestion and grouping.

Input : data/task2_research_gaps.txt  (text of our Task 2 submission, "Research Gaps and
        Limitations Identified in the Base Paper")
Output: outputs/stage0_gaps.json        (15 structured gap records + groups)

Steps
1. Parse the bullet list of limitations into records G1..G15 (deterministic, no LLM).
2. Lexical cross-check: TF-IDF vectors of the gap texts + agglomerative clustering. This is a
   cheap, model-free view of which gaps talk about the same things.
3. LLM grouping (1 call, role=judge): tag every gap (which MVFA component it concerns, what
   kind of gap it is, whether it can be addressed in code on BMAD with one Colab GPU) and
   group gaps that a single research problem could address together.

Why group at all? The research-problem prompts of ResearchAgent take "the literature" as
input. Feeding it 15 unrelated limitations at once produces the kind of everything-at-once
problem statement our Task 2 ended with. Feeding it one coherent group at a time produces
fine-grained problems, which is what the assignment asks for.
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import LLM
from utils import parse_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "stage0_gaps.json"


def parse_gaps(text: str) -> list[dict]:
    sec = text.split("2. Research Gaps and Limitations Identified in the Base Paper")[1]
    sec = sec.split("3. Newly Identified Research Problem")[0]
    # drop page numbers and blank lines, keep bullet structure
    lines = [l.strip() for l in sec.splitlines()]
    lines = [l for l in lines if l and not re.fullmatch(r"\d+", l)]
    body = " ".join(lines)
    items = [b.strip() for b in body.split("•")[1:]]
    gaps = []
    for i, it in enumerate(items, 1):
        it = re.sub(r"\s+", " ", it)
        title, _, desc = it.partition(". ")
        gaps.append(dict(id=f"G{i}", title=title.strip(), text=desc.strip()))
    return gaps


def lexical_clusters(gaps, n_clusters=6):
    docs = [g["title"] + ". " + g["text"] for g in gaps]
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
    S = cosine_similarity(X)
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(X.toarray())
    return S, labels


SYSTEM = (
    "You are an AI assistant whose primary goal is to organise the limitations of a scientific paper "
    "so that researchers can turn them into well-scoped follow-up research problems. You are precise, "
    "you do not invent limitations, and you only use the information given."
)

USER = """The base paper is:
Title: {title}
Abstract: {abstract}

A previous critical-appraisal step produced the following {n} limitations of the base paper:

{gap_list}

A lexical (TF-IDF) clustering of the same texts, given only as a weak hint, grouped them as:
{lexical}

Your tasks:
1. Tag every limitation with
   - "component": the part of the base method it concerns (one of: supervision/data, prompting/text branch,
     visual adaptation/architecture, score fusion/aggregation, evaluation protocol, metrics/reporting,
     efficiency/privacy, clinical scope/validation),
   - "kind": "method" (needs a new technique), "evaluation" (needs a new protocol/metric), or "reporting"
     (only missing information),
   - "codable": "high" / "medium" / "low" — can it be addressed by modifying the public base-paper code on the
     public BMAD benchmark with a single Colab GPU (A100/L4) within a few weeks?
2. Group the limitations into 4–6 groups such that ONE fine-grained research problem could address all members
   of a group together. A limitation may appear in at most one group; "reporting"-only limitations may be left
   ungrouped. For each group give a short name, its member ids, and one sentence on the shared root cause.
3. Say which 2–3 groups are the most promising starting points for fine-grained, codable research problems
   and why (one sentence each).

Return ONLY valid JSON of the form
{{"tags": {{"G1": {{"component": "...", "kind": "...", "codable": "..."}}, ...}},
  "groups": [{{"name": "...", "members": ["G..", ...], "root_cause": "..."}}, ...],
  "ungrouped": ["G..", ...],
  "promising": [{{"group": "...", "why": "..."}}, ...]}}"""


def main():
    text = (ROOT / "data" / "task2_research_gaps.txt").read_text()
    gaps = parse_gaps(text)
    assert len(gaps) == 15, len(gaps)
    S, labels = lexical_clusters(gaps)

    lex = {}
    for g, l in zip(gaps, labels):
        lex.setdefault(int(l), []).append(g["id"])
    lexical = "; ".join(f"cluster {c}: {', '.join(v)}" for c, v in sorted(lex.items()))

    base = json.loads((ROOT / "data" / "base_paper.json").read_text())
    gap_list = "\n".join(f"{g['id']}. {g['title']}: {g['text']}" for g in gaps)
    user = USER.format(title=base["title"], abstract=base["abstract"], n=len(gaps), gap_list=gap_list, lexical=lexical)

    resp = LLM("judge").call(SYSTEM, user, tag="stage0/group_gaps", temperature=0.2)
    res = parse_json(resp)

    for g in gaps:
        g.update(res["tags"][g["id"]])
    out = dict(gaps=gaps, groups=res["groups"], ungrouped=res.get("ungrouped", []), promising=res.get("promising", []),
               lexical_clusters=lex, tfidf_similarity=np.round(S, 3).tolist())
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for grp in res["groups"]:
        print(f"- {grp['name']}: {grp['members']}")


if __name__ == "__main__":
    main()
