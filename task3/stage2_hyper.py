"""
Stage 2 — Literature-grounded seed ideas with provenance (HypER, Vasu et al. 2025).

Part A: reasoning-chain construction (HypER Section 3, released code RAG_temporal_extension_llama.py)
  * Source paper p_k = MVFA (2024). Citing papers come from data/s2_citations.json (Semantic Scholar, as in HypER Step 2).
  * Step 2: keep citing papers with an abstract published in the two years after the current paper
    (code: years current+1 .. current+2), grouped by year, in chunks of 10.
  * Step 3: LLM relevancy scoring of every chunk with HypER's prompt: 0 irrelevant / 1 inspired / 2 dependent,
    plus the model's top-3 of the chunk.
  * Step 4: candidates = top-3 papers with relevance 1 or 2; score = 0.7 * relevance + 0.3 * citations / max citations
    (code: compute_score, w_r=0.7, w_c=0.3).
  * Step 5: the best candidate becomes the new source and the procedure repeats on its citing papers.
  ADAPTATIONS
  * HypER builds one greedy chain per source paper and obtains many chains from many source papers. We have one
    source paper, so we branch at the first hop: the B best first-hop candidates (by HypER's score) each start a chain,
    which is then extended greedily as in HypER.
  * HypER's code drops papers with zero citations unless they are from the two most recent years; with MVFA's citing
    papers being from 2025-2026, every paper falls under that exemption, so no citation filter applies.
  * Second-hop citing papers were fetched for the 40 most-cited first-hop papers only (fetch_citations.py --hop2-top 40),
    and, to bound cost, only the 20 second-hop papers most similar (TF-IDF cosine of abstracts) to their parent are
    scored. A chain stops when no eligible citing papers exist (e.g., a 2026 paper has no citers in 2027-2028 yet).
  * Citation contexts are added to the relevancy prompt (see prompts/hyper.py).

Part B: hypothesis/idea generation (HypER Listing 4) — for each Stage 1 research problem and each chain, one seed
idea whose Provenance lists the chain papers it builds on. The 2 x B seeds form HypoEvolve's initial population.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import LLM
from prompts import hyper as H
from prompts.researchagent import PROTOCOL_C
from utils import parse_json

ROOT = Path(__file__).resolve().parent
OUT_CHAINS = ROOT / "outputs" / "stage2_chains.json"
OUT_SEEDS = ROOT / "outputs" / "stage2_seeds.json"
BRANCHES = 6          # = HypoEvolve population size mu
HOP2_PREFILTER = 20
W_R, W_C = 0.7, 0.3


def paper(x):
    return dict(paperId=x["paperId"], title=x["title"], abstract=x.get("abstract"), year=x.get("year"),
                citation_count=x.get("citationCount") or 0, citation_contexts=x.get("citation_contexts") or [],
                url=x.get("url"))


def eligible(citers, year):
    out = defaultdict(list)
    for c in citers:
        if c.get("abstract") and c.get("year") and year < c["year"] <= year + 2:
            out[c["year"]].append(paper(c))
    return out


def score_citers(llm, source, by_year, tag):
    """HypER Steps 2-3: chunks of 10 per year -> relevancy prompt. Returns all scored papers and top-3 candidates."""
    reqs, meta = [], []
    for year in sorted(by_year):
        papers = by_year[year]
        for ci in range(0, len(papers), 10):
            chunk = papers[ci:ci + 10]
            reqs.append(dict(system=H.RELEVANCY_SYSTEM, user=H.relevancy_prompt(source, year, chunk),
                             tag=f"{tag}/{year}/chunk{ci // 10}", temperature=0.2))
            meta.append(chunk)
    scored, top = [], []
    for chunk, resp in zip(meta, llm.batch(reqs)):
        js = parse_json(resp)
        for key, v in js.get("paper_list", {}).items():
            m = re.match(r"\s*(\d+)\.", key)
            if m and 1 <= int(m.group(1)) <= len(chunk):
                scored.append(dict(chunk[int(m.group(1)) - 1], relevance=int(v["relevance"]), explanation=v["explanation"]))
        for key, v in (js.get("top3_relevant_papers") or {}).items():
            m = re.match(r"\s*(\d+)\.", key)
            if not m:
                continue
            # the top-3 list numbers papers 1..3, so match it back to the chunk by title
            t = re.sub(r"^\s*\d+\.\s*", "", key).lower()[:60]
            hit = [p for p in chunk if p["title"].lower()[:60] == t] or \
                  [p for p in chunk if t[:40] and t[:40] in p["title"].lower()]
            if hit and int(v["relevance"]) in (1, 2):
                top.append(dict(hit[0], relevance=int(v["relevance"]), explanation=v["explanation"]))
    return scored, top


def rank(cands):
    if not cands:
        return []
    mx = max(c["citation_count"] for c in cands) or 1
    for c in cands:
        c["hyper_score"] = round(W_R * c["relevance"] + W_C * c["citation_count"] / mx, 4)
    seen, out = set(), []
    for c in sorted(cands, key=lambda c: -c["hyper_score"]):
        if c["paperId"] not in seen:
            seen.add(c["paperId"])
            out.append(c)
    return out


def prefilter(parent, papers_by_year, k):
    flat = [p for ps in papers_by_year.values() for p in ps]
    if len(flat) <= k:
        return papers_by_year
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([parent["abstract"]] + [p["abstract"] for p in flat])
    sims = cosine_similarity(X[0], X[1:])[0]
    keep = {flat[i]["paperId"] for i in sims.argsort()[::-1][:k]}
    return {y: [p for p in ps if p["paperId"] in keep] for y, ps in papers_by_year.items() if any(p["paperId"] in keep for p in ps)}


def build_chains(judge):
    s2 = json.loads((ROOT / "data" / "s2_citations.json").read_text())
    b = s2["base"]
    source = dict(paperId=b["paperId"], title=b["title"], abstract=b["abstract"], year=b["year"],
                  citation_count=b.get("citationCount") or 0)
    hop1 = eligible(s2["hop1"], source["year"])
    scored1, top1 = score_citers(judge, source, hop1, "stage2/relevance/hop1")
    ranked1 = rank(top1)
    branches = ranked1[:BRANCHES]

    # second hop for every branch (independent -> score all branches before choosing)
    hop2_inputs = {}
    for p1 in branches:
        citers = s2["hop2"].get(p1["paperId"], [])
        hop2_inputs[p1["paperId"]] = prefilter(p1, eligible(citers, p1["year"]), HOP2_PREFILTER)
    results2 = {}
    reqs_pending = []
    for p1 in branches:
        if hop2_inputs[p1["paperId"]]:
            try:
                results2[p1["paperId"]] = score_citers(judge, p1, hop2_inputs[p1["paperId"]],
                                                       f"stage2/relevance/hop2/{p1['paperId'][:8]}")
            except Exception as e:  # collect pending calls of all branches in one go
                from llm_client import PendingCalls
                if isinstance(e, PendingCalls):
                    reqs_pending += e.keys
                else:
                    raise
    if reqs_pending:
        from llm_client import PendingCalls
        raise PendingCalls(reqs_pending)

    chains = []
    for i, p1 in enumerate(branches, 1):
        chain = [dict(p1, hop=1)]
        note = "extended"
        if p1["paperId"] not in s2["hop2"]:
            note = "stopped: second-hop citations not fetched (not among the 40 most-cited first-hop papers)"
        elif not hop2_inputs[p1["paperId"]]:
            note = "stopped: no citing papers with abstracts in the two following years"
        else:
            r2 = rank(results2[p1["paperId"]][1])
            if r2:
                chain.append(dict(r2[0], hop=2))
            else:
                note = "stopped: no relevant (score 1 or 2) citing paper"
        chains.append(dict(id=f"C{i}", papers=chain, note=note))
    return dict(source=source, hop1_scored=scored1, hop1_candidates=ranked1, chains=chains,
                hop2_scored={k: v[0] for k, v in results2.items()})


def fmt_chain(chain):
    s = ""
    for i, p in enumerate(chain["papers"], 1):
        s += f"\n{i}. Title: {p['title']}\n   Abstract: {p['abstract']}\n   Year: {p['year']}"
    return s


def main():
    judge, gen = LLM("judge"), LLM("generator")
    data = build_chains(judge)
    OUT_CHAINS.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT_CHAINS}")
    for c in data["chains"]:
        print(c["id"], " -> ".join(f"{p['title'][:45]} ({p['year']}, rel {p['relevance']})" for p in c["papers"]), "|", c["note"])

    s1 = json.loads((ROOT / "outputs" / "stage1_problems.json").read_text())
    problems = [c for c in s1["candidates"] if c["id"] in s1["proposed"]]
    reqs, meta = [], []
    for pr in problems:
        v = pr["versions"][pr["best_version"]]
        for ch in data["chains"]:
            user = H.GEN_USER.format(n_rel=len(ch["papers"]), title=data["source"]["title"],
                                     abstract=data["source"]["abstract"], year=data["source"]["year"],
                                     related=fmt_chain(ch), problem=v["problem"], code_map=H.CODE_MAP,
                                     protocol=PROTOCOL_C)
            reqs.append(dict(system=H.GEN_SYSTEM, user=user, tag=f"stage2/seed/{pr['id']}/{ch['id']}", temperature=0.8))
            meta.append((pr["id"], ch))
    seeds = []
    for (pid, ch), resp in zip(meta, gen.batch(reqs)):
        js = parse_json(resp)
        seeds.append(dict(id=f"{pid}-{ch['id']}", problem=pid, chain=ch["id"],
                          chain_papers=[p["title"] for p in ch["papers"]], **{k.lower().replace(" ", "_"): v for k, v in js.items()}))
    OUT_SEEDS.write_text(json.dumps(dict(problems=[p["id"] for p in problems], seeds=seeds), indent=1))
    print(f"wrote {OUT_SEEDS}: {len(seeds)} seeds")
    for s in seeds:
        print(s["id"], "|", s["title"])


if __name__ == "__main__":
    main()
