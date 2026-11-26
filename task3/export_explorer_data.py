"""
export_explorer_data.py — collects every stage's outputs into one compact JSON used by the explorer page
(outputs/explorer_data.json) and by the report generator.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
O = ROOT / "outputs"


def main():
    s0 = json.loads((O / "gaps_verified.json").read_text())
    ev = json.loads((O / "stage0b_code_verification.json").read_text())
    s1 = json.loads((O / "stage1_problems.json").read_text())
    s2c = json.loads((O / "stage2_chains.json").read_text())
    problems = {}
    for c in s1["candidates"]:
        v = c["versions"][c["best_version"]]
        problems[c["id"]] = dict(id=c["id"], group=c["group"], selected=c["id"] in s1["proposed"], problem=v["problem"],
                                 rationale=v["rationale"], scores={m: r["rating"] for m, r in v["reviews"].items()},
                                 trajectory=[x["mean"] for x in c["versions"]])
    evo, rank = {}, {}
    for pid in s1["proposed"]:
        e = json.loads((O / f"stage3_evolution_{pid}.json").read_text())
        r = json.loads((O / f"stage4_ranking_{pid}.json").read_text())
        rk = {x["id"]: x for x in r["ranking"]}
        cands = {}
        for cid, c in e["candidates"].items():
            cands[cid] = dict(id=cid, gen=c["gen"], parents=c["parents"], operators=c["operators"], title=c["title"],
                              summary=c["summary"], hypothesis=c["hypothesis"], rationale=c.get("rationale") or "",
                              method=c["method"], experiment=c["experiment"], provenance=c.get("provenance") or [],
                              fitness=c["fitness"], chain=c.get("chain"))
            if cid in rk:
                x = rk[cid]
                cands[cid].update(rank=x["overall_rank"], final=x["final"], scores=x["scores"], family=x["family"],
                                  review_mean=x["review_mean"], reviews={k: v["review"] for k, v in x["reviews"].items()},
                                  novelty=x["novelty"], criterion_rank=x["criterion_rank"])
        evo[pid] = dict(history=[dict(generation=h["generation"], population=h["population"], fitness=h["fitness"],
                                      offspring=h.get("offspring", []), pool_fitness=h.get("pool_fitness", h["fitness"]))
                                 for h in e["history"]],
                        candidates=cands, best=e["best"], final_population=e["final_population"],
                        comparisons=len(e["comparisons"]), settings=e["settings"])
        rank[pid] = dict(top5=r["top5"], shortlist=r["shortlist"], families=r["families"],
                         skipped=r["skipped_for_diversity"], weights=r["weights"])
    chains = [dict(id=c["id"], note=c["note"], papers=[dict(title=p["title"], year=p["year"], relevance=p["relevance"],
                                                          explanation=p["explanation"], score=p.get("hyper_score"))
                                                     for p in c["papers"]]) for c in s2c["chains"]]
    from collections import Counter
    rel = Counter(p["relevance"] for p in s2c["hop1_scored"])
    calls = Counter()
    for line in (ROOT / "logs" / "calls.jsonl").read_text().splitlines():
        tag = json.loads(line)["tag"]
        if not tag.startswith(("rejudge/", "hypercheck/")):   # independent checks are not part of the run
            calls[tag.split("/")[0]] += 1
    # calls answered from the cache of an earlier run because their prompts did not change: in run 3 only the
    # second-hop relevance chunks of the chain that was extended (MadCLIP), whose prompt depends only on that paper
    reused = Counter()
    hop2_prefixes = tuple(f"stage2/relevance/hop2/{k[:8]}/" for k in s2c.get("hop2_scored", {}))
    logged = {json.loads(l)["key"] for l in (ROOT / "logs" / "calls.jsonl").read_text().splitlines() if l.strip()}
    seen = set()
    for f in (ROOT / "logs" / "cache").glob("*.json"):
        r = json.loads(f.read_text())
        if hop2_prefixes and r["tag"].startswith(hop2_prefixes) and r["key"] not in logged and r["tag"] not in seen:
            seen.add(r["tag"])
            reused[r["tag"].split("/")[0]] += 1
    import sys
    sys.path.insert(0, str(ROOT))
    from corpus import load_surveygen, mvfa_citers, review_set
    P = load_surveygen()
    cit = mvfa_citers(P)
    src = Counter("+".join(x["sources"]) for x in cit)
    inputs = dict(manifest=json.loads((ROOT / "data" / "inputs_manifest.json").read_text()),
                  task1_papers=len(P), task1_review=len(review_set(P)),
                  task1_origins=dict(Counter(o for p in P for o in (p["origin"] if isinstance(p["origin"], list) else [p["origin"]]))),
                  citers=len(cit), citers_both=src.get("s2+task1", 0), citers_task1_only=src.get("task1", 0),
                  citers_s2_only=src.get("s2", 0), citers_eligible=len(s2c["hop1_scored"]),
                  task2_limitations=sum(1 for g in s0["gaps"] if g.get("source") == "task2"),
                  code_gaps=sum(1 for g in s0["gaps"] if g["id"].startswith("C")))
    data = dict(gaps=s0["gaps"], groups=s0["groups"], ungrouped=s0.get("ungrouped", []), evidence=ev["evidence"],
                later_evidence=ev.get("later_findings", []), verdicts=ev.get("verdicts", {}),
                problems=problems, selected_problems=s1["proposed"], entities=s1["entities"], related=s1["related"],
                chains=chains, hop1_relevance=dict(rel), evolution=evo, ranking=rank, calls=dict(calls),
                total_calls=sum(calls.values()), reused_calls=dict(reused), run="run 3 (Task 1 v6 corpus, Task 2 39 limitations)",
                inputs=inputs, corpus=dict(papers=len(s1.get("related", [])), entities=s1["entities"]))
    (O / "explorer_data.json").write_text(json.dumps(data, indent=1))
    print("wrote", O / "explorer_data.json", f"({len(json.dumps(data)) // 1024} KB)")
    tpl = (ROOT / "explorer" / "template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (ROOT / "explorer" / "mvfa_idea_explorer.html").write_text(html)
    print("wrote", ROOT / "explorer" / "mvfa_idea_explorer.html")


if __name__ == "__main__":
    main()
