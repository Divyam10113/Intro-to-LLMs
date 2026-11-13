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
        calls[json.loads(line)["tag"].split("/")[0]] += 1
    # calls answered from the cache of run 1 because their prompts did not change (Stage 0 grouping, Stage 2 relevance)
    reused = Counter()
    for f in (ROOT / "logs" / "cache").glob("*.json"):
        r = json.loads(f.read_text())
        if r["tag"].startswith(("stage0/", "stage2/relevance")):
            reused[r["tag"].split("/")[0]] += 1
    data = dict(gaps=s0["gaps"], groups=s0["groups"], evidence=ev["evidence"], later_evidence=ev.get("later_findings", []),
                problems=problems, selected_problems=s1["proposed"], entities=s1["entities"], related=s1["related"],
                chains=chains, hop1_relevance=dict(rel), evolution=evo, ranking=rank, calls=dict(calls),
                total_calls=sum(calls.values()), reused_calls=dict(reused), run="run 2 (QUAL-SG corpus)",
                corpus=dict(papers=len(s1.get("related", [])), entities=s1["entities"]))
    (O / "explorer_data.json").write_text(json.dumps(data, indent=1))
    print("wrote", O / "explorer_data.json", f"({len(json.dumps(data)) // 1024} KB)")
    tpl = (ROOT / "explorer" / "template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (ROOT / "explorer" / "mvfa_idea_explorer.html").write_text(html)
    print("wrote", ROOT / "explorer" / "mvfa_idea_explorer.html")


if __name__ == "__main__":
    main()
