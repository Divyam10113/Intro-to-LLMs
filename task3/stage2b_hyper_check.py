"""
Stage 2b (optional) — HypER's distilled checker as a second opinion on Stage 2.

Input: outputs/hypercheck_phi3.json, written on Colab by colab/train_hyper_phi3.py (Phi-3-mini fine-tuned with LoRA on
HypER's released chains), and, if present, outputs/hypercheck_eval_lora.json / outputs/hypercheck_eval_base.json (the
checker's and the base model's scores on HypER's own test split).
Output: outputs/stage2b_hyper_check.json with
  * relevance agreement between the checker and our Stage 2 judge on every scored MVFA-citing paper (hop 1) and
    second-hop paper: exact agreement, quadratic-weighted kappa, confusion matrix, and the papers that started our
    chains but that the checker scores 0;
  * the checker's validity verdict on each of our six chains;
  * for each seed idea, whether the checker finds that its chain supports the seed's hypothesis.
No LLM calls; runs in seconds.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "colab"))
import hyper_checker as H  # noqa: E402

IN = ROOT / "outputs" / "hypercheck_phi3.json"
OUT = ROOT / "outputs" / "stage2b_hyper_check.json"


def main():
    if not IN.exists():
        print(f"{IN} not found: run colab/T3_hyper_checker_phi3.ipynb first (optional stage)")
        return
    d = json.loads(IN.read_text())
    rows = d["rows"]
    chains = json.loads((ROOT / "outputs" / "stage2_chains.json").read_text())
    starters = {c["papers"][0]["title"] for c in chains["chains"]}
    rel = {}
    for kind in ("hop1", "hop2"):
        rs = [r for r in rows if r["kind"] == kind]
        pairs = [(r["ours"], r["parsed"].get("relevance_score")) for r in rs]
        ok = [(a, b) for a, b in pairs if b in (0, 1, 2)]
        conf = [[sum(1 for a, b in ok if a == i and b == j) for j in range(3)] for i in range(3)]
        rel[kind] = dict(n=len(rs), parsed=len(ok),
                         exact=round(sum(a == b for a, b in ok) / len(ok), 3) if ok else None,
                         qwk=H.qwk([a for a, _ in ok], [b for _, b in ok]) if ok else None,
                         confusion_ours_rows_checker_cols=conf,
                         checker_distribution={str(k): sum(1 for _, b in ok if b == k) for k in (0, 1, 2)},
                         ours_distribution={str(k): sum(1 for a, _ in ok if a == k) for k in (0, 1, 2)})
    rel["chain_starters_scored_0_by_checker"] = [r["paper"] for r in rows if r["kind"] == "hop1" and r["paper"] in starters
                                                 and r["parsed"].get("relevance_score") == 0]
    out = dict(model=d.get("model"), train_info=d.get("train_info"), relevance=rel,
               chains=[dict(chain=r["chain"], n_papers=r["n_papers"], validity=r["parsed"].get("validity"),
                            invalid_paper_ids=r["parsed"].get("invalid_paper_ids")) for r in rows if r["kind"] == "chain"],
               seeds=[dict(seed=r["seed"], chain=r["chain"], supported=r["parsed"].get("validity") == "valid",
                           answer=r["parsed"]) for r in rows if r["kind"] == "seed"])
    for name in ("lora", "base"):
        p = ROOT / "outputs" / f"hypercheck_eval_{name}.json"
        if p.exists():
            out[f"hyper_test_{name}"] = json.loads(p.read_text())["metrics"]
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)
    print(json.dumps({k: v for k, v in rel.items()}, indent=1))


if __name__ == "__main__":
    main()
