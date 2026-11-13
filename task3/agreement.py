#!/usr/bin/env python3
"""
agreement.py — human-model agreement for the final ideas (ResearchAgent-style validation of the LLM judge).

Input: a JSON export of the explorer page's human ratings, a list of records
    {"rater": "...", "idea": "<idea id>", "scores": {"Novelty": 1-5, "Feasibility": ..., "Clarity": ..., "Impact": ..., "Validity": ...}}
(download it from the explorer page, "Export ratings").

Output: per criterion, Spearman correlation and quadratic-weighted Cohen's kappa between each rater and the model
reviews of Stage 4, and between raters when there are two or more (ResearchAgent Table 1 reports both kinds).

    python3 agreement.py ratings.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent
CRITERIA = ["Novelty", "Feasibility", "Clarity", "Impact", "Validity"]


def model_scores():
    out = {}
    for pid in ("A1", "B1"):
        for r in json.loads((ROOT / "outputs" / f"stage4_ranking_{pid}.json").read_text())["ranking"]:
            out[r["id"]] = r["scores"]
    return out


def agree(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None, None
    return round(spearmanr(a, b).correlation, 3), round(cohen_kappa_score(a, b, weights="quadratic"), 3)


def main(path):
    ratings = json.loads(Path(path).read_text())
    model = model_scores()
    by_rater = {}
    for r in ratings:
        by_rater.setdefault(r["rater"], {})[r["idea"]] = r["scores"]
    report = {}
    for rater, ideas in by_rater.items():
        common = [i for i in ideas if i in model]
        for c in CRITERIA:
            h = [ideas[i][c] for i in common if ideas[i].get(c)]
            m = [model[i][c] for i in common if ideas[i].get(c)]
            report.setdefault(f"human:{rater} vs model", {})[c] = dict(n=len(h), spearman_kappa=agree(h, m))
    for r1, r2 in combinations(by_rater, 2):
        common = [i for i in by_rater[r1] if i in by_rater[r2]]
        for c in CRITERIA:
            a = [by_rater[r1][i][c] for i in common]
            b = [by_rater[r2][i][c] for i in common]
            report.setdefault(f"human:{r1} vs human:{r2}", {})[c] = dict(n=len(a), spearman_kappa=agree(a, b))
    print(json.dumps(report, indent=1))
    (ROOT / "outputs" / "human_model_agreement.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
