"""
hyper_checker.py — data, prompts and metrics for the HypER small-model checker (Phi-3 + LoRA).

HypER (Vasu et al., EMNLP 2025, github.com/rosnikv/HypER) distils its large-model chain construction into
Phi-3-mini-128k-instruct with multi-task instruction tuning. Our pipeline ran HypER's chain construction with a
large model (the teacher role) but never trained the student. This module rebuilds the student's training data
from HypER's released chains and prepares the same task formats for OUR MVFA chains, so the trained student can be
used as an independent second opinion on Stage 2:

  hyper-1-hop          source paper + target paper  -> {"relevance_score": 0|1|2}
  hyper-multi-hop2_c1  reasoning chain               -> {"validity": "valid"|"invalid", "invalid_paper_ids": [...]}
  hyper-multi-hop2_c2  chain (without the last paper) + target hypotheses -> same output as c1

Faithful to HypER's multi-task-data_prep.py: the three instruction texts are copied verbatim, the 1-hop task takes
every relevance-0 pair of a chain (otherwise one random pair with relevance 1 or 2), invalid paper ids are the
1-based positions of relevance-0 papers, and 2024 papers are removed from the training split. Differences: HypER's
fourth task (hypothesis generation, hyper-hg) needs GPT-4o-extracted target hypotheses and is left out; the 'input'
dict is rendered as plain text sections (the released code does not show how axolotl serialised it); prompts use
axolotl's alpaca template, which HypER's Phi-3 config uses (type: alpaca).

Standard library only, so the data part can be tested without a GPU:
    python3 colab/hyper_checker.py /path/to/HypER/dataset/training_data/balanced_splits_w_hyp
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ALPACA = ("Below is an instruction that describes a task, paired with an input that provides further context. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

INS_1HOP = "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They state something that the scientific process will attempt to evaluate, corroborate, verify, or falsify. Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make. You are a scientist. Your task is to evaluate the relevance of a target paper to a source paper in a one-hop citation context. You are given a source paper and a target paper that followed from it. Your job is to determine the degree of relevance between these papers based on scientific progression, logical dependence, and hypothesis inspiration.\n\nAssign a relevance score to the target paper based on the following criteria:\n- 0: No meaningful connection between the target paper and the source paper. This includes review papers (e.g., systematic reviews, meta-analyses) that summarize literature without introducing novel hypotheses or findings.\n- 1: The key hypothesis in the target paper is inspired by the hypothesis or findings from the source paper.\n- 2: The key hypothesis in the target paper directly depends on the findings of the source paper. This means the source paper contains sub-hypotheses or foundational results that contribute directly to the target paper's hypothesis.\n\nYour task:\n1. Assign a relevance score (0, 1, or 2) based on the relationship between the source and target paper.\nAnswer must follow the example output format in JSON:\n```json\n{ \"relevance_score\": <0, 1, or 2> }\n```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required."

INS_C1 = "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper. \n\nA reasoning chain is considered 'valid' if:\n- All papers in the sequence are logically connected.\n- The final hypothesis in the last paper is logically inspired by or dependent on the source paper, either directly or through intermediate papers.\n\nA reasoning chain is considered 'invalid' if:\n- if any logical break exists in the chain, such as unrelated intermediate papers, or  if the final hypothesis does not logically depend on the previous papers.\n\nYour task:\n1. Determine whether the reasoning chain is 'valid' or 'invalid'.\n2. If invalid, return a list of paper IDs that break logical progression in the chain.\n\nAnswer must follow the example output format in JSON:\n- If the chain is valid, return:\n  ```json\n  { \"validity\": \"valid\", \"invalid_paper_ids\": [] }\n  ```\n- If the chain is invalid, return:\n  ```json\n  { \"validity\": \"invalid\", \"invalid_paper_ids\": [3, 4, 7] }\n  ```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required."

INS_C2 = "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper.\n\nA reasoning chain is considered 'valid' if:\n- All papers in the sequence are logically connected.\n- The final hypotheses in the last paper is logically inspired by or dependent on the source paper, either directly or through intermediate papers.\n- The reasoning chain logically supports the given target hypotheses.\n\nA reasoning chain is considered 'invalid' if:\n- Any logical break exists in the chain, such as unrelated intermediate papers, or if the final hypotheses do not logically depend on the previous papers.\n- The reasoning chain does not logically support the given target hypotheses.\n\nYour task:\n1. Determine whether the reasoning chain supports the target hypotheses and is 'valid' or 'invalid'.\n2. If invalid, return a list of paper IDs that break logical progression in the chain.\n\nAnswer must follow the example output format in JSON:\n- If the chain is valid, return:\n  ```json\n  { \"validity\": \"valid\", \"invalid_paper_ids\": [] }\n  ```\n- If the chain is invalid, return:\n  ```json\n  { \"validity\": \"invalid\", \"invalid_paper_ids\": [3, 4, 7] }\n  ```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required."


# ------------------------------------------------------------------------------------------------ formatting
def fmt_paper(p):
    return f"Title: {p.get('title') or 'No Title Provided'}\nAbstract: {p.get('abstract') or 'No Abstract Provided'}\nYear: {p.get('year') or 'No Year Provided'}"


def fmt_chain(content):
    """As in HypER's format_reasoning_chain."""
    return "\n".join(f"{i}. PAPER TITLE: {p.get('title') or 'No Title Provided'}\n   ABSTRACT: {p.get('abstract') or 'No Abstract Provided'}\nYear: {p.get('year') or 'No Year Provided'}"
                     for i, p in enumerate(content, start=1))


def ex_1hop(source, target, label=None, **meta):
    return dict(task="hyper-1-hop", instruction=INS_1HOP,
                input=f"Source paper:\n{fmt_paper(source)}\n\nTarget paper:\n{fmt_paper(target)}",
                output=None if label is None else json.dumps({"relevance_score": label}), **meta)


def ex_c1(content, validity=None, invalid_ids=None, **meta):
    out = None if validity is None else json.dumps({"validity": validity, "invalid_paper_ids": invalid_ids or []})
    return dict(task="hyper-multi-hop2_c1", instruction=INS_C1, input=f"Reasoning chain:\n{fmt_chain(content)}", output=out, **meta)


def ex_c2(content, hypotheses, validity=None, invalid_ids=None, **meta):
    out = None if validity is None else json.dumps({"validity": validity, "invalid_paper_ids": invalid_ids or []})
    return dict(task="hyper-multi-hop2_c2", instruction=INS_C2,
                input=f"Reasoning chain:\n{fmt_chain(content)}\n\nTarget hypotheses:\n{hypotheses}", output=out, **meta)


def prompt(ex):
    return ALPACA.format(instruction=ex["instruction"], input=ex["input"])


# ------------------------------------------------------------------------------------------------ HypER data
def build_split(path, split, seed=0):
    """Multi-task examples from one of HypER's balanced splits (train_hyp.json, valid_hyp.json, test_hyp.json)."""
    rng = random.Random(seed)
    data = json.loads(Path(path).read_text())
    out = []
    for e in data:
        content = e["content"]
        if split == "train":   # as in HypER: 2024 papers are removed from the training chains
            content = [p for p in content if p.get("year") != 2024]
        if not content:
            continue
        label = "invalid" if e["chain_label"] in {"invalid_type1", "invalid_type2"} else e["chain_label"]
        meta = dict(chain=e["file_name"], chain_label=e["chain_label"])
        zero = [(content[i], content[i + 1]) for i in range(len(content) - 1) if content[i + 1].get("relevance") == 0]
        other = [(content[i], content[i + 1]) for i in range(len(content) - 1) if content[i + 1].get("relevance") in (1, 2)]
        pairs = zero if zero else (rng.sample(other, 1) if other else [])
        for s, t in pairs:
            out.append(ex_1hop(s, t, t.get("relevance"), **meta))
        bad = [i + 1 for i, p in enumerate(content) if p.get("relevance") == 0]
        out.append(ex_c1(content, label, bad if label == "invalid" else [], **meta))
        th = e.get("target_hypothesis")
        hyps = th.get("Target Hypotheses", ["NA"]) if isinstance(th, dict) else ["NA"]
        hyp_text = " ".join(f"{i + 1}. {h}" for i, h in enumerate(hyps))
        c2_content = content[:-1] if len(content) > 1 else content
        out.append(ex_c2(c2_content, hyp_text, label, bad if label == "invalid" else [], **meta))
    return out


# ------------------------------------------------------------------------------------------------ our MVFA data
def build_mvfa(task3_root):
    """The same task formats for our Stage 2 output: every scored hop-1 and hop-2 paper (1-hop task, with our
    judge's 0/1/2 score as the reference), every chain (c1), and every seed idea with its chain (c2: does the chain
    support the seed's hypothesis?)."""
    root = Path(task3_root)
    ch = json.loads((root / "outputs" / "stage2_chains.json").read_text())
    seeds = json.loads((root / "outputs" / "stage2_seeds.json").read_text())["seeds"]
    src = ch["source"]
    out = []
    for p in ch["hop1_scored"]:
        out.append(ex_1hop(src, p, None, kind="hop1", paper=p["title"], ours=p["relevance"]))
    parents = {c["papers"][0]["paperId"]: c["papers"][0] for c in ch["chains"]}
    for pid, scored in ch.get("hop2_scored", {}).items():
        par = parents.get(pid)
        if par:
            for p in scored:
                out.append(ex_1hop(par, p, None, kind="hop2", parent=par["title"], paper=p["title"], ours=p["relevance"]))
    chains = {c["id"]: [src] + c["papers"] for c in ch["chains"]}
    for cid, content in chains.items():
        out.append(ex_c1(content, None, None, kind="chain", chain=cid, n_papers=len(content)))
    for s in seeds:
        out.append(ex_c2(chains[s["chain"]], s.get("hypothesis", ""), None, None, kind="seed", seed=s["id"], chain=s["chain"]))
    return out


# ------------------------------------------------------------------------------------------------ parsing, metrics
def parse(text):
    """Parse the model's JSON answer; tolerant of code fences and trailing text."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        d = {}
        r = re.search(r'"relevance_score"\s*:\s*(\d)', m.group(0))
        if r:
            d["relevance_score"] = int(r.group(1))
        v = re.search(r'"validity"\s*:\s*"(valid|invalid)"', m.group(0))
        if v:
            d["validity"] = v.group(1)
        return d


def prf(gold, pred, labels):
    """Per-label precision/recall/F1, macro F1 and accuracy (predictions that failed to parse count as wrong)."""
    res, f1s = {}, []
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        res[str(l)] = dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(f1, 3), support=sum(1 for g in gold if g == l))
        f1s.append(f1)
    res["macro_f1"] = round(sum(f1s) / len(f1s), 3)
    res["accuracy"] = round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 3) if gold else None
    res["unparsed"] = sum(1 for p in pred if p is None)
    return res


def qwk(x, y, k=3):
    n = len(x)
    if not n:
        return None
    O = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        O[a][b] += 1
    hx = [sum(O[i]) for i in range(k)]
    hy = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum(((i - j) ** 2) * O[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2) * hx[i] * hy[j] / n for i in range(k) for j in range(k))
    return round(1 - num / den, 3) if den else None


def evaluate(examples, answers):
    """Metrics on HypER's test split, as in the paper: F1 per task, Jaccard of invalid paper ids."""
    by = {}
    for ex, a in zip(examples, answers):
        by.setdefault(ex["task"], []).append((json.loads(ex["output"]), parse(a)))
    out = {}
    for t, rows in by.items():
        if t == "hyper-1-hop":
            g = [r[0]["relevance_score"] for r in rows]
            p = [r[1].get("relevance_score") if r[1].get("relevance_score") in (0, 1, 2) else None for r in rows]
            out[t] = prf(g, p, [0, 1, 2])
        else:
            g = [r[0]["validity"] for r in rows]
            p = [r[1].get("validity") if r[1].get("validity") in ("valid", "invalid") else None for r in rows]
            m = prf(g, p, ["valid", "invalid"])
            jac = []
            for gold, pred in rows:
                if gold["validity"] == "invalid" and pred.get("validity") == "invalid":
                    a, b = set(gold["invalid_paper_ids"]), set(pred.get("invalid_paper_ids") or [])
                    jac.append(len(a & b) / len(a | b) if a | b else 1.0)
            m["invalid_ids_jaccard"] = round(sum(jac) / len(jac), 3) if jac else None
            out[t] = m
        out[t]["n"] = len(rows)
    return out


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("HypER/dataset/training_data/balanced_splits_w_hyp")
    for sp in ("train", "valid", "test"):
        ex = build_split(d / f"{sp}_hyp.json", sp)
        lens = sorted(len(prompt(e)) + len(e["output"]) for e in ex)
        print(sp, len(ex), dict(Counter(e["task"] for e in ex)),
              f"chars median {lens[len(lens) // 2]}, 95% {lens[int(len(lens) * 0.95)]}, max {lens[-1]}")
    root = Path(__file__).resolve().parents[1]
    m = build_mvfa(root)
    print("mvfa", len(m), dict(Counter(e["kind"] for e in m)))
