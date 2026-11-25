"""Step 3 - baselines + evaluation.

  python evaluate.py reference             # build reference limitations from held-out citing papers
  python evaluate.py baselines             # zero-shot + our old 5-agent pipeline (same models)
  python evaluate.py score full full_nocite zero_shot old_pipeline

Metrics
  * Ground-truth coverage C_GT (recall), C_LLM (share of generated items matching a reference item) and F1 -
    pointwise LLM-as-judge matching (BAGELS / arXiv:2601.11578).
  * Faithfulness / Soundness / Importance, 1-5, reference-free (LimitGen rubric, arXiv:2507.02694).
The judge uses EVAL_MODELS (a different model family first) to limit self-preference bias.
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import (DATA_DIR, EVAL_MODELS, GEN_PROVIDER, OUT_DIR, TASK1_DIR, base_paper_text, chat, chat_json, read_jsonl,
                    write_json)

REF_PATH = OUT_DIR / "reference_limitations.json"
# baselines per generation run: RUN_SUFFIX (e.g. "_sol") or "_nim" for the NIM provider
SUFFIX = os.getenv("RUN_SUFFIX", "_nim" if GEN_PROVIDER == "nim" else "")


# -----------------------------------------------------------------------------
# Reference set from held-out citing papers
# -----------------------------------------------------------------------------
def build_reference():
    mentions = read_jsonl(DATA_DIR / "heldout_mentions.jsonl")
    by_paper = defaultdict(list)
    for m in mentions:
        by_paper[(m["pid"], m["title"])].append(m["text"])
    print(f"[reference] {len(by_paper)} held-out citing papers, {len(mentions)} passages")

    sys_p = ("You read passages from a paper that discuss MVFA (Huang et al., CVPR 2024, 'Adapting Visual-Language "
             "Models for Generalizable Anomaly Detection in Medical Images'; it may appear as a citation number). "
             "Extract every weakness or limitation OF MVFA that the passages state or clearly imply (e.g. what it "
             "requires, cannot handle, ignores, or does worse and why). Do NOT include: plain descriptions of MVFA, "
             "numbers without a stated reason, or weaknesses of other methods. If none, return []. "
             'Return JSON list: [{"limitation": "one sentence", "quote": "supporting text from the passage"}]')

    def extract(item):
        (pid, title), texts = item
        out = chat_json(sys_p, f"PAPER: {title}\n\nPASSAGES:\n" + "\n---\n".join(texts),
                        models=EVAL_MODELS, temperature=0, tag=f"ref-{pid}")
        return [dict(r, pid=pid, paper=title) for r in out if isinstance(r, dict) and r.get("limitation")]

    with ThreadPoolExecutor(max_workers=4) as ex:
        raw = [r for rs in ex.map(extract, by_paper.items()) for r in rs]
    print(f"[reference] {len(raw)} raw critiques")
    if not raw:
        sys.exit("No critiques found in held-out passages - lower --heldout_frac or add papers.")

    merged = chat_json(
        "Merge duplicate limitations (same underlying issue) into one statement each; keep all supporting papers. "
        'Return JSON list: [{"limitation": "...", "papers": ["title", ...]}]',
        json.dumps([{"limitation": r["limitation"], "paper": r["paper"]} for r in raw], indent=1),
        models=EVAL_MODELS, temperature=0, tag="ref-merge")
    for i, r in enumerate(merged, 1):
        r["rid"] = f"R{i}"
    write_json(REF_PATH, {"raw": raw, "reference": merged})
    print(f"[reference] {len(merged)} reference limitations -> {REF_PATH}")


# -----------------------------------------------------------------------------
# Baselines (same generation models as the full system)
# -----------------------------------------------------------------------------
def baselines():
    paper = base_paper_text()

    # (a) zero-shot single prompt
    zs = chat_json("You are an expert reviewer. List the limitations of the following research paper. "
                   'Return a JSON list of strings, one limitation per string.',
                   paper, temperature=0.3, tag="baseline-zeroshot")
    _save_list("zero_shot" + SUFFIX, zs)

    # (b) our previous pipeline (Intro-to-LLMs/multi_agent_limitations.py), re-run with the same models
    review_p = TASK1_DIR / "item3_lit_review.md"
    review = review_p.read_text(encoding="utf-8") if review_p.exists() else ""
    ext = chat("You are the Extractor Agent. Your task is to extract explicitly stated limitations, future work, and "
               "methodological constraints directly from the provided text of a research paper. Do not infer "
               "anything; only report what is explicitly stated by the authors.",
               f"--- Base Paper ---\n{paper}", tag="old-extractor")
    ana = chat("You are the Analyzer Agent. Your task is to critically analyze the methodology and experiment "
               "sections of the provided research paper to identify implicit limitations, potential flaws, "
               "unrealistic assumptions, heavy data requirements, or scalability issues that the authors did not "
               "explicitly state.", f"--- Base Paper ---\n{paper}", tag="old-analyzer")
    rev = chat("You are the Reviewer Agent. Your task is to compare the provided research paper against the provided "
               "literature review. Identify research gaps, missing clinical considerations, or aspects where the "
               "base paper falls short of the broader field's goals as described in the literature.",
               f"--- Literature Review ---\n{review}\n\n--- Base Paper ---\n{paper}", tag="old-reviewer")
    jud = chat("You are the Judge Agent (Master Agent). Your task is to synthesize the findings from the Extractor, "
               "Analyzer, and Reviewer agents. You must:\n1. Deduplicate the findings and resolve any "
               "contradictions.\n2. Output a cohesive, bulleted list of research gaps and limitations.\n3. Follow "
               "the bulleted list with a single, highly technical paragraph defining a new research problem that "
               "addresses these specific gaps.",
               f"--- Extractor Findings ---\n{ext}\n\n--- Analyzer Findings ---\n{ana}\n\n--- Reviewer Findings "
               f"---\n{rev}\n\nPlease synthesize these into the final required output format.", tag="old-judge")
    fin = chat("You are the Self-feedback Agent. Your task is to critique and refine the output generated by the "
               "Judge Agent. Review the bulleted list of limitations and the research problem definition. Ensure "
               "the language is academic, objective, highly technical, and strictly follows the required format. "
               "Output the refined, final version directly.",
               f"--- Judge Agent Output ---\n{jud}\n\nPlease critique and refine this output.", tag="old-selffeedback")
    (OUT_DIR / ("old_pipeline" + SUFFIX)).mkdir(exist_ok=True)
    (OUT_DIR / ("old_pipeline" + SUFFIX) / "output.md").write_text(fin, encoding="utf-8")
    items = chat_json("Extract the bulleted limitations / research gaps from this text (not the research-problem "
                      "paragraph). Return a JSON list of strings, one per limitation, wording unchanged.",
                      fin, temperature=0, tag="old-parse")
    _save_list("old_pipeline" + SUFFIX, items)


def _save_list(name, items):
    d = OUT_DIR / name
    d.mkdir(exist_ok=True)
    items = [i if isinstance(i, str) else i.get("limitation", json.dumps(i)) for i in items]
    write_json(d / "limitations_list.json", items)
    print(f"[baseline] {name}: {len(items)} limitations")


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def load_system(name):
    d = OUT_DIR / name
    if (d / "final_limitations.json").exists():
        return [f"{x['limitation']} {x.get('explanation', '')}".strip()
                for x in json.load(open(d / "final_limitations.json"))]
    return json.load(open(d / "limitations_list.json"))


def coverage(ref, gen, name):
    numbered = "\n".join(f"G{i + 1}. {g}" for i, g in enumerate(gen))
    sys_p = ("You judge whether generated limitations cover a reference limitation of a paper. A generated item "
             "matches if it identifies the same underlying issue (same topic and same direction), even if worded "
             "differently or more specifically. A broader generic statement that merely includes the topic does "
             "NOT match. Return JSON: {\"matches\": [\"G3\", ...], \"reason\": \"...\"} (empty list if none).")

    def judge_one(r):
        out = chat_json(sys_p, f"REFERENCE: {r['limitation']}\n\nGENERATED:\n{numbered}",
                        models=EVAL_MODELS, temperature=0, tag=f"cov-{name}-{r['rid']}")
        return r["rid"], [m for m in out.get("matches", []) if isinstance(m, str)]

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(judge_one, ref))
    c_gt = sum(1 for v in res.values() if v) / len(ref)
    matched_gen = {m for v in res.values() for m in v}
    c_llm = len(matched_gen) / max(1, len(gen))
    f1 = 2 * c_gt * c_llm / (c_gt + c_llm) if c_gt + c_llm else 0.0
    return {"C_GT": c_gt, "C_LLM": c_llm, "F1": f1, "matches": res}


FSI_RUBRIC = """\
Faithfulness (1-5): accurately represents the paper's content; no misinformation or contradictions with the paper.
Soundness (1-5): detailed, specific, logically coherent; suggestions practical. 1 = vague/generic.
Importance (1-5): 5 = critical issue affecting main findings/contributions; 1 = trivial."""


def fsi(paper, gen, name):
    numbered = "\n".join(f"G{i + 1}. {g}" for i, g in enumerate(gen))
    out = chat_json("You are an expert reviewer scoring generated limitations of the paper below.\n" + FSI_RUBRIC +
                    '\nReturn JSON list: [{"id": "G1", "faithfulness": int, "soundness": int, "importance": int}] '
                    "for every item.",
                    f"=== PAPER ===\n{paper}\n=== END PAPER ===\n\nGENERATED LIMITATIONS:\n{numbered}",
                    models=EVAL_MODELS, temperature=0, tag=f"fsi-{name}")
    rows = [r for r in out if isinstance(r, dict)]
    avg = lambda k: sum(float(r.get(k, 0)) for r in rows) / max(1, len(rows))  # noqa: E731
    return {"faithfulness": avg("faithfulness"), "soundness": avg("soundness"), "importance": avg("importance"),
            "per_item": rows}


def score(systems):
    ref = json.load(open(REF_PATH))["reference"]
    paper = base_paper_text()
    results = {}
    for name in systems:
        gen = load_system(name)
        print(f"[score] {name}: {len(gen)} generated vs {len(ref)} reference")
        results[name] = {"n_generated": len(gen), **coverage(ref, gen, name), **fsi(paper, gen, name)}
    write_json(OUT_DIR / "evaluation.json", results)
    lines = ["| system | #lims | C_GT (recall) | C_LLM | F1 | Faith. | Sound. | Import. |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['n_generated']} | {r['C_GT']:.1%} | {r['C_LLM']:.1%} | {r['F1']:.3f} | "
                     f"{r['faithfulness']:.2f} | {r['soundness']:.2f} | {r['importance']:.2f} |")
    table = "\n".join(lines)
    (OUT_DIR / "evaluation.md").write_text(
        f"# Evaluation\n\nReference: {len(ref)} limitations of MVFA stated in held-out citing papers "
        f"(never indexed for retrieval).\nJudge models: {', '.join(EVAL_MODELS)}\n\n{table}\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "reference":
        build_reference()
    elif cmd == "baselines":
        baselines()
    elif cmd == "score":
        score(sys.argv[2:] or ["full", "full_nocite", "zero_shot", "old_pipeline"])
    else:
        print(__doc__)
