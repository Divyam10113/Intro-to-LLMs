"""Run 3, Stage 2 hop-1 relevance (judge role). Papers already judged in run 2 (same source paper, same prompt
wording) keep their run-2 score and explanation, so the judge is consistent across runs; the 48 papers new in run 3
(mostly found only by Task 1) were read and scored here. Top-3 per chunk: highest relevance, ties broken by
citation count."""
import json, glob, re, sys
sys.path.insert(0, "/home/claude/task3")
from corpus import load_surveygen, mvfa_citers
from utils import parse_json
D0 = "No connection between this paper's key hypothesis and the source paper; the source is not used beyond a generic mention, if at all."
REV = "Review or survey paper: it summarises existing literature and has no novel hypothesis of its own."
NEW = {
 ("2026/chunk1", 1): (1, "Zero-shot detection for industrial and medical images by aligning hierarchical text prompts with multi-level image patches, inspired by the source paper's multi-level visual-language alignment."),
 ("2025/chunk7", 2): (1, "Few-shot medical anomaly detection with cross-layer interaction of CLIP visual features and learnable text prompts, inspired by the source paper's multi-level adaptation of CLIP features for medical images."),
 ("2025/chunk7", 3): (0, "Evaluates BiomedCLIP for scoliosis classification; no anomaly-detection hypothesis derived from the source paper."),
 ("2025/chunk7", 4): (0, REV), ("2025/chunk7", 7): (0, REV), ("2025/chunk7", 9): (0, REV),
 ("2025/chunk7", 5): (0, "Unsupervised reconstruction/prototype Mamba model for medical anomaly detection; it does not build on vision-language adaptation."),
 ("2025/chunk7", 6): (0, "Industrial zero-shot detection with detail-feature and text-refinement modules; the source paper is at most a generic reference."),
 ("2025/chunk7", 8): (1, "Unifies zero- and few-shot detection by combining text-prompt scores with a memory bank filtered by zero-shot priors, an idea inspired by the source paper's fusion of a zero-shot text branch and a few-shot memory branch."),
 ("2025/chunk7", 10): (0, "Clinical evaluation of general multimodal chat models on fundus photographs; unrelated to the source paper's method."),
 ("2026/chunk4", 3): (1, "Adds a hypergraph spatial-consistency loss for cross-modal medical anomaly detection and plugs its score into zero-shot text comparison and few-shot memory mechanisms of the kind the source paper introduced."),
 ("2026/chunk4", 4): (2, "An adapter framework on a frozen CLIP backbone for zero/few-shot medical anomaly detection on BMAD, evaluated against CLIP-based adapter baselines under identical backbones; its hypothesis (decoupling anatomy from adapter activations) presupposes the source paper's adapter-based adaptation."),
 ("2026/chunk4", 5): (2, "Analyses what the text anchors contribute to few-/full-shot vision-language detectors with trained visual adapters and tests the source paper's model directly, so its key hypothesis depends on the source paper's design."),
 ("2026/chunk4", 6): (2, "Target-free zero-shot medical anomaly detection with CLIP adapters under a leave-one-domain-out protocol on six medical datasets — the source paper's protocol — hypothesising that source-domain dependency of such adapters limits transfer."),
 ("2026/chunk4", 7): (1, "Cross-domain zero-shot medical detection that adds normal-variation prompts to the normal/abnormal text pair, inspired by the text-anchored patch scoring used by the source paper."),
 ("2026/chunk4", 8): (0, "Unsupervised OCT anomaly detection by reconstruction and triplet learning on visual foundation models; no vision-language adaptation."),
 ("2026/chunk4", 9): (0, REV), ("2026/chunk4", 10): (0, REV),
 ("2025/chunk0", 7): (1, "Benchmarks CLIP-based zero-shot detectors, including medical-adapted variants, on brain metastases; the finding that domain-specific fine-tuning is needed follows the source paper's motivation."),
 ("2026/chunk6", 1): (0, "Contextual anomaly detection with vision-language models; unrelated to medical adaptation."),
 ("2026/chunk6", 2): (0, "Pseudo-label refinement for unsupervised CLIP classification; not anomaly detection."),
 ("2026/chunk6", 3): (0, REV), ("2026/chunk6", 5): (0, REV), ("2026/chunk6", 7): (0, REV),
 ("2026/chunk6", 4): (0, "Reverse-distillation industrial anomaly detection; no dependence on the source paper."),
 ("2026/chunk6", 6): (0, "Transformer for histopathology tumour pattern recognition; not anomaly detection."),
 ("2026/chunk6", 8): (0, "RGB-depth multimodal fusion bias in industrial detection; unrelated."),
 ("2026/chunk5", 1): (1, "Zero-shot detection with anomaly-aware prompts and multi-scale feature adapters, extended to few-shot, on industrial and medical data; inspired by the source paper's multi-level adapters."),
 ("2026/chunk5", 2): (0, REV), ("2026/chunk5", 5): (0, REV), ("2026/chunk5", 10): (0, REV),
 ("2026/chunk5", 3): (0, "Generalist detection by text-guided residual features; the source paper is not the basis of its hypothesis."),
 ("2026/chunk5", 4): (0, "Industrial zero-shot detection with LLM-generated prompts and multi-scale fusion; generic relation only."),
 ("2026/chunk5", 6): (0, "Anatomy-aware image-report pretraining for radiology interpretation; not anomaly detection."),
 ("2026/chunk5", 7): (1, "Cross-domain detection (industrial, semantic, medical) with a feature adaptor that tightens the normal boundary, inspired by adapter-based cross-domain transfer as in the source paper."),
 ("2026/chunk5", 8): (1, "Zero-shot detection that reweights multi-scale CLIP representations per image, an extension of the multi-level feature use popularised by the source paper."),
 ("2026/chunk5", 9): (0, "Structured prompt learning for zero-shot detection on industrial data; generic relation only."),
 ("2026/chunk0", 6): (0, "Training-free detection that builds its normal reference from the unlabelled test set; it argues against trained adaptation rather than building on the source paper."),
 ("2025/chunk8", 4): (0, "Video anomaly detection thesis; unrelated."), ("2025/chunk8", 6): (0, REV), ("2025/chunk8", 7): (0, REV),
}
prev = json.load(open("/home/claude/task3/logs/answers/run3/run2_hop1_judgments.json"))
cites = {x["title"].lower().strip(): x.get("citationCount") or 0 for x in mvfa_citers(load_surveygen())}
out = {}
for f in glob.glob("/home/claude/task3/logs/pending/*.json"):
    r = json.load(open(f))
    if not r["tag"].startswith("stage2/relevance/hop1/"):
        continue
    ck = r["tag"].split("hop1/")[1]
    lst = r["user"][r["user"].index("Papers from the Year"):]
    titles = re.findall(r"\n\n(\d+)\. Title: (.*?) Abstract:", lst)
    pl, rel = {}, []
    for n, t in titles:
        k = t.lower().strip()
        if k in prev:
            sc, ex = prev[k][0][0], prev[k][0][1]
        else:
            sc, ex = NEW.get((ck, int(n)), (0, D0))
        pl[f"{n}.{t}"] = {"explanation": ex, "relevance": sc}
        if sc > 0:
            rel.append((sc, cites.get(k, 0), t, ex))
    rel.sort(key=lambda x: (-x[0], -x[1]))
    top = {f"{j}.{t}": {"explanation": ex, "relevance": sc} for j, (sc, _, t, ex) in enumerate(rel[:3], 1)}
    out[r["tag"]] = "```json\n" + json.dumps({"paper_list": pl, "top3_relevant_papers": top}, indent=1) + "\n```"
json.dump(out, open("/home/claude/task3/logs/answers/run3/s2_hop1_relevance.json", "w"), indent=1)
print(len(out))
