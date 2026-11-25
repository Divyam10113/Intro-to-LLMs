import json, glob, re
S = {  # tag -> {index: (relevance, explanation)} ; unlisted indices -> 0 with a default explanation
"stage2/relevance/hop1/2025/chunk0": {
 1:(1,"Inserts lightweight residual adapters into the CLIP encoder and fuses multi-level intermediate features for zero-shot AD, following the source paper's multi-level residual adaptation idea; it compares directly against MVFA-AD."),
 2:(1,"Its cross-domain few-shot design builds on adapter-based alignment of prior VLM methods including the source paper, adding LLM prompts and multi-attention fusion on medical and industrial data."),
 3:(0,"Cites the source paper only in related work; its hypothesis (frequency-guided refinement) is independent."),
 8:(1,"Explicitly motivated by the source paper's multi-level visual adaptation and prompt ensembling, transferring them to handwriting-based Alzheimer's screening."),
 9:(0,"Review paper.")},
"stage2/relevance/hop1/2025/chunk1": {
 6:(2,"MLCA directly extends the source paper: same multi-level residual adapters and pixel-level visual-text alignment loss, plus a ConvTokenizer; its hypothesis depends on the source's finding that multi-level adaptation transfers CLIP to medical AD."),
 3:(0,"Survey of video anomaly detection."),
 4:(0,"Uses the source paper only as a few-shot baseline; the normal-abnormal generalist hypothesis is independent.")},
"stage2/relevance/hop1/2025/chunk2": {
 10:(1,"Combines visual adapters (as introduced by the source paper for medical AD) with prompt learning and partial optimal transport; inspired by, but not dependent on, the source's findings."),
 1:(0,"Lists the source paper in a comparison table only; universal ultrasound lesion detection is an independent hypothesis."),
 4:(0,"Unsupervised autoencoder approach; the source paper is only a compared method."),
 8:(0,"Survey paper.")},
"stage2/relevance/hop1/2025/chunk3": {
 1:(2,"MadCLIP builds on the source paper's adapters in the CLIP vision encoder for medical few-shot AD and modifies them into normal/abnormal dual branches with SigLIP loss and no memory bank, so its hypothesis partly depends on the source's findings."),
 3:(1,"Few-shot CLIP adaptation for ultrasound inspired by the source line of work; it compares with MVFA and targets fine-grained classification."),
 5:(1,"Medical zero-/few-shot AD that positions itself against the source paper's multi-level adaptation and proposes instance-aware query embeddings instead."),
 4:(0,"Review paper."),
 6:(0,"3D Med-VQA dataset; cites the source only for the natural-medical domain gap.")},
"stage2/relevance/hop1/2025/chunk4": {
 3:(1,"Multi-anomaly few-shot medical AD that follows the source paper's evaluation protocol and extends it with binary-prediction and multi-category protocols."),
 1:(0,"Train surface defects; the source paper appears only in related work."),
 7:(0,"Tumor segmentation with diffusion models; the source paper is only a baseline."),
 8:(0,"Systematic review."),
 9:(0,"Reconstruction-based medical AD; cites the source paper only for context."),
 10:(0,"Hierarchical prompt alignment for ZSAD; no evidence of dependence on the source paper.")},
"stage2/relevance/hop1/2025/chunk5": {
 1:(1,"Hierarchical adaptation of intermediate visual-encoder layers for zero-shot AD, inspired by the multi-level feature adaptation introduced by the source paper."),
 6:(1,"AA-CLIP refines patch-level visual features with adapters in the line of the source paper and adds anomaly-aware text anchors; evaluated on the same medical datasets."),
 10:(0,"MLLM-based anomaly reasoning; independent of the source paper's hypothesis.")},
"stage2/relevance/hop1/2025/chunk6": {
 1:(0,"Survey paper."), 2:(0,"Review paper."),
 3:(0,"Borrows prompt templates only; the personalization hypothesis is independent."),
 8:(0,"Mentions the source paper in related work only."),
 9:(0,"Cites adapter modules generally; LoRA-based adaptation is independent.")},
"stage2/relevance/hop1/2026/chunk0": {
 1:(2,"Spatial-FAD adopts the source paper's CLIP ViT-L/14 setting and residual adapters and extends them with a vision-foundation-model spatial prior; its hypothesis depends on the source's adapter framework."),
 6:(2,"Fed-LoRA fine-tunes MVFA-AD itself as the final anomaly detector on its augmented data, so its results depend directly on the source paper."),
 2:(1,"Inspired by the multi-level visual feature aggregation of VAND and the source paper; proposes visually calibrated text prototypes and compares with MVFA-AD.")},
"stage2/relevance/hop1/2026/chunk1": {
 2:(1,"Few-shot medical AD that builds on the source paper's hierarchical adaptation and adds a text-feature adapter and fine-grained descriptions."),
 6:(1,"Motivated by the source paper's reliance on auxiliary datasets with pixel-level anomaly labels; proposes normal-only adaptation with screened pseudo-anomalies."),
 7:(1,"Layer-wise residual adapters at multiple visual levels with Dice and Focal losses, in the style of the source paper, applied to industrial ZSAD."),
 5:(0,"Self-supervised multi-task medical AD; cites the source paper's cross-modality setting but its hypothesis is independent."),
 9:(0,"Medical VLM for clinical intelligence; cites multi-level aggregation generally."),
 10:(0,"Reconstruction-based universal AD; the source paper appears in related work only.")},
"stage2/relevance/hop1/2026/chunk2": {
 2:(1,"Supervised CLIP adaptation for medical AD that departs from the source paper's averaged handcrafted prompts and simple fusion; inspired by, but not dependent on, its findings."),
 1:(0,"Review paper."), 5:(0,"Distillation-based few-shot AD; the source paper is only a baseline.")},
"stage2/relevance/hop1/2026/chunk3": {
 3:(1,"Industrial VLM AD inspired by the source paper's multi-level trainable adapters for calibrating CLIP features; compares with MVFA."),
 2:(0,"Review paper."), 4:(0,"Survey paper.")},
}
TOP = {
"stage2/relevance/hop1/2025/chunk0":[2,1,8], "stage2/relevance/hop1/2025/chunk1":[6], "stage2/relevance/hop1/2025/chunk2":[10],
"stage2/relevance/hop1/2025/chunk3":[1,5,3], "stage2/relevance/hop1/2025/chunk4":[3], "stage2/relevance/hop1/2025/chunk5":[6,1],
"stage2/relevance/hop1/2025/chunk6":[], "stage2/relevance/hop1/2026/chunk0":[6,1,2], "stage2/relevance/hop1/2026/chunk1":[6,2,7],
"stage2/relevance/hop1/2026/chunk2":[2], "stage2/relevance/hop1/2026/chunk3":[3]}
out = {}
for f in glob.glob("/home/claude/task3/logs/pending/*.json"):
    r = json.load(open(f))
    if r["tag"] not in S: continue
    lst = r["user"][r["user"].index("Papers from the Year"):]
    titles = re.findall(r"\n\n(\d+)\. Title: (.*?) Abstract:", lst)
    pl, top = {}, {}
    for n, t in titles:
        rel, ex = S[r["tag"]].get(int(n), (0, "No connection between this paper's key hypothesis and the source paper; the source is not used beyond a generic mention, if at all."))
        pl[f"{n}.{t}"] = {"explanation": ex, "relevance": rel}
    for j, n in enumerate(TOP[r["tag"]], 1):
        t = dict(titles)[str(n)]; rel, ex = S[r["tag"]][n]
        top[f"{j}.{t}"] = {"explanation": ex, "relevance": rel}
    out[r["tag"]] = "```json\n" + json.dumps({"paper_list": pl, "top3_relevant_papers": top}, indent=1) + "\n```"
json.dump(out, open("/home/claude/task3/logs/answers/stage2_hop1_relevance.json","w"), indent=1)
print(len(out))
