import json, glob, re
S = {
"stage2/relevance/hop2/7592c305/2026/chunk0": {
 1:(1,"Its two-stage prompt framework separates acquiring anomaly semantics from adapting to normal appearance, inspired by the source paper's anomaly-aware text anchors aligned before visual adaptation."),
 2:(2,"Adopts the source paper as its baseline for alignment (BCE for classification, Dice+Focal for segmentation, orthogonality of anchors) and adds SAM2 patch prompts and multi-semantic regularisation, so its hypothesis depends on the source's anchor-alignment findings."),
 5:(1,"Builds on the source paper's diagnosis of anomaly-unaware text anchors and targets anchor collapse, noisy dense responses and unstable outlier-dominated image-level scoring."),
 6:(1,"Motivated by the source paper's residual adapters in both text and visual branches; argues that joint alignment causes a granularity mismatch and decouples it."),
 7:(1,"Extends the idea of explicit normal/abnormal anchors from text-only anchors to hierarchical image anchors."),
 3:(0,"Uses the source paper only as a compared method."), 8:(0,"Mentions the source paper in related work only."),
 9:(0,"Image artifact detection; cites the source generally."), 10:(0,"Related-work mention only.")},
"stage2/relevance/hop2/7592c305/2026/chunk1": {},
"stage2/relevance/hop2/a794096c/2026/chunk0": {
 1:(1,"Few-shot medical AD that, like the source paper, uses learnable prompts and few-shot visual information, adding vision-foundation-model spatial priors."),
 2:(1,"Replaces the static normal/abnormal references (text prompts or learned visual tokens, as in the source paper) with bounded, test-image-conditioned prototypes for cross-domain medical AD."),
 3:(1,"Adopts the source paper's dual-branch design that disentangles normal and abnormal representations with layer-wise adapters, transferred to surgical bleeding detection."),
 8:(1,"Positions supervised CLIP adaptation for medical AD against the source paper's few-shot design, adding margin-based calibration."),
 9:(0,"Uses the source paper only as a baseline.")},
}
TOP = {"stage2/relevance/hop2/7592c305/2026/chunk0":[2,5,1], "stage2/relevance/hop2/7592c305/2026/chunk1":[],
       "stage2/relevance/hop2/a794096c/2026/chunk0":[2,1,8]}
out = {}
for f in glob.glob("/home/claude/task3/logs/pending/*.json"):
    r = json.load(open(f))
    lst = r["user"][r["user"].index("Papers from the Year"):]
    titles = re.findall(r"\n\n(\d+)\. Title: (.*?) Abstract:", lst)
    pl, top = {}, {}
    for n, t in titles:
        rel, ex = S[r["tag"]].get(int(n), (0, "Cites the source paper only as a compared method or in related work; its key hypothesis is independent of the source paper."))
        pl[f"{n}.{t}"] = {"explanation": ex, "relevance": rel}
    for j, n in enumerate(TOP[r["tag"]], 1):
        t = dict(titles)[str(n)]; rel, ex = S[r["tag"]][n]
        top[f"{j}.{t}"] = {"explanation": ex, "relevance": rel}
    out[r["tag"]] = "```json\n" + json.dumps({"paper_list": pl, "top3_relevant_papers": top}, indent=1) + "\n```"
json.dump(out, open("/home/claude/task3/logs/answers/stage2_hop2_relevance.json","w"), indent=1)
