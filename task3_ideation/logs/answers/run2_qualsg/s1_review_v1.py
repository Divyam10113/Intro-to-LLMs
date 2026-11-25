import json
R = {
"A1": {
 "Clarity": (5, "The pseudo-mask rule is now operational (feature level, distance, threshold fixed on source validation data), and the outcome is a precise label-cost curve.", "Keep the threshold percentile as a reported hyper-parameter with a small sensitivity check."),
 "Relevance": (5, "Directly addresses G3 and makes use of the three image-only BMAD sources; the zero-shot focus is now explicit.", "None."),
 "Originality": (3, "Using image-level labels to restrict pseudo-masks to known abnormal images is a sensible distinction from CoDeGraph's unlabelled batches, but nearest-neighbour pseudo-masks and weakly supervised localisation remain established ideas; the main novelty is the question asked of MVFA.", "The originality would rise if the pseudo-masks were refined by MVFA itself during training (self-training) rather than fixed, but that would widen the problem."),
 "Feasibility": (4, "Computationally cheap and well planned, with an upfront pseudo-mask quality check. Pseudo-mask quality on ChestXray and HIS remains a risk for the image-only extension.", "Treat the image-only extension as secondary so the main result does not depend on it."),
 "Significance": (4, "A label-cost curve would directly tell practitioners how much expert annotation MVFA needs; useful for the field regardless of outcome.", "None."),
},
"A2": {
 "Clarity": (4, "The branch question and the fixed synthesis recipe make the problem well defined; the outcome is clear. The problem statement still mixes two questions (which branch, how much retained).", "Make the retained fraction the outcome of the branch comparison rather than a second question."),
 "Relevance": (5, "Directly addresses G4, including the single-draw issue, and the lack of a normal-only route in G3.", "None."),
 "Originality": (3, "Asking which of MVFA's two branches should consume synthetic supervision is specific to the target method and not answered by MediCLIP or SpatioPrompt; the underlying normal-only setting is still known.", "None needed for this scope."),
 "Feasibility": (5, "Minutes per run; all public.", "None."),
 "Significance": (3, "Practical, but the likely answer (both branches, partial recovery) has limited impact beyond MVFA.", "Report the minimal K at which normal-only matches K+K, which is the actionable number."),
},
"B1": {
 "Clarity": (4, "The ordered ablation and the invariance measurement make the study well defined. The statement is long and contains two questions (dependence on transductive normalisation and the inductive replacement).", "Lead with the single question (the inductive replacement) and present the dependence measurement as its baseline."),
 "Relevance": (5, "Addresses G8, G9 and G11 directly, on the component every published few-shot number of MVFA depends on.", "None."),
 "Originality": (3, "The invariance measurement is a new and useful diagnostic for medical CLIP detectors, but inductive normalisation and calibration are established techniques.", "None needed; the measurement is the contribution."),
 "Feasibility": (5, "Inference-only on trained checkpoints.", "None."),
 "Significance": (4, "Quantifies a deployment gap in a widely cited method and supplies thresholded metrics that clinical readers need.", "None."),
},
"B2": {
 "Clarity": (4, "Clear variants and hypothesis; the k-to-lesion-area rule needs a definition for datasets without masks.", "State how k is set for the image-only datasets (e.g. median over the masked sources)."),
 "Relevance": (4, "Addresses G17 and G9; G11 through the robustness test.", "None."),
 "Originality": (3, "Relating the pooling choice to lesion size across modalities gives a testable hypothesis beyond a sweep.", "None."),
 "Feasibility": (5, "Four retrainings with one-line changes.", "None."),
 "Significance": (3, "Useful for attribution and robustness, with modest expected effect sizes.", "None."),
},
"D1": {
 "Clarity": (4, "Clear options, thresholds and scaling axes; the Gaussianity test is well motivated.", "Name the Gaussianity test (e.g. Mardia's or a per-dimension check) to keep it reproducible."),
 "Relevance": (4, "Addresses G13 and the compactness part of G7.", "None."),
 "Originality": (3, "Testing whether adapted features are more Gaussian than raw CLIP features is a new diagnostic; the compact models themselves are known.", "None."),
 "Feasibility": (5, "No retraining; fits are seconds.", "None."),
 "Significance": (3, "K = 64 and 448-pixel inputs are outside MVFA's protocol, so the practical relevance of the cost gains is limited.", "Justify the extended regime with a clinical scenario."),
},
"D2": {
 "Clarity": (4, "Clear set-up with frozen adapters, fixed token count and a fixed set of wordings.", "Specify how the ten wordings are chosen (e.g. clinical vs generic)."),
 "Relevance": (4, "Addresses G5; G7 is no longer claimed, which is appropriate.", "None."),
 "Originality": (3, "Cross-modality transfer of learned prompts under a modality-disjoint split is not isolated in the review set, although learned prompts themselves are common.", "None."),
 "Feasibility": (5, "Training only prompt tokens on frozen adapters is cheap.", "None."),
 "Significance": (3, "Sensitivity reduction is useful, but the impact is limited to the text side of one method.", "None."),
},
}
out = {}
for cid, crits in R.items():
    for m, (score, review, fb) in crits.items():
        out[f"stage1/review/{cid}/v1/{m}"] = f"Review: {review}\nFeedback: {fb}\nRating (1-5): {score}"
json.dump(out, open("logs/answers/run2/s1_review_v1.json", "w"), indent=1)
