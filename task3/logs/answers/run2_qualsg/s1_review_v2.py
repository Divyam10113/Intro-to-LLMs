import json
R = {
"A1": {
 "Clarity": (5, "One question with an operational pseudo-mask rule, a threshold fixed on source data, and a precisely defined primary outcome (label-cost curve) with a secondary analysis clearly separated.", "None."),
 "Relevance": (5, "Directly addresses G3, the dependence of MVFA's localisation on dense masks.", "None."),
 "Originality": (3, "The label-restricted pseudo-mask construction is a reasonable twist on known nearest-neighbour pseudo-labelling; the question asked of MVFA is new but the mechanism is not.", "None at this stage."),
 "Feasibility": (5, "Pseudo-masks computed once; training cost equals MVFA's; the risky image-only extension is now secondary, and the quality check makes negative results interpretable.", "None."),
 "Significance": (4, "Tells practitioners how much expert annotation MVFA needs and whether image-level archives suffice.", "None."),
},
"A2": {
 "Clarity": (5, "A single comparison (which branch receives synthetic supervision) with a fixed recipe and an actionable outcome (smallest K matching K+K).", "None."),
 "Relevance": (5, "Directly addresses G4, including the single-draw issue, and the absence of a normal-only route in G3.", "None."),
 "Originality": (3, "The branch question is specific to MVFA's two-branch design; the normal-only synthetic setting is established.", "None."),
 "Feasibility": (5, "Minutes per run; public code and data.", "None."),
 "Significance": (3, "Actionable for MVFA users, but the impact on the wider field is limited because the setting is already studied in single-branch methods.", "None."),
},
"B1": {
 "Clarity": (4, "The lead question is now clear, and the component-wise evaluation is specified. The statement remains long and carries several metrics and resampling settings in one sentence.", "Move the resampling settings into the rationale to keep the problem statement short."),
 "Relevance": (5, "Addresses G8, G9 and G11 directly, on the scoring component behind all published few-shot results.", "None."),
 "Originality": (3, "The score-invariance diagnostic is new for medical CLIP detectors; inductive normalisation itself is established.", "None."),
 "Feasibility": (5, "Inference-only on trained checkpoints.", "None."),
 "Significance": (4, "Quantifies the deployment gap of a widely cited method and adds clinically interpretable metrics.", "None."),
},
"B2": {
 "Clarity": (4, "Clear variants, hypothesis and stratification; the injection of spurious pixels needs a quantitative specification (number, intensity).", "Specify the injection protocol numerically."),
 "Relevance": (4, "Addresses G17 and G9 directly.", "None."),
 "Originality": (3, "The lesion-size hypothesis turns a sweep into a testable question.", "None."),
 "Feasibility": (5, "Four retrainings with one-line changes.", "None."),
 "Significance": (3, "Resolves a paper/code discrepancy and informs pooling choices, with modest expected effects.", "None."),
},
"D1": {
 "Clarity": (4, "Clear options, test and scaling axes; the 'stored patient information' outcome is not operationalised.", "Define it, e.g. as bytes stored per patient or feasibility of reconstructing patches from stored statistics."),
 "Relevance": (4, "Addresses G13 and the compactness aspect of G7.", "None."),
 "Originality": (3, "The Gaussianity diagnostic on adapted features is new; compact normal models are known.", "None."),
 "Feasibility": (5, "No retraining.", "None."),
 "Significance": (3, "Mostly relevant to long-term deployment scenarios outside MVFA's few-shot protocol.", "None."),
},
"D2": {
 "Clarity": (5, "Frozen adapters, fixed token count, conditioning option and a fixed set of ten wordings make the study precise.", "None."),
 "Relevance": (4, "Addresses G5 directly.", "None."),
 "Originality": (3, "Cross-modality transfer of learned prompts under a modality-disjoint split is not isolated in prior work, though learned prompts are common.", "None."),
 "Feasibility": (5, "Cheap prompt-only training.", "None."),
 "Significance": (3, "Limited to the text side of one method; useful mainly as a robustness finding.", "None."),
},
}
out = {}
for cid, crits in R.items():
    for m, (score, review, fb) in crits.items():
        out[f"stage1/review/{cid}/v2/{m}"] = f"Review: {review}\nFeedback: {fb}\nRating (1-5): {score}"
json.dump(out, open("logs/answers/run2/s1_review_v2.json", "w"), indent=1)
