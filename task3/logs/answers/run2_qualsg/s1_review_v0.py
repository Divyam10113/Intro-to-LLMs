import json
R = {
"A1": {
 "Clarity": (4, "The component (segmentation-adapter targets), the supervision (image-level labels on six sources), the pseudo-mask rule and the outcome (fraction of mask-supervised pixel AUROC and Dice recovered) are all named. The pseudo-mask rule leaves its threshold ('no close neighbour') and the feature level it is computed on undefined.", "State the pseudo-mask rule operationally: which CLIP level, which distance, and how the threshold is chosen without target labels (for example a percentile of normal-to-normal distances on source validation data)."),
 "Relevance": (5, "Directly addresses G3, the dependence of localisation on dense masks from three datasets, and uses the three image-only BMAD sources that MVFA currently cannot use for localisation. It builds on CoDeGraph's pseudo-mask supervision and the weak-supervision line in the review set.", "Make the link to G4 explicit only if the same mechanism is also tested in the few-shot setting; otherwise keep the focus on zero-shot source training."),
 "Originality": (3, "Pseudo-masks from patch-level nearest-neighbour discrepancy are known (PatchCore-style scoring, CoDeGraph uses batch-level graphs to supervise prompt-driven VLMs), and weakly supervised localisation is a mature area. The new element is the question of whether MVFA's multi-level adapters can be trained this way and what the image-only datasets contribute; that is a useful but moderate step.", "Sharpen what is new relative to CoDeGraph: for example, that the pseudo-masks come from labelled abnormal images rather than unlabeled batches, so image-level labels can veto false positives."),
 "Feasibility": (4, "Pseudo-masks are computed once with frozen features and then replace the mask loader; training cost equals MVFA's. The risk is pseudo-mask quality on low-contrast modalities (ChestXray, HIS), which could make the comparison uninformative.", "Plan a pseudo-mask quality check on the three masked datasets (pseudo vs true mask IoU) before training, so a negative result can be attributed."),
 "Significance": (4, "Removing the need for dense masks would let MVFA use any image-level labelled archive and would show how much expert annotation actually buys; either outcome is informative for clinical adoption.", "Report results as a label-cost curve (no masks, masks on one dataset, masks on three) to make the significance concrete."),
},
"A2": {
 "Clarity": (4, "Clear setting (few-shot, K in 2-16), clear change (drop abnormal masked supports, add synthesised lesions) and clear outcome (fraction of K+K AUROC retained). The synthesis recipe is described only loosely.", "Fix the synthesis families and their parameters in the problem, or state that they follow MediCLIP's published recipe."),
 "Relevance": (5, "Targets G4 directly: the few-shot protocol's hidden reliance on masked abnormal images is the main gap between MVFA's claims and clinical few-shot reality.", "Mention that the same 3-draw protocol answers the single-draw part of G4."),
 "Originality": (2, "Normal-only few-shot adaptation of CLIP with anomaly synthesis is exactly what MediCLIP and SpatioPrompt study; the problem transfers their setting to MVFA's adapters. The contribution is an ablation of MVFA's protocol rather than a new challenge.", "Add a distinctive angle, for example whether synthetic supervision should be applied only to the segmentation branch or to both branches, or how synthesis interacts with MVFA's zero-shot pre-training."),
 "Feasibility": (5, "Few-shot fine-tuning on K images is minutes per run; synthesis is cheap; all comparisons use public code and data.", "None beyond fixing the synthesis recipe."),
 "Significance": (3, "Quantifying the retained performance is practically useful, but the expected result (synthesis recovers part of the gap) is predictable from MediCLIP and SpatioPrompt.", "Explain what decision the result would change, e.g. whether clinics need any annotated abnormal studies at all."),
},
"B1": {
 "Clarity": (4, "The three parts of the scoring pipeline that are replaced (test-set min-max, fixed 0.5 weight, max pooling) and the measurable criteria are precise. Bundling three changes makes it less clear which one drives any effect.", "Keep one problem but require an ablation order: normalisation first, then fusion weight, then aggregation."),
 "Relevance": (5, "Directly addresses G8 (transductive normalisation verified in the code), G9 (max-pixel image score) and G11 (no thresholded metrics). Every published few-shot number of MVFA depends on the component in question.", "None."),
 "Originality": (3, "Inductive score normalisation with validation statistics is standard practice in anomaly-detection tooling, and calibration of CLIP scores appears in CANOR. What is new is identifying that MVFA's reported results rely on transductive normalisation and measuring the deployment gap, including invariance to test-set composition.", "Emphasise the invariance test (re-scoring an image under resampled test sets) as the novel measurement."),
 "Feasibility": (5, "Inference-only: it reuses trained checkpoints and needs a few hours on one GPU.", "None."),
 "Significance": (4, "Turns MVFA into a detector that can score one image at a time and reports thresholded clinical metrics; if AUROC drops, it also corrects the published comparison.", "Report the per-image score shift as a headline number; it is what a deploying team needs."),
},
"B2": {
 "Clarity": (5, "One operator, applied consistently in training and testing, four named variants, two named metrics including a controlled spurious-pixel stress test.", "None."),
 "Relevance": (4, "Addresses G17 and G9 directly; G11 only through the robustness test.", "Tie the stress test to a clinical artefact (e.g. bright vessels or acquisition noise) to strengthen relevance."),
 "Originality": (2, "Choosing among max, mean, top-k and noisy-OR is a standard ablation; CANOR already proposes noisy-OR aggregation for CLIP chest X-ray scores. The paper/code mismatch is a reproducibility finding rather than a research challenge.", "Frame a hypothesis about why one pooling suits MVFA (e.g. lesion size distribution per modality) to go beyond a sweep."),
 "Feasibility": (5, "One-line changes in training and testing; four retrainings.", "None."),
 "Significance": (3, "Resolves an attribution question for MVFA and gives a robustness number, but the likely effect sizes are modest.", "Report results stratified by lesion size, where the choice should matter most."),
},
"D1": {
 "Clarity": (4, "Clear component (few-shot memory bank), two named replacements, clear outcomes including memory and latency as functions of K.", "Specify how prototypes are obtained (k-means per level, number of prototypes) so the comparison is reproducible."),
 "Relevance": (4, "Addresses G13 (memory, retrieval cost, storage of patient features) and part of G7 (explicit compact normal model); G5 is not touched.", "State the privacy benefit precisely: a parametric model stores statistics rather than raw patch features."),
 "Originality": (3, "Gaussian normal models (PaDiM) and compact prototypes (Spatial-FAD) are known; applying them to MVFA's adapted multi-level features with a K-scaling analysis adds a moderate new angle.", "Add the question of whether the adapted features are more Gaussian than raw CLIP features, which would explain when compact models suffice."),
 "Feasibility": (5, "No retraining; fitting a Gaussian or k-means on support features is seconds.", "None."),
 "Significance": (3, "Useful for deployment cost and privacy, but the bank is small at K <= 16, so practical gains may be limited.", "Include a high-resolution or larger-K setting where the bank's cost actually matters."),
},
"D2": {
 "Clarity": (4, "Clear change (learnable, visually conditioned prompt tokens), clear setting (zero-shot, held-out modality) and two outcomes including prompt-wording sensitivity.", "Define the conditioning mechanism and the size of the token set, and the fixed list of alternative wordings."),
 "Relevance": (4, "Addresses G5 directly and G7 only loosely.", "Drop the G7 claim or explain how conditioned prompts model the normal/abnormal asymmetry."),
 "Originality": (2, "Learnable and visually conditioned prompts are the dominant design in the review set (AnomalyCLIP, AdaCLIP, CoPS, VCF-CLIP, IQE-CLIP). Inserting one into MVFA is an incremental combination.", "Focus on an under-studied aspect, e.g. whether learned prompts transfer across modalities in the leave-one-modality-out split, which none of these papers isolates."),
 "Feasibility": (4, "Requires training text tokens jointly with adapters; feasible on one GPU but adds hyper-parameters.", "Freeze the visual adapters from MVFA's checkpoint and train only prompts to keep cost low."),
 "Significance": (3, "Expected gains are modest because MVFA's adapters already align features with fixed prompts.", "Report the sensitivity reduction as the main outcome; it is the part not covered by prior work."),
},
}
out = {}
for cid, crits in R.items():
    for m, (score, review, fb) in crits.items():
        out[f"stage1/review/{cid}/v0/{m}"] = f"Review: {review}\nFeedback: {fb}\nRating (1-5): {score}"
json.dump(out, open("logs/answers/run2/s1_review_v0.json", "w"), indent=1)
