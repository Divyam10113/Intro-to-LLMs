import json
def R(review, feedback, rating):
    return f"Review: {review}\nFeedback: {feedback}\nRating (1-5): {rating}"
A = {}
# ---------------- A1: image-level (MIL) supervision for MVFA adapters
A["stage1/review/A1/v0/Clarity"] = R(
 "The problem names the exact component (the alignment loss of the four adapter stages), the change (image-level labels only, masks discarded), the setting (zero-shot, leave-one-modality-out on BMAD) and the outcomes (image AUROC, pixel AUROC, PRO, retention ratio versus mask-supervised MVFA). The scope and boundaries are easy to grasp.",
 "Fix the mechanism less loosely: 'for example, multiple-instance pooling' leaves open whether the study is about MIL specifically or about any weak-label objective. State the primary pooling operator (e.g., top-k mean of patch scores) and treat others as ablations. Also specify which BMAD source datasets actually have image-level labels for both classes in their training split.",
 4)
A["stage1/review/A1/v0/Relevance"] = R(
 "Annotation cost is the central obstacle named in the target paper's own motivation and in most related medical works (MedicalCLIP, MedCLIP-SAMv2, MadCLIP). The problem builds directly on MVFA's architecture and benchmark and asks a question the field has avoided: recent follow-ups (MedSAD-CLIP, MLCA) increase rather than decrease pixel supervision.",
 "Make the connection to weakly-supervised segmentation literature explicit so that the contribution is positioned as transferring weak supervision to multi-level VLM adaptation rather than inventing weak supervision.",
 5)
A["stage1/review/A1/v0/Originality"] = R(
 "Learning localisation from image-level labels is well established outside this area (multiple-instance learning, class-activation-map based weakly supervised segmentation, MIL in video anomaly detection). What is new is applying it to the patch-to-text similarity maps of a multi-level CLIP adapter and testing whether the resulting localisation transfers to unseen modalities. That is a moderately novel combination rather than a new challenge.",
 "Sharpen the novel element: for instance, ask whether text-anchored patch similarities make MIL localisation transfer better than visual-only MIL, or whether multi-level supervision changes where localisation emerges. Without such a hook the problem risks reading as 'MIL applied to MVFA'.",
 3)
A["stage1/review/A1/v0/Feasibility"] = R(
 "Highly feasible: the change is a loss replacement in public code, BMAD provides image-level labels for every dataset, training cost equals MVFA's (which ran on one RTX 3090), and the evaluation reuses MVFA's metrics. The only risk is that MIL training collapses to trivial maps, which is itself a measurable outcome.",
 "Plan a short pilot on a single held-out modality to verify that the weak loss produces non-trivial maps before running all six leave-one-out folds and three seeds.",
 5)
A["stage1/review/A1/v0/Significance"] = R(
 "If a large fraction of mask-supervised zero-shot performance can be retained from image-level labels, MVFA-style adaptation becomes trainable on report-labelled hospital data, which substantially widens its practical applicability. A negative result is also informative because it quantifies the value of masks.",
 "Strengthen significance by also measuring what happens when image-labelled data is plentiful (more images without masks versus fewer images with masks), which is the realistic trade-off for clinical sites.",
 4)
# ---------------- A2: transductive self-training on unlabelled target images
A["stage1/review/A2/v0/Clarity"] = R(
 "The setting (unlabelled target batch, pseudo-labels from confident regions, adapters only, confident-pixel losses) and outcomes (AUROC, PRO as functions of N and thresholds, versus zero-shot and few-shot MVFA) are clearly stated.",
 "Define how the target data is split into the unlabelled adaptation pool and the disjoint test set for each BMAD modality, since some BMAD test sets are small, and define the confidence thresholds (quantiles of the zero-shot scores) up front.",
 4)
A["stage1/review/A2/v0/Relevance"] = R(
 "Unlabelled target data under modality shift is a realistic deployment condition and connects to G4. The problem is well connected to AnoCLIP's test-time adaptation and to MVFA's zero-/few-shot comparison.",
 "Relate it to source-free domain adaptation, which is the established name for this setting.",
 4)
A["stage1/review/A2/v0/Originality"] = R(
 "Self-training on pseudo-labels and test-time adaptation are standard; AnoCLIP already performs pseudo-label adaptation for CLIP-based localisation. Moving it to multi-level medical adapters and studying confirmation bias under modality shift adds some insight but the core idea is familiar.",
 "Identify an element that is specific to the medical, multi-level setting — for example, using agreement across the four feature levels as the confidence criterion — to differentiate the problem from generic self-training.",
 3)
A["stage1/review/A2/v0/Feasibility"] = R(
 "Implementable in the MVFA code and cheap to run, but several BMAD modalities have few test images, so carving out an unlabelled adaptation pool reduces the test set and widens confidence intervals; pseudo-label quality on modalities where zero-shot AUROC is already low (e.g., chest X-ray) may be too poor to bootstrap.",
 "Report how the adaptation/test split interacts with variance, and include a modality with weak zero-shot performance as a stress test.",
 4)
A["stage1/review/A2/v0/Significance"] = R(
 "Useful for deployment, but the expected outcome is an incremental shift along the zero-shot/few-shot continuum; it does not change what supervision MVFA needs for its source training (G3), which is the larger barrier.",
 "Clarify what decision a clinic could make from the result (e.g., how many unlabelled scans are needed to match K labelled ones).",
 3)
# ---------------- B1: support-set normalisation + robust pooling (few-shot)
A["stage1/review/B1/v0/Clarity"] = R(
 "The defect being fixed is described precisely (scales [0,1] and [0,2], fixed weight 0.5, max pooling), and the outcomes are concrete (ECE, sensitivity at 95% specificity, F1 at a pre-fixed threshold, AUROC). However, the problem bundles two interventions — branch normalisation and aggregation — and it is not stated which is the primary question.",
 "Make the primary question the transferability of a single threshold, and frame normalisation and robust pooling as the two factors of a 2x2 design so their individual effects are identifiable. State how the 'threshold fixed before seeing any target label' is chosen (on source modalities? on support normals?).",
 4)
A["stage1/review/B1/v0/Relevance"] = R(
 "Directly addresses G8, G9 and G11 of the target paper and the general lack of thresholded evaluation in the related CLIP-based anomaly detection works. Relevant to anyone deploying few-shot detectors.",
 "Connect explicitly to score normalisation practice in PatchCore-style methods and to AnomalyGPT's threshold-free claim.",
 4)
A["stage1/review/B1/v0/Originality"] = R(
 "Score standardisation before fusion (z-normalisation) is a classical technique, and robust pooling such as top-k mean is common in anomaly detection. The genuinely fresh part is using leave-one-out scores on the K normal support images as a label-free per-modality reference and evaluating threshold transfer, which most papers in this area ignore.",
 "Put the threshold-transfer question and the leave-one-out support reference at the centre to make the contribution clearly distinct from generic normalisation.",
 3)
A["stage1/review/B1/v0/Feasibility"] = R(
 "Inference-only on existing MVFA checkpoints; all quantities are computable from BMAD labels; the multi-seed, multi-draw protocol is cheap because no retraining is needed.",
 "None significant; ensure that K normal support images exist for K=2 (the leave-one-out distribution from two images is degenerate — consider pooling patch-level scores).",
 5)
A["stage1/review/B1/v0/Significance"] = R(
 "A practical improvement for deployment of few-shot detectors, but restricted to the few-shot regime and to score post-processing; the contribution to the field is useful but narrow.",
 "Broaden the significance by showing that the same normalisation makes MVFA comparable with other few-shot methods at fixed operating points, or that it changes conclusions drawn from AUROC alone.",
 3)
# ---------------- B2: conformal calibration under modality shift (zero-shot)
A["stage1/review/B2/v0/Clarity"] = R(
 "Clear target (realised FPR versus a nominal bound on an unseen modality), clear procedure (split conformal on held-out source modalities), clear comparison (max versus robust pooling) and clear metrics (FPR, sensitivity, coverage violation, ECE).",
 "Specify the calibration set precisely: normal images from which source modalities, and whether calibration is pooled across modalities or done per modality and then aggregated. State the nominal levels to be tested (e.g., 1%, 5%, 10%).",
 4)
A["stage1/review/B2/v0/Relevance"] = R(
 "Deployment of zero-shot detectors requires thresholds without target labels; this is the natural, under-studied counterpart to the AUROC-centred evaluation of MVFA and all related CLIP-based works.",
 "Relate to the distribution-shift conformal literature (weighted conformal prediction) to show awareness of existing corrections.",
 4)
A["stage1/review/B2/v0/Originality"] = R(
 "Conformal anomaly detection exists, but asking whether conformal false-positive guarantees survive a modality shift for VLM-based medical anomaly scores, and whether pixel-aggregation choice affects the violation, is a distinct perspective not explored in the target paper or the related works.",
 "Consider adding a corrective element (e.g., using the most dissimilar source modality as calibration, or a modality-similarity weighting) so the problem is not only diagnostic.",
 4)
A["stage1/review/B2/v0/Feasibility"] = R(
 "Inference-only; calibration requires only normal images from source modalities, which BMAD provides; computation is trivial on one GPU.",
 "Ensure enough normal calibration images per source modality for stable quantiles at the 1% level.",
 5)
A["stage1/review/B2/v0/Significance"] = R(
 "A safety-relevant question: whether a zero-shot medical detector can be deployed with a controlled false-alarm rate. Results generalise beyond MVFA to any CLIP-based zero-shot detector, and a negative result (systematic violation) is an important warning.",
 "Report the result per modality pair to make it actionable (which shifts break the guarantee).",
 4)
# ---------------- D1: prototype memory for few-shot branch
A["stage1/review/D1/v0/Clarity"] = R(
 "The replacement (M learned prototypes per level instead of all support patches) and outcomes (AUROC versus M, memory, retrieval time, comparison of prototype versus nearest-neighbour distances) are well defined.",
 "Specify how prototypes are fitted (k-means versus learned) and which M values are tested.",
 4)
A["stage1/review/D1/v0/Relevance"] = R(
 "Relevant to G7 and G13 and to the few-shot branch of MVFA; related to APRIL-GAN and MadCLIP's memory-free design.",
 "Explain why compressing a few-shot memory bank matters in MVFA's regime.",
 4)
A["stage1/review/D1/v0/Originality"] = R(
 "Memory-bank compression is established: PatchCore's coreset subsampling and prototype-based normality models (including intrinsic normal prototype methods) already address exactly this. Applying k-means prototypes to MVFA's adapted features is a minor variation.",
 "Find an angle that existing compression methods do not cover, for example prototypes shared across modalities, or prototypes learned jointly with the adapters.",
 2)
A["stage1/review/D1/v0/Feasibility"] = R(
 "Straightforward to implement and evaluate on BMAD with one GPU.",
 "None.",
 5)
A["stage1/review/D1/v0/Significance"] = R(
 "The motivation is weak in MVFA's actual regime: with K between 2 and 16 support images at 240x240 resolution, the memory bank holds only a few thousand vectors per level, so memory and retrieval cost are not a practical bottleneck. The privacy argument is also limited, because centroids of a handful of images can still be close to individual patches. Gains are therefore likely to be small.",
 "Either move to a regime where memory matters (larger K, higher resolution, 3D volumes) or reframe the problem around modelling the compactness of normality, which is the more significant question in G7.",
 2)
# ---------------- D2: image-conditioned text anchor + template sensitivity
A["stage1/review/D2/v0/Clarity"] = R(
 "Clear description of the static anchor, the proposed conditioned anchor and the two-part study (sensitivity analysis, then conditioning).",
 "Two questions are bundled (template sensitivity and conditioning); make the sensitivity analysis a precondition and state the hypothesis for the conditioning step.",
 4)
A["stage1/review/D2/v0/Relevance"] = R(
 "Addresses G5 directly and is in line with the field's move to adaptive text sides (AnomalyCLIP, HieClip, AULoRA, text-adapted few-shot).",
 "None beyond positioning against those works.",
 4)
A["stage1/review/D2/v0/Originality"] = R(
 "Image-conditioned prompts (CoCoOp-style meta-networks) and visual-context injection into text (AULoRA, AdaCLIP's hybrid prompts) are already widespread; the problem mostly re-tests a known idea inside MVFA. The template-sensitivity analysis is useful but not novel.",
 "Differentiate by conditioning on something other than the image itself (e.g., modality metadata or a normal support prototype) or by studying asymmetric normal/abnormal anchors.",
 2)
A["stage1/review/D2/v0/Feasibility"] = R(
 "Small meta-network and cheap training; fully feasible on BMAD with one GPU.",
 "None.",
 5)
A["stage1/review/D2/v0/Significance"] = R(
 "Likely to produce modest AUROC gains in line with existing prompt-learning work; the sensitivity analysis has value for reproducibility of MVFA but limited broader impact.",
 "Explain what new understanding — beyond a performance number — the study would deliver.",
 3)
json.dump(A, open("/home/claude/task3/logs/answers/stage1_review_v0.json", "w"), indent=1)
print(len(A))
