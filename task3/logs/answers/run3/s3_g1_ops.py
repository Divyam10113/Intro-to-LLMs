import json
MAD = "MadCLIP: Few-shot Medical Anomaly Detection with CLIP"
RECAP = "Beyond Static Anchors: Bounded Prototype Conditioning for Language-Free Medical Anomaly Detection"
MLCA = "Fine-Tuning CLIP Models with Convolutional Features for Medical Anomaly Detection"
SFAD = "Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Images"
FED = "Fed-LoRA: Cross-Domain Generative Knowledge Feedback for Few-Shot Medical Anomaly Detection"
ADAN = "Anatomic-Decoupling Adaptation: A Unified Framework for Zero/Few-Shot Medical Anomaly Detection"
DWM = "Do Words Matter? What Text Contributes to Few and Full-Shot Vision-Language Anomaly Detection"
PA = ("Leave-one-modality-out zero-shot training on BMAD (RESC and OCT17 held out together); all source masks discarded; checkpoints "
      "selected on source-modality validation by image AUROC; mask-supervised MVFA and the problem's plain top-k MIL re-run under the "
      "same rule; 3 seeds, mean and std; pixel AUROC and AUPRO on held-out BrainMRI, LiverCT, RESC, image AUROC on all folds. ")
PB = ("Leave-one-modality-out on BMAD (RESC and OCT17 held out together), MVFA retrained with source-validation selection; K in "
      "{2, 4, 8, 16}; 3 seeds x 3 support draws; 0.5 fusion and max pooling unchanged. Baselines: transductive MVFA pipeline, "
      "support percentiles (problem statement). ")
O = {}
def off(tag, t, s, h, r, m, e, p):
    O[tag] = "```json\n" + json.dumps({"Title": t, "Summary": s, "Hypothesis": h, "Rationale": r, "Method": m, "Experiment": e,
                                       "Provenance": p}, indent=1, ensure_ascii=False) + "\n```"

off("stage3/A1/g1/o1/inspiration",
 "Cross-level consensus MIL: choosing instances by agreement of the four adapter levels",
 "Select the positive instances of an abnormal image as the patches on which all four segmentation levels agree (geometric mean of level probabilities) and apply the MIL loss to those patches at every level, in a single training stage.",
 "Training MVFA's segmentation adapters with consensus MIL — top-k (k = 1%) selected on the geometric mean of the four level maps rather than per level — raises held-out pixel AUROC by at least 2 points and AUPRO by at least 3 points over per-level top-k MIL, because single-level top-k selections disagree and each level reinforces its own spurious maxima.",
 "The self-distillation parent shows that cross-level agreement filters noisy evidence, but uses it only after training, in a second stage that can lock in errors. The anchor-frame parent shows that all four levels read their patches in the same fixed anchor frame, so their probabilities are directly comparable across levels. Together they suggest a different mechanism: use cross-level agreement inside the MIL loss itself, as the instance selector. Per-level top-k lets each level pick its own (often anatomical) maxima; a consensus selector picks only patches that every level finds anomalous, which are more likely to be lesions, and then trains every level on those patches. This is one-stage, needs no pseudo-masks, and is a direct modification of the problem's operator.",
 "1) loss.py: consensus_mil(p_levels, label, k_frac=0.01): c = exp(mean_l log p_l) per patch; idx = topk(c); positive loss for abnormal images = mean over levels of BCE(mean(p_l[idx]), 1); normal images: BCE(top-k mean of p_l, 0) per level as in the problem. 2) train_zero.py: replace the problem's per-level top-k MIL with consensus_mil; selection indices are detached. 3) Hyper-parameters: k_frac {0.5%, 1%, 2%}; consensus by geometric mean (ablation: minimum over levels); released learning rate and epochs. 4) Inference unchanged.",
 PA + "Ablations: geometric mean vs. minimum; consensus over levels 2-4 only. Diagnostics: overlap of per-level top-k selections; fraction of selected patches inside true masks on masked source images. Falsified if consensus MIL does not beat per-level MIL in AUPRO.",
 [MLCA, DWM])

off("stage3/A1/g1/o3/inspiration",
 "Equivariance-regularised MIL: dense self-supervision from geometric transforms",
 "Densify the sparse MIL signal by requiring the segmentation maps of an image and of its flipped/rotated/cropped copy to agree after inverse transformation, using no pseudo-masks and no extra backbone.",
 "Adding a transformation-equivariance consistency loss (weight 1.0) to top-k MIL increases held-out AUPRO by at least 3 points and reduces the number of connected components per abnormal map by at least 30%, matching or exceeding two-stage pseudo-mask refinement without its extra stage.",
 "Both parents densify MIL supervision through an external target — pseudo-masks from the model itself or DINOv2 affinities — and both add a stage whose errors are frozen into the next. A different source of dense supervision is available in every image: a lesion moves with the image under flips, small rotations and crops, so the correct anomaly map is equivariant. Requiring the predicted maps of two views to agree after alignment constrains every patch, not only the top-k, and spreads MIL evidence over the whole lesion without any target map. It is a single-stage, single-backbone change.",
 "1) train_zero.py: for each image draw a transform T (horizontal flip, rotation in [-15, 15] degrees, random resized crop 0.8-1.0); forward both x and T(x) through the seg branch. 2) loss.py: equi_loss = mean over levels of || T(p_l(x)) - p_l(T(x)) ||_1 over valid (overlapping) pixels, stop-gradient on neither. 3) Total seg loss = top-k MIL (problem) + 1.0 x equi_loss. 4) Hyper-parameters: weight {0.3, 1, 3}; transform family ablation. Training cost roughly doubles per step; epochs as released.",
 PA + "Baselines also include two-stage MIL-to-mask self-distillation. Metrics: pixel AUROC, AUPRO, connected components per map, image AUROC. Falsified if AUPRO does not improve over MIL alone.",
 [MLCA, SFAD])

off("stage3/A1/g1/o5/combination",
 "Anatomy-debiased MIL under text and random anchor frames",
 "Train the segmentation adapters with top-k MIL plus hard-negative mining of the top-k patches of normal images, under MVFA's text anchors and under scale-matched random anchors, to test both whether debiasing improves weak localisation and whether its effect depends on the meaning of the anchors.",
 "Normal hard-negative mining raises held-out pixel AUROC of MIL-trained segmentation adapters by at least 2 points over plain top-k MIL, and the gain is the same (within 1 point) under text and scale-matched random anchors, showing that it acts on the visual adapters rather than on anchor semantics.",
 "The anatomy parent identifies a concrete failure of MIL (top-k patches on salient anatomy) and corrects it with image labels alone; its positional prior is fragile on unaligned modalities, so only the hard-negative term is kept, which needs no alignment. The anchor-frame parent provides the control that separates the contribution of text meaning from that of a fixed frame and its scale. Combining them gives one improvement and one explanation in a 2 x 2 design whose cells are each one flag in the training script.",
 "1) loss.py: mil_hardneg(p_l, label, k_frac=0.01, w_hard=2): abnormal images: BCE(top-k mean, 1); normal images: BCE on all patches towards 0 plus w_hard x BCE(top-k mean, 0). 2) utils.py: make_anchors('text' | 'rand'), random pair rescaled to the text pair's difference length, 3 draws. 3) train_zero.py: flags --loss {mil, mil_hardneg} --anchors {text, rand}. 4) Hyper-parameters: w_hard {1, 2, 4}; released learning rate and epochs.",
 PA + "2 x 2 design (loss x anchors) plus mask-supervised MVFA with text anchors. Diagnostic: fraction of top-k patches of abnormal source images inside true masks. Falsified if hard negatives do not improve pixel AUROC, or if their gain differs by more than 1 point between frames.",
 [ADAN, DWM])

off("stage3/B1/g1/o1/combination",
 "Position-wise low anchor and synthetic high anchor for inductive MVFA scoring",
 "Normalise each branch per grid position between a shrunk position-wise median of leave-one-out support scores (low anchor) and the median score of synthetic lesions pasted into the supports (high anchor), keeping MVFA's 0.5 fusion and max pooling.",
 "On spatially aligned held-out modalities, the position-wise/synthetic two-anchor normaliser keeps image and pixel AUROC within 1 point of the transductive pipeline at K >= 4 and improves pixel AUROC by at least 1 point over global support percentiles.",
 "The position-wise parent removes anatomy-driven offsets but has no principled upper scale; the synthetic-anchor parent gives an upper scale but treats all positions alike. Using the position-wise statistic as the low anchor and the synthetic lesion score as a global high anchor gives a normaliser that is both anatomy-aware and on a common scale across branches, without any test-set statistic.",
 "1) utils.py: positional_low(support_maps): per-position LOO medians with 3x3 neighbourhood and shrinkage 20 towards the global median; synthetic_high(model, supports): 20 Poisson-blended lesions per support (1-5% area), median lesion-pixel score per branch and level. 2) test_few.py: z = clip((s_j - lo_j)/(hi - lo_j), -1, 3) rescaled to [0, 1]; 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baselines: position-wise only; synthetic two-anchor only. Metrics: image/pixel AUROC, sensitivity at 95% specificity, per-position false-positive maps. Falsified if no gain over global percentiles on aligned modalities.",
 [ADAN, FED])

off("stage3/B1/g1/o2/combination",
 "Position-wise leave-one-out empirical CDF with neighbourhood pooling",
 "Map each patch score through the empirical CDF of leave-one-out support scores from the same grid neighbourhood, mixed with the global support CDF according to how many reference values the neighbourhood holds.",
 "At K >= 4 on spatially aligned held-out modalities, the position-wise empirical CDF improves pixel AUROC by at least 1 point over the global empirical CDF and keeps image AUROC within 1 point of the transductive pipeline, while every score stays exactly invariant to the test set.",
 "The empirical-CDF parent gives an assumption-free, test-set-invariant scale but ignores anatomy; the position-wise parent removes anatomy offsets but relies on mean/standard-deviation statistics from very few values. Their combination keeps the rank-based CDF and makes the reference set local: for position j the references are all LOO support patch scores within a 3x3 (level 4: 5x5) neighbourhood, and the local CDF is mixed with the global one with weight n_j/(n_j + 50).",
 "1) utils.py: build_local_refs(support_maps, radius): sorted arrays per position; global array as in the parent. 2) cdf_mix(s, R_local, R_global) = w F_local(s) + (1 - w) F_global(s), w = n/(n + 50). 3) test_few.py: apply per branch and level, then 0.5 fusion and max pooling unchanged; test-set min-max removed. Hyper-parameters: radius {1, 2}, mixing constant {10, 50, 200}. No training.",
 PB + "Additional baseline: global empirical CDF. Metrics: image/pixel AUROC, sensitivity at 95% specificity. Falsified if the local CDF is not better than the global CDF in pixel AUROC on aligned modalities.",
 [MAD, RECAP, ADAN])

off("stage3/B1/g1/o3/combination",
 "Empirical CDF of anatomy residuals for inductive MVFA scoring",
 "Subtract a shrunk position-wise median of support scores from every patch score and map the residual through one global leave-one-out empirical CDF of support residuals, then fuse with MVFA's unchanged rule.",
 "Mapping position-wise residuals through a global empirical CDF matches the global empirical CDF in image AUROC (within 1 point) and improves pixel AUROC by at least 1 point on aligned modalities, with lower variance across support draws at K = 2 than the position-wise CDF.",
 "This combination takes the positional part from the position-wise parent only as a location correction, and the scale from the empirical-CDF parent as a single global reference. Separating location (local, needs few values) from scale (global, needs many) avoids estimating local spreads from a handful of supports, which is the weak point of both a position-wise z-score and a local CDF at K = 2.",
 "1) utils.py: positional_median(support_maps) with 3x3 neighbourhood and shrinkage 20; residual reference set R = {s - med_j} over all LOO support patches. 2) test_few.py: r = s_j - med_j; F(r) via searchsorted on R; 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baselines: global empirical CDF; position-wise CDF. Metric of stability: standard deviation of AUROC over support draws. Falsified if residual CDF is not more stable than the position-wise CDF at K = 2.",
 [MAD, RECAP, ADAN])

off("stage3/B1/g1/o4/inspiration",
 "Matched-support normalisation: normalise each test image against its most similar supports",
 "Instead of per-position statistics (which assume alignment) or a global analytic scale, normalise each test image's branch scores with the leave-one-out scores of the m supports whose global CLIP embedding is closest to it.",
 "Normalising with the m = min(3, K) nearest supports improves pixel AUROC by at least 1 point over global support percentiles at K >= 8 on all six held-out datasets, including the unaligned ChestXray and HIS where position-wise normalisation does not apply.",
 "The position-wise parent captures that normal score levels vary with anatomy, but assumes images are aligned; the anchor-geometry parent shows that a branch's scale is set by global properties of the representation. A different mechanism follows: anatomy and acquisition vary per image, and the supports that look most like a test image carry the most relevant normal score distribution. Choosing the reference per test image by global embedding similarity handles variation without spatial alignment and stays inductive, since it uses only the supports.",
 "1) utils.py: support_cls(supports): CLS embeddings of the supports (frozen CLIP). 2) test_few.py: for a test image, cosine similarity of its CLS to the supports; select top m; reference set = their LOO patch scores per branch and level; normalise with 1st/99th percentiles of that set; 0.5 fusion and max pooling unchanged; test-set min-max removed. 3) Hyper-parameters: m in {1, 3, 5}. No training.",
 PB + "Additional baseline: position-wise normalisation. Metrics: image/pixel AUROC, sensitivity at 95% specificity. Falsified if no gain over global percentiles at K >= 8.",
 [ADAN, DWM])

off("stage3/B1/g1/o6/inspiration",
 "Analytic zero-shot scale plus support CDF for the few-shot branch",
 "Normalise MVFA's zero-shot branch without any target data, from the anchor geometry and source-validation statistics, and only the few-shot branch from the K supports via a leave-one-out empirical CDF.",
 "With the zero-shot branch normalised analytically and the few-shot branch by the support CDF, fused image AUROC stays within 1 point of the transductive pipeline at K >= 4, and at K = 2 it loses at least 1 point less than normalising both branches from the supports.",
 "The anchor-geometry parent shows that the zero-shot branch reads the anchors only through their difference, so its scale is a property of the trained model, not of the target. The CDF parent gives a robust support-based scale where target data are needed. A different allocation follows: the zero-shot branch needs no target statistics at all (its normaliser can be fixed on source validation), and the scarce K supports are spent only on the memory-bank branch, whose scale genuinely depends on the target. This halves the number of quantities estimated from K images.",
 "1) calibrate_zero.py: on source validation splits, compute zero-shot patch projections on the unit anchor-difference direction per level; store median and MAD of normal patches. 2) test_few.py: zero-shot z = (proj - median_src)/MAD_src mapped by a logistic; few-shot via LOO empirical CDF of support scores; 0.5 fusion and max pooling unchanged; test-set min-max removed. No adapter training.",
 PB + "Additional baselines: both branches by support CDF; both by support percentiles. Falsified if the hybrid is not better than both-support normalisation at K = 2.",
 [DWM, MAD, RECAP])

json.dump(O, open("/home/claude/task3/logs/answers/run3/s3_g1_ops.json", "w"), indent=1, ensure_ascii=False)
print(len(O))
