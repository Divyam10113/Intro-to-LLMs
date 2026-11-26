import json
exec(open("/home/claude/task3/logs/answers/run3/s3_g1_ops.py").read().split("O = {}")[0])
O = {}
def off(tag, t, s, h, r, m, e, p):
    O[tag] = "```json\n" + json.dumps({"Title": t, "Summary": s, "Hypothesis": h, "Rationale": r, "Method": m, "Experiment": e,
                                       "Provenance": p}, indent=1, ensure_ascii=False) + "\n```"

off("stage3/A1/g3/o2/inspiration",
 "Per-image adaptive k from normal-bank exceedances",
 "Set the MIL instance count of each abnormal image to the number of its patches whose distance to a bank of normal patches of the same dataset exceeds the 99th percentile of normal-image distances, so k follows the lesion size of that image rather than a fixed or per-dataset value.",
 "Per-image k from normal-bank exceedances raises held-out pixel AUROC by at least 1.5 points over per-dataset witness-rate k and by at least 2 points over fixed k = 1%, and the per-image k correlates with true lesion area on masked source images (Spearman >= 0.5).",
 "The gated-MIL parent uses a normal bank to decide which patches to select; the witness-rate parent decides how many, but only per dataset. Lesion size varies strongly within a dataset (small and large tumours in BrainMRI, small and large fluid regions in RESC), so the right k is a property of the image. Counting the patches that are novel relative to the normal bank gives a label-free, per-image estimate of how many patches the lesion occupies; selection by anomaly score then trains the adapters on that many patches.",
 "1) build_normal_bank.py: per dataset, coreset of frozen level-4 normal patch features; threshold t_d = 99th percentile of leave-one-image-out distances of normal patches. 2) loss.py: k_i = clip(#{j: 1 - max cos(f_j, bank_d) > t_d}, 3, 30% of patches) for abnormal images (computed once per epoch, detached); MIL positive term = BCE(top-k_i mean of p, 1); normal images: top-k BCE towards 0 with k = 1%. 3) train_zero.py: per-image k used at every level. Inference unchanged.",
 PA + "Baselines: fixed k; per-dataset witness-rate k; gated MIL. Diagnostic: Spearman correlation of k_i with true lesion area on masked source images. Falsified if per-image k is not better than per-dataset k.",
 [ADAN, MAD, RECAP])

off("stage3/A1/g3/o4/inspiration",
 "Label-proportion MIL on mosaics: abnormal mass must scale with the number of abnormal quadrants",
 "Build mosaics with a known number of abnormal quadrants and add a learning-from-label-proportions loss: the predicted abnormal mass of a mosaic must equal the sum of the per-image abnormal masses of its quadrants' source images when each is seen alone, which ties the segmentation maps' area to image content across contexts.",
 "Adding the label-proportion consistency loss to mosaic MIL reduces the error of predicted lesion area (relative to true mask area on held-out masked modalities) by at least 25% and raises AUPRO by at least 2 points, while image AUROC changes by less than 1 point.",
 "Mosaic MIL gives free quadrant labels, and witness-rate MIL shows that the amount of lesion per image matters. A different mechanism combines these hints: learning from label proportions. If a mosaic contains n abnormal images, its abnormal mass should be the sum of the masses of those images seen individually, and the normal quadrants should contribute none. This consistency between single images and their mosaics constrains how much area the adapters mark as abnormal — a quantity no image-level loss controls and that AUPRO directly rewards — without any mask.",
 "1) dataset/mosaic.py as in mosaic MIL (2x2, 120 px quadrants, 1-3 abnormal). 2) For each mosaic, also forward its four source images individually (resized to 120 px, no gradient through the single-image pass). 3) loss.py: lp_loss = | sum_j p_j(mosaic) - sum_q sum_j p_j(image_q) | / number of patches, plus BCE towards 0 on normal quadrants and consensus top-k MIL inside abnormal quadrants. 4) Hyper-parameters: lp weight {0.1, 0.5}; mosaic probability 0.5. Inference unchanged.",
 PA + "Baselines: mosaic MIL; consensus MIL. Metric added: relative lesion-area error on held-out masked datasets. Falsified if the area error is not reduced.",
 [MLCA, DWM])

off("stage3/A1/g3/o3/combination",
 "Mosaic MIL with normal-bank-gated instance selection",
 "Train the segmentation adapters on four-image mosaics whose normal quadrants are pixel-level negatives, and select the positive instances inside abnormal quadrants by anomaly probability gated by distance to a normal-patch bank, so both the spatial contrast and the instance choice avoid normal anatomy.",
 "Gated mosaic MIL raises held-out pixel AUROC by at least 1.5 points over mosaic MIL with plain consensus selection and by at least 3 points over single-image top-k MIL.",
 "Mosaic MIL contributes free spatial contrast (normal quadrants as negatives); gated MIL contributes alignment-free debiasing of instance selection. Their failure modes are complementary: mosaics still select instances by raw score inside abnormal quadrants, and gated MIL never sees an image in which normal and abnormal regions coexist with known labels. Combining them addresses both in one single-stage loss.",
 "1) build_normal_bank.py as in gated MIL. 2) dataset/mosaic.py as in mosaic MIL. 3) loss.py: inside each abnormal quadrant select top-k (1% of quadrant patches) by p_j x g_j with g_j = 1 - max cos(f_j, bank_d) (detached); BCE towards 0 on all patches of normal quadrants; single images use gated MIL. 4) train_zero.py: mosaic probability 0.5. Inference unchanged.",
 PA + "Baselines: mosaic MIL; gated MIL; top-k MIL. Falsified if gated mosaic MIL does not beat both parents.",
 [MLCA, ADAN, MAD, RECAP])

off("stage3/A1/g3/o5/combination",
 "MPE-anchored posterior MIL",
 "Fix the mixing proportion of the posterior-weighted MIL mixture to the witness rate estimated by positive-unlabelled mixture-proportion estimation, and fit only the lesion component, which makes the per-patch posteriors stable early in training while keeping soft targets.",
 "Anchoring the mixture proportion to the MPE estimate reduces the epoch-to-epoch variation of the fitted lesion posterior by at least 50% and raises held-out AUPRO by at least 1 point over free-proportion posterior MIL.",
 "The witness-rate parent estimates the proportion of lesion patches robustly but uses it only as a hard k; the posterior parent uses soft targets but must estimate proportion and lesion component jointly, which is unstable when the two distributions overlap early in training. Using the MPE estimate as the fixed proportion and fitting only the lesion component combines the robust part of one with the soft targets of the other.",
 "1) Every 2 epochs per source dataset and level: pi_d from the best-bin MPE estimator; f0 from normal-image patch scores (kernel density); f1 Gaussian fitted by EM with pi fixed to pi_d. 2) Posterior weights as in posterior MIL; weighted BCE for abnormal images; BCE towards 0 for normal images. 3) First 2 epochs: top-k MIL. Hyper-parameters: refit interval 2 epochs; pi clip [0.005, 0.3].",
 PA + "Baselines: posterior MIL with free proportion; witness-rate top-k MIL. Diagnostics: posterior stability across epochs; posterior AUROC against masks on source images. Falsified if AUPRO does not improve over free-proportion posterior MIL.",
 [MAD, RECAP])

off("stage3/B1/g3/o5/combination",
 "Exact leave-one-out, alignment-adaptive residual CDF",
 "Combine the alignment-adaptive residual CDF (positional correction weighted by measured alignment) with a strictly leave-one-out construction of the support residual references, so the method is safe on unaligned modalities and unbiased at small K.",
 "The exact leave-one-out alignment-adaptive residual CDF matches the alignment-adaptive residual CDF in AUROC (within 0.5 points) and reduces the gap between nominal and realised specificity at the support-derived 95% threshold by at least 3 points at K <= 4.",
 "The alignment-adaptive parent decides how much positional correction to apply; the residual parent's weak point at small K is that a support's residual reference uses a positional median that includes the support itself, which makes support residuals too small and thresholds too low. Building every support's residuals with the median, alignment estimate and memory bank of the other K-1 supports removes that bias while keeping the adaptive safety of the first parent.",
 "1) For i in 1..K: from the other K-1 supports compute bank_{-i}, positional median med_{-i}, alignment rho_{-i}; residuals of support i: s_ij - rho_{-i} med_{-i,j}; pool into R. 2) Test image: med and rho from all K supports; r = s_j - rho med_j; F(r) by searchsorted on R. 3) 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baselines: alignment-adaptive residual CDF (non-exact); global CDF. Metrics: AUROC, nominal-vs-realised specificity gap. Falsified if the specificity gap is not reduced at K <= 4.",
 [MAD, RECAP, ADAN])

json.dump(O, open("/home/claude/task3/logs/answers/run3/s3_g3_ops.json", "w"), indent=1, ensure_ascii=False)
print(len(O))
