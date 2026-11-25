import json
exec(open("/home/claude/task3/logs/answers/run3/s3_g1_ops.py").read().split("O = {}")[0])
O = {}
def off(tag, t, s, h, r, m, e, p):
    O[tag] = "```json\n" + json.dumps({"Title": t, "Summary": s, "Hypothesis": h, "Rationale": r, "Method": m, "Experiment": e,
                                       "Provenance": p}, indent=1, ensure_ascii=False) + "\n```"

# ---------------------------------------------------------------- A1
off("stage3/A1/g3/o1/out_of_box",
 "Learnable log-sum-exp pooling per dataset replaces fixed top-k in MIL",
 "Replace the hard top-k mean of the MIL operator by log-sum-exp pooling with a learnable temperature per source dataset, which interpolates smoothly between max and mean pooling and lets gradient descent find each dataset's effective lesion fraction from image labels.",
 "Per-dataset learnable log-sum-exp pooling matches or exceeds witness-rate-adaptive top-k MIL in held-out pixel AUROC (difference >= 0) and the learned temperatures order the masked source datasets by their true lesion area fraction.",
 "Witness-rate MIL estimates the lesion fraction outside the network with a statistical estimator and then applies a hard cut. An analogous but end-to-end principle exists in weakly supervised localisation: smooth pooling operators whose sharpness is a parameter. With a learnable temperature r_d per dataset, LSE pooling (1/r) log mean exp(r p_j) behaves like max pooling for large r and like mean pooling for small r, so the image-level loss itself learns how concentrated each dataset's lesions are. Every patch receives gradient in proportion to its softmax weight, which also avoids the discontinuity of top-k.",
 "1) loss.py: lse_pool(p, r) = (1/r) log(mean_j exp(r p_j)); r_d = softplus(rho_d) + 1, rho_d learnable per source dataset (init so that r = 20). 2) MIL loss: BCE(lse_pool(p_l, r_d), y) per level for every image (normal images included). 3) train_zero.py: rho_d in the seg optimizer. 4) On the held-out modality, r is not needed at test time (inference unchanged). Hyper-parameters: init r {5, 20, 50}.",
 PA + "Baselines: fixed top-k MIL; witness-rate top-k MIL. Diagnostic: learned r_d vs. true lesion area fraction. Falsified if LSE pooling is worse than witness-rate top-k.",
 [MAD, RECAP])

off("stage3/A1/g3/o4/substitution",
 "Mosaic-to-single-image map consistency",
 "Keep the mosaic framing but replace the label-proportion (mass) consistency with dense map consistency: the segmentation map of each quadrant of a mosaic must equal the map of its source image computed alone at the same resolution.",
 "Dense mosaic-to-single map consistency reduces relative lesion-area error by at least 25% and raises AUPRO by at least 2 points over mosaic MIL, at least matching the label-proportion loss on both metrics.",
 "The parent's core component constrains only a scalar (total abnormal mass) per mosaic. Its purpose — making the adapters' maps independent of context so that area is consistent — is served more directly by requiring that each quadrant's map in the mosaic equals the map of the same image seen alone. This dense constraint uses the same forward passes, keeps the problem framing (image labels only), targets the same quantity (map area and shape) and keeps the evaluation.",
 "1) dataset/mosaic.py as in the parent. 2) Forward the four source images individually at 120 px (stop-gradient) and the mosaic. 3) loss.py: cons = mean over quadrants and levels of || p(mosaic)[quadrant q] - p(image_q) ||_1, plus BCE towards 0 on normal quadrants and consensus MIL inside abnormal quadrants. 4) Hyper-parameters: consistency weight {0.5, 1}; mosaic probability 0.5. Inference unchanged.",
 PA + "Baselines: mosaic MIL; label-proportion mosaic MIL. Metrics: pixel AUROC, AUPRO, relative lesion-area error. Falsified if dense consistency is not at least as good as the mass consistency.",
 [MLCA, DWM])

off("stage3/A1/g3/o6/out_of_box",
 "Feature-space anomaly mixing: graded synthetic positives from normal-bank patches",
 "Create synthetic training patches in feature space by mixing the features of normal-bank patches with those of the gated top patches of abnormal images at random ratios, and train the segmentation adapters to output the mixing ratio, giving dense, graded patch-level supervision without image-space artefacts.",
 "Adding feature-space mixing (weight 0.5) to gated MIL raises held-out AUPRO by at least 2 points over gated MIL alone, and outperforms image-space synthetic lesions (Poisson blending) under the same training budget.",
 "Gated MIL selects a few positive patches per abnormal image and leaves most of the map unsupervised. Image-space synthetic lesions add supervision but teach the model blending artefacts. An analogous trick from representation learning (mixup in feature space) avoids both: a convex combination of a normal patch feature and a likely-lesion patch feature at ratio lambda is a patch whose abnormality is known to be lambda, and such graded examples shape the adapters' output scale over the whole range, not only at 0 and 1. Because mixing happens at the input of the adapter at each level, it needs no new images and no masks.",
 "1) Per batch and level: take the gated top-k patch features of abnormal images (as in gated MIL) and an equal number of random normal-bank patch features of the same dataset. 2) Mixed feature f_mix = lambda f_abn + (1 - lambda) f_nor with lambda ~ U(0, 1); pass through the level's segmentation adapter. 3) loss.py: mix_loss = BCE(p(f_mix), lambda) (soft target) added to gated MIL with weight 0.5. 4) Hyper-parameters: weight {0.25, 0.5, 1}; number of mixed patches per batch 64. Inference unchanged.",
 PA + "Baselines: gated MIL; gated MIL + image-space synthetic lesions. Falsified if feature mixing does not improve AUPRO over gated MIL.",
 [ADAN, MAD, RECAP, FED])

# ---------------------------------------------------------------- B1
off("stage3/B1/g3/o2/out_of_box",
 "Foreground-size-aware Gumbel normalisation of the max-pooled image score",
 "Because the maximum of N normal patch scores grows with N, normalise each image's max fused score by a Gumbel distribution of leave-one-out support maxima whose location is adjusted for the image's number of foreground patches, so large-anatomy images are not systematically scored as more anomalous.",
 "Adjusting the max-pooled image score for foreground size with a support-fitted Gumbel model increases image AUROC by at least 1 point on held-out datasets whose foreground area varies strongly between images (LiverCT, ChestXray), and changes it by less than 0.5 points elsewhere.",
 "The EVT parent extends the patch-level CDF tail, but the image score is the maximum over all patches, and a maximum of more patches is larger even if every patch is normal. In BMAD the foreground area differs strongly between slices and images, so with max pooling fixed by the problem, images with more tissue get higher scores for no pathological reason. Block-maxima theory says that the maximum of N i.i.d. normal scores is approximately Gumbel with location increasing in log N. Fitting this relation on the supports (leave-one-out maxima of support images and of their random sub-regions of different sizes) gives an inductive, size-aware normaliser of the image score.",
 "1) Foreground mask per image by Otsu on intensity (patch-level). 2) For each support i (LOO): maxima of the fused normalised map over random sub-regions of sizes N in {25%, 50%, 100%} of its foreground; fit Gumbel location mu(N) = a + b log N and scale beta by maximum likelihood. 3) Test image: z = (max - mu(N_fg)) / beta, image score = Gumbel CDF(z). 4) Pixel maps unchanged; patch normaliser = support percentiles (problem statement). No training.",
 PB + "Additional baseline: support-percentile normaliser without size correction. Diagnostic: correlation of image score with foreground size among normal test images before and after correction. Falsified if no AUROC gain on LiverCT and ChestXray.",
 [MAD, RECAP])

off("stage3/B1/g3/o4/out_of_box",
 "Content-conditioned expected-normal-score regression",
 "Train on the source modalities a small regressor that predicts the expected normal anomaly score of a patch from its frozen CLIP feature, and normalise target scores by the residual from this prediction (re-centred with the supports), removing anatomy-driven offsets without assuming spatial alignment.",
 "Content-conditioned residual normalisation improves pixel AUROC by at least 1 point over global support percentiles on all six held-out datasets, including the unaligned ChestXray and HIS where positional residuals do not apply.",
 "The residual parent removes anatomy offsets by grid position, which only works where images are aligned. The underlying cause is that some normal tissue types score systematically higher; this is a property of the patch's content, not of its position. A regressor from frozen CLIP features to the score of normal patches, trained on the source modalities' normal images, predicts this offset for any patch, and the residual is then mapped through the support CDF. Because the regressor is trained on sources only and re-centred with the target supports, the method stays inductive and leave-one-modality-out.",
 "1) train_offset.py (per fold): on normal source training images, collect (frozen level-l patch feature, branch score) pairs; fit a 2-layer MLP (768-128-1) per branch and level with MSE, 5 epochs. 2) Target: r = s - g(f) - c, where c is the median residual of the supports (LOO). 3) F(r) via the LOO support CDF of residuals; 0.5 fusion and max pooling unchanged; test-set min-max removed. Hyper-parameters: hidden size 128; ablation with a linear regressor.",
 PB + "Additional baselines: positional residual CDF; global CDF. Falsified if no gain on ChestXray and HIS over global percentiles.",
 [ADAN, MAD, RECAP])

off("stage3/B1/g3/o3/out_of_box",
 "Optimal-transport correspondence for patch-wise normal references",
 "Instead of grid positions or rigid shifts, match every test patch to support patches by entropic optimal transport over feature and position costs, and normalise its score against the leave-one-out scores of the support patches it is matched to.",
 "Optimal-transport-matched references improve pixel AUROC by at least 1 point over the alignment-adaptive residual CDF on BrainMRI, LiverCT and RESC, and by at least 0.5 points on the unaligned datasets, at less than 2x inference time.",
 "Alignment-adaptive correction decides how much positional correspondence to trust but cannot repair it. Optimal transport, used for patch-prompt alignment in recent medical anomaly detection, gives a soft correspondence between two sets of patches that respects both appearance and position, and works on deformed or partially aligned anatomy. The scores of the matched support patches then form a reference that is specific to the anatomy of each test patch.",
 "1) For a test image and each support: cost C_ij = (1 - cos(f_i, f_j)) + 0.1 ||pos_i - pos_j||; Sinkhorn with entropy 0.05, 50 iterations (level 2 features). 2) Matched reference of test patch i: transport-weighted LOO scores of support patches; normalised score = weighted rank of s_i within them; mix with the global CDF (weight 0.2). 3) 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baselines: alignment-adaptive residual CDF; feature-registered references. Metrics: AUROC, inference time. Falsified if no gain over the alignment-adaptive residual CDF on the aligned datasets.",
 [MAD, RECAP, ADAN, SFAD])

off("stage3/B1/g3/o1/out_of_box",
 "Meta-selected normaliser from simulated source episodes",
 "Simulate few-shot episodes on the source modalities, record which inductive normaliser (global CDF, residual CDF, alignment-adaptive residual CDF) works best as a function of support-measurable features (alignment rho, K, score spread), and use this rule to pick the normaliser for the unseen modality from its supports alone.",
 "The meta-selected normaliser is within 0.3 AUROC points of the best normaliser in hindsight on at least five of six held-out datasets, and on average better than any single fixed normaliser.",
 "The alignment-adaptive parent blends two normalisers with one statistic. A more general principle — algorithm selection by meta-learning — chooses among several normalisers using features that can be measured on the K supports. The source modalities provide labelled episodes on which the choice can be learned, and leave-one-modality-out evaluation tests whether that choice transfers.",
 "1) meta_select.py: on each source modality's validation split, 50 episodes per K; features: rho, K, coefficient of variation of LOO support scores, fraction of foreground patches; label: best normaliser by episode AUROC. 2) Fit a depth-2 decision tree on the pooled episodes of the training fold. 3) Target: compute features from the supports, choose the normaliser; 0.5 fusion and max pooling unchanged; test-set min-max removed. No adapter training.",
 PB + "Additional baselines: each fixed normaliser; oracle choice in hindsight. Falsified if meta-selection is not better on average than the best fixed normaliser.",
 [MAD, RECAP, ADAN])

off("stage3/B1/g3/o6/out_of_box",
 "Multi-scale positional residuals",
 "Subtract positional medians computed at several grid resolutions (full grid, 2x2-pooled, 4x4-pooled) and map the combined residual through one leave-one-out support CDF, so coarse anatomy offsets and fine structure offsets are removed without estimating each from a few values.",
 "Multi-scale positional residuals improve pixel AUROC by at least 0.5 points over the single-scale residual CDF on aligned held-out datasets at K <= 4, with smaller variance across support draws.",
 "The single-scale residual estimates one median per grid position, which at K = 2 is noisy. Multi-resolution decompositions (as in image pyramids and wavelets) estimate coarse components from many values and fine components from few, and combine them with shrinkage. Applying this to positional normal offsets gives a better bias-variance trade-off at small K while keeping the method inductive.",
 "1) For each branch and level: medians at pooling factors {1, 2, 4} from LOO support maps; combined offset o_j = sum_s w_s med_s(j) with w_s proportional to the number of values per cell (shrinkage). 2) r = s_j - o_j; F(r) from the LOO support residual reference. 3) 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baseline: single-scale residual CDF. Metric: mean and standard deviation of AUROC over support draws. Falsified if no gain at K <= 4 on aligned datasets.",
 [MAD, RECAP, ADAN])

json.dump(O, open("/home/claude/task3/logs/answers/run3/s3_g3_mut.json", "w"), indent=1, ensure_ascii=False)
print(len(O))
