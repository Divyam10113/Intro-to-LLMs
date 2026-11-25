import json
exec(open("/home/claude/task3/logs/answers/run3/s3_g1_ops.py").read().split("O = {}")[0])
O = {}
def off(tag, t, s, h, r, m, e, p):
    O[tag] = "```json\n" + json.dumps({"Title": t, "Summary": s, "Hypothesis": h, "Rationale": r, "Method": m, "Experiment": e,
                                       "Provenance": p}, indent=1, ensure_ascii=False) + "\n```"

off("stage3/A1/g2/o3/combination",
 "Consensus MIL with an anchor-frame control",
 "Train the segmentation adapters with cross-level consensus MIL under MVFA's text anchors and under scale-matched random anchors, testing whether consensus selection improves weak localisation and whether its benefit relies on the shared anchor frame rather than on the anchors' meaning.",
 "Consensus MIL (top-k on the geometric mean of the four level maps) raises held-out AUPRO by at least 3 points over per-level MIL under both text and scale-matched random anchors, with a difference of less than 1 point between the two frames, whereas giving each level its own random frame removes at least half of the consensus gain.",
 "The consensus parent argues that the four levels' probabilities are comparable because they are read in one anchor frame; the anchor-control parent shows, following Do Words Matter?, that this frame matters through its geometry, not its words. Combining them makes the mechanism of consensus MIL testable: if the shared frame is what makes level agreement meaningful, scale-matched random anchors should work as well as text anchors, and separate random frames per level should break the consensus gain.",
 "1) loss.py: consensus_mil as in the parent (geometric mean of level probabilities, top-k = 1%, selection detached; normal images: per-level top-k BCE towards 0). 2) utils.py: make_anchors('text' | 'rand' | 'rand_per_level'), random pairs rescaled to the text pair's difference length, 3 draws. 3) train_zero.py: flags --loss {mil, consensus} --anchors {text, rand, rand_per_level}. 4) Hyper-parameters: k_frac 1%; released learning rate and epochs.",
 PA + "Design: loss {per-level MIL, consensus} x anchors {text, shared random, per-level random}. Diagnostics: overlap of per-level top-k sets; fraction of selected patches inside true masks. Falsified if consensus does not beat per-level MIL, or if per-level random frames keep the full gain.",
 [MLCA, DWM])

off("stage3/B1/g2/o1/combination",
 "Alignment-adaptive residual CDF: weighting the positional correction by measured anatomy alignment",
 "Use the residual empirical CDF (positional median subtracted, global CDF of residuals) but weight the positional correction by how much the supports' normal scores actually vary with position, so unaligned modalities fall back to the plain global CDF automatically.",
 "The alignment-adaptive residual CDF is never worse than the global empirical CDF by more than 0.3 pixel-AUROC points on any held-out dataset, and on spatially aligned datasets it keeps at least 80% of the full residual-CDF gain, at all K.",
 "The global-CDF parent is safe but ignores anatomy; the residual-CDF parent removes anatomy offsets but assumes alignment. Their combination measures alignment from the supports themselves: the share of variance of leave-one-out support scores explained by grid position (an intraclass correlation, rho). The positional correction is applied with weight rho, so a modality whose normal score level does not depend on position (ChestXray crops, histopathology tiles) receives the plain global CDF and a well-aligned one receives the residual CDF.",
 "1) utils.py: rho = var_between_positions / (var_between + var_within) of LOO support scores per branch and level (positions pooled in 3x3 neighbourhoods). 2) Residual r = s_j - rho x med_j (med_j shrunk positional median); reference set built from support LOO residuals computed the same way. 3) test_few.py: F(r) via searchsorted; 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baselines: global empirical CDF; residual CDF with rho = 1. Report rho per dataset. Falsified if the adaptive variant is worse than the global CDF by more than 0.3 points on any dataset.",
 [MAD, RECAP, ADAN])

off("stage3/B1/g2/o5/combination",
 "Exact leave-one-out residual CDF",
 "Make the residual empirical CDF strictly leave-one-out: each support's residual reference scores are computed with a positional median and a memory bank that exclude that support, so the reference distribution matches what a new test image sees.",
 "At K <= 4, the exact leave-one-out residual CDF reduces the gap between the support-derived and the realised specificity at a 95%-specificity threshold by at least 3 points relative to the non-exact residual CDF, while keeping AUROC equal within 0.5 points.",
 "Both parents build references from leave-one-out patch scores, but the residual variant subtracts a positional median that includes the very support whose scores form the reference. At small K this shrinks the residuals of supports relative to those of a new image, so thresholds set on supports are too low and specificity is overestimated. Combining the global-CDF parent's strict leave-one-out construction with the residual parent's positional correction removes this bias at no extra cost.",
 "1) utils.py: for i in 1..K: bank_{-i} and med_{-i,j} from the other K-1 supports; residuals of support i = s_ij - med_{-i,j}; pool all residuals into R. 2) Test image: med_j from all K supports; r = s_j - med_j; F(r) by searchsorted on R. 3) 0.5 fusion and max pooling unchanged; test-set min-max removed. No training.",
 PB + "Additional baseline: non-exact residual CDF. Metrics: AUROC, gap between nominal and realised specificity at the support-derived 95% threshold. Falsified if the specificity gap is not reduced at K <= 4.",
 [MAD, RECAP, ADAN])

off("stage3/B1/g2/o3/inspiration",
 "Augmented-support reference sets for inductive normalisation at small K",
 "Enlarge the leave-one-out reference set of every normaliser by scoring label-preserving augmentations of the K normal supports (flips, small shifts and rotations, mild intensity jitter), so the reference distribution reflects normal variability even at K = 2.",
 "At K = 2, building the empirical-CDF reference from 16 augmentations per support reduces the standard deviation of image AUROC across support draws by at least 30% and increases mean pixel AUROC by at least 1 point over the unaugmented reference, with no benefit expected at K = 16.",
 "Both parents spend all their effort on how to map scores, while their common weakness is how few normal scores are available: at K = 2 the reference consists of two images scored against a one-image bank. A different lever is the reference itself. Normal anatomy varies by position, orientation and intensity; label-preserving augmentations of the supports sample this variation and give many more leave-one-out reference scores without any extra data. Augmented views are scored against the bank of the other supports only, so the scheme stays strictly inductive.",
 "1) utils.py: augment(x, n=16): horizontal flip (where anatomically valid), shifts up to 8 px, rotations up to 10 degrees, gamma in [0.8, 1.2]. 2) build_refs: for each support i and each augmented view, few-shot scores against the bank of the other supports (unaugmented) and zero-shot scores; pool into the reference arrays. 3) test_few.py: empirical CDF per branch and level; 0.5 fusion and max pooling unchanged; test-set min-max removed. Hyper-parameters: n in {4, 16, 64}. No training.",
 PB + "Additional baseline: unaugmented empirical CDF. Metrics: mean and standard deviation of image/pixel AUROC across support draws per K. Falsified if variance at K = 2 is not reduced by 30%.",
 [MAD, RECAP])

off("stage3/B1/g2/o6/combination",
 "Conformal thresholds on residual-CDF scores",
 "Normalise each branch with the residual empirical CDF and set the operating threshold as a split-conformal quantile of the supports' leave-one-out image scores, giving an anatomy-aware inductive score and a distribution-free false-positive bound.",
 "On held-out BMAD datasets with K >= 8, residual-CDF scoring with a conformal threshold at alpha = 0.1 keeps realised false-positive rate at or below alpha + 1/(K+1) and achieves at least 3 points higher sensitivity than the same conformal threshold on globally normalised scores.",
 "The conformal parent guarantees a false-positive bound for any per-image score, but the power of the resulting test (its sensitivity) depends on how well the score separates abnormal from normal images. The residual-CDF parent produces a more separating score on aligned modalities by removing anatomy offsets. Combining them keeps the guarantee and raises the sensitivity at the guaranteed false-positive rate.",
 "1) Residual CDF per branch and level as in the parent (exact leave-one-out construction). 2) Fused map with 0.5 weights; image score = max. 3) LOO image scores of the K supports; tau = sorted(s)[ceil((K+1)(1 - alpha)) - 1]. 4) Report realised FPR and sensitivity at alpha in {0.1, 0.2}. No training.",
 PB + "Additional baseline: conformal threshold on global-CDF scores. Metrics: realised FPR vs. bound, sensitivity, AUROC. Falsified if sensitivity is not higher than with global-CDF scores at equal guaranteed FPR.",
 [DWM, MAD, RECAP, ADAN])

json.dump(O, open("/home/claude/task3/logs/answers/run3/s3_g2_ops.json", "w"), indent=1, ensure_ascii=False)
print(len(O))
