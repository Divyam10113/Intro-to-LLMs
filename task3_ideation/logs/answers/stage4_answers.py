import json
def J(d): return "```json\n" + json.dumps(d, indent=1) + "\n```"
A = {}
A["stage4/A1/families"] = J({"families": {
 "Patch-level MIL objectives": ["A1-g0-C1","A1-g1-o6","A1-g2-o5","A1-g3-o1","A1-g3-o4","A1-g0-C6","A1-g3-o5"],
 "Prior-based patch label estimation (PU, OT assignment, noisy labels)": ["A1-g2-o6","A1-g3-o3","A1-g3-o2","A1-g3-o6"],
 "Normal-reference pseudo-masks": ["A1-g1-o1","A1-g2-o4","A1-g2-o2","A1-g2-o1"],
 "Decision-attribution pseudo-masks": ["A1-g0-C2","A1-g1-o2","A1-g1-o4","A1-g2-o3","A1-g1-o3"],
 "Auxiliary priors (learned anchors, DINO, synthetic anomalies)": ["A1-g0-C5","A1-g0-C4","A1-g1-o5","A1-g0-C3"]}})
A["stage4/B1/families"] = J({"families": {
 "Support-anchored p-value combination": ["B1-g0-C2","B1-g2-o6","B1-g3-o5","B1-g3-o3","B1-g1-o6"],
 "Support-standardised score normalisation": ["B1-g0-C1","B1-g1-o2","B1-g1-o1","B1-g0-C4","B1-g1-o3","B1-g0-C6","B1-g2-o5"],
 "Learned calibration from source episodes or synthetic data": ["B1-g1-o4","B1-g2-o1","B1-g2-o3","B1-g0-C3"],
 "Statistical decision guarantees and uncertainty": ["B1-g3-o1","B1-g2-o4","B1-g3-o2"],
 "Distribution tests and density ratios": ["B1-g1-o5","B1-g2-o2"],
 "Aggregation and robustness add-ons": ["B1-g0-C5","B1-g3-o4","B1-g3-o6"]}})
def R(pid, cid, n, f, c, i, v):
    A[f"stage4/{pid}/review/{cid}"] = J({"Novelty": {"review": n[1], "score": n[0]}, "Feasibility": {"review": f[1], "score": f[0]},
        "Clarity": {"review": c[1], "score": c[0]}, "Impact": {"review": i[1], "score": i[0]}, "Validity": {"review": v[1], "score": v[0]}})
# ------------------------------------------------------------------ A1
R("A1","A1-g2-o6",
 (4,"PU learning is established theory, but the search found no application to patch-level supervision of CLIP-based medical anomaly segmentation; framing normal-image patches as labelled negatives and abnormal-image patches as unlabelled is a well-differentiated perspective relative to MIL heuristics."),
 (4,"Requires a mixture-proportion estimate per dataset and balanced batches (MVFA trains with batch size 1), both manageable on one GPU; nnPU is a short loss function."),
 (4,"The overall procedure is clear, but the risk expression in step 3 is left half-written and deferred to 'the standard estimator'; the prior-estimation step and balanced batching are well specified."),
 (4,"If it works, localisation becomes trainable from image labels with a principled objective that also applies to other CLIP adapters."),
 (4,"Sound formulation that matches the data structure; the main validity risk is bias in the estimated class prior for small lesions, which the prior-misspecification ablation addresses."))
R("A1","A1-g3-o3",
 (4,"Instance-dependent priors exist in PU theory, and memory banks are common, but their combination for mask-free medical segmentation was not found."),
 (3,"Combines position-aware banks, prior estimation and weighted nnPU; feasible but heavier engineering and more hyper-parameters (temperature, clipping, radius)."),
 (3,"The prior construction (softmax over bank distances rescaled to the dataset prior, then clipped) is described but somewhat ad hoc and not fully justified."),
 (4,"Potentially sharper localisation than single-prior PU, especially for small lesions."),
 (3,"Bank-derived priors are themselves noisy estimates; using them as instance priors can bias the risk estimator, and unbiasedness of the weighted nnPU under estimated instance priors is not argued."))
R("A1","A1-g0-C1",
 (3,"MIL-based weakly supervised lesion segmentation exists (e.g. FMaMIL), and treating normal-image patches as negatives is natural; the SigLIP patch loss and the cross-modality zero-shot setting give moderate novelty."),
 (5,"A single loss replacement in train_zero.py with standard hyper-parameters; highly feasible."),
 (5,"Exceptionally precise: logits, normal and abnormal terms, k, weights, levels, optimiser and ablations are all specified."),
 (4,"Directly removes MVFA's dependence on masks for localisation; a clear baseline for the whole direction."),
 (4,"Sound and directly aligned with the verified seg-loss limitation; top-k selection can pick salient normal anatomy, which is acknowledged through ablations."))
R("A1","A1-g3-o5",
 (3,"Gating MIL positives with memory-bank evidence is a sensible combination of known components; related pixel-labelling-efficiency work exists in industrial AD."),
 (4,"Requires per-dataset banks and a gated loss; straightforward."),
 (4,"Gating rule, fallback and weighting are clearly specified."),
 (4,"Likely to reduce MIL's selection of salient normal structures, improving PRO."),
 (4,"Agreement of two independent evidence sources is a well-founded way to raise positive precision."))
R("A1","A1-g2-o4",
 (2,"Using spatial position as the normality cue for brain lesions is close to Patch2Loc (AISTATS 2026) and to position-conditioned normal models such as PaDiM; the incremental novelty is low."),
 (4,"Offline banks per grid position are easy to build."),
 (4,"Clear radius, coreset and threshold choices, including a rule for unaligned datasets."),
 (3,"Benefits mainly spatially aligned datasets (registered brain slices); limited for OCT, chest X-ray and histopathology."),
 (3,"Assumes spatial alignment that holds only for part of BMAD; the per-dataset radius rule is reasonable but the claim of alignment for liver CT is not verified."))
R("A1","A1-g1-o1",
 (3,"Memory-bank localisation is standard (PatchCore-style); using it offline to create pseudo-masks for training a CLIP segmentation branch from image labels is a moderately new use."),
 (5,"Offline feature extraction and coreset search on source datasets; very feasible."),
 (4,"Clear procedure with thresholds defined on held-out normal images."),
 (4,"Simple, model-agnostic way to obtain dense supervision without masks."),
 (3,"Frozen-CLIP nearest-neighbour distances may not localise subtle medical lesions; the falsification threshold on pseudo-mask IoU is a good safeguard."))
R("A1","A1-g3-o6",
 (4,"OT-based label assignment exists for point-supervised segmentation (Point2Mask) and detection, but not for distributing image-level anomaly labels over patches."),
 (4,"Sinkhorn on 289 patches per image is cheap."),
 (4,"Costs, budget and schedule are specified."),
 (3,"Likely gains over PU are uncertain; the main benefit is sharper per-image labels."),
 (3,"Depends on the lesion-fraction budget and a neighbourhood cost whose weight is hand-set; risk of collapse to contiguous but wrong regions."))
R("A1","A1-g1-o6",
 (3,"The MIL ranking loss is well known from video anomaly detection (Sultani et al.); transferring it to image patches with cross-image batches is a moderate novelty."),
 (5,"Balanced batches and a hinge loss; easy to implement."),
 (4,"Loss, regularisers and weights are clear."),
 (4,"Cross-image comparability of scores benefits both localisation and image scoring."),
 (4,"Sound transfer of a validated weakly supervised objective; also fixes the batch-size-1 limitation."))
R("A1","A1-g3-o2",
 (3,"Co-teaching and noise-robust training for medical segmentation exist; applying them to image-label-derived patch labels is moderately new."),
 (4,"Two adapter copies double training cost but remain feasible."),
 (4,"Schedule and selection rule are clear."),
 (3,"Avoids explicit priors, but the expected gain over PU is modest."),
 (3,"The forget-rate schedule implicitly encodes the lesion fraction, so the 'no prior needed' claim is only partly true."))
R("A1","A1-g1-o4",
 (4,"Counterfactual perturbation for weakly supervised anomaly localisation appears in other domains (logs) but not for medical image patches in CLIP adapters."),
 (4,"Batched replacement at the scoring head is cheap."),
 (4,"Procedure is clear, including the batching trick."),
 (3,"Causal localisation is attractive but gains are uncertain."),
 (3,"Replacing post-adapter tokens without propagating through later layers changes only the scoring head; with a mean-pooled score single-patch drops may be tiny for small lesions."))
R("A1","A1-g2-o3",
 (3,"CAM-based pseudo-masks from CLIP exist (e.g. CLISC); counterfactual verification adds some novelty."),
 (3,"Two pseudo-mask stages plus group replacement; more engineering."),
 (4,"Clear proposal-verification pipeline."),
 (3,"Improves pseudo-mask precision but builds on a weaker attribution signal."),
 (4,"Verification is a sound remedy for noisy attributions."))
R("A1","A1-g2-o5",
 (3,"Adversarial erasing is established in weakly supervised segmentation; new to MVFA's setting."),
 (4,"Additional training rounds; feasible."),
 (4,"Clear rounds and stopping rule."),
 (3,"Targets full lesion extent, which matters for PRO."),
 (4,"Sound, with a sensible safeguard against forcing evidence on erased images."))
# ------------------------------------------------------------------ B1
R("B1","B1-g2-o6",
 (3,"Combining layer-wise p-values, including dependence-robust rules, has been used for DNN input anomaly detection (arXiv 2007.15147); using support-anchored p-values to replace MVFA's verified transductive fusion is the new element."),
 (5,"Inference-only changes to the scoring code; trivial compute."),
 (4,"Clear statistic, pooling and thresholds; the correlation analysis is a good addition."),
 (4,"Delivers deployable per-image scores with a statistically grounded threshold, fixing two verified defects."),
 (4,"The harmonic mean p-value is approximately valid under dependence; empirical p-values are discrete at small K, which limits resolution at very low levels."))
R("B1","B1-g3-o5",
 (3,"Empirical Brown's method and layer-wise p-value combination exist; the combination with support-anchored nulls for MVFA is new in application."),
 (5,"Inference-only; covariance estimation is cheap."),
 (4,"Clear formulas for scale and degrees of freedom."),
 (4,"Could keep Fisher's power with corrected validity."),
 (3,"Moment matching from strongly spatially correlated pixels of very few supports may be unstable at K = 2, and Brown's approximation can be poor in the far tail."))
R("B1","B1-g0-C2",
 (3,"Fisher combination of empirical p-values is standard (and used for layer-wise DNN anomaly detection); application to MVFA's fusion is the new part."),
 (5,"Inference-only and simple."),
 (4,"Clear statistic and thresholds."),
 (4,"Fixes the transductive fusion with an interpretable statistic."),
 (3,"Fisher assumes independent p-values; MVFA's levels and branches are strongly dependent, so the statistic is anti-conservative — a flaw later evolution addressed."))
R("B1","B1-g3-o1",
 (3,"Conformal and FDR-controlled segmentation with guarantees exist (industrial defects 2504.17721; medical instance segmentation 2504.04482); the per-image BY procedure on support-anchored p-values for MVFA is an adaptation."),
 (5,"Inference-only; BY on 289 patches is trivial."),
 (4,"Clear q levels, statistics and evaluation."),
 (4,"Gives the anomaly map a clinically meaningful guarantee on flagged regions."),
 (3,"Patch p-values come from a pooled support null across positions, so exchangeability between support and test patches at different anatomical positions is questionable; BY may be conservative for small lesions."))
R("B1","B1-g1-o2",
 (2,"Normalising anomaly scores with held-out normal statistics is standard practice in tooling such as Anomalib, and leave-one-out calibration of anomaly detectors exists; the contribution is applying it to MVFA, not a new technique."),
 (5,"Inference-only with a small new module."),
 (5,"Exceptionally clear: statistics, clipping, weights, pooling, thresholds and stability analyses are fully specified."),
 (4,"Directly removes both verified scoring defects and quantifies how much published few-shot numbers depend on test-set composition."),
 (4,"Sound and assumption-light; Gaussian z-scores are a rough tail approximation but adequate for ranking."))
R("B1","B1-g2-o1",
 (4,"Episodic few-shot adaptation exists in other anomaly detection domains (audio), but learning p-value weights from simulated source-modality episodes for medical VLM detectors was not found."),
 (4,"Requires scoring many source episodes; moderate compute but feasible on one GPU."),
 (4,"Clear features, constraints and thresholds."),
 (4,"Uses real labelled source data to calibrate, potentially improving both AUROC and calibration."),
 (3,"Assumes calibration relationships transfer from source to unseen modalities; non-negative weights over correlated -log p terms may be unstable."))
R("B1","B1-g3-o3",
 (3,"Conformal e-values for novelty detection exist (2302.07294); application to dependent multi-level evidence in MVFA is new in application."),
 (5,"Inference-only."),
 (4,"Clear calibrator and decision rule."),
 (3,"Exact guarantees but likely power loss limits practical impact."),
 (4,"Valid by construction under arbitrary dependence."))
R("B1","B1-g1-o4",
 (3,"Episodic calibration on source data is a moderate adaptation of known ideas."),
 (4,"Requires episode simulation on source modalities."),
 (4,"Clear features and model."),
 (3,"Improves calibration if transfer holds."),
 (3,"Calibration transfer across modalities is assumed rather than argued; hand-chosen features."))
R("B1","B1-g3-o2",
 (3,"Bootstrap and leave-one-out conformal anomaly detectors exist (2402.16388)."),
 (4,"Thirty resamples per image multiply scoring cost moderately."),
 (4,"Clear procedure."),
 (3,"Useful per-image uncertainty, but does not change the fusion rule."),
 (3,"With K = 2 the bootstrap distribution is degenerate (three distinct resamples), weakening the method exactly where it is most needed."))
R("B1","B1-g0-C1",
 (2,"Validation-statistic normalisation and leave-one-out calibration are established; low novelty."),
 (5,"Inference-only."),
 (4,"Clear, with an awkwardly phrased probability-mapping step."),
 (3,"Fixes fusion but keeps outlier-sensitive pooling."),
 (4,"Sound and simple."))
R("B1","B1-g2-o2",
 (4,"Combining a target-support normal density with a cross-modality abnormal density learned on source data is not found in the related literature on generalist AD with normal and abnormal references."),
 (4,"PCA and Gaussian mixtures are cheap; requires collecting source abnormal patches."),
 (4,"Clear models and scoring."),
 (4,"A two-sided test could give real power gains at small K."),
 (3,"Relies on the abnormal density transferring across modalities, which is the central and untested assumption."))
R("B1","B1-g2-o5",
 (3,"Score normalisation for domain generalisation in anomaly detection exists (2509.10951); importing S-norm from speaker verification is a moderately new angle."),
 (4,"Cohort banks from source modalities are easy to build."),
 (4,"Clear Z-norm, T-norm and S-norm definitions."),
 (3,"Could make one threshold work across modalities, a useful but narrower goal."),
 (3,"Benefit depends on cohort similarity to the target, which is not guaranteed for distant modalities."))
json.dump(A, open("/home/claude/task3/logs/answers/stage4_answers.json","w"), indent=1); print(len(A))
