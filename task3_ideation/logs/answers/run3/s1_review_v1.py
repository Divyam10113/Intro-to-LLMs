import json
def R(review, feedback, rating): return f"Review: {review}\nFeedback: {feedback}\nRating (1-5): {rating}"
A = {}
c = "stage1/review/A1/v1/"
A[c+"Clarity"] = R("The revision fixes the open details (all four levels, original weights, detection branch unchanged, pixel metrics only on masked folds) and states a testable hypothesis. It has, however, grown: the core comparison, a level-wise analysis and a two-point label-budget curve are now in one problem, which blurs which result is primary.",
 "Mark the mask-free versus mask-supervised comparison as primary and the level-wise analysis and label-budget points as secondary analyses.", 4)
A[c+"Relevance"] = R("Unchanged and high: it addresses the verified mask dependence of MVFA's localisation (L3, L5, L18) and the unused image-labelled half of the benchmark.",
 "None.", 5)
A[c+"Originality"] = R("The added hypothesis — that weak supervision through similarities to modality-shared text anchors gives localisation that transfers to an unseen modality, emerging mainly at deeper levels — turns a known technique (MIL) into a specific, testable claim about this architecture that the related work ([22], [29], [40], [58]) does not address.",
 "To strengthen the claim, include a control in which the text anchors are replaced by a learned linear head, so that the role of the text anchors is isolated.", 4)
A[c+"Feasibility"] = R("Still a one-term change at the original compute, with a sensible collapse check. The two label-budget variants add two more trainings per fold and seed, roughly tripling the training cost; on one GPU over a few weeks this is manageable but tight with 3 seeds.",
 "Run the label-budget variants with one seed on two folds if time is short.", 4)
A[c+"Significance"] = R("If confirmed, MVFA-style adaptation becomes trainable from report-level labels on all five source datasets; the label-budget curve makes the result directly usable for annotation planning.",
 "None of substance.", 4)

c = "stage1/review/A2/v1/"
A[c+"Clarity"] = R("The synthesis recipe, the memory-bank content, the K grid and the hypothesis are now explicit. Two questions remain bundled (can the abnormal supports be replaced, and does the gap depend on lesion type), and 'focal' versus 'diffuse' is asserted for each dataset rather than defined.",
 "Define the focal/diffuse grouping operationally (e.g., median lesion area from masks where available) or present it as a post-hoc analysis.", 4)
A[c+"Relevance"] = R("Addresses L4 and L18 directly; the decision-rule framing makes the deployment relevance explicit.",
 "None.", 4)
A[c+"Originality"] = R("The lesion-type hypothesis adds a question the synthetic-anomaly literature ([22], [43], [53], [55]) leaves open, but the core manipulation remains a well-known substitution of real by synthetic anomalies.",
 "Consider testing a property specific to MVFA, e.g., whether synthetic supports harm the zero-shot text branch through the shared adapters.", 3)
A[c+"Feasibility"] = R("The reduced grid fits one GPU; synthesis is cheap; all losses and scripts exist.",
 "None.", 4)
A[c+"Significance"] = R("A decision rule about when annotated abnormal cases are worth collecting is useful for practice, and a dataset-dependent answer would be informative; still, the expected result is partly anticipated by [22] and [55].",
 "Report the annotation time saved per site alongside the AUROC gap.", 4)

c = "stage1/review/B1/v1/"
A[c+"Clarity"] = R("Primary, motivating and secondary outcomes are now separated, the leave-one-out statistic is defined (1st/99th percentiles of patch scores), and the fixed parts (fusion weight, max pooling) are explicit. The problem is precise and readable.",
 "None of substance.", 5)
A[c+"Relevance"] = R("High: it addresses the verified test-set dependence of MVFA's scores (C1, L10) and the missing operating-point evaluation (L8).",
 "None.", 5)
A[c+"Originality"] = R("Framed as an evaluation-validity study with an inductive fix, it targets a practice the CLIP anomaly-detection literature has not examined ([17], [56], [57]); the fix itself (normalising by reference statistics) is standard.",
 "Show whether other BMAD baselines share the transductive normalisation; that would make the finding broader than MVFA.", 3)
A[c+"Feasibility"] = R("Inference-only, confined to one script; the patch-level leave-one-out statistic resolves the K = 2 problem. Compute is minimal.",
 "None.", 5)
A[c+"Significance"] = R("A per-image score with a support-fixed threshold is a prerequisite for deployment, and quantifying the test-set dependence affects how published BMAD numbers are read.",
 "None of substance.", 4)

c = "stage1/review/B2/v1/"
A[c+"Clarity"] = R("The map being pooled, the k grid, the selection rule and the robustness test are now defined. The central hypothesis relies on the 'typical lesion area' of a modality, which cannot be measured on the three datasets without masks, so the claimed relation cannot be tested on half of the benchmark.",
 "State how lesion area is estimated where masks are absent, or restrict the hypothesis test to the masked datasets.", 4)
A[c+"Relevance"] = R("Addresses L38, C3 and L30; the selection-from-source-modalities framing ties it to the generalisation claim of the target paper.",
 "None.", 4)
A[c+"Originality"] = R("Selecting the pooling rule from source modalities and relating it to lesion size goes a little beyond a routine ablation ([12], [15]), but remains a modest extension.",
 "None of substance.", 3)
A[c+"Feasibility"] = R("Inference-only and simple.",
 "None.", 5)
A[c+"Significance"] = R("A consistent, transferable pooling rule improves reproducibility; the effect on detection AUROC is likely small.",
 "None of substance.", 3)
json.dump(A, open("/home/claude/task3/logs/answers/run3/s1_review_v1.json", "w"), indent=1)
