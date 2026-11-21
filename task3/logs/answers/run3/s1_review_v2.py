import json
def R(review, feedback, rating): return f"Review: {review}\nFeedback: {feedback}\nRating (1-5): {rating}"
A = {}
c = "stage1/review/A1/v2/"
A[c+"Clarity"] = R("The primary question, primary outcomes and the three secondary analyses are now clearly separated, and every design choice is specified. The problem statement is long, and three secondary analyses stretch the 'one well-delimited question' requirement, although the primary comparison is unambiguous.",
 "Consider moving the per-level breakdown into the rationale or the evaluation plan to keep the problem statement focused.", 4)
A[c+"Relevance"] = R("High and unchanged: verified mask dependence of localisation (L3, L5, L18) and the unused image-labelled half of the benchmark.",
 "None.", 5)
A[c+"Originality"] = R("The text-anchor control makes the hypothesis falsifiable and specific to MVFA's architecture; the combination of MIL supervision of text-anchored multi-level maps with an unseen-modality test is not covered by [22], [29], [40], [54] or [58].",
 "None of substance.", 4)
A[c+"Feasibility"] = R("The primary comparison costs the same as MVFA's own training; the text-anchor control adds one training per fold and seed, and the label-budget points are restricted to one seed and two folds. Manageable on one GPU in a few weeks, with the collapse pilot as a sensible first step.",
 "Prioritise the primary comparison and the text-anchor control; drop the label-budget points if time runs short.", 4)
A[c+"Significance"] = R("Removing the need for dense masks from MVFA's source training would make the method trainable from report-level labels; the result is useful either way.",
 "None.", 4)

c = "stage1/review/A2/v2/"
A[c+"Clarity"] = R("The primary question is clear and the post-hoc analyses are defined operationally (median lesion area from masks, text-branch AUROC change). The problem remains somewhat long.",
 "None of substance.", 4)
A[c+"Relevance"] = R("Addresses L4 and L18 directly.",
 "None.", 4)
A[c+"Originality"] = R("The check of whether synthetic supports distort the shared text branch is specific to MVFA's design, but the core manipulation is the well-known substitution of real by synthetic anomalies ([22], [43], [53], [55]).",
 "None of substance.", 3)
A[c+"Feasibility"] = R("Cheap synthesis, existing losses and a reduced grid that fits one GPU.",
 "None.", 4)
A[c+"Significance"] = R("Useful decision rule for deployment at a new site; partly anticipated by the synthetic-anomaly literature.",
 "None.", 4)

c = "stage1/review/B1/v2/"
A[c+"Clarity"] = R("The main problem remains precise, but the added scope check is loosely worded: 'the strongest BMAD baseline evaluated with MVFA (APRIL-GAN-style memory fusion)' does not name a specific code base, so it is unclear what will be inspected and when the inductive normaliser would be applied to it.",
 "Name the baseline and its repository, or drop the scope check and mention it as future work.", 4)
A[c+"Relevance"] = R("High: verified test-set dependence (C1, L10) and missing operating-point evaluation (L8).",
 "None.", 5)
A[c+"Originality"] = R("Unchanged from the previous version: an evaluation-validity study with a standard inductive fix; the scope check adds breadth but no new idea.",
 "None of substance.", 3)
A[c+"Feasibility"] = R("Inference-only and confined to one script; the scope check adds code reading only.",
 "None.", 5)
A[c+"Significance"] = R("A per-image, support-calibrated score is a prerequisite for deployment and affects how BMAD results are read.",
 "None.", 4)

c = "stage1/review/B2/v2/"
A[c+"Clarity"] = R("The transfer question, the selection procedure and the robustness test are clear, and the lesion-area hypothesis is now restricted to the masked datasets.",
 "None of substance.", 4)
A[c+"Relevance"] = R("Addresses L38, C3 and L30.",
 "None.", 4)
A[c+"Originality"] = R("A modest extension of a standard pooling ablation ([12], [15]) with a transfer-selection step.",
 "None.", 3)
A[c+"Feasibility"] = R("Inference-only and simple.",
 "None.", 5)
A[c+"Significance"] = R("Improves reproducibility of MVFA's detection numbers; expected AUROC effects are small.",
 "None.", 3)
json.dump(A, open("/home/claude/task3/logs/answers/run3/s1_review_v2.json", "w"), indent=1)
