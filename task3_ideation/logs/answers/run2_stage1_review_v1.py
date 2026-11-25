import json
def R(review, feedback, rating): return f"Review: {review}\nFeedback: {feedback}\nRating (1-5): {rating}"
A = {}
c="stage1/review/A1/v1/"
A[c+"Clarity"]=R("The operator, levels, hypothesis (at least 80% retention, text-anchored mechanism) and outcomes are now specified. The problem has however grown to four outcomes including a label-budget curve, so it is less concise than before.",
 "Mark the retention question as primary and the per-level analysis and label-budget curve as secondary analyses.",4)
A[c+"Relevance"]=R("Remains central to the target paper's clinical applicability and to annotation-efficient medical anomaly detection.","None.",5)
A[c+"Originality"]=R("The explicit hypothesis — that text-anchored patch similarities let weak image-level supervision yield localisation that transfers across modalities — and the label-budget comparison distinguish this from generic weakly supervised localisation and from MVFA's existing mean-pooled detection loss.",
 "Include a visual-only MIL control (same operator on patch features without text anchors) to test the text-anchoring hypothesis directly.",4)
A[c+"Feasibility"]=R("The core comparison is cheap, but the label-budget curve multiplies training runs (several mask fractions x six folds x three seeds), which is heavy for one Colab GPU within a few weeks.",
 "Restrict the label-budget curve to two held-out modalities and three mask fractions.",4)
A[c+"Significance"]=R("A positive result makes MVFA-style localisation trainable from report-level labels; the label-budget curve adds decision value for clinical sites.","None.",4)
c="stage1/review/A2/v1/"
A[c+"Clarity"]=R("Well specified: rounds, adapters updated, comparison with single-level confidence, pseudo-label precision.","State the confidence quantile choice as fixed in advance.",4)
A[c+"Relevance"]=R("Relevant to the corrected few-shot requirement and to source-free adaptation.","None.",4)
A[c+"Originality"]=R("Now framed as a test of cross-level agreement as a confidence signal, which is a modest but specific contribution; the underlying self-training paradigm is standard.","None.",3)
A[c+"Feasibility"]=R("Feasible; the 50/50 split still reduces test sizes for the smaller BMAD datasets.","Report confidence intervals.",4)
A[c+"Significance"]=R("Useful for deployment but incremental relative to the other problems in this group.","None.",3)
c="stage1/review/B1/v1/"
A[c+"Clarity"]=R("The primary question (inductive versus transductive fusion AUROC), the exact leave-one-out construction, pooling, threshold rule, and the separation of primary, motivating and secondary outcomes are all specified with no ambiguity.",
 "None of substance.",5)
A[c+"Relevance"]=R("Directly grounded in verified properties of the released code that affect both the validity of the published few-shot results and deployability.","None.",5)
A[c+"Originality"]=R("The evaluation-validity angle (dependence of reported AUROC on test-set composition) and the support-based inductive reference are uncommon in this literature, though the statistical tools are classical.","None.",4)
A[c+"Feasibility"]=R("Inference-only on baseline checkpoints re-run under the shared protocol; all analyses are cheap.","None.",5)
A[c+"Significance"]=R("Determines whether MVFA's few-shot results hold under deployable scoring; applicable to other methods that normalise over the test set.","Check and report whether common baselines share the practice.",4)
c="stage1/review/B2/v1/"
A[c+"Clarity"]=R("Clear and complete, including the explicit transductive-in-features assumption.","None.",4)
A[c+"Relevance"]=R("Relevant to zero-shot deployment of the target method.","None.",4)
A[c+"Originality"]=R("The relation between guarantee violation and source-target feature distance adds a useful predictive element beyond applying conformal and weighted conformal prediction.","None.",4)
A[c+"Feasibility"]=R("Inference-only, trivial compute.","None.",5)
A[c+"Significance"]=R("Safety-relevant and transferable to other zero-shot detectors.","None.",4)
c="stage1/review/D1/v1/"
A[c+"Clarity"]=R("Initialisation, margin rule, scoring and comparisons are specified.","None.",4)
A[c+"Relevance"]=R("Relevant to G7 and G13.","None.",4)
A[c+"Originality"]=R("Tying prototypes to the normal text embedding is a modest vision-language twist on established compactness objectives.","None.",3)
A[c+"Feasibility"]=R("Tuning is now limited (two values of M, margin fixed from source statistics).","None.",5)
A[c+"Significance"]=R("Expected gains remain modest.","None.",3)
c="stage1/review/D2/v1/"
A[c+"Clarity"]=R("Clear, with an explicit hypothesis and comparison.","None.",4)
A[c+"Relevance"]=R("Relevant to G5.","None.",4)
A[c+"Originality"]=R("Support versus test-image conditioning is a small but testable twist.","None.",3)
A[c+"Feasibility"]=R("Cheap on one GPU.","None.",5)
A[c+"Significance"]=R("Modest; the sensitivity analysis is the most useful part.","None.",3)
json.dump(A, open("/home/claude/task3/logs/answers/run2_stage1_review_v1.json","w"), indent=1)
