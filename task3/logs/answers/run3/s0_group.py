import json
T = {}
def t(ids, comp, kind, cod):
    for i in ids.split(): T[i] = dict(component=comp, kind=kind, codable=cod)
t("L1 L11 L20 L22 L27 L37", "evaluation protocol", "evaluation", "high")
t("L2", "evaluation protocol", "evaluation", "medium")
t("L3 L4", "supervision/data", "method", "high")
t("L5 L18", "supervision/data", "method", "high")
t("L6", "score fusion/aggregation", "method", "high")
t("L7 L12 L26 L37 L39", "metrics/reporting", "reporting", "high")
t("L8", "metrics/reporting", "evaluation", "high")
t("L9", "prompting/text branch", "method", "high")
t("L10 L30 L38", "score fusion/aggregation", "method", "high")
t("L13", "evaluation protocol", "reporting", "medium")
t("L14 L24 L25", "visual adaptation/architecture", "evaluation", "high")
t("L15", "efficiency/privacy", "evaluation", "high")
t("L16", "visual adaptation/architecture", "method", "medium")
t("L17", "efficiency/privacy", "method", "medium")
t("L19", "metrics/reporting", "reporting", "high")
t("L21", "clinical scope/validation", "evaluation", "low")
t("L23", "visual adaptation/architecture", "evaluation", "high")
t("L28", "metrics/reporting", "reporting", "high")
t("L29", "metrics/reporting", "reporting", "high")
t("L31", "evaluation protocol", "evaluation", "high")
t("L32", "visual adaptation/architecture", "evaluation", "medium")
t("L33", "visual adaptation/architecture", "method", "medium")
t("L34", "evaluation protocol", "evaluation", "low")
t("L35", "clinical scope/validation", "reporting", "low")
t("L36", "prompting/text branch", "method", "medium")
T["L37"] = dict(component="evaluation protocol", kind="evaluation", codable="high")
assert len(T) == 39, len(T)
R = {"tags": T,
 "groups": [
  {"name": "Label-efficient adaptation", "members": ["L3", "L4", "L5", "L18"],
   "root_cause": "Both the zero-shot source training and the few-shot adaptation rely on dense pixel masks and labelled abnormal images, whose amount and necessity are never quantified, and no route exists for supervising the segmentation branch with cheaper labels."},
  {"name": "Score calibration and robust aggregation", "members": ["L8", "L10", "L30", "L38"],
   "root_cause": "Anomaly evidence from two differently scaled branches is turned into image scores by unspecified fixed rules (a 0.5/0.5 sum, a max over patches) and judged only by threshold-free AUC, so calibration, operating points and outlier sensitivity are never assessed."},
  {"name": "Leakage-free generalisation protocol", "members": ["L1", "L2", "L11", "L20", "L22", "L27", "L37"],
   "root_cause": "The evaluation protocol lets target information or chance leak into the results: a single support draw, no validation-based model selection, overlapping retinal-OCT sources, unverified patient-level splits and single folds."},
  {"name": "Compact normality modelling", "members": ["L6", "L15", "L17"],
   "root_cause": "Normality is represented by a per-target memory bank of raw normal patch features with per-target retraining; abnormal references are unused at test time and the memory and compute costs are never quantified."},
  {"name": "Attribution of the adapter design", "members": ["L14", "L23", "L24", "L25", "L31"],
   "root_cause": "The gains of the multi-level residual adapters are not separated from added capacity, the residual path, untuned constants or a simple supervised baseline."},
  {"name": "Text-branch and domain-invariant alignment", "members": ["L9", "L36"],
   "root_cause": "The alignment target is a fixed, unaudited pair of generic prompts, and nothing prevents the adapted features from keying on source-specific appearance instead of abnormality."},
  {"name": "Localisation resolution", "members": ["L16", "L33"],
   "root_cause": "Anomaly maps are limited to the 14-pixel ViT patch grid of independent 2D slices, so small lesions, boundaries and through-plane context are lost."}],
 "ungrouped": ["L7", "L12", "L13", "L19", "L21", "L26", "L28", "L29", "L32", "L34", "L35", "L39"],
 "promising": [
  {"group": "Score calibration and robust aggregation", "why": "The scoring rules are a small, self-contained part of inference that can be changed without retraining and measured with additional thresholded and calibration metrics on BMAD."},
  {"group": "Label-efficient adaptation", "why": "Replacing mask supervision of the segmentation branch is a single change to the training targets whose effect is directly measurable as a label-cost curve on the three masked datasets."},
  {"group": "Compact normality modelling", "why": "The memory bank can be replaced at inference time and its memory, latency and accuracy compared without retraining."}]}
json.dump({"stage0/group_gaps_task2": json.dumps(R, indent=1)}, open("logs/answers/run3/s0_group.json", "w"), indent=1)
