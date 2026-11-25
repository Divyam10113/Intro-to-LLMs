import sys; sys.path.insert(0, "/home/claude/task3/logs/answers")
from judge_lib import write_verdicts
exec(open("/home/claude/task3/logs/answers/stage3_g2_compare.py").read().split("A1_RANK = [")[0])  # A1, B1 dicts incl. gen2
A1.update({
 "A1-g3-o3": ("injects anatomy-aware per-patch evidence into PU learning through instance-dependent class priors, addressing PU's single-prior weakness within one objective",
              "adds the position-aware bank machinery on top of PU, so part of any gain may come from the bank rather than the PU formulation"),
 "A1-g3-o5": ("selects MIL positives only where text-based and memory-based evidence agree, a cheap way to remove salient normal structures from the positive set",
              "can fall back to plain top-k when the bank flags too few patches, leaving its benefit dataset-dependent"),
 "A1-g3-o6": ("distributes each image's abnormal label over patches by optimal transport with a lesion-mass budget, an established label-assignment idea from object detection",
              "depends on the lesion-fraction budget and adds Sinkhorn hyper-parameters"),
 "A1-g3-o2": ("needs neither a class prior nor k by treating patch labels as one-sided label noise and using co-teaching small-loss selection",
              "relies on a forget-rate schedule that implicitly encodes the lesion fraction anyway"),
 "A1-g3-o4": ("makes atlas-based MIL selection robust to skewed scores with rank statistics",
              "is an incremental refinement of a selection heuristic with limited expected gain"),
 "A1-g3-o1": ("adds a label-free equivariance constraint known from weakly supervised segmentation",
              "is a generic regulariser that does not change the supervision signal itself"),
})
B1.update({
 "B1-g3-o5": ("keeps Fisher's power while correcting its null for MVFA's correlated levels with Brown's method, using dependence estimated from the support normals",
              "estimates covariances of eight p-value terms from very few supports, which may be unstable at K = 2"),
 "B1-g3-o1": ("gives the anomaly map itself a statistical guarantee by controlling the false discovery rate of flagged patches under dependence",
              "may be conservative for small lesions because Benjamini-Yekutieli is strict under arbitrary dependence"),
 "B1-g3-o3": ("combines correlated evidence exactly with e-values and gives a guaranteed false-positive bound",
              "is likely to lose noticeable power relative to p-value combinations"),
 "B1-g3-o2": ("quantifies per-image uncertainty from the support draw by bootstrapping and decides on a lower bound",
              "addresses decision reliability rather than the fusion rule and multiplies scoring cost"),
 "B1-g3-o4": ("suppresses unstable single-view peaks by requiring consensus across augmented views",
              "is a generic robustness add-on that multiplies inference cost without changing fusion"),
 "B1-g3-o6": ("guards support-anchored scoring against acquisition shift between supports and test images",
              "targets acquisition shift rather than the transductive fusion and threshold transfer that define the problem"),
})
A1_RANK = ["A1-g2-o6","A1-g3-o3","A1-g0-C1","A1-g3-o5","A1-g2-o4","A1-g1-o1","A1-g3-o6","A1-g1-o6","A1-g3-o2","A1-g2-o5","A1-g3-o4","A1-g3-o1"]
B1_RANK = ["B1-g2-o6","B1-g3-o5","B1-g0-C2","B1-g3-o1","B1-g1-o2","B1-g2-o1","B1-g3-o3","B1-g1-o4","B1-g3-o2","B1-g2-o5","B1-g3-o4","B1-g3-o6"]
print(write_verdicts({**A1, **B1}, A1_RANK + B1_RANK, "/home/claude/task3/logs/answers/stage3_g3_compare.json"))
