import sys, json
sys.path.insert(0, "/home/claude/task3/logs/answers"); from judge_lib import write_verdicts
A = {
 "A1-g0-C2": ("fixes a verified property of MVFA, the 14-pixel patch grid at 240 px that makes patch pseudo-masks blocky, with a cheap, training-free snapping step and a direct pseudo-mask Dice test",
           "adds only boundary refinement and cannot recover lesions that the seed rule missed"),
 "A1-g0-C4": ("targets the fragmentation caused by the 99th-percentile seed rule and uses DINOv2 affinities whose boundary adherence Spatial-FAD documents, with a CLIP-affinity ablation that isolates the prior",
           "depends on a second backbone and a propagation threshold that may leak into healthy anatomy on low-contrast CT"),
 "A1-g0-C1": ("introduces an ignore band so the segmentation adapters are trained only on confident pseudo-labels, a principled answer to label noise",
           "stacks three changes (re-centred prototypes, ignore band, new loss), and re-centring prototypes toward an abnormal image's mean can pull them toward the lesion"),
 "A1-g0-C3": ("combines realistic-but-noisy pseudo-masks with exact synthetic masks and makes the mixing ratio the testable variable",
           "relies on synthetic lesions whose realism on medical images is uncertain, so the gain may come from synthesis rather than from the pseudo-masks"),
 "A1-g0-C6": ("cleans isolated false positives with a small, cheap CRF on the patch grid",
           "is a generic smoothing step justified only by an industrial paper, and a CRF on a 17x17 grid has little spatial resolution to work with"),
 "A1-g0-C5": ("uses SAM2 to turn point evidence into object-shaped masks",
           "relies on SAM2, whose masks on medical modalities such as OCT and CT are unreliable without medical fine-tuning, and on evidence from industrial and endoscopy data only"),
 "B1-g0-C5": ("acts on a verified mechanism: MVFA's zero-shot branch is already a two-way softmax probability that test-set min-max destroys, so restoring it with a source-fitted temperature and fusing in log-odds is principled and directly testable through threshold drift",
           "requires source-episode fitting for the few-shot branch"),
 "B1-g0-C1": ("replaces test-set min-max with a support-defined empirical CDF that makes every image's score exactly invariant to the test set, with a clear ordered ablation",
           "inherits the noise of small support sets at K = 2, where the CDF has very few reference values"),
 "B1-g0-C2": ("learns a stable calibration scale on source-modality episodes and uses the target supports only for location, which directly addresses ECE",
           "assumes that a calibration fitted on source modalities transfers to an unseen modality, which the leave-one-modality-out split makes doubtful"),
 "B1-g0-C6": ("addresses the max-pixel aggregation with a region-level score and a spike-injection test",
           "addresses aggregation more than the transductive normalisation that the problem leads with"),
 "B1-g0-C3": ("keeps MVFA's two-anchor normalisation logic but takes both anchors from the target supports",
           "depends on synthetic lesions as the upper anchor, whose scores may not represent real lesions"),
 "B1-g0-C4": ("gives the few-shot distance an intrinsic scale through prototype radii",
           "changes the few-shot scoring itself as well as the normalisation, which confounds the comparison with MVFA's bank"),
}
n = write_verdicts(A, ["A1-g0-C2", "A1-g0-C4", "A1-g0-C1", "A1-g0-C3", "A1-g0-C6", "A1-g0-C5"], "/home/claude/task3/logs/answers/run2_qualsg/tmpA.json")
m = write_verdicts(A, ["B1-g0-C5", "B1-g0-C1", "B1-g0-C2", "B1-g0-C6", "B1-g0-C3", "B1-g0-C4"], "/home/claude/task3/logs/answers/run2_qualsg/tmpB.json")
a = json.load(open("/home/claude/task3/logs/answers/run2_qualsg/tmpA.json")); b = json.load(open("/home/claude/task3/logs/answers/run2_qualsg/tmpB.json"))
a = {k: v for k, v in a.items() if k.startswith("stage3/A1")}; b = {k: v for k, v in b.items() if k.startswith("stage3/B1")}
json.dump({**a, **b}, open("/home/claude/task3/logs/answers/run2_qualsg/s3_g0_compare.json", "w"), indent=1)
print(len(a), len(b))
