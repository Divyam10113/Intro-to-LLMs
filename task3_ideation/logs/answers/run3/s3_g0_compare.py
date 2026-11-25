import sys, json
sys.path.insert(0, "/home/claude/task3/logs/answers"); from judge_lib import write_verdicts
D = "/home/claude/task3/logs/answers/run3/"
A = {
 "A1-g0-C1": ("improves the problem's own MIL operator in a principled way — every patch of a normal image is a free pixel label — with a one-function change and a direct comparison against symmetric MIL",
              "is an incremental change to the loss whose gain may be small if normal images already constrain the maps through the detection branch"),
 "A1-g0-C6": ("turns the problem's central hypothesis (text anchors make weak localisation transfer) into a sharp, falsifiable test grounded in a published proof that adapters read anchors only through difference and scale, at almost no implementation cost",
              "is a control study that explains rather than improves localisation"),
 "A1-g0-C5": ("targets a concrete medical failure mode of MIL — top-k patches landing on salient anatomy — with two training-only corrections derived from image labels",
              "relies on a positional prior that assumes spatially aligned images, which does not hold for ChestXray crops or histopathology"),
 "A1-g0-C2": ("densifies MIL supervision through cross-level-agreement pseudo-masks with MVFA's existing losses",
              "is a two-stage self-training scheme that can reinforce the MIL model's own errors"),
 "A1-g0-C3": ("uses DINOv2 affinities, whose boundary adherence Spatial-FAD documents, to grow MIL seeds into regions",
              "adds a second backbone, a propagation step with several hyper-parameters and an extra training stage"),
 "A1-g0-C4": ("combines exact synthetic masks with realistic image-level MIL on real lesions",
              "confounds the effect of image-level labels with that of synthetic-lesion design, so it does not answer the problem's question cleanly"),
 "B1-g0-C1": ("replaces test-set min-max with a support-defined empirical CDF that uses all leave-one-out patch scores, makes every score exactly test-set invariant and keeps fusion and pooling fixed as the problem requires",
              "inherits the variability of the support set itself at K = 2"),
 "B1-g0-C5": ("exploits the spatial alignment of most BMAD modalities to remove anatomy-driven score offsets per position, a medical-specific and testable improvement over a global normaliser",
              "estimates per-position statistics from very few supports and does not apply to unaligned modalities"),
 "B1-g0-C6": ("derives the zero-shot branch's scale analytically from the anchor geometry, which Do Words Matter? shows is all the score depends on",
              "changes the zero-shot branch's representation as well as its normalisation, and its threshold-transfer claim is speculative"),
 "B1-g0-C2": ("learns a stable scale on simulated source episodes and sets only the location from the target supports, addressing K = 2 noise",
              "assumes that a scale fitted on source modalities transfers to an unseen modality, which the leave-one-modality-out split makes doubtful"),
 "B1-g0-C4": ("gives min-max a genuine high anchor through synthetic lesions in the target's own supports",
              "depends on synthetic lesions whose scores may not represent real lesions"),
 "B1-g0-C3": ("gives the few-shot distance an intrinsic scale through prototype radii and saves memory",
              "changes the few-shot scoring itself, which confounds the comparison with MVFA's memory bank"),
}
write_verdicts(A, ["A1-g0-C1", "A1-g0-C6", "A1-g0-C5", "A1-g0-C2", "A1-g0-C3", "A1-g0-C4"], D + "tmp/a.json")
write_verdicts(A, ["B1-g0-C1", "B1-g0-C5", "B1-g0-C6", "B1-g0-C2", "B1-g0-C4", "B1-g0-C3"], D + "tmp/b.json")
a = {k: v for k, v in json.load(open(D + "tmp/a.json")).items() if k.startswith("stage3/A1")}
b = {k: v for k, v in json.load(open(D + "tmp/b.json")).items() if k.startswith("stage3/B1")}
json.dump({**a, **b}, open(D + "s3_g0_compare.json", "w"), indent=1)
print(len(a), len(b))
