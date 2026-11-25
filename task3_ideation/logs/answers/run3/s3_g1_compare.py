import sys, json
sys.path.insert(0, "/home/claude/task3/logs/answers"); from judge_lib import write_verdicts
D = "/home/claude/task3/logs/answers/run3/"
src = open(D + "s3_g0_compare.py").read()
A = {}
exec(src[src.index("A = {"):src.index("write_verdicts(A")])
A.update({
 "A1-g1-o2": ("keeps the anatomy-debiasing idea but replaces the alignment-dependent positional prior with distances to a bank of normal source patches, so it applies to every BMAD modality and is checkable against the masks that exist",
              "needs a per-dataset normal patch bank during training"),
 "A1-g1-o6": ("removes the arbitrary fixed k of the problem's operator with a per-dataset witness rate estimated from image labels by mixture-proportion estimation, and validates the estimate against true lesion areas",
              "relies on a mixture-proportion estimator that can be noisy when normal and abnormal score distributions overlap early in training"),
 "A1-g1-o1": ("uses cross-level agreement inside the MIL loss as a one-stage instance selector, a small and direct change to the problem's operator",
              "assumes the four levels' errors are independent, which may not hold for adapters trained jointly"),
 "A1-g1-o5": ("tests anatomy hard-negative mining in a clean 2 x 2 design with the anchor-frame control, giving both an improvement and an explanation",
              "keeps only the hard-negative part of the debiasing and so offers a smaller expected gain"),
 "A1-g1-o4": ("replaces correlational top-k selection with counterfactual occlusion evidence from the image-label detector, an original idea",
              "requires an extra training stage and token-level occlusion by normal-bank patches whose implementation and cost are uncertain"),
 "A1-g1-o3": ("densifies MIL with a transformation-equivariance consistency loss in a single stage",
              "uses a generic consistency regulariser known from weakly supervised segmentation and doubles the training cost"),
 "B1-g1-o3": ("separates location (local, positional median) from scale (one global leave-one-out empirical CDF), which keeps scores exactly test-set invariant and avoids estimating local spreads from a handful of supports at K = 2",
              "still assumes approximate spatial alignment for its location term"),
 "B1-g1-o5": ("turns the inductive score into a deployable decision rule with a distribution-free false-positive bound from the K supports, and measures support/test shift where the bound fails",
              "adds a thresholding layer rather than a better normaliser, so it does not change AUROC"),
 "B1-g1-o4": ("chooses each test image's normal reference among the supports by global similarity, which handles anatomical variation without assuming alignment",
              "uses only m of K supports, which makes the reference smaller and noisier at small K"),
 "B1-g1-o6": ("spends the scarce supports only on the few-shot branch and normalises the zero-shot branch from source statistics and anchor geometry",
              "assumes that the zero-shot branch's scale on the source modalities holds for the unseen modality"),
 "B1-g1-o2": ("localises the empirical CDF by neighbourhood with a principled mixing towards the global CDF",
              "estimates local CDFs from few values and adds two hyper-parameters"),
 "B1-g1-o1": ("combines an anatomy-aware low anchor with a synthetic high anchor",
              "inherits the uncertain realism of synthetic lesions and the alignment assumption at the same time"),
})
write_verdicts(A, ["A1-g1-o2", "A1-g1-o6", "A1-g0-C1", "A1-g1-o1", "A1-g0-C6", "A1-g1-o5", "A1-g1-o4", "A1-g0-C5", "A1-g1-o3",
                   "A1-g0-C2", "A1-g0-C3", "A1-g0-C4"], D + "tmp/a.json")
write_verdicts(A, ["B1-g1-o3", "B1-g1-o5", "B1-g0-C1", "B1-g1-o4", "B1-g1-o6", "B1-g1-o2", "B1-g0-C5", "B1-g0-C6", "B1-g0-C2",
                   "B1-g1-o1", "B1-g0-C4", "B1-g0-C3"], D + "tmp/b.json")
a = {k: v for k, v in json.load(open(D + "tmp/a.json")).items() if k.startswith("stage3/A1")}
b = {k: v for k, v in json.load(open(D + "tmp/b.json")).items() if k.startswith("stage3/B1")}
json.dump({**a, **b}, open(D + "s3_g1_compare.json", "w"), indent=1)
print(len(a), len(b))
