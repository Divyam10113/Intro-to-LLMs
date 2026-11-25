import sys; sys.path.insert(0, "/home/claude/task3/logs/answers")
from judge_lib import write_verdicts
exec(open("/home/claude/task3/logs/answers/stage3_g1_compare.py").read().split("A1_RANK")[0])  # A1 dict
exec("B1 = " + open("/home/claude/task3/logs/answers/stage3_g1_compare.py").read().split("B1 = ")[1].split("B1_RANK")[0])
A1.update({
 "A1-g2-o6": ("casts mask-free training as patch-level positive-unlabelled learning with an unbiased non-negative risk estimator, using every patch rather than a top-k heuristic and estimating the needed class prior without masks",
              "depends on estimating the patch-level class prior, which can be biased for very small lesions"),
 "A1-g2-o4": ("fixes the normal-bank weakness of matching lesions to normal tissue elsewhere by restricting matches to the same anatomical neighbourhood, exploiting BMAD's registered brain slices",
              "helps only where images are spatially aligned and needs a per-dataset radius"),
 "A1-g2-o5": ("recovers full lesion extent with adversarial erasing, a proven remedy for partial weakly supervised localisation that matters for PRO",
              "adds training rounds and a heuristic for when erased images remain positive"),
 "A1-g2-o3": ("pairs cheap attribution proposals with counterfactual verification so that each component covers the other's weakness",
              "stacks two pseudo-mask stages, making the gain harder to attribute"),
 "A1-g2-o2": ("localises lesions as unmatched mass against a retrieved normal twin with unbalanced optimal transport",
              "assumes a sufficiently similar normal twin exists for every abnormal image and adds an OT pipeline"),
 "A1-g2-o1": ("captures context-dependent abnormality with a masked feature predictor trained on normal images",
              "requires training and tuning an additional predictor whose surprisal may be dominated by normal anatomical variability"),
})
B1.update({
 "B1-g2-o6": ("keeps per-image, support-anchored p-value fusion and region pooling but replaces Fisher's independence assumption with the harmonic mean p-value, which remains valid for MVFA's strongly correlated levels and branches",
              "can lose some power relative to Fisher when evidence is spread weakly across many levels"),
 "B1-g2-o1": ("lets real labelled source episodes learn how much each level and branch p-value should count, while inputs stay support-anchored and comparable across modalities",
              "adds a learned weighting whose transfer still rests on source-target similarity"),
 "B1-g2-o5": ("imports cohort score normalisation from speaker verification to make one threshold work across modalities",
              "requires cohort banks from source modalities and its benefit depends on cohort similarity"),
 "B1-g2-o2": ("applies the Neyman-Pearson lemma with a support-normal density and a source-abnormal density",
              "assumes that an abnormal feature density learned on source modalities transfers to the unseen one"),
 "B1-g2-o4": ("adds a risk-controlled defer option and risk-coverage evaluation",
              "reframes the output rather than fixing per-image fusion itself"),
 "B1-g2-o3": ("transfers calibration from the most similar source modalities",
              "builds on per-modality calibrators and a similarity heuristic, stacking assumptions"),
})
A1_RANK = ["A1-g2-o6","A1-g0-C1","A1-g2-o4","A1-g1-o1","A1-g1-o6","A1-g2-o5","A1-g1-o4","A1-g2-o3","A1-g2-o2","A1-g1-o2","A1-g2-o1","A1-g0-C2"]
B1_RANK = ["B1-g2-o6","B1-g0-C2","B1-g1-o2","B1-g2-o1","B1-g1-o4","B1-g2-o5","B1-g0-C1","B1-g2-o2","B1-g1-o6","B1-g2-o4","B1-g1-o3","B1-g2-o3"]
print(write_verdicts({**A1, **B1}, A1_RANK + B1_RANK, "/home/claude/task3/logs/answers/stage3_g2_compare.json"))
