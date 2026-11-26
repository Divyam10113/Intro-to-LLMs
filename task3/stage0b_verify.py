"""
Stage 0b — Code verification of the Task 2 limitations (model-free).

Why: Task 2's limitations were extracted from the MVFA paper and later literature, not from the released code.
Several say that the paper leaves something unspecified; the code answers them, and in some cases the answer is
a more serious problem than the limitation states. ResearchAgent builds problems from whatever descriptions it is
given, and the next assignment modifies this code, so every implementation-level claim is checked here.

Input : data/mvfa_code/  (snapshot of github.com/MediaBrain-SJTU/MVFA-AD, commit a30bf7b, 2024-04-06)
        outputs/stage0_gaps.json (Task 2's limitations L1..L39 and their groups)
Output: outputs/stage0b_code_verification.json   (evidence E1..E20: file, line, code, finding)
        outputs/gaps_verified.json                (limitations with a code-check note where one applies, plus
                                                   code-only limitations C1..C4, and the groups)

Every piece of evidence is located by a regular expression, so the script fails loudly if the code
does not contain what we claim. Verdicts: confirmed (the code shows the limitation holds), resolved (the code
answers what the paper leaves unspecified), worse (the code answers it and reveals a more serious problem),
corrected (the code differs from what the paper or the limitation states).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "data" / "mvfa_code"
COMMIT = "a30bf7bb2c51772a36e7be58d3c072be480620fe"

CHECKS = [
    dict(id="E1", file="train_zero.py", pattern=r"score = test\(args, model, test_loader",
         finding="Zero-shot training evaluates the model on the TARGET dataset's test loader five times per epoch "
                 "and saves the checkpoint with the best test score (model selection on the test set)."),
    dict(id="E2", file="train_zero.py", pattern=r"if score >= save_score:",
         finding="The saved zero-shot checkpoint is the one maximising test AUC (image AUC + pixel AUC)."),
    dict(id="E3", file="train_few.py", pattern=r"result = test\(args, model, test_loader",
         finding="Few-shot training likewise evaluates on the target test set every epoch ..."),
    dict(id="E4", file="train_few.py", pattern=r"if result > best_result:",
         finding="... and keeps the best-on-test checkpoint."),
    dict(id="E5", file="train_zero.py", pattern=r"anomaly_score = torch\.mean\(anomaly_map, dim=-1\)",
         finding="The image-level training loss pools the patch anomaly map with a MEAN, whereas Eq. (3) of the "
                 "paper states a MAX over h x w."),
    dict(id="E6", file="train_zero.py", pattern=r"det_loss \+= loss_bce\(anomaly_score, image_label\)",
         finding="Image-level BCE on the detection (classification) branch is applied to every training image."),
    dict(id="E7", file="train_zero.py", pattern=r"if seg_idx > 0:",
         finding="Dice + Focal pixel losses on the segmentation branch are applied only to datasets with masks "
                 "(seg_idx > 0: Brain, Liver, RESC)."),
    dict(id="E8", file="train_zero.py", pattern=r"loss = det_loss\s*$",
         finding="Datasets without masks (OCT17, Chest, Histopathology; seg_idx < 0) train only the detection "
                 "adapters with the image-level loss — they are used, but never supervise localisation."),
    dict(id="E9", file="test_few.py",
         pattern=r"seg_score_map_zero = \(seg_score_map_zero - seg_score_map_zero\.min\(\)\) / \(seg_score_map_zero\.max\(\)",
         finding="Before fusion, each branch's maps are min-max normalised using the minimum and maximum over the "
                 "ENTIRE TEST SET (transductive normalisation; not computable for a single incoming image)."),
    dict(id="E10", file="test_few.py", pattern=r"segment_scores = 0\.5 \* seg_score_map_zero \+ 0\.5 \* seg_score_map_few",
         finding="The normalised zero-shot and few-shot pixel maps are fused with a fixed weight of 0.5."),
    dict(id="E11", file="test_few.py", pattern=r"roc_auc_im = roc_auc_score\(gt_list, np\.max\(segment_scores_flatten, axis=1\)\)",
         finding="For datasets with masks, the image-level score is the MAX over the fused pixel map "
                 "(the detection branch is not used for image scores on these datasets)."),
    dict(id="E12", file="test_few.py",
         pattern=r"det_image_scores_zero = \(det_image_scores_zero - det_image_scores_zero\.min\(\)\)",
         finding="For datasets without masks, image scores are the test-set min-max normalised detection-branch "
                 "scores (mean-pooled), fused with weight 0.5."),
    dict(id="E13", file="dataset/medical_few.py", pattern=r"fewshot_seed/\{self\.class_name\}/\{self\.shot\}-shot\.txt",
         finding="Few-shot support images are read from one fixed file per dataset and shot number (a single draw)."),
    dict(id="E14", file="train_few.py", pattern=r"default=111",
         finding="A single default random seed (111)."),
    dict(id="E15", file="dataset/medical_few.py", pattern=r"img_dir = os\.path\.join\(self\.dataset_path, 'valid', 'good', 'img'\)",
         finding="Few-shot NORMAL support images are drawn from BMAD's valid/ split ..."),
    dict(id="E16", file="dataset/medical_few.py", pattern=r"img_dir = os\.path\.join\(self\.dataset_path, 'valid', 'Ungood', 'img'\)",
         finding="... and K ABNORMAL support images WITH masks are drawn from the same valid/ split, so 'K-shot' "
                 "means K normal + K abnormal annotated images (2K images). A validation split therefore exists "
                 "but is not used for model selection."),
    dict(id="E17", file="dataset/fewshot_seed/Brain/4-shot.txt", pattern=r"^a-0:",
         finding="Each seed file contains a single draw (a-0 / n-0) per dataset and shot number."),
    dict(id="E18", file="train_zero.py", pattern=r'"--learning_rate", type=float, default=0\.0001',
         finding="Zero-shot training defaults to a learning rate of 1e-4, while the paper reports 1e-3 "
                 "(train_few.py uses 1e-3)."),
    dict(id="E19", file="train_zero.py", pattern=r"train_loader = torch\.utils\.data\.DataLoader\(train_dataset, batch_size=1",
         finding="Training uses a batch size of 1 (hard-coded), while the paper reports a batch size of 16."),
    dict(id="E20", file="train_few.py", pattern=r"train_loader = torch\.utils\.data\.DataLoader\(train_dataset, batch_size=1",
         finding="Few-shot training also uses a hard-coded batch size of 1."),
]

CODE_NOTES = {  # limitation id -> (verdict, note appended to the limitation for Stage 1, evidence ids)
    "L1": ("confirmed", "The released code has one fixed support file per dataset and shot number containing a single "
           "draw, and a single default seed (111) [E13, E14, E17].", ["E13", "E14", "E17"]),
    "L3": ("confirmed", "In the released code every source image trains the detection adapters with an image-level BCE "
           "loss [E6], while the segmentation adapters, which produce the anomaly maps, are trained with Dice and Focal "
           "losses only on the datasets that have masks [E7, E8].", ["E6", "E7", "E8"]),
    "L4": ("resolved", "The released code answers this: 'K-shot' means K normal AND K abnormal images, the abnormal ones "
           "with pixel masks, both drawn from BMAD's valid/ split [E15, E16], from one fixed file per dataset and shot "
           "number [E13, E17].", ["E13", "E15", "E16", "E17"]),
    "L5": ("resolved", "In the released code the Dice and Focal losses are applied only to datasets with masks "
           "(BrainMRI, LiverCT, RESC) [E7]; OCT17, ChestXray and HIS train only the detection adapters [E8]; no "
           "pseudo-masks are used.", ["E7", "E8"]),
    "L18": ("confirmed", "Confirmed in the released code: the K abnormal support images are loaded together with their "
            "masks [E16].", ["E16"]),
    "L10": ("worse", "The released code shows what the paper leaves open: each branch's maps are min-max normalised with "
            "the minimum and maximum over the ENTIRE TEST SET and then fused with a fixed weight of 0.5 [E9, E10, E12] "
            "(see C1).", ["E9", "E10", "E12"]),
    "L30": ("resolved", "In the released code the image score on datasets with masks is the MAX of the fused pixel map "
            "[E11]; on datasets without masks it is the test-set-normalised, mean-pooled detection score [E12]; "
            "training mean-pools the detection map [E5].", ["E5", "E11", "E12"]),
    "L38": ("confirmed", "Confirmed in the released code for datasets with masks: the image score is the max over the "
            "fused pixel map [E11], so a single spurious pixel can decide the label.", ["E11"]),
    "L20": ("worse", "The released code does select a checkpoint, but on the TARGET TEST SET: training evaluates on the "
            "target test loader and keeps the best-on-test checkpoint [E1-E4], although BMAD's valid/ split exists and "
            "is used only to draw few-shot supports [E15, E16] (see C2).", ["E1", "E2", "E3", "E4", "E15", "E16"]),
    "L19": ("corrected", "The released code differs from the reported recipe: zero-shot training defaults to a learning "
            "rate of 1e-4 rather than 1e-3 [E18], and both training scripts hard-code a batch size of 1 rather than 16 "
            "[E19, E20] (see C4).", ["E18", "E19", "E20"]),
}

NEW_GAPS = [  # problems visible only in the code; each joins the group of the limitation it extends
    dict(id="C1", linked="L10", title="Scores are normalised with statistics of the whole test set",
         text="Before the two branches are fused, each branch's maps are min-max normalised using the minimum and "
              "maximum over the ENTIRE TEST SET, then combined with a fixed weight of 0.5 [E9, E10, E12]. This "
              "normalisation is transductive: it cannot score a single incoming image, and the fused score of an "
              "image changes with the composition (prevalence, outliers) of the rest of the test set.",
         evidence=["E9", "E10", "E12"], component="score fusion/aggregation", kind="method", codable="high"),
    dict(id="C2", linked="L20", title="Model selection on the target test set",
         text="Both the zero-shot and few-shot training scripts evaluate on the target dataset's test set during "
              "training and keep the checkpoint with the best test AUC [E1-E4]. BMAD's valid/ split exists and is "
              "used to draw few-shot supports [E15, E16], but not for model selection. In the zero-shot setting the "
              "target modality, which is supposed to be unseen, is used to choose the model, so the reported results "
              "are optimistically biased.",
         evidence=["E1", "E2", "E3", "E4", "E15", "E16"], component="evaluation protocol", kind="evaluation",
         codable="high"),
    dict(id="C3", linked="L38", title="Paper and code disagree on image-level pooling",
         text="Eq. (3) of the paper defines the image-level loss with a max over the h x w anomaly map, while the "
              "released code mean-pools [E5]. Results in the paper therefore cannot be attributed to the method as "
              "described, and the effect of the pooling choice on detection is unknown.",
         evidence=["E5"], component="score fusion/aggregation", kind="reporting", codable="high"),
    dict(id="C4", linked="L19", title="Training hyper-parameters in the code differ from the paper",
         text="Zero-shot training defaults to a learning rate of 1e-4 while the paper reports 1e-3 [E18], and both "
              "training scripts hard-code a batch size of 1 while the paper reports 16 [E19, E20].",
         evidence=["E18", "E19", "E20"], component="metrics/reporting", kind="reporting", codable="high"),
]

LATER_CHECKS = []  # run 3: all checks, including E18-E20 found during run 1, are regular checks


def locate(check):
    lines = (CODE / check["file"]).read_text().splitlines()
    hits = [i for i, l in enumerate(lines, 1) if re.search(check["pattern"], l)]
    if not hits:
        raise SystemExit(f"{check['id']}: pattern not found in {check['file']}")
    return [dict(line=i, code=lines[i - 1].strip()) for i in hits]


def main():
    evidence = []
    for c in CHECKS:
        locs = locate(c)
        evidence.append(dict(id=c["id"], file=c["file"], lines=[l["line"] for l in locs], code=locs[0]["code"],
                             finding=c["finding"],
                             url=f"https://github.com/MediaBrain-SJTU/MVFA-AD/blob/{COMMIT}/{c['file']}#L{locs[0]['line']}"))
    s0 = json.loads((ROOT / "outputs" / "stage0_gaps.json").read_text())
    gaps = []
    for g in s0["gaps"]:
        g = dict(g, code_check=None, evidence=[])
        if g["id"] in CODE_NOTES:
            verdict, note, ev = CODE_NOTES[g["id"]]
            g.update(code_check=dict(verdict=verdict, note=note), evidence=ev,
                     text=g["text"] + " Code check (released MVFA code): " + note)
        gaps.append(g)
    member_of = {m: grp["name"] for grp in s0["groups"] for m in grp["members"]}
    groups = [dict(grp, members=list(grp["members"])) for grp in s0["groups"]]
    ungrouped = list(s0["ungrouped"])
    for ng in NEW_GAPS:
        gaps.append(dict(ng, source="stage0b code verification", code_check=dict(verdict="code-only", note=ng["text"])))
        home = member_of.get(ng["linked"])
        if home:
            next(grp for grp in groups if grp["name"] == home)["members"].append(ng["id"])
        else:
            ungrouped.append(ng["id"])
    (ROOT / "outputs" / "stage0b_code_verification.json").write_text(json.dumps(
        dict(repository="https://github.com/MediaBrain-SJTU/MVFA-AD", commit=COMMIT, evidence=evidence,
             later_findings=[], verdicts={k: v[0] for k, v in CODE_NOTES.items()}), indent=1))
    (ROOT / "outputs" / "gaps_verified.json").write_text(json.dumps(
        dict(gaps=gaps, groups=groups, ungrouped=ungrouped, source="stage0 (Task 2 limitations) + stage0b code verification"),
        indent=1))
    for e in evidence:
        print(f"{e['id']:4} {e['file']}:{e['lines']}  {e['code'][:80]}")
    print("code checks:", {k: v[0] for k, v in CODE_NOTES.items()})
    for grp in groups:
        print(f"- {grp['name']}: {grp['members']}")
    print("ungrouped:", ungrouped)


if __name__ == "__main__":
    main()
