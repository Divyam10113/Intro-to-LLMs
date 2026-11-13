"""
Stage 0b — Code verification of the Task 2 gaps (model-free).

Why: our Task 2 limitations were inferred from the MVFA paper only. ResearchAgent builds problems
from whatever gap descriptions it is given, so a gap that misdescribes the released code produces a
problem that collapses when it is implemented in that code (the next assignment). This stage checks
each implementation-level claim against the public MVFA repository and writes corrected gap records.

Input : data/mvfa_code/  (snapshot of github.com/MediaBrain-SJTU/MVFA-AD, commit a30bf7b, 2024-04-06)
        outputs/stage0_gaps.json
Output: outputs/stage0b_code_verification.json   (evidence: file, line, code, finding)
        outputs/gaps_verified.json                (gap records + groups used by Stage 1 onwards)

Every piece of evidence is located by a regular expression, so the script fails loudly if the code
does not contain what we claim.
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
]

CORRECTIONS = {
    "G3": dict(
        title="Localisation supervision requires dense masks",
        text="MVFA trains two branches. The detection (classification) adapters receive an image-level BCE loss on "
             "every source image, including the three BMAD datasets that have no masks (OCT17, ChestXray, HIS) "
             "[E6, E8]. The segmentation adapters, which produce the anomaly maps, are trained only with Dice and "
             "Focal losses on pixel-level masks, i.e. only on BrainMRI, LiverCT and RESC [E7]. Localisation therefore "
             "depends entirely on dense expert masks from three datasets; no image-level, sparse or normal-only "
             "supervision of the segmentation branch is formulated.",
        evidence=["E6", "E7", "E8"], change="corrected"),
    "G4": dict(
        title="The few-shot setting needs labelled abnormal images with masks and is evaluated on one draw",
        text="In the released code, 'K-shot' means K normal AND K abnormal images, the abnormal ones with pixel masks, "
             "both drawn from BMAD's valid/ split [E15, E16]; the adapters are fine-tuned on them per target task. "
             "Clinical few-shot supervision may instead contain only normal or unlabelled images, which the method "
             "does not support. Only one fixed draw per dataset is provided [E17]. Training time and compute for the "
             "per-task fine-tuning are not reported.",
        evidence=["E15", "E16", "E17"], change="corrected"),
    "G8": dict(
        title="Score fusion relies on test-set statistics and fixed weights; hyper-parameters are under-ablated",
        text="Numerous design choices (gamma = 0.1, lambda = 1.0, four feature levels, input resolution 240, learning "
             "rate 1e-3, 50 epochs, fixed prompt templates) are presented without sensitivity analysis. In the few-shot "
             "branch fusion, the paper describes adding scores on different scales; the released code instead min-max "
             "normalises each branch using the minimum and maximum over the ENTIRE TEST SET before fusing them with a "
             "fixed weight of 0.5 [E9, E10, E12]. This normalisation is transductive: it cannot score a single "
             "incoming image, and the fused score of an image changes with the composition (e.g. prevalence, "
             "outliers) of the rest of the test set.",
        evidence=["E9", "E10", "E12"], change="corrected"),
    "G9": dict(
        title="Image-level aggregation is inconsistent and outlier-sensitive",
        text="The image-level score is computed differently in training and testing and across datasets: training "
             "mean-pools the detection map [E5]; for datasets with masks, the reported image AUC uses the MAX over the "
             "fused segmentation map, so a single spurious high-scoring pixel decides the image label [E11]; for "
             "datasets without masks, mean-pooled detection scores are used [E12]. No robust aggregation, "
             "calibration or uncertainty estimate is used.",
        evidence=["E5", "E11", "E12"], change="corrected"),
    "G10": dict(
        title="Evaluation lacks patient-level leakage controls and variance estimates",
        text="Patient-level splits are not reported. Few-shot support images come from a single fixed file per "
             "dataset and shot number [E13, E17] with one default seed [E14]; no standard deviations or confidence "
             "intervals are reported, although few-shot selection is known to be variance-sensitive.",
        evidence=["E13", "E14", "E17"], change="evidence added"),
}

NEW_GAPS = [
    dict(id="G16", title="Model selection on the target test set",
         text="Both the zero-shot and few-shot training scripts evaluate on the target dataset's test set during "
              "training and keep the checkpoint with the best test AUC [E1-E4]. BMAD's valid/ split exists and is "
              "used to draw few-shot supports [E15, E16], but not for model selection. In the "
              "zero-shot setting this means the target modality, which is supposed to be unseen, is used to choose "
              "the model, so the reported zero-shot and few-shot results are optimistically biased and are not "
              "comparable with methods that select models without test data.",
         evidence=["E1", "E2", "E3", "E4", "E15", "E16"], component="evaluation protocol", kind="evaluation", codable="high",
         change="new"),
    dict(id="G17", title="Paper and code disagree on image-level pooling",
         text="Eq. (3) of the paper defines the image-level loss with a max over the h x w anomaly map, while the "
              "released code mean-pools [E5]. Results in the paper therefore cannot be attributed to the method as "
              "described, and the effect of the pooling choice on detection is unknown.",
         evidence=["E5"], component="score fusion/aggregation", kind="reporting", codable="high", change="new"),
]

# Found while preparing Stage 2 (after Stage 1 had run). Recorded as evidence, but deliberately NOT merged into
# the gap texts, so that the Stage 1 prompts (and their cached responses) remain exactly what was run.
LATER_CHECKS = [
    dict(id="E18", file="train_zero.py", pattern=r'"--learning_rate", type=float, default=0\.0001',
         finding="Zero-shot training defaults to a learning rate of 1e-4, while the paper reports 1e-3 "
                 "(train_few.py uses 1e-3)."),
    dict(id="E19", file="train_zero.py", pattern=r"train_loader = torch\.utils\.data\.DataLoader\(train_dataset, batch_size=1",
         finding="Training uses a batch size of 1 (hard-coded), while the paper reports a batch size of 16."),
    dict(id="E20", file="train_few.py", pattern=r"train_loader = torch\.utils\.data\.DataLoader\(train_dataset, batch_size=1",
         finding="Few-shot training also uses a hard-coded batch size of 1."),
]

GROUP_UPDATES = {  # new gaps join the groups whose root cause they share
    "Score calibration and robust aggregation": ["G17"],
    "Leakage-free generalisation protocol": ["G16"],
}
GROUP_ROOT_CAUSE = {
    "Score calibration and robust aggregation":
        "Anomaly evidence is turned into image scores and fused with rules that depend on the whole test set "
        "(min-max normalisation), a fixed weight and a max over pixels, and is evaluated only by threshold-free AUC, "
        "so per-image deployability, calibration and outlier sensitivity are never measured.",
    "Leakage-free generalisation protocol":
        "Model selection on the target test set, same-modality data (RESC/OCT17), possible patient overlap and "
        "single few-shot draws all leak target information into 'zero-shot' evaluation.",
}


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
        g = dict(g, change="unchanged", evidence=[])
        if g["id"] in CORRECTIONS:
            g.update(CORRECTIONS[g["id"]])
        gaps.append(g)
    gaps += NEW_GAPS
    groups = []
    for grp in s0["groups"]:
        grp = dict(grp)
        grp["members"] = grp["members"] + GROUP_UPDATES.get(grp["name"], [])
        if grp["name"] in GROUP_ROOT_CAUSE:
            grp["root_cause"] = GROUP_ROOT_CAUSE[grp["name"]]
        groups.append(grp)
    later = []
    for c in LATER_CHECKS:
        locs = locate(c)
        later.append(dict(id=c["id"], file=c["file"], lines=[l["line"] for l in locs], code=locs[0]["code"],
                          finding=c["finding"], related_gaps=["G8", "G17"],
                          url=f"https://github.com/MediaBrain-SJTU/MVFA-AD/blob/{COMMIT}/{c['file']}#L{locs[0]['line']}"))
    (ROOT / "outputs" / "stage0b_code_verification.json").write_text(json.dumps(
        dict(repository="https://github.com/MediaBrain-SJTU/MVFA-AD", commit=COMMIT, evidence=evidence,
             later_findings=later,
             later_findings_note="Found after Stage 1 ran; recorded here and in the report, not merged into gap texts."),
        indent=1))
    for e in later:
        print(f"{e['id']:4} {e['file']}:{e['lines']}  {e['code'][:80]}  (later finding)")
    (ROOT / "outputs" / "gaps_verified.json").write_text(json.dumps(
        dict(gaps=gaps, groups=groups, ungrouped=s0["ungrouped"], source="stage0 + stage0b code verification"), indent=1))
    for e in evidence:
        print(f"{e['id']:4} {e['file']}:{e['lines']}  {e['code'][:80]}")
    print("changed gaps:", [g["id"] + ":" + g["change"] for g in gaps if g["change"] != "unchanged"])
    for grp in groups:
        print(f"- {grp['name']}: {grp['members']}")


if __name__ == "__main__":
    main()
