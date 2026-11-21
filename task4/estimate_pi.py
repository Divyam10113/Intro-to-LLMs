"""Run the plan's pi estimator once per target with the R0 model, on each
source's held-out pi lists (never trained on), at the deepest level.

    python estimate_pi.py --target Liver --branch seg   # mask-less sources -> results/pi_estimated_seg.json
    python estimate_pi.py --target Liver --branch det   # all sources       -> results/pi_estimated_det.json

REPORT ONLY: these values no longer set pi for any run (report/DECISIONS.md
D1, D2, D6), because the estimator's mixture assumption fails on MVFA's patch
scores (issues log B11). Training reads pi.json / pi_maskfree.json from set_pi.py.
Each source dataset is scored with its own text prompt, as in training; the
estimator works on logits z = s1 - s0 (rank-equivalent to probabilities, never saturates).
"""
import argparse
import json

import numpy as np
import torch

from common import (DEEPEST, MASKLESS, RESULTS_DIR, autocast, build_model, level_scores, load_image,
                    load_splits, sources_for, text_features, write_json)
from pu import estimate_pi, two_class_to_logit


@torch.no_grad()
def patch_probs(model, tfeat, rels, branch, device, bs=16):
    out = []
    for i in range(0, len(rels), bs):
        x = torch.stack([load_image(p) for p in rels[i:i + bs]]).to(device)
        with autocast(device):
            _, seg, det = model(x)
        s = level_scores(seg if branch == "seg" else det, tfeat)[DEEPEST]
        out.append(two_class_to_logit(s).float().cpu().numpy().ravel())
    return np.concatenate(out)


def run(target, branch, device, out_path=None, model=None, clip_model=None):
    if model is None:
        model, clip_model = build_model(target, device)
    tf = text_features(clip_model, device)
    datasets = [d for d in sources_for(target) if (branch == "det" or d in MASKLESS)]
    out_path = out_path or RESULTS_DIR / f"pi_estimated_{branch}.json"
    allpi = json.loads(out_path.read_text()) if out_path.exists() else {}
    allpi[target] = {}
    for d in datasets:
        s = load_splits(d)
        pn = patch_probs(model, tf[d], s["pi_normal"], branch, device)
        pa = patch_probs(model, tf[d], s["pi_abnormal"], branch, device)
        pi, raw = estimate_pi(pn, pa)
        allpi[target][d] = {"pi": pi, "pi_raw": raw, "branch": branch, "level": "deepest (24)",
                            "n_normal_imgs": len(s["pi_normal"]), "n_abnormal_imgs": len(s["pi_abnormal"]),
                            "n_normal_patches": int(pn.size), "n_abnormal_patches": int(pa.size)}
        print(f"{target} {branch} {d}: pi={pi:.4f} (raw {raw:.4f})")
    write_json(out_path, allpi)
    return allpi[target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--branch", choices=["seg", "det"], required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run(a.target, a.branch, a.device)


if __name__ == "__main__":
    main()
