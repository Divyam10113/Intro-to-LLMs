"""Check the pi estimator against ground truth on the MASKED source datasets.

For Brain / RESC / Liver the true anomalous-patch fraction of abnormal images is
known from the masks (patch = 14x14 px cell, anomalous if > 50% covered). This
runs the same estimator as estimate_pi.py on R0's patch logits, for every level
and both branches, and prints estimate vs truth. CPU is fine (~3 min).

    python validate_pi.py --target Liver --device cpu
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import (FEATURES, build_model, has_masks, level_scores, load_image, load_mask, load_splits,
                    sources_for, text_features, write_json, RESULTS_DIR)
from pu import estimate_pi, two_class_to_logit
from set_pi import true_pi


@torch.no_grad()
def logits(model, tf, rels, branch, device):
    out = [[] for _ in FEATURES]
    for i in range(0, len(rels), 16):
        x = torch.stack([load_image(p) for p in rels[i:i + 16]]).to(device)
        _, seg, det = model(x)
        for l, sc in enumerate(level_scores(seg if branch == "seg" else det, tf)):
            out[l].append(two_class_to_logit(sc).float().cpu().numpy().ravel())
    return [np.concatenate(v) for v in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Liver")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, cm = build_model(a.target, a.device)
    tf = text_features(cm, a.device)
    res = {}
    for d in [d for d in sources_for(a.target) if has_masks(d)]:
        s = load_splits(d)
        res[d] = {"true_patch_pi": true_pi(d)}
        for br in ("seg", "det"):
            n, ab = logits(model, tf[d], s["pi_normal"], br, a.device), logits(model, tf[d], s["pi_abnormal"], br, a.device)
            res[d][br] = {f"L{f}": estimate_pi(n[l], ab[l])[1] for l, f in enumerate(FEATURES)}
            print(f"R0-{a.target} source {d} [{br}] true {res[d]['true_patch_pi']:.3f} | "
                  + "  ".join(f"{k}: {v:.3f}" for k, v in res[d][br].items()))
    write_json(RESULTS_DIR / f"pi_validation_{a.target}.json", res)


if __name__ == "__main__":
    main()
