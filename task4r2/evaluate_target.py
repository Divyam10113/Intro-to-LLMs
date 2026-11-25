"""The ONLY round 2 code that reads a target test set (plan §5, rule 1).

Called by train2.py after training has ended and the checkpoint has been chosen
on source validation; nothing computed here feeds back into any choice.
Datasets with pixel masks (Brain, Liver, RESC): image AUROC, pixel AUROC, AUPRO,
exactly as round 1 (MVFA's scoring). Datasets without masks (OCT17, Chest, HIS):
image AUROC only, and no maps are kept (Chest has 17k test images).

    python evaluate_target.py --run_dir /mnt/HDD1/divyam/task4/r2/runs/<run_id>   # re-evaluate a saved run
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common2 import has_masks
from common import (autocast, level_scores, list_test_set, load_image, load_mask, scores_to_image_score,
                    scores_to_maps)
from metrics import aupro, image_auroc, pixel_auroc

_READS = []          # audit trail for the tests: which datasets' test sets were read


class _Test(Dataset):
    def __init__(self, dataset, limit=None):
        self.items, self.masks = list_test_set(dataset), has_masks(dataset)
        if limit:
            n = [x for x in self.items if x[1] == 0][: limit // 2]
            self.items = n + [x for x in self.items if x[1] == 1][: limit - len(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        m = load_mask(p, y)[0] if self.masks else torch.zeros(0)
        return load_image(p), m, y


@torch.no_grad()
def evaluate(model, tfeat, dataset, device, batch_size=32, limit=None, save_maps=None, workers=4):
    _READS.append(dataset)
    ds = _Test(dataset, limit)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=torch.device(device).type == "cuda")
    labels, scores, maps, masks = [], [], [], []
    for x, m, y in dl:
        x = x.to(device, non_blocking=True)
        with autocast(device):
            _, seg, det = model(x)
        scores.append(scores_to_image_score(level_scores(det, tfeat)).float().cpu())
        labels.append(y)
        if ds.masks:
            maps.append(scores_to_maps(level_scores(seg, tfeat)).half().cpu())
            masks.append(m.bool())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    res = {"img_auroc": image_auroc(labels, scores), "n_test": len(labels)}
    if ds.masks:
        maps, masks = torch.cat(maps).float().numpy(), torch.cat(masks).numpy()
        res["pix_auroc"] = pixel_auroc(masks, maps)
        res["aupro"] = aupro(masks, maps)
        if save_maps:
            np.savez_compressed(save_maps, maps=maps.astype(np.float16), labels=labels, img_scores=scores,
                                paths=np.array([p for p, _ in ds.items]))
    return res


def main():
    from pathlib import Path
    from common import text_features
    from train2 import build_r2_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import json
    meta = json.load(open(Path(a.run_dir) / "run.json"))
    model, clip_model = build_r2_model(meta["seed"], a.device)
    ck = torch.load(Path(a.run_dir) / "adapters.pth", map_location=a.device)
    model.seg_adapters.load_state_dict(ck["seg_adapters"])
    model.det_adapters.load_state_dict(ck["det_adapters"])
    tf = text_features(clip_model, a.device)
    from common2 import held_out
    for d in held_out(meta["fold"]):
        print(d, evaluate(model, tf[d], d, a.device))


if __name__ == "__main__":
    main()
