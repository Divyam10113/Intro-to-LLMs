"""Evaluate a model on a target's test set: image AUROC, pixel AUROC, AUPRO.

    python evaluate.py --target Liver                 # R0 (released checkpoint)
    python evaluate.py --target Liver --ckpt runs/X/seg_adapters.pth

Scores exactly as MVFA's test(): image score = sum over levels of the mean
detection-branch patch probability; pixel map = sum over levels of the
upsampled segmentation-branch probability. MVFA min-max normalises both
before the AUROC, which changes nothing (monotone), so it is skipped here.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import os

from common import (RESULTS_DIR, autocast, build_model, file_lock, level_scores, list_test_set, load_image,
                    load_mask, scores_to_image_score, scores_to_maps, text_features)
from metrics import aupro, image_auroc, pixel_auroc

RESULT_FIELDS = ["run_id", "config", "target", "seed", "steps", "lambda", "pi_json",
                 "img_auroc", "pix_auroc", "aupro", "gpu_minutes", "notes"]


class _TestSet(Dataset):
    def __init__(self, target, limit=None):
        self.items = list_test_set(target)
        if limit:   # smoke tests only: keep both classes
            n = [x for x in self.items if x[1] == 0][: limit // 2]
            a = [x for x in self.items if x[1] == 1][: limit - len(n)]
            self.items = n + a

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return load_image(p), load_mask(p, y)[0], y


@torch.no_grad()
def predict(model, tfeat, target, device, batch_size=32, limit=None,
            workers=int(os.environ.get("TASK4_WORKERS", 4))):
    ds = _TestSet(target, limit)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=torch.device(device).type == "cuda")
    labels, img_scores, maps, masks = [], [], [], []
    for x, m, y in dl:
        x = x.to(device, non_blocking=True)
        with autocast(device):
            _, seg, det = model(x)
        img_scores.append(scores_to_image_score(level_scores(det, tfeat)).float().cpu())
        maps.append(scores_to_maps(level_scores(seg, tfeat)).float().cpu())
        masks.append(m.bool())
        labels.append(y)
    return (torch.cat(labels).numpy(), torch.cat(img_scores).numpy(),
            torch.cat(maps).numpy(), torch.cat(masks).numpy(), [p for p, _ in ds.items])


def compute_metrics(labels, img_scores, maps, masks):
    return {"img_auroc": image_auroc(labels, img_scores),
            "pix_auroc": pixel_auroc(masks, maps),
            "aupro": aupro(masks, maps)}


def evaluate(model, tfeat, target, device, save_maps=None, limit=None):
    t0 = time.time()
    labels, s, maps, masks, paths = predict(model, tfeat, target, device, limit=limit)
    gpu_s = time.time() - t0
    if save_maps:
        Path(save_maps).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_maps, maps=maps.astype(np.float16), labels=labels,
                            img_scores=s, paths=np.array(paths))
    res = compute_metrics(labels, s, maps, masks)
    res["eval_gpu_seconds"] = gpu_s
    return res


def append_result(row, path=None):
    path = Path(path or RESULTS_DIR / "results.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):              # many concurrent jobs append here
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--ckpt", default=None, help="seg_adapters.pth from train.py; default: R0")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_maps", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images")
    ap.add_argument("--no_record", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    model, clip_model = build_model(a.target, a.device)
    if a.ckpt:
        model.seg_adapters.load_state_dict(torch.load(a.ckpt, map_location=a.device))
        det = Path(a.ckpt).with_name("det_adapters.pth")
        if det.exists():   # fully mask-free M / B0 (DECISIONS D3)
            model.det_adapters.load_state_dict(torch.load(det, map_location=a.device))
    tfeat = text_features(clip_model, a.device)[a.target]
    res = evaluate(model, tfeat, a.target, a.device, save_maps=a.save_maps, limit=a.limit)
    print({k: round(v, 4) for k, v in res.items()})
    if not a.no_record and not a.limit:
        append_result({"run_id": f"R0_{a.target}" if not a.ckpt else Path(a.ckpt).parent.name,
                       "config": "R0" if not a.ckpt else "eval", "target": a.target,
                       "img_auroc": res["img_auroc"], "pix_auroc": res["pix_auroc"], "aupro": res["aupro"],
                       "gpu_minutes": round((time.time() - t0) / 60, 2), "notes": ""})


if __name__ == "__main__":
    main()
