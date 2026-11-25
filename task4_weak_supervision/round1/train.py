"""Task 4 fine-tunes: M-h, M-h-ctrl, M (mask-free PU), B0 (mask-free top-k MIL).

    python train.py --config mh      --target Liver --seed 0
    python train.py --config mh_ctrl --target Liver --seed 0
    python train.py --config m       --target Liver --seed 0
    python train.py --config b0      --target Liver --seed 0
    python train.py --config mh --target Liver --seed 0 --lam 0.5 --tag lam0.5    # the one allowed exception
    python train.py --config m  --target Liver --seed 0 --steps 4000 --tag 2S     # optional under-training check
    python train.py --config mh --target Liver --seed 0 --pi_override 0.30 --tag pi0.30   # sensitivity (DECISIONS D1)
    python train.py --config m  --target Liver --seed 0 --det_init r0 --tag r0det         # plan's M (DECISIONS D3)

Always: CLIP frozen; Adam(lr 1e-4, betas (0.5, 0.999)) as MVFA's zero-shot
script; fixed number of steps; the LAST step is the final model (no checkpoint
selection).
  mh, mh_ctrl: R0 start; only the segmentation adapters train; detection frozen.
  m, b0 (--det_init fresh, default; DECISIONS D3): BOTH adapter sets
           re-initialised; segmentation adapters trained with PU / top-k MIL,
           detection adapters with MVFA's detection loss (image labels); no mask
           is ever loaded. --det_init r0 keeps R0's detection adapters frozen
           (the plan's original M; not strictly mask-free, issues log B2).
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (MVFA_DIR, RESULTS_DIR, RUNS_DIR, autocast, build_model, level_scores, load_image,
                    load_mask, sources_for, text_features, write_json)
from evaluate import append_result, compute_metrics, predict
from pu import pu_risk, topk_mil_loss, two_class_to_logit
from schedule import build_schedule

import sys
sys.path.insert(0, str(MVFA_DIR))
from loss import BinaryDiceLoss, FocalLoss  # noqa: E402

MASK_FREE = ("m", "b0")
_focal, _dice = FocalLoss(), BinaryDiceLoss()


class MaskLeakError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- losses
def mvfa_mask_loss(scores, mask):
    """MVFA's segmentation loss, unchanged: per level, upsample the 2-class
    scores to 240, softmax, Focal + Dice against the mask; summed over levels."""
    import torch.nn.functional as F
    loss = 0
    for s in scores:
        B, L, _ = s.shape
        H = int(round(L ** 0.5))
        m = F.interpolate(s.permute(0, 2, 1).reshape(B, 2, H, H), size=mask.shape[-1],
                          mode="bilinear", align_corners=True)
        m = torch.softmax(m, dim=1)
        loss = loss + _focal(m, mask) + _dice(m[:, 1], mask)
    return loss


def mvfa_det_loss(det_scores, labels):
    """MVFA's detection loss, unchanged (train_zero.py:125-131): per level, the
    mean patch anomaly probability goes through BCEWithLogitsLoss against the
    image label, summed over levels. Note MVFA feeds a probability where a
    logit is expected; replicated as is (DECISIONS D4, issues log B12)."""
    bce = torch.nn.BCEWithLogitsLoss()
    loss = 0
    for s in det_scores:
        score = torch.softmax(s.float(), dim=-1)[..., 1].mean(dim=-1)
        loss = loss + bce(score, labels)
    return loss


def compute_loss(config, kind, scores, mask=None, pi=None, lam=1.0):
    """scores: list over levels of [B, L, 2].
    kind 'mask': B = 1, MVFA mask loss. kind 'pair': B = 2 (normal, abnormal)."""
    if config in MASK_FREE and mask is not None:
        raise MaskLeakError(f"a mask tensor reached the loss in mask-free config {config!r}")
    if kind == "mask":
        assert config in ("mh", "mh_ctrl") and mask is not None
        return mvfa_mask_loss(scores, mask)
    assert kind == "pair" and config != "mh_ctrl"
    total = 0
    for s in scores:
        z = two_class_to_logit(s.float())
        if config in ("mh", "m"):
            total = total + pu_risk(z[0], z[1], pi)
        else:  # b0
            total = total + 0.5 * (topk_mil_loss(z[0], 0) + topk_mil_loss(z[1], 1))
    return lam * total if config == "mh" else total


# ----------------------------------------------------------------------------- data
class _SchedSet(Dataset):
    def __init__(self, sched, load_masks):
        self.sched, self.load_masks = sched, load_masks

    def __len__(self):
        return len(self.sched)

    def __getitem__(self, i):
        kind, d, a, b = self.sched[i]
        if kind == "mask":
            assert self.load_masks, "mask item in a mask-free schedule"
            return kind, d, load_image(a)[None], load_mask(a, b)[None]
        return kind, d, torch.stack([load_image(a), load_image(b)]), torch.zeros(0)


def _collate(x):
    return x


# ----------------------------------------------------------------------------- main
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def run(a):
    t_start = time.time()
    setup_seed(a.seed)
    dev = torch.device(a.device)
    run_id = f"{a.config}_{a.target}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out = RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)

    train_det = a.config in MASK_FREE and a.det_init == "fresh"
    model, clip_model = build_model(a.target, a.device, load_r0=True, reinit_seg=a.config in MASK_FREE,
                                    reinit_det=train_det)
    for p in model.seg_adapters.parameters():
        p.requires_grad_(True)
    if train_det:
        for p in model.det_adapters.parameters():
            p.requires_grad_(True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    allowed = ("seg_adapters.", "det_adapters.") if train_det else ("seg_adapters.",)
    assert trainable and all(n.startswith(allowed) for n in trainable), f"only {allowed} may train"
    # two optimizers, as MVFA (same hyperparameters)
    opts = [torch.optim.Adam(model.seg_adapters.parameters(), lr=a.lr, betas=(0.5, 0.999))]
    if train_det:
        opts.append(torch.optim.Adam(model.det_adapters.parameters(), lr=a.lr, betas=(0.5, 0.999)))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    tf = text_features(clip_model, a.device)

    pi, pi_file = {}, ""
    if a.config in ("mh", "m"):
        if a.pi_override is not None:
            pi_file = f"override {a.pi_override}"
            pi = {d: a.pi_override for d in sources_for(a.target)}
        else:
            pi_file = str(RESULTS_DIR / ("pi.json" if a.config == "mh" else "pi_maskfree.json"))
            pi = {d: v["pi"] for d, v in json.load(open(pi_file))[a.target].items()}

    sched = build_schedule(a.config, a.target, a.seed, a.steps)
    dl = DataLoader(_SchedSet(sched, load_masks=a.config not in MASK_FREE), batch_size=None,
                    shuffle=False, num_workers=a.workers, collate_fn=_collate,
                    pin_memory=dev.type == "cuda", prefetch_factor=8 if a.workers else None)

    losses = []
    t0 = time.time()
    for step, (kind, d, x, m) in enumerate(dl):
        x = x.to(dev, non_blocking=True)
        mask = m.to(dev) if kind == "mask" else None
        with autocast(a.device):
            _, seg, det = model(x)
        loss = compute_loss(a.config, kind, level_scores(seg, tf[d]), mask=mask, pi=pi.get(d), lam=a.lam)
        if train_det:   # pair = (normal, abnormal): image labels only
            loss = loss + mvfa_det_loss(level_scores(det, tf[d]), torch.tensor([0.0, 1.0], device=dev))
        for o in opts:
            o.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        for o in opts:
            scaler.step(o)
        scaler.update()
        losses.append((step, kind, d, float(loss.detach())))
        if a.log_every and step % a.log_every == 0:
            print(f"[{run_id}] step {step}/{len(sched)} {kind:4s} {d:15s} loss {losses[-1][3]:.4f} "
                  f"({(time.time() - t0) / (step + 1):.3f}s/step)", flush=True)
    train_s = time.time() - t0
    torch.save(model.seg_adapters.state_dict(), out / "seg_adapters.pth")
    if train_det:
        torch.save(model.det_adapters.state_dict(), out / "det_adapters.pth")
    with open(out / "loss.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "kind", "dataset", "loss")] + losses)

    # under-training diagnostic: last 10% vs the 10% before it
    lv = np.array([l for *_, l in losses])
    n10 = max(1, len(lv) // 10)
    falling = len(lv) >= 20 and lv[-n10:].mean() < 0.95 * lv[-2 * n10:-n10].mean()

    notes = []
    res = {}
    if not a.no_eval:
        t1 = time.time()
        labels, s, maps, masks, paths = predict(model, tf[a.target], a.target, a.device, limit=a.eval_limit)
        eval_s = time.time() - t1
        gpu_min = (time.time() - t_start) / 60
        del model, clip_model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if a.save_maps:
            np.savez_compressed(out / "maps.npz", maps=maps.astype(np.float16), labels=labels,
                                img_scores=s, paths=np.array(paths))
        res = compute_metrics(labels, s, maps, masks)
        if falling:
            notes.append("loss still falling (last 10% < 0.95x previous 10%): possibly under-trained")
        if a.config in MASK_FREE:
            notes.append(f"det_init={a.det_init}" + ("" if train_det else " (R0 det adapters: not strictly mask-free)"))
        if a.pi_override is not None:
            notes.append(f"pi_override={a.pi_override} (sensitivity)")
        slots = int(os.environ.get("TASK4_SLOTS", 1))
        if slots > 1:
            notes.append(f"GPU shared by up to {slots} jobs: gpu_minutes is wall time on a shared card")
        if a.eval_limit:
            notes.append(f"SMOKE: eval on {a.eval_limit} images only")
        row = {"run_id": run_id, "config": a.config, "target": a.target, "seed": a.seed, "steps": a.steps,
               "lambda": a.lam if a.config == "mh" else (0 if a.config == "mh_ctrl" else ""),
               "pi_json": pi_file, **res, "gpu_minutes": round(gpu_min, 2), "notes": "; ".join(notes)}
        if not a.eval_limit:
            append_result(row)
        res.update(eval_seconds=eval_s)
    meta = {**vars(a), "run_id": run_id, "jobs_per_gpu": int(os.environ.get("TASK4_SLOTS", 1)), "pi": pi, "train_seconds": train_s, "n_updates": len(sched),
            "seconds_per_update": train_s / max(1, len(sched)), "loss_still_falling": bool(falling),
            "total_seconds": time.time() - t_start, **res, "notes": notes}
    write_json(out / "run.json", meta)
    print(json.dumps({k: meta[k] for k in ("run_id", "n_updates", "seconds_per_update", "total_seconds")
                      + tuple(k for k in ("img_auroc", "pix_auroc", "aupro") if k in meta)}, indent=1))
    return meta


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["mh", "mh_ctrl", "m", "b0"], required=True)
    ap.add_argument("--target", choices=["Liver", "Brain", "Retina_RESC"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000, help="S iterations")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TASK4_WORKERS", 4)))
    ap.add_argument("--tag", default="")
    ap.add_argument("--det_init", choices=["fresh", "r0"], default="fresh",
                    help="m/b0 only: fresh = detection adapters re-initialised and trained with image labels")
    ap.add_argument("--pi_override", type=float, default=None, help="same pi for every source (sensitivity runs)")
    ap.add_argument("--save_maps", action="store_true", help="save test maps (for Liver heat maps)")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--eval_limit", type=int, default=None, help="smoke test only")
    ap.add_argument("--log_every", type=int, default=100)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
