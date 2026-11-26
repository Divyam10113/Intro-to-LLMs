"""Round 2 training: every arm from scratch under Task 3's protocol.

    python train2.py --arm mvfa --fold Brain --seed 0
    python train2.py --arm pu   --fold Liver --seed 0 --pi 0.05 --tag pi5
    python train2.py --arm g_m100 --fold RESC_OCT --seed 1

Arms (decisions G1-n in report/ISSUES_LOG.md):
  mvfa     mask-supervised MVFA re-run: masked sources -> Dice+Focal + det loss;
           mask-less sources -> det loss only (MVFA steps only the det optimizer then)
  mil      plain top-k MIL (Task 3's A1 problem): every source image -> BCE(top-k mean p, y)
           at each level + det loss; no mask
  g        idea #1, normal-bank-gated MIL (A1-g1-o2): as mil, but an abnormal image's
           top-k is chosen by p*g (g from the frozen layer-24 bank, fixed), normal
           images' top-k by p with weight 2; no mask
  pu       refined idea #3: nnPU (round 1's pu_risk) with fixed pi on normal/abnormal
           pairs from one dataset + det loss on both; no mask
  pu_m100, g_m100   masked sources exactly as mvfa; mask-less sources get the pu / g loss
  g_m10, g_m25      budget points: a fixed seeded 10 % / 25 % of the masked sources'
           abnormal images keep their masks (mvfa loss); every other image gets the g loss
Common: CLIP frozen; both adapter sets freshly initialised from the run seed
(G1-10); Adam lr 1e-4 betas (0.5, 0.999), two optimizers as MVFA; fp16 autocast +
GradScaler (round 1 B8); --epochs passes over round 1's subsample (G1-8); after
each epoch, image AUROC on the sources' valid/ lists; the best epoch (ties -> later)
is the final model (G1-5). Target test sets are then read by evaluate_target.py only.
"""
import argparse
import copy
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common2 import (BANKS, FOLDS, K_FRAC, RESULTS2, RUNS2, assert_no_heldout, file_lock, has_masks, held_out,
                     load_gates, sources, train_lists, valid_lists, write_json)
from common import (autocast, build_model, level_scores, load_image, load_mask, scores_to_image_score,
                    text_features)
from metrics import image_auroc
from pu import pu_risk, two_class_to_logit
from train import MaskLeakError, mvfa_det_loss, mvfa_mask_loss
import evaluate_target

ARMS = ["mvfa", "mil", "g", "pu", "pu_m100", "g_m100", "g_m10", "g_m25"]
MASK_BUDGET = {"mvfa": 1.0, "pu_m100": 1.0, "g_m100": 1.0, "g_m10": 0.10, "g_m25": 0.25}
WEAK = {"mil": "mil", "g": "g", "pu": "pu", "pu_m100": "pu", "g_m100": "g", "g_m10": "g", "g_m25": "g"}
FIELDS = ["run_id", "arm", "fold", "eval_dataset", "seed", "epochs", "pi", "mask_budget", "best_epoch",
          "val_img_auroc", "val_curve", "img_auroc", "pix_auroc", "aupro", "diag_inmask_p", "diag_inmask_pg",
          "n_updates", "train_minutes", "total_minutes", "notes"]


# ----------------------------------------------------------------------------- losses
def topk_k(n):
    return max(3, int(round(K_FRAC * n)))


def mil_loss(z, y):
    """Task 3's plain MIL at one level: BCE(mean of the top-k patch probabilities, y)."""
    p = torch.sigmoid(z.float().reshape(-1))
    bag = p.topk(topk_k(p.numel())).values.mean().clamp(1e-6, 1 - 1e-6)
    return torch.nn.functional.binary_cross_entropy(bag, torch.tensor(float(y), device=p.device))


def gated_idx(p, gate, y):
    """Indices chosen by idea #1: abnormal -> top-k of p*g, normal -> top-k of p. No gradient."""
    with torch.no_grad():
        key = p.detach() * gate.to(p.device) if y == 1 else p.detach()
        return key.topk(topk_k(p.numel())).indices


def gated_mil_loss(z, y, gate):
    """Idea #1 at one level (A1-g1-o2 step 2): positive BCE on the mean p of the
    p*g-selected patches; normal images: top-k BCE towards 0 with weight 2."""
    p = torch.sigmoid(z.float().reshape(-1))
    bag = p[gated_idx(p, gate.reshape(-1).float(), y)].mean().clamp(1e-6, 1 - 1e-6)
    bce = torch.nn.functional.binary_cross_entropy(bag, torch.tensor(float(y), device=p.device))
    return 2.0 * bce if y == 0 else bce


# ----------------------------------------------------------------------------- schedule
def build_epoch(arm, fold, seed, epoch, lists, mask_set):
    """Items for one epoch: ("mask"|"weak"|"det", d, path, y) single images, or
    ("pair", d, normal_path, abnormal_path). One seeded shuffle of the whole list;
    mvfa / mil / g with the same seed get identical images in identical order (G1-14)."""
    rng = np.random.default_rng([seed, epoch, 7])
    weak = WEAK.get(arm)
    items = []
    for d in sources(fold):
        normal, abnormal = lists[d]
        pairs = weak == "pu" and not (arm == "pu_m100" and has_masks(d))
        if pairs:   # per dataset max(#n, #a) pairs, shorter class cycling
            r = np.random.default_rng([seed, epoch, 11, len(items)])
            n_ = [normal[i] for i in r.permutation(len(normal))]
            a_ = [abnormal[i] for i in r.permutation(len(abnormal))]
            m = max(len(n_), len(a_))
            items += [("pair", d, n_[i % len(n_)], a_[i % len(a_)]) for i in range(m)]
            continue
        for p, y in [(p, 0) for p in normal] + [(p, 1) for p in abnormal]:
            if has_masks(d) and arm in MASK_BUDGET and (MASK_BUDGET[arm] == 1.0 or p in mask_set):
                kind = "mask"
            elif weak is None:
                kind = "det"         # mvfa on a mask-less source
            else:
                kind = "weak"
            items.append((kind, d, p, y))
    items.sort(key=lambda t: (t[1], t[2], t[3] if isinstance(t[3], str) else str(t[3])))
    return [items[i] for i in rng.permutation(len(items))]


def mask_budget_set(arm, fold, seed, lists):
    """Budget arms: a fixed seeded share of each masked source's abnormal images keeps its mask."""
    b = MASK_BUDGET.get(arm, 0.0)
    if b in (0.0, 1.0):
        return set()
    out = set()
    for d in sources(fold):
        if has_masks(d):
            ab = lists[d][1]
            r = np.random.default_rng([seed, 99, len(ab)])
            out |= {ab[i] for i in r.choice(len(ab), int(round(b * len(ab))), replace=False)}
    return out


class _Items(Dataset):
    def __init__(self, items, allow_masks, gates):
        self.items, self.allow_masks, self.gates = items, allow_masks, gates

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, d, a, b = self.items[i]
        if kind == "pair":
            return kind, d, torch.stack([load_image(a), load_image(b)]), None, torch.tensor([0.0, 1.0]), None
        m = None
        if kind == "mask":
            if not self.allow_masks:
                raise MaskLeakError(f"mask item in a mask-free arm: {a}")
            m = load_mask(a, b)[None]
        g = self.gates[d][a] if (self.gates and kind == "weak" and d in self.gates) else None
        return kind, d, load_image(a)[None], m, torch.tensor([float(b)]), g


def _collate(x):
    return x


# ----------------------------------------------------------------------------- validation / diagnostic
@torch.no_grad()
def source_validation(model, tf, fold, seed, device, limit=None, bs=32):
    """Image AUROC on each source's valid/ list with its own prompt; equal-weight mean."""
    per = {}
    for d in sources(fold):
        items = valid_lists(d, seed)
        if limit:
            items = [x for x in items if x[1] == 0][:limit] + [x for x in items if x[1] == 1][:limit]
        assert_no_heldout([p for p, _ in items], fold)
        s = []
        for i in range(0, len(items), bs):
            x = torch.stack([load_image(p) for p, _ in items[i:i + bs]]).to(device)
            with autocast(device):
                _, _, det = model(x)
            s.append(scores_to_image_score(level_scores(det, tf[d])).float().cpu())
        per[d] = image_auroc([y for _, y in items], torch.cat(s).numpy())
    return float(np.mean(list(per.values()))), per


@torch.no_grad()
def inmask_diagnostic(model, tf, fold, seed, device, gates, limit=None):
    """Task 3's diagnostic for #1 (G1-19), report only, after training: on the masked
    sources' valid/ abnormal images, fraction of the top-k patches (by p, and by p*g)
    whose 14x14 cell is > 50 % inside the true mask; mean over images and levels."""
    from common import IMG_SIZE
    fp, fpg = [], []
    for d in sources(fold):
        if not has_masks(d) or d not in gates:
            continue
        items = [(p, y) for p, y in valid_lists(d, seed) if y == 1 and p in gates[d]][:limit]
        for p, _ in items:
            x = load_image(p)[None].to(device)
            m = load_mask(p, 1)                                   # [1, 240, 240]
            H = 17
            cell = torch.nn.functional.adaptive_avg_pool2d(m[None], H)[0, 0].reshape(-1) > 0.5
            if cell.sum() == 0:
                continue
            with autocast(device):
                _, seg, _ = model(x)
            g = gates[d][p].float().to(device)
            for s in level_scores(seg, tf[d]):
                pr = torch.sigmoid(two_class_to_logit(s[0].float()))
                k = topk_k(pr.numel())
                fp.append(cell[pr.topk(k).indices.cpu()].float().mean().item())
                fpg.append(cell[(pr * g).topk(k).indices.cpu()].float().mean().item())
    return (float(np.mean(fp)) if fp else float("nan")), (float(np.mean(fpg)) if fpg else float("nan"))


# ----------------------------------------------------------------------------- main
def build_r2_model(seed, device):
    """Fresh seg + det adapters (MVFA's init) from the run seed; CLIP frozen (G1-6, G1-10)."""
    return build_model("Liver", device, load_r0=False, reinit_seg=True, reinit_det=True, reinit_seed=seed)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def append_row(row, path=None):
    path = path or RESULTS2 / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def run(a):
    t_start = time.time()
    setup_seed(a.seed)
    dev = torch.device(a.device)
    run_id = f"{a.arm}_{a.fold}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out = RUNS2 / run_id
    out.mkdir(parents=True, exist_ok=True)
    weak = WEAK.get(a.arm)
    allow_masks = a.arm in MASK_BUDGET

    lists = {}
    for d in sources(a.fold):
        n, ab = train_lists(d)
        if a.max_per_source:
            n, ab = n[:a.max_per_source], ab[:a.max_per_source]
        assert_no_heldout(n + ab, a.fold)
        lists[d] = (n, ab)
    gates = {}
    if weak == "g" or not a.no_diag:
        for d in sources(a.fold):
            if (BANKS / f"{d}_gates.pt").exists():
                gates[d] = load_gates(d)
            elif weak == "g":
                raise FileNotFoundError(f"no normal bank for {d}: run build_normal_bank.py")
    assert not set(gates) & set(held_out(a.fold)), "bank of a held-out dataset loaded"
    mask_set = mask_budget_set(a.arm, a.fold, a.seed, lists)

    model, clip_model = build_r2_model(a.seed, a.device)
    for p in list(model.seg_adapters.parameters()) + list(model.det_adapters.parameters()):
        p.requires_grad_(True)
    assert all(n.startswith(("seg_adapters.", "det_adapters.")) for n, p in model.named_parameters()
               if p.requires_grad)
    opt_seg = torch.optim.Adam(model.seg_adapters.parameters(), lr=a.lr, betas=(0.5, 0.999))
    opt_det = torch.optim.Adam(model.det_adapters.parameters(), lr=a.lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    tf = text_features(clip_model, a.device)

    best, best_state, curve, losses, n_upd = -1.0, None, [], [], 0
    t0 = time.time()
    for ep in range(a.epochs):
        items = build_epoch(a.arm, a.fold, a.seed, ep, lists, mask_set)
        if not allow_masks and any(k == "mask" for k, *_ in items):
            raise MaskLeakError(f"mask item scheduled in mask-free arm {a.arm}")
        if a.max_steps:
            items = items[:a.max_steps]
        dl = DataLoader(_Items(items, allow_masks, gates if weak == "g" else {}), batch_size=None, shuffle=False,
                        num_workers=a.workers, collate_fn=_collate, pin_memory=dev.type == "cuda",
                        prefetch_factor=8 if a.workers else None, persistent_workers=False)
        te = time.time()
        for step, (kind, d, x, m, y, g) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with autocast(a.device):
                _, seg, det = model(x)
            loss = mvfa_det_loss(level_scores(det, tf[d]), y)
            seg_loss = None
            if kind == "mask":
                seg_loss = mvfa_mask_loss(level_scores(seg, tf[d]), m.to(dev))
            elif kind == "pair":
                seg_loss = sum(pu_risk(z[0], z[1], a.pi) for z in
                               (two_class_to_logit(s.float()) for s in level_scores(seg, tf[d])))
            elif kind == "weak":
                yi = int(y.item())
                zs = [two_class_to_logit(s.float())[0] for s in level_scores(seg, tf[d])]
                if weak == "mil":
                    seg_loss = sum(mil_loss(z, yi) for z in zs)
                else:
                    seg_loss = sum(gated_mil_loss(z, yi, g) for z in zs)
            opt_seg.zero_grad(set_to_none=True)
            opt_det.zero_grad(set_to_none=True)
            if seg_loss is not None:
                loss = loss + seg_loss
            scaler.scale(loss).backward()
            if seg_loss is not None:
                scaler.step(opt_seg)
            scaler.step(opt_det)        # mvfa "det" items: only the det optimizer steps (as MVFA)
            scaler.update()
            n_upd += 1
            losses.append((ep, step, kind, d, float(loss.detach())))
            if a.log_every and step % a.log_every == 0:
                print(f"[{run_id}] ep {ep} step {step}/{len(items)} {kind:4s} {d:15s} loss {losses[-1][4]:.4f} "
                      f"({(time.time() - te) / (step + 1):.3f}s/step)", flush=True)
        ep_s = time.time() - te
        v, per = source_validation(model, tf, a.fold, a.seed, a.device, limit=a.val_limit)
        curve.append(round(v, 5))
        print(f"[{run_id}] epoch {ep} done in {ep_s:.0f}s; source-val image AUROC {v:.4f} "
              f"{ {k: round(x, 3) for k, x in per.items()} }", flush=True)
        if v >= best:            # ties -> later epoch
            best, best_ep = v, ep
            best_state = {"seg_adapters": copy.deepcopy({k: t.cpu() for k, t in model.seg_adapters.state_dict().items()}),
                          "det_adapters": copy.deepcopy({k: t.cpu() for k, t in model.det_adapters.state_dict().items()})}
    train_s = time.time() - t0
    torch.save(best_state, out / "adapters.pth")
    with open(out / "loss.csv", "w", newline="") as f:
        csv.writer(f).writerows([("epoch", "step", "kind", "dataset", "loss")] + losses)
    model.seg_adapters.load_state_dict(best_state["seg_adapters"])
    model.det_adapters.load_state_dict(best_state["det_adapters"])
    model.eval()
    # ---- training and selection are finished; from here nothing is chosen
    dp, dpg = (float("nan"), float("nan")) if a.no_diag else \
        inmask_diagnostic(model, tf, a.fold, a.seed, a.device, gates, limit=a.val_limit)
    rows = []
    for ds in ([] if a.no_eval else held_out(a.fold)):
        sm = out / f"maps_{ds}.npz" if a.save_maps else None
        res = evaluate_target.evaluate(model, tf[ds], ds, a.device, limit=a.eval_limit, save_maps=sm,
                                       workers=a.workers)
        notes = []
        if a.epochs < 5:
            notes.append(f"{a.epochs} epochs")
        if a.max_per_source or a.max_steps or a.eval_limit:
            notes.append("SMOKE")
        notes.append(f"jobs_per_gpu={os.environ.get('TASK4_SLOTS', 1)}")
        row = {"run_id": run_id, "arm": a.arm, "fold": a.fold, "eval_dataset": ds, "seed": a.seed,
               "epochs": a.epochs, "pi": a.pi if weak == "pu" else "",
               "mask_budget": MASK_BUDGET.get(a.arm, 0.0), "best_epoch": best_ep, "val_img_auroc": round(best, 5),
               "val_curve": json.dumps(curve), "img_auroc": res["img_auroc"], "pix_auroc": res.get("pix_auroc", ""),
               "aupro": res.get("aupro", ""), "diag_inmask_p": dp, "diag_inmask_pg": dpg, "n_updates": n_upd,
               "train_minutes": round(train_s / 60, 2), "total_minutes": round((time.time() - t_start) / 60, 2),
               "notes": "; ".join(notes)}
        rows.append(row)
        if not (a.max_per_source or a.max_steps or a.eval_limit):
            append_row(row)
    meta = {**vars(a), "run_id": run_id, "best_epoch": best_ep, "val_curve": curve, "n_updates": n_upd,
            "train_seconds": train_s, "seconds_per_update": train_s / max(1, n_upd),
            "total_seconds": time.time() - t_start, "results": rows, "diag": [dp, dpg],
            "eval_reads": list(evaluate_target._READS)}
    write_json(out / "run.json", meta)
    print(json.dumps({k: meta[k] for k in ("run_id", "best_epoch", "val_curve", "n_updates", "seconds_per_update",
                                            "total_seconds")}, indent=1))
    for r in rows:
        print({k: r[k] for k in ("eval_dataset", "img_auroc", "pix_auroc", "aupro")})
    return meta


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--fold", choices=list(FOLDS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--pi", type=float, default=0.01, help="pu arms: fixed pi (G1-3)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TASK4_WORKERS", 4)))
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_maps", action="store_true")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_diag", action="store_true")
    ap.add_argument("--log_every", type=int, default=500)
    # smoke / timing only (any of these marks the run SMOKE and keeps it out of results.csv)
    ap.add_argument("--max_per_source", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
