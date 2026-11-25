"""Round 2 shared definitions: folds, data lists, normal-bank features.

Reuses round 1's code (../task4: MVFA forward, data loading, losses, metrics)
unchanged; only what round 2 adds lives here. Decisions referenced as G1-n are
in report/ISSUES_LOG.md.
"""
import json
import os
import sys
from pathlib import Path

R2 = Path(__file__).resolve().parent                 # proj/task4r2
R1 = R2.parent / "task4"
HDD2 = Path(os.environ.get("TASK4R2_HDD", "/mnt/HDD1/divyam/task4/r2"))
os.environ.setdefault("TASK4_RUNS", str(HDD2 / "runs"))
sys.path.insert(0, str(R1))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import (CLASS_INDEX, DATA_DIR, DATASETS, file_lock, has_masks, load_splits,  # noqa: E402,F401
                    write_json)

RUNS2 = Path(os.environ["TASK4_RUNS"])
BANKS = HDD2 / "banks"
RESULTS2 = Path(os.environ.get("TASK4R2_RESULTS", R2 / "results"))
SPLITS2 = R2 / "splits"
VAL_CAP = 250            # images per class per source for validation (G1-5)
BANK_SIZE = 20000        # G1-2
K_FRAC = 0.01            # G1-13: k = max(3, round(1% of 289)) = 3

# G1-7: five leave-one-modality-out folds; RESC and OCT17 held out together.
FOLDS = {"Brain": ["Brain"], "Liver": ["Liver"], "RESC_OCT": ["Retina_RESC", "Retina_OCT2017"],
         "Chest": ["Chest"], "HIS": ["Histopathology"]}
MASKED_FOLDS = ["Brain", "Liver", "RESC_OCT"]


def held_out(fold):
    return FOLDS[fold]


def sources(fold):
    return [d for d in DATASETS if d not in FOLDS[fold]]


def train_lists(dataset):
    """Round 1's subsample, train + pi lists together (G1-8): <=500 + <=500."""
    s = load_splits(dataset)
    return sorted(s["train_normal"] + s["pi_normal"]), sorted(s["train_abnormal"] + s["pi_abnormal"])


def valid_lists(dataset, seed):
    """Source validation (G1-5): valid/ minus files byte-identical to any test file,
    at most VAL_CAP per class drawn with the run seed (only Chest exceeds it)."""
    v = json.load(open(SPLITS2 / "valid.json"))[dataset]
    out = []
    for y, key in ((0, "normal"), (1, "abnormal")):
        files = v[key]
        if len(files) > VAL_CAP:
            rng = np.random.default_rng([seed, CLASS_INDEX[dataset] + 10, y])
            files = sorted(files[i] for i in rng.choice(len(files), VAL_CAP, replace=False))
        out += [(p, y) for p in files]
    return out


def assert_no_heldout(paths, fold):
    """No file of a held-out dataset may be used for training, bank or validation."""
    bad = [p for p in paths for h in FOLDS[fold] if str(p).startswith(f"{h}_AD/")]
    if bad:
        raise AssertionError(f"fold {fold}: held-out files used: {bad[:3]}")


# ----------------------------------------------------------------------------- frozen features (G1-1)
@torch.no_grad()
def frozen_l24(model, x):
    """Patch tokens after block 24 of plain frozen CLIP (no adapters), L2-normalised.
    x: [B, 3, 240, 240] -> [B, 289, 1024] float32."""
    enc = model.image_encoder
    x = enc.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([enc.class_embedding.to(x.dtype)
                   + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    x = x + enc.positional_embedding.to(x.dtype)
    x = enc.patch_dropout(x)
    x = enc.ln_pre(x)
    x = x.permute(1, 0, 2)
    for i in range(24):
        x, _ = enc.transformer.resblocks[i](x, attn_mask=None)
    t = x.permute(1, 0, 2)[:, 1:, :].float()
    return t / t.norm(dim=-1, keepdim=True)


def greedy_coreset(feats, n, seed=0, proj_dim=128):
    """Greedy k-centre coreset (PatchCore): distances on a fixed random projection,
    returns indices of n selected rows of feats [N, C]."""
    N = feats.shape[0]
    if n >= N:
        return torch.arange(N)
    g = torch.Generator().manual_seed(seed)
    P = (torch.randn(feats.shape[1], proj_dim, generator=g) / proj_dim ** 0.5).to(feats.device)
    z = feats.float() @ P
    first = int(torch.randint(N, (1,), generator=g))
    sel = torch.empty(n, dtype=torch.long, device=feats.device)
    sel[0] = first
    mind = ((z - z[first]) ** 2).sum(1)
    for j in range(1, n):
        i = torch.argmax(mind)
        sel[j] = i
        mind = torch.minimum(mind, ((z - z[i]) ** 2).sum(1))
    return sel.cpu()


def gate_from_bank(f, bank, chunk=4096):
    """g = 1 - max cosine similarity to the bank. f: [L, C] normalised, bank: [M, C] normalised."""
    best = torch.full((f.shape[0],), -1.0, device=f.device)
    for i in range(0, bank.shape[0], chunk):
        best = torch.maximum(best, (f @ bank[i:i + chunk].T.to(f.dtype)).max(dim=1).values)
    return (1.0 - best).clamp(min=0.0)


def load_gates(dataset):
    """{img_rel: [289] gate} for every subsample and valid image of `dataset`."""
    return torch.load(BANKS / f"{dataset}_gates.pt")
