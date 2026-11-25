"""Shared paths, data loading and model utilities for Task 4.

Everything here mirrors MVFA-AD (commit a30bf7b) unless a comment says otherwise:
images are resized to 240x240 bicubic with no normalisation, masks nearest,
patch scores are 100 * <normalised token, text feature>, levels = [6, 12, 18, 24].
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent            # task4_weak_supervision/round1 (proj/task4 on the server)
# MVFA-AD checkout: $MVFA_DIR (see shared/config.sh and .env.example); default: the folder that holds
# this repository, i.e. proj/MVFA-AD on the server, the same checkout as before the restructure.
MVFA_DIR = Path(os.environ.get("MVFA_DIR", ROOT.parents[2] / "MVFA-AD"))
sys.path.insert(0, str(MVFA_DIR))

from CLIP.adapter import CLIP_Inplanted, ClipAdapter  # noqa: E402
from CLIP.clip import create_model  # noqa: E402
from prompt import REAL_NAME  # noqa: E402
from utils import encode_text_with_prompt_ensemble  # noqa: E402

HDD = Path(os.environ.get("TASK4_HDD", "/mnt/HDD1/divyam/task4"))
DATA_DIR = Path(os.environ.get("TASK4_DATA", HDD / "data/extracted"))
R0_CKPT_DIR = HDD / "ckpt/zero-shot"
RUNS_DIR = Path(os.environ.get("TASK4_RUNS", HDD / "runs"))
SPLITS_DIR = Path(os.environ.get("TASK4_SPLITS", ROOT / "splits"))
RESULTS_DIR = Path(os.environ.get("TASK4_RESULTS", ROOT / "results"))

CLASS_INDEX = {"Brain": 3, "Liver": 2, "Retina_RESC": 1,
               "Retina_OCT2017": -1, "Chest": -2, "Histopathology": -3}
DATASETS = list(CLASS_INDEX)
TARGETS = ["Liver", "Brain", "Retina_RESC"]           # BMAD datasets with pixel masks
MASKLESS = ["Retina_OCT2017", "Chest", "Histopathology"]
IMG_SIZE = 240
FEATURES = [6, 12, 18, 24]
DEEPEST = len(FEATURES) - 1


def sources_for(target):
    assert target in TARGETS, target
    return [d for d in DATASETS if d != target]


def has_masks(dataset):
    return CLASS_INDEX[dataset] > 0


# ----------------------------------------------------------------------------- data
_tx = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), transforms.ToTensor()])
_tm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE), Image.NEAREST), transforms.ToTensor()])


def abs_path(rel):
    return str(DATA_DIR / rel)


def load_image(rel):
    return _tx(Image.open(abs_path(rel)).convert("RGB"))


def mask_rel(img_rel):
    """MVFA: mask path = image path with the 'img' directory -> 'anomaly_mask'."""
    p = Path(img_rel)
    assert p.parent.name == "img", img_rel
    return str(p.parent.parent / "anomaly_mask" / p.name)


def load_mask(img_rel, label):
    """Binary [1, H, W] mask; zeros for normal images (as MVFA)."""
    if label == 0:
        return torch.zeros(1, IMG_SIZE, IMG_SIZE)
    m = _tm(Image.open(abs_path(mask_rel(img_rel))).convert("L"))
    return (m > 0.5).float()


def list_test_set(dataset):
    """MVFA's test set for a dataset: test/good (label 0) + test/Ungood (label 1), sorted."""
    out = []
    for sub, y in (("good", 0), ("Ungood", 1)):
        d = DATA_DIR / f"{dataset}_AD" / "test" / sub / "img"
        out += [(str(Path(f"{dataset}_AD/test/{sub}/img") / f), y) for f in sorted(os.listdir(d))]
    return out


def load_splits(dataset):
    with open(SPLITS_DIR / f"{dataset}.json") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- model
class MVFA(CLIP_Inplanted):
    """CLIP_Inplanted with the same forward, minus the unused attention map
    (MVFA builds it with a hard-coded .to('cuda'), which breaks CPU runs).
    tests/test_model.py checks the outputs are identical to the original."""

    def forward(self, x):
        enc = self.image_encoder
        x = enc.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([enc.class_embedding.to(x.dtype)
                       + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + enc.positional_embedding.to(x.dtype)
        x = enc.patch_dropout(x)
        x = enc.ln_pre(x)
        x = x.permute(1, 0, 2)
        seg_tokens, det_tokens = [], []
        for i in range(24):
            x, _ = enc.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:
                j = self.features.index(i + 1)
                seg_med, seg_out = self.seg_adapters[j](x)
                det_med, det_out = self.det_adapters[j](x)
                x = 0.8 * x + 0.1 * seg_out + 0.1 * det_out
                seg_tokens.append(seg_med.permute(1, 0, 2))
                det_tokens.append(det_med.permute(1, 0, 2))
        return None, seg_tokens, det_tokens


def build_model(target, device, load_r0=True, reinit_seg=False, reinit_det=False, reinit_seed=0):
    """CLIP ViT-L/14-336 at 240 px + MVFA adapters; everything frozen.

    load_r0:    load MVFA's released zero-shot checkpoint for `target` (R0).
    reinit_seg / reinit_det: replace the segmentation / detection adapters with
                freshly initialised ones (MVFA's own init, i.e. ClipAdapter
                defaults, built seg-then-det from one RNG stream as MVFA does),
                seeded by `reinit_seed` and NOT by the run seed (the run seed
                controls only data order). The seg init is the same whether or
                not det is re-initialised.
    """
    clip_model = create_model(model_name="ViT-L-14-336", img_size=IMG_SIZE, device=device,
                              pretrained="openai", require_pretrained=True)
    clip_model.eval()
    model = MVFA(clip_model=clip_model, features=FEATURES).to(device)
    model.eval()
    if load_r0:
        ck = torch.load(R0_CKPT_DIR / f"{target}.pth", map_location=device)
        model.seg_adapters.load_state_dict(ck["seg_adapters"])
        model.det_adapters.load_state_dict(ck["det_adapters"])
    if reinit_seg or reinit_det:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(reinit_seed)
            fresh_seg = torch.nn.ModuleList([ClipAdapter(1024, bottleneck=768) for _ in FEATURES])
            fresh_det = torch.nn.ModuleList([ClipAdapter(1024, bottleneck=768) for _ in FEATURES])
        if reinit_seg:
            model.seg_adapters = fresh_seg.to(device)
        if reinit_det:
            model.det_adapters = fresh_det.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, clip_model


@torch.no_grad()
def text_features(clip_model, device):
    """{dataset: [768, 2] text features}, MVFA's prompt ensemble."""
    out = {}
    with torch.autocast(device_type=torch.device(device).type, enabled=torch.device(device).type == "cuda"):
        for d in DATASETS:
            out[d] = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[d], device).float()
    return out


def level_scores(tokens, text_feat):
    """tokens: list over levels of [B, 1+L, C]; returns list of [B, L, 2] two-class scores."""
    out = []
    for t in tokens:
        t = t[:, 1:, :].float()
        t = t / t.norm(dim=-1, keepdim=True)
        out.append(100.0 * t @ text_feat)
    return out


def scores_to_maps(scores):
    """MVFA's pixel map: per level, upsample 2-class scores to 240 bilinear,
    softmax, take class 1; sum over levels. Returns [B, 240, 240]."""
    total = 0
    for s in scores:
        B, L, _ = s.shape
        H = int(round(L ** 0.5))
        m = F.interpolate(s.permute(0, 2, 1).reshape(B, 2, H, H), size=IMG_SIZE,
                          mode="bilinear", align_corners=True)
        total = total + torch.softmax(m, dim=1)[:, 1]
    return total


def scores_to_image_score(scores):
    """MVFA's image score: sum over levels of the mean patch anomaly probability."""
    return sum(torch.softmax(s, dim=-1)[..., 1].mean(dim=-1) for s in scores)


def autocast(device):
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


class file_lock:
    """Exclusive inter-process lock (fcntl) on <path>.lock; results live on the
    SSD, where flock works. Used for read-modify-write of shared result files
    when many jobs run at once."""

    def __init__(self, path):
        self.path = Path(str(path) + ".lock")

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
