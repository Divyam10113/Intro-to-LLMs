"""Idea #1's normal bank (Task 3 A1-g1-o2, step 1), one per dataset.

    python build_normal_bank.py --dataset Chest [--device cuda:0]

Per dataset: frozen CLIP layer-24 patch features (G1-1) of its <= 500 normal
subsample images, greedy coreset to 20,000 vectors (G1-2). Then, because the
gate g = 1 - max cos(f, bank) depends only on the frozen features and the bank,
it is precomputed for every subsample image and every valid/ image of the
dataset (training only looks it up; this changes nothing in the method).

Outputs (HDD): banks/<dataset>.pt  [M, 1024] fp16, normalised
               banks/<dataset>_gates.pt  {img_rel: [289] float16}
               banks/<dataset>.json      sizes, timing
A dataset's bank is built from its own images only, so it is a valid bank in
every fold where that dataset is a source; train2.py never loads the bank of a
held-out dataset.
"""
import argparse
import json
import time

import torch

from common2 import BANKS, BANK_SIZE, greedy_coreset, frozen_l24, gate_from_bank, train_lists, write_json
from common import autocast, build_model, load_image


def feats_of(model, paths, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_image(p) for p in paths[i:i + bs]]).to(device)
        with autocast(device):
            out.append(frozen_l24(model, x).half())
    return torch.cat(out)            # [N, 289, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bank_size", type=int, default=BANK_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="CPU smoke test: first N images per list")
    ap.add_argument("--out", default=str(BANKS))
    a = ap.parse_args()
    t0 = time.time()
    from pathlib import Path
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = build_model("Liver", a.device, load_r0=False)   # adapters unused here: plain CLIP
    model.eval()

    normal, abnormal = train_lists(a.dataset)
    vj = json.load(open(Path(__file__).resolve().parent / "splits" / "valid.json"))[a.dataset]
    valid = vj["normal"] + vj["abnormal"]
    if a.limit:
        normal, abnormal, valid = normal[:a.limit], abnormal[:a.limit], valid[:a.limit]

    fn = feats_of(model, normal, a.device)
    flat = fn.reshape(-1, fn.shape[-1])
    idx = greedy_coreset(flat.to(a.device), a.bank_size, seed=0)
    bank = flat[idx.to(flat.device)].to(a.device)
    torch.save(bank.cpu(), out / f"{a.dataset}.pt")
    t_bank = time.time() - t0

    gates = {}
    for paths, f in ((normal, fn), (abnormal, None), (valid, None)):
        f = f if f is not None else feats_of(model, paths, a.device)
        for p, fi in zip(paths, f):
            gates[p] = gate_from_bank(fi.to(a.device).float(), bank.float()).half().cpu()
    torch.save(gates, out / f"{a.dataset}_gates.pt")
    import numpy as np
    gn = torch.stack([gates[p] for p in normal]).float()
    ga = torch.stack([gates[p] for p in abnormal]).float()
    info = {"dataset": a.dataset, "n_normal_images": len(normal), "n_vectors": int(flat.shape[0]),
            "bank_size": int(bank.shape[0]), "n_gated_images": len(gates),
            "gate_mean_normal_train": float(gn.mean()), "gate_mean_abnormal_train": float(ga.mean()),
            "gate_p99_abnormal_train": float(np.percentile(ga.numpy(), 99)),
            "seconds_bank": t_bank, "seconds_total": time.time() - t0}
    write_json(out / f"{a.dataset}.json", info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
