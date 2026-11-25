"""Write the pi values the training runs use (report/DECISIONS.md D1, D2). CPU only.

    python set_pi.py

results/pi.json         (M-h)  for target T, every mask-less source gets
                               pi_T = mean over T's masked sources of the true
                               anomalous-patch fraction of their abnormal
                               training images (from their masks).
results/pi_maskfree.json (M)   0.30 for every source of every target: the value
                               the plan's estimator procedure returns (clip),
                               uses no mask information.
"""
import torch
import torch.nn.functional as F

from common import MASKLESS, RESULTS_DIR, TARGETS, has_masks, load_mask, load_splits, sources_for, write_json
from pu import PI_CLIP

PATCH = 14          # 240 px / 17 patches ~ 14 px per ViT patch cell
COVER = 0.5         # cell counts as anomalous if > 50% covered by the mask
PI_MASKFREE = PI_CLIP[1]


def true_pi(dataset):
    """True anomalous-patch fraction of a masked dataset's abnormal images
    (subsampled train + pi lists)."""
    assert has_masks(dataset), dataset
    s = load_splits(dataset)
    cells = [F.avg_pool2d(load_mask(p, 1)[None], PATCH, PATCH)[0, 0] > COVER
             for p in s["train_abnormal"] + s["pi_abnormal"]]
    return float(torch.stack(cells).float().mean())


def main():
    truth = {}
    pi, pi_mf = {}, {}
    for t in TARGETS:
        masked = [d for d in sources_for(t) if has_masks(d)]
        for d in masked:
            truth.setdefault(d, true_pi(d))
        value = sum(truth[d] for d in masked) / len(masked)
        pi[t] = {d: {"pi": value, "method": "D1: mean true patch fraction of masked sources",
                     "from": {m: truth[m] for m in masked}}
                 for d in sources_for(t) if d in MASKLESS}
        pi_mf[t] = {d: {"pi": PI_MASKFREE, "method": "D2: fixed, = plan estimator's clip value"}
                    for d in sources_for(t)}
        print(f"{t}: M-h pi = {value:.4f} from {', '.join(f'{m} {truth[m]:.4f}' for m in masked)}; "
              f"M pi = {PI_MASKFREE}")
    write_json(RESULTS_DIR / "pi.json", pi)
    write_json(RESULTS_DIR / "pi_maskfree.json", pi_mf)
    write_json(RESULTS_DIR / "pi_true_masked_sources.json", truth)


if __name__ == "__main__":
    main()
