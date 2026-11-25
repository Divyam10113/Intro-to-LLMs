"""Liver heat maps for round 2, with round 1's fixed rules (task4/make_paper_inputs.py):
the same 4 images (positions 1/8, 3/8, 5/8, 7/8 of the sorted abnormal-with-mask
Liver test list), a per-column colour scale (1st-99.5th percentile of that model's
maps), and the ground-truth contour in cyan. Reads saved maps only; no GPU.

    python make_figures.py
"""
import numpy as np

from common2 import R2, RUNS2
from common import HDD, load_image, load_mask

COLS = [("R0", HDD / "runs/R0_Liver/maps.npz"), ("MVFA", RUNS2 / "mvfa_Liver_s0/maps_Liver.npz"),
        ("MIL", RUNS2 / "mil_Liver_s0/maps_Liver.npz"), ("PU", RUNS2 / "pu_Liver_s0/maps_Liver.npz"),
        ("G", RUNS2 / "g_Liver_s0/maps_Liver.npz"), ("PU+100%", RUNS2 / "pu_m100_Liver_s0/maps_Liver.npz"),
        ("G+100%", RUNS2 / "g_m100_Liver_s0/maps_Liver.npz")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [(n, p) for n, p in COLS if p.exists()]
    data = {n: np.load(p) for n, p in cols}
    paths, labels = data[cols[0][0]]["paths"], data[cols[0][0]]["labels"]
    for n, _ in cols:
        assert (data[n]["paths"] == paths).all()
    cand = sorted((i for i in range(len(paths)) if labels[i] == 1 and load_mask(str(paths[i]), 1).sum() > 0),
                  key=lambda i: str(paths[i]))
    pick = [cand[int(len(cand) * q)] for q in (1 / 8, 3 / 8, 5 / 8, 7 / 8)]
    rng = np.random.default_rng(0)
    scale = {n: np.percentile(data[n]["maps"][rng.integers(0, len(paths), 200)].astype(np.float32).ravel(), [1, 99.5])
             for n, _ in cols}
    names = ["image"] + [n for n, _ in cols]
    fig, ax = plt.subplots(len(pick), len(names), figsize=(2 * len(names), 2 * len(pick)))
    for r, i in enumerate(pick):
        gt = load_mask(str(paths[i]), 1)[0].numpy()
        for c, n in enumerate(names):
            a = ax[r, c]
            if n == "image":
                a.imshow(load_image(str(paths[i])).permute(1, 2, 0).numpy())
            else:
                a.imshow(data[n]["maps"][i].astype(np.float32), cmap="inferno", vmin=scale[n][0], vmax=scale[n][1])
            a.contour(gt, levels=[0.5], colors="cyan", linewidths=0.8)
            a.set_xticks([]), a.set_yticks([])
            if r == 0:
                a.set_title(n, fontsize=10)
    fig.tight_layout()
    out = R2 / "report"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"liver_heatmaps_r2.{ext}", dpi=150)
    (out / "liver_heatmaps_r2_images.txt").write_text("\n".join(str(paths[i]) for i in pick) + "\n")
    print("wrote", out / "liver_heatmaps_r2.png", [n for n, _ in cols])


if __name__ == "__main__":
    main()
