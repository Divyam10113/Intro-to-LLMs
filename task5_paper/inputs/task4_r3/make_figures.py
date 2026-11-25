"""Round 3 figures (plan §6): Liver heat maps (seed-3 runs) and the label-budget curve.

Heat maps: round 2's script unchanged except the columns (round 3 arms, seed 3) and the
output names. Budget curve: AUPRO and pixel-AUROC retention at 0 / 10 / 25 / 100 % masks
(MIL, MIL+10 %, MIL+25 %, MIL+100 %) per target, MVFA as a reference line, from
results/results.csv via analyze3.load.

Round 2's description: Liver heat maps for round 2, with round 1's fixed rules (task4/make_paper_inputs.py):
the same 4 images (positions 1/8, 3/8, 5/8, 7/8 of the sorted abnormal-with-mask
Liver test list), a per-column colour scale (1st-99.5th percentile of that model's
maps), and the ground-truth contour in cyan. Reads saved maps only; no GPU.

    python make_figures.py
"""
import numpy as np

from common2 import R2, RUNS2
from common import HDD, load_image, load_mask

COLS = [("R0", HDD / "runs/R0_Liver/maps.npz")] + [
    (n, RUNS2 / f"{a}_Liver_s3/maps_Liver.npz") for n, a in
    (("MVFA", "mvfa"), ("MIL", "mil"), ("PU+100%", "pu_m100"), ("MIL+100%", "mil_m100"), ("MIL+10%", "mil_m10"),
     ("MIL+25%", "mil_m25"))]


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
        fig.savefig(out / f"liver_heatmaps_r3.{ext}", dpi=150)
    (out / "liver_heatmaps_r3_images.txt").write_text("\n".join(str(paths[i]) for i in pick) + "\n")
    print("wrote", out / "liver_heatmaps_r3.png", [n for n, _ in cols])


def budget_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import analyze3 as A
    d = A.load(A.RESULTS2 / "results.csv")
    pts = [(0, "mil"), (10, "mil_m10"), (25, "mil_m25"), (100, "mil_m100")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for t, c in zip(A.T, ("#1f77b4", "#d62728", "#2ca02c")):
        xs = [x for x, a in pts if A.vals(d, a, t, "aupro")]
        if not xs:
            continue
        au = [A.mean(d, a, t, "aupro") for x, a in pts if A.vals(d, a, t, "aupro")]
        sd = [np.std(A.vals(d, a, t, "aupro"), ddof=1) if len(A.vals(d, a, t, "aupro")) > 1 else 0
              for x, a in pts if A.vals(d, a, t, "aupro")]
        rt = [(A.mean(d, a, t, "pix_auroc") - 50) / (A.mean(d, "mvfa", t, "pix_auroc") - 50)
              for x, a in pts if A.vals(d, a, t, "aupro")]
        ax[0].errorbar(xs, au, yerr=sd, marker="o", color=c, label=t.replace("Retina_", ""), capsize=3)
        ax[0].axhline(A.mean(d, "mvfa", t, "aupro"), color=c, ls="--", lw=0.8)
        ax[1].plot(xs, rt, marker="o", color=c, label=t.replace("Retina_", ""))
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    for a, yl in zip(ax, ("AUPRO (dashed: MVFA)", "pixel-AUROC retention vs MVFA")):
        a.set_xscale("symlog", linthresh=10)
        a.set_xticks([0, 10, 25, 100])
        a.set_xticklabels(["0", "10", "25", "100"])
        a.set_xlim(-1, 130)
        a.set_xlabel("% abnormal masked-source images with masks\n(plain MIL on the rest)", fontsize=9)
        a.set_ylabel(yl)
        a.legend(fontsize=8)
    fig.tight_layout()
    out = R2 / "report"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"budget_curve_r3.{ext}", dpi=150)
    print("wrote", out / "budget_curve_r3.png")


if __name__ == "__main__":
    main()
    budget_curve()
