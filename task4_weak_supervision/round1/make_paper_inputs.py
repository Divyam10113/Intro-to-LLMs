"""Step 6 hand-off: build the two results tables and the Liver heat-map figure,
and copy everything the paper needs to report/paper_inputs/.

    python make_paper_inputs.py

Reads only results/ and the saved Liver maps (runs/<run>/maps.npz); no GPU.

Heat-map figure, fixed rules (chosen before looking at any map):
- images: the Liver test images with label 1 and a non-empty mask, sorted by
  path; take the ones at positions 1/8, 3/8, 5/8 and 7/8 of that list;
- colour scale: per model, from the 1st to the 99.5th percentile of that
  model's map values over all Liver test pixels (a fixed random subsample),
  so each column uses one scale for every row; maps are not comparable in
  absolute value across models (MVFA's maps are sums of 4 level probabilities);
- ground truth drawn as a cyan contour on every panel.
"""
import csv
import json
import shutil
from collections import defaultdict

import numpy as np

from common import HDD, RESULTS_DIR, ROOT, TARGETS, load_image, load_mask

OUT = ROOT / "report" / "paper_inputs"
RUNS = HDD / "runs"
PTS = 100.0
MAP_RUNS = [("R0", "R0_Liver"), ("M-h", "mh_Liver_s0"), ("M-h-ctrl", "mh_ctrl_Liver_s0"),
            ("M", "m_Liver_s0"), ("B0", "b0_Liver_s0")]


def rows():
    rs = list(csv.DictReader(open(RESULTS_DIR / "results.csv")))
    for r in rs:
        for k in ("img_auroc", "pix_auroc", "aupro"):
            r[k] = float(r[k]) * PTS
    return rs


def primary(rs, cfg):
    """Untagged runs only (the pre-registered ones): {(target, seed): row}."""
    return {(r["target"], int(r["seed"])): r for r in rs
            if r["config"] == cfg and r["run_id"] == f"{cfg}_{r['target']}_s{r['seed']}"}


def ms(v):
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def write_table(name, header, body):
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")


def table_a(rs):
    paper = {r["target"]: r for r in csv.DictReader(open(RESULTS_DIR / "paper_numbers.csv"))}
    r0 = {r["target"]: r for r in rs if r["config"] == "R0"}
    pi = json.load(open(RESULTS_DIR / "pi.json"))
    mh, ctrl = primary(rs, "mh"), primary(rs, "mh_ctrl")
    f = lambda m, s: f"{m:.2f}" if np.isnan(s) else f"{m:.2f} ± {s:.2f}"
    head = ["target", "model", "n", "img AUROC", "pix AUROC", "AUPRO"]
    body = []
    for t in TARGETS:
        body.append([t, "R0 (paper)", "–", paper[t]["img_auroc"], paper[t]["pix_auroc"], "–"])
        body.append([t, "R0 (reproduced)", 1] + [f"{r0[t][k]:.2f}" for k in ("img_auroc", "pix_auroc", "aupro")])
        for name, d in (("M-h", mh), ("M-h-ctrl", ctrl)):
            runs = [d[(t, s)] for s in (0, 1, 2) if (t, s) in d]
            body.append([t, name, len(runs)] + [f(*ms([r[k] for r in runs])) for k in ("img_auroc", "pix_auroc", "aupro")])
    write_table("table_a_main", head, body)

    dhead = ["target", "seed", "Δ img AUROC", "Δ pix AUROC", "Δ AUPRO"]
    dbody = []
    for t in TARGETS:
        for s in (0, 1, 2):
            if (t, s) in mh and (t, s) in ctrl:
                dbody.append([t, s] + [f"{mh[(t, s)][k] - ctrl[(t, s)][k]:+.2f}" for k in ("img_auroc", "pix_auroc", "aupro")])
    write_table("table_a_deltas", dhead, dbody)

    phead = ["target", "mask-less source", "π used (D1)", "π estimator raw, seg branch (report only)"]
    est = json.load(open(RESULTS_DIR / "pi_estimated_seg.json"))
    pbody = [[t, d, f"{v['pi']:.4f}", f"{est[t][d]['pi_raw']:.3f}"] for t in TARGETS for d, v in pi[t].items()]
    write_table("table_a_pi", phead, pbody)


def table_b(rs):
    r0 = {r["target"]: r for r in rs if r["config"] == "R0"}
    m, b0 = primary(rs, "m"), primary(rs, "b0")
    est = json.load(open(RESULTS_DIR / "pi_estimated_det.json"))
    pim = json.load(open(RESULTS_DIR / "pi_maskfree.json"))
    head = ["target", "model", "n", "img AUROC", "pix AUROC", "AUPRO",
            "retention (pix−50)/(R0−50)", "AUPRO / R0 AUPRO"]
    body = []
    for t in TARGETS:
        R = r0[t]
        for name, d, seeds in (("M", m, (0, 1, 2) if t == "Liver" else (0,)), ("B0", b0, (0,))):
            runs = [d[(t, s)] for s in seeds if (t, s) in d]
            vals = {k: ms([r[k] for r in runs]) for k in ("img_auroc", "pix_auroc", "aupro")}
            fm = lambda k: f"{vals[k][0]:.2f}" if len(runs) == 1 else f"{vals[k][0]:.2f} ± {vals[k][1]:.2f}"
            s0 = d[(t, 0)]
            body.append([t, name + (" (seeds 0-2)" if len(runs) > 1 else " (seed 0)"), len(runs),
                         fm("img_auroc"), fm("pix_auroc"), fm("aupro"),
                         f"{(s0['pix_auroc'] - 50) / (R['pix_auroc'] - 50):.3f} (s0)",
                         f"{s0['aupro'] / R['aupro']:.3f} (s0)"])
    write_table("table_b_maskfree", head, body)

    phead = ["target", "source", "π used (D2)", "π estimator raw, det branch (report only)"]
    pbody = [[t, d, f"{pim[t][d]['pi']:.2f}", f"{est[t][d]['pi_raw']:.3f}"] for t in TARGETS for d in pim[t]]
    write_table("table_b_pi", phead, pbody)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {name: np.load(RUNS / run / "maps.npz") for name, run in MAP_RUNS}
    paths = data["R0"]["paths"]
    labels = data["R0"]["labels"]
    for name, _ in MAP_RUNS:  # same images, same order in every run
        assert (data[name]["paths"] == paths).all()
    cand = sorted(i for i in range(len(paths)) if labels[i] == 1 and load_mask(str(paths[i]), 1).sum() > 0)
    cand.sort(key=lambda i: str(paths[i]))
    pick = [cand[int(len(cand) * q)] for q in (1 / 8, 3 / 8, 5 / 8, 7 / 8)]

    rng = np.random.default_rng(0)
    scale = {}
    for name, _ in MAP_RUNS:
        mp = data[name]["maps"]
        sub = mp[rng.integers(0, len(mp), 200)].astype(np.float32).ravel()
        scale[name] = np.percentile(sub, [1, 99.5])

    cols = ["image"] + [n for n, _ in MAP_RUNS]
    fig, ax = plt.subplots(len(pick), len(cols), figsize=(2 * len(cols), 2 * len(pick)))
    for r, i in enumerate(pick):
        img = load_image(str(paths[i])).permute(1, 2, 0).numpy()
        gt = load_mask(str(paths[i]), 1)[0].numpy()
        for c, name in enumerate(cols):
            a = ax[r, c]
            if name == "image":
                a.imshow(img)
            else:
                lo, hi = scale[name]
                a.imshow(data[name]["maps"][i].astype(np.float32), cmap="inferno", vmin=lo, vmax=hi)
            a.contour(gt, levels=[0.5], colors="cyan", linewidths=0.8)
            a.set_xticks([]), a.set_yticks([])
            if r == 0:
                a.set_title(name, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"liver_heatmaps.{ext}", dpi=150)
    (OUT / "liver_heatmaps_images.txt").write_text(
        "Rows (top to bottom), Liver test images at positions 1/8, 3/8, 5/8, 7/8 of the "
        "sorted abnormal-with-mask list:\n" + "\n".join(str(paths[i]) for i in pick) + "\n"
        "Colour scale per column: 1st-99.5th percentile of that model's maps over the Liver test set.\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows()
    table_a(rs)
    table_b(rs)
    figure()
    for f in ("results.csv", "paper_numbers.csv", "pi.json", "pi_maskfree.json",
              "pi_estimated_seg.json", "pi_estimated_det.json", "analysis.md"):
        shutil.copy(RESULTS_DIR / f, OUT / f)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
