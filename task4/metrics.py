"""Image AUROC, pixel AUROC and AUPRO (integrated to FPR 0.3, normalised).

AUPRO follows the MVTec-AD definition (Bergmann et al. 2021):
  * FPR(t) = fraction of all GT-normal pixels (every image) with score >= t;
  * PRO(t) = mean over GT connected components (8-connectivity) of the fraction
    of the component's pixels with score >= t;
  * AUPRO = integral of PRO over FPR in [0, 0.3], divided by 0.3.
It is computed one image at a time against a fixed threshold grid, so memory
stays O(image + thresholds) rather than O(all pixels x thresholds).
"""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

STRUCT8 = np.ones((3, 3), dtype=int)


def image_auroc(labels, scores) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def pixel_auroc(masks, maps) -> float:
    """masks, maps: [N, H, W]. Exact, via sklearn (as MVFA does)."""
    m = (np.asarray(masks) > 0).ravel()
    s = np.asarray(maps, dtype=np.float32).ravel()
    return float(roc_auc_score(m, s))


def _thresholds(masks, maps, max_fpr, n_grid, rng):
    """Thresholds whose FPRs cover [0, max_fpr] finely and [max_fpr, 1] coarsely.

    Chosen as quantiles of a subsample of GT-normal pixel scores, plus one value
    above the largest normal score (so FPR = 0 is always on the curve).
    """
    sample, max_norm = [], -np.inf
    per_img = max(1, 2_000_000 // max(1, len(maps)))
    for m, s in zip(masks, maps):
        v = s[m == 0]
        if v.size:
            max_norm = max(max_norm, float(v.max()))
            sample.append(v if v.size <= per_img else rng.choice(v, per_img, replace=False))
    sample = np.concatenate(sample)
    fine = np.linspace(0.0, max_fpr * 1.2, n_grid)       # target FPRs
    coarse = np.linspace(max_fpr * 1.2, 1.0, 50)
    q = 1.0 - np.concatenate([fine, coarse])
    ts = np.quantile(sample, np.clip(q, 0, 1))
    ts = np.concatenate([ts, [np.nextafter(max_norm, np.inf)]])
    return np.unique(ts)  # ascending


def aupro(masks, maps, max_fpr: float = 0.3, n_grid: int = 1000, seed: int = 0) -> float:
    """masks, maps: sequences/arrays of [H, W]. Returns normalised AUPRO."""
    rng = np.random.default_rng(seed)
    masks = [(np.asarray(m) > 0) for m in masks]
    maps = [np.asarray(s, dtype=np.float32) for s in maps]
    ts = _thresholds(masks, maps, max_fpr, n_grid, rng)

    fp = np.zeros(ts.size, dtype=np.int64)       # normal pixels >= t, summed over images
    n_normal = 0
    pro_sum = np.zeros(ts.size, dtype=np.float64)
    n_regions = 0
    for m, s in zip(masks, maps):
        neg = np.sort(s[~m])
        n_normal += neg.size
        fp += neg.size - np.searchsorted(neg, ts, side="left")
        if m.any():
            lab, n = ndimage.label(m, structure=STRUCT8)
            for r in range(1, n + 1):
                reg = np.sort(s[lab == r])
                pro_sum += (reg.size - np.searchsorted(reg, ts, side="left")) / reg.size
            n_regions += n
    if n_regions == 0 or n_normal == 0:
        return float("nan")
    fpr = fp / n_normal
    pro = pro_sum / n_regions
    # curve ordered by increasing FPR; append (1, 1) end point (t = -inf)
    order = np.argsort(fpr, kind="stable")
    fpr, pro = np.concatenate([fpr[order], [1.0]]), np.concatenate([pro[order], [1.0]])
    # integrate to max_fpr with linear interpolation at the cut
    keep = fpr <= max_fpr
    x, y = fpr[keep], pro[keep]
    if x.size == 0 or x[-1] < max_fpr:
        y_cut = np.interp(max_fpr, fpr, pro)
        x, y = np.concatenate([x, [max_fpr]]), np.concatenate([y, [y_cut]])
    if x[0] > 0:
        x, y = np.concatenate([[0.0], x]), np.concatenate([[0.0], y])
    return float(np.trapezoid(y, x) / max_fpr)
