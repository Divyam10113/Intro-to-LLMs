"""Step 3.2 checks: AUPRO and AUROCs."""
import numpy as np
from sklearn.metrics import roc_auc_score

from metrics import aupro, image_auroc, pixel_auroc


def _masks(n=12, h=64, seed=0):
    g = np.random.default_rng(seed)
    masks = np.zeros((n, h, h), bool)
    for i in range(n // 2, n):                     # half the images anomalous, 1-3 regions each
        for _ in range(g.integers(1, 4)):
            y, x, r = g.integers(8, h - 8, size=2).tolist() + [int(g.integers(2, 7))]
            masks[i, y - r:y + r, x - r:x + r] = True
    return masks


def test_aupro_perfect_is_one():
    m = _masks()
    assert abs(aupro(m, m.astype(np.float32)) - 1.0) < 1e-9


def test_aupro_inverted_is_about_zero():
    m = _masks()
    assert aupro(m, (~m).astype(np.float32)) < 0.01


def test_aupro_random_is_low_and_noisy_perfect_is_high():
    g = np.random.default_rng(1)
    m = _masks()
    rnd = g.random(m.shape).astype(np.float32)
    good = m + 0.3 * g.random(m.shape)
    a_r, a_g = aupro(m, rnd), aupro(m, good)
    # random map: PRO ~= FPR, so normalised AUPRO ~= 0.15
    assert 0.10 < a_r < 0.20, a_r
    assert a_g > 0.95, a_g


def test_aupro_small_region_counts_equally():
    # one big region fully found + one small region missed -> PRO ~ 0.5, not ~1
    m = np.zeros((2, 64, 64), bool)
    m[1, 5:45, 5:45] = True
    m[1, 55:58, 55:58] = True
    g = np.random.default_rng(3)
    s = (0.5 * g.random(m.shape)).astype(np.float32)   # normal pixels in [0, 0.5)
    s[1, 5:45, 5:45] = 1.0                              # big region: found at FPR 0
    s[1, 55:58, 55:58] = -1.0                           # small region: found only at FPR 1
    assert abs(aupro(m, s) - 0.5) < 1e-6


def test_aurocs_match_sklearn():
    g = np.random.default_rng(2)
    y = g.integers(0, 2, 500)
    s = g.random(500) + 0.3 * y
    assert image_auroc(y, s) == roc_auc_score(y, s)
    m = g.random((6, 32, 32)) > 0.8
    mp = (g.random((6, 32, 32)) + 0.2 * m).astype(np.float32)   # maps are float32 in practice
    assert abs(pixel_auroc(m, mp) - roc_auc_score(m.ravel(), mp.ravel())) < 1e-12
