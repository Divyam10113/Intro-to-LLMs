"""PU loss, top-k MIL loss and the pi (anomalous-patch fraction) estimator.

Conventions
-----------
z : patch logit for "anomalous". In MVFA a patch's two-class score is
    s = 100 * <token, text_features>  (shape [L, 2]) and p = softmax(s)[:, 1],
    so z = s[:, 1] - s[:, 0] and sigmoid(z) == p exactly.
N : patches of a normal image (all normal).
U : patches of an abnormal image from the same dataset (a mixture: a fraction
    pi of them anomalous, 1 - pi normal).
"""
import numpy as np
import torch
import torch.nn.functional as F

PI_CLIP = (0.005, 0.30)


def two_class_to_logit(scores: torch.Tensor) -> torch.Tensor:
    """[..., 2] two-class scores -> [...] binary logit for class 1."""
    return scores[..., 1] - scores[..., 0]


def _bce(z: torch.Tensor, target: float) -> torch.Tensor:
    """Mean BCE-with-logits of all entries of z against a constant target."""
    return F.binary_cross_entropy_with_logits(z, torch.full_like(z, target))


def pu_risk(z_n: torch.Tensor, z_u: torch.Tensor, pi: float) -> torch.Tensor:
    """Non-negative PU risk (anomalies = the unlabelled positive class).

    risk = (1 - pi) * mean_N l(z, 0) + max(0, mean_U l(z, 1) - (1 - pi) * mean_N l(z, 1))

    The second term estimates pi * E_anomalous[l(z, 1)] from the mixture U minus
    its normal part, clamped at 0 (nnPU, Kiryo et al. 2017). If U is empty the
    second term is 0, so with pi = 0 the risk is plain BCE on N.
    """
    z_n = z_n.float().reshape(-1)
    z_u = z_u.float().reshape(-1)
    neg = (1.0 - pi) * _bce(z_n, 0.0)
    if z_u.numel() == 0:
        return neg
    pos = _bce(z_u, 1.0) - (1.0 - pi) * _bce(z_n, 1.0)
    return neg + torch.clamp(pos, min=0.0)


def topk_k(num_patches: int) -> int:
    return max(3, int(round(0.01 * num_patches)))


def topk_mil_loss(z: torch.Tensor, label: int) -> torch.Tensor:
    """BCE(mean of the top-k patch probabilities, image label), k = max(3, 1%)."""
    p = torch.sigmoid(z.float().reshape(-1))
    k = topk_k(p.numel())
    bag = p.topk(k).values.mean().clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(bag, torch.tensor(float(label), device=bag.device))


def estimate_pi(p_norm, p_abn, lo: float = 0.10, hi: float = 0.50, k_se: float = 1.0,
                clip=PI_CLIP):
    """Mixture-proportion estimate of the anomalous-patch fraction in abnormal images.

    F_abn = (1 - pi) F_norm + pi F_anom. Anomalous patches score high, so in the
    lower tail F_anom(t) / F_norm(t) -> 0 and F_abn(t) / F_norm(t) -> 1 - pi:

        pi = 1 - F_abn(t*) / F_norm(t*),
        t* = argmin_t [ F_abn(t)/F_norm(t) + k_se * SE(t) ],  F_norm(t) in [lo, hi].

    Two deviations from the plan, both logged in ISSUES_LOG.md:
    1. The plan wrote min_t (1 - F_abn)/(1 - F_norm) (upper tails). That ratio
       is >= 1 everywhere, so it returns ~1 (clipped to 0.30), never pi.
    2. A raw min over thresholds down to F_norm = 0.01 is dominated by noise
       in the sparse tail and overestimates pi (0.19 +- 0.05 for a true 0.10 at
       ~14.5k patches per class). Restricting to F_norm in [0.10, 0.50] and
       choosing t by a 1-SE upper bound (as in BBE, Garg et al. 2021) gives
       0.12 +- 0.02 at that size.

    Returns (pi_clipped, pi_raw).
    """
    p_norm = np.sort(np.asarray(p_norm, dtype=np.float64).ravel())
    p_abn = np.sort(np.asarray(p_abn, dtype=np.float64).ravel())
    ts = np.quantile(p_norm, np.linspace(lo, hi, 500))
    f_norm = np.searchsorted(p_norm, ts, side="right") / p_norm.size
    f_abn = np.searchsorted(p_abn, ts, side="right") / p_abn.size
    ratio = f_abn / f_norm
    se = ratio * np.sqrt(1.0 / (p_abn.size * np.maximum(f_abn, 1e-12)) + 1.0 / (p_norm.size * f_norm))
    i = int(np.argmin(ratio + k_se * se))
    pi_raw = float(1.0 - ratio[i])
    return float(np.clip(pi_raw, *clip)), pi_raw


def estimate_pi_plan_literal(p_norm, p_abn, min_mass: float = 0.01, clip=PI_CLIP):
    """The plan's formula verbatim, kept only to document that it fails its check."""
    p_norm = np.sort(np.asarray(p_norm, dtype=np.float64).ravel())
    p_abn = np.sort(np.asarray(p_abn, dtype=np.float64).ravel())
    ts = np.quantile(p_norm, np.linspace(0.0, 1.0 - min_mass, 2000))
    s_norm = 1 - np.searchsorted(p_norm, ts, side="right") / p_norm.size
    s_abn = 1 - np.searchsorted(p_abn, ts, side="right") / p_abn.size
    ok = s_norm >= min_mass
    pi_raw = float((s_abn[ok] / s_norm[ok]).min())
    return float(np.clip(pi_raw, *clip)), pi_raw
