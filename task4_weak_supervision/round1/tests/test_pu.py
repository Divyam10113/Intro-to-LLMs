"""Step 3.2 checks: PU loss and pi estimator."""
import math

import numpy as np
import torch

from pu import estimate_pi, estimate_pi_plan_literal, pu_risk, topk_mil_loss, two_class_to_logit


def _l(z, y):  # hand BCE-with-logits
    p = 1 / (1 + math.exp(-z))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def test_pu_matches_hand_calculation():
    zn, zu, pi = [-2.0, 0.5, 1.0], [3.0, -1.0], 0.2
    neg = (1 - pi) * np.mean([_l(z, 0) for z in zn])
    pos = np.mean([_l(z, 1) for z in zu]) - (1 - pi) * np.mean([_l(z, 1) for z in zn])
    want = neg + max(0.0, pos)
    got = pu_risk(torch.tensor(zn), torch.tensor(zu), pi).item()
    assert abs(got - want) < 1e-6, (got, want)


def test_pu_nonnegative_clamp_active():
    # normal patches low, unlabelled patches all high (the overfitting regime):
    # mean_U l(z,1) - (1-pi) mean_N l(z,1) < 0 -> clamped to 0
    zn, zu, pi = [-3.0, -3.0], [3.0, 3.0], 0.1
    got = pu_risk(torch.tensor(zn), torch.tensor(zu), pi).item()
    want = (1 - pi) * _l(-3.0, 0)
    assert abs(got - want) < 1e-6


def test_pu_pi0_no_u_is_plain_bce():
    zn = torch.tensor([-1.0, 0.3, 2.0])
    want = torch.nn.functional.binary_cross_entropy_with_logits(zn, torch.zeros(3)).item()
    assert abs(pu_risk(zn, torch.tensor([]), 0.0).item() - want) < 1e-7


def test_pu_gradients_flow():
    zn = torch.tensor([-1.0, 0.0], requires_grad=True)
    zu = torch.tensor([2.0, -2.0], requires_grad=True)
    pu_risk(zn, zu, 0.1).backward()
    assert zn.grad.abs().sum() > 0 and zu.grad.abs().sum() > 0


def test_two_class_logit_matches_softmax():
    s = torch.randn(10, 2) * 5
    assert torch.allclose(torch.sigmoid(two_class_to_logit(s)), torch.softmax(s, -1)[:, 1], atol=1e-6)


def test_topk_mil():
    z = torch.full((289,), -5.0)
    z[:3] = 5.0                           # k = max(3, round(2.89)) = 3
    p = torch.sigmoid(torch.tensor(5.0)).item()
    assert abs(topk_mil_loss(z, 1).item() - (-math.log(p))) < 1e-5
    assert abs(topk_mil_loss(z, 0).item() - (-math.log(1 - p))) < 1e-4


def _synthetic(n, seed):
    g = np.random.default_rng(seed)
    pn = g.normal(0, 1, n)
    anom = g.random(n) < 0.10
    pa = np.where(anom, g.normal(3, 1, n), g.normal(0, 1, n))
    return pn, pa


def test_pi_estimator_synthetic_plan_check():
    # Plan: normal ~ N(0,1); abnormal = 90% N(0,1) + 10% N(3,1) -> 0.10 +- 0.03.
    # n = 14,450 patches per class = 50 held-out images x 289 patches (our real size).
    est = [estimate_pi(*_synthetic(14450, s))[0] for s in range(20)]
    assert abs(np.mean(est) - 0.10) <= 0.03, np.mean(est)
    big = estimate_pi(*_synthetic(200_000, 123))[0]
    assert abs(big - 0.10) <= 0.03, big


def test_plan_literal_formula_fails_its_own_check():
    # Documents ISSUES_LOG entry: the formula as written returns ~1 (-> clip 0.30).
    _, raw = estimate_pi_plan_literal(*_synthetic(50_000, 0))
    assert raw > 0.9
