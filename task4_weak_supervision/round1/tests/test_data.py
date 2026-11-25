"""Step 3.2 checks: no leakage; schedules; no masks in M / B0."""
import pytest
import torch

from common import MASKLESS, TARGETS, has_masks, sources_for
from make_splits import leakage_check
from schedule import build_schedule
from train import MaskLeakError, compute_loss


def test_no_leakage_into_any_target():
    assert leakage_check(content_hash=True)


@pytest.mark.parametrize("target", TARGETS)
def test_schedules(target):
    S = 200
    mh = build_schedule("mh", target, 0, S)
    ctrl = build_schedule("mh_ctrl", target, 0, S)
    assert len(mh) == S and len(ctrl) == S // 2
    # ctrl = exactly M-h's mask items, same order
    assert ctrl == [x for x in mh if x[0] == "mask"]
    assert all(x[0] == "mask" for x in mh[0::2]) and all(x[0] == "pair" for x in mh[1::2])
    # M-h pairs only from mask-less sources; mask items only from masked sources
    assert {x[1] for x in mh if x[0] == "pair"} <= set(MASKLESS)
    assert all(has_masks(x[1]) for x in mh if x[0] == "mask")
    for cfg in ("m", "b0"):
        s = build_schedule(cfg, target, 0, S)
        assert len(s) == S and all(x[0] == "pair" for x in s)
        assert {x[1] for x in s} == set(sources_for(target))      # all sources used
    # nothing from the target, ever
    for cfg in ("mh", "mh_ctrl", "m", "b0"):
        for x in build_schedule(cfg, target, 0, S):
            assert not any(str(v).startswith(f"{target}_AD/") for v in x[2:])
    # seeds change the order, same seed reproduces it
    assert build_schedule("mh", target, 0, S) == mh
    assert build_schedule("mh", target, 1, S) != mh


@pytest.mark.parametrize("cfg", ["m", "b0"])
def test_mask_free_configs_reject_masks(cfg):
    scores = [torch.randn(2, 289, 2) for _ in range(4)]
    with pytest.raises(MaskLeakError):
        compute_loss(cfg, "pair", scores, mask=torch.ones(1, 1, 240, 240), pi=0.1)
    assert torch.isfinite(compute_loss(cfg, "pair", scores, mask=None, pi=0.1))


def test_mh_lambda_scales_only_pair_loss():
    scores = [torch.randn(2, 289, 2) for _ in range(4)]
    a = compute_loss("mh", "pair", scores, pi=0.1, lam=1.0)
    b = compute_loss("mh", "pair", scores, pi=0.1, lam=0.5)
    assert torch.allclose(b, 0.5 * a)
    one = [torch.randn(1, 289, 2) for _ in range(4)]
    mask = (torch.rand(1, 1, 240, 240) > 0.9).float()
    assert torch.allclose(compute_loss("mh", "mask", one, mask=mask, lam=1.0),
                          compute_loss("mh", "mask", one, mask=mask, lam=0.5))


def test_set_pi_values():
    """DECISIONS D1/D2: pi.json from masked sources' masks; pi_maskfree.json fixed 0.30."""
    import json
    from common import RESULTS_DIR
    from set_pi import true_pi
    truth = {d: true_pi(d) for d in ("Liver", "Brain", "Retina_RESC")}
    assert abs(truth["Liver"] - 0.0080) < 5e-4 and abs(truth["Brain"] - 0.0103) < 5e-4
    assert abs(truth["Retina_RESC"] - 0.1191) < 5e-4
    pi = json.load(open(RESULTS_DIR / "pi.json"))
    mf = json.load(open(RESULTS_DIR / "pi_maskfree.json"))
    for t in TARGETS:
        masked = [d for d in sources_for(t) if has_masks(d)]
        want = sum(truth[d] for d in masked) / len(masked)
        assert set(pi[t]) == {d for d in sources_for(t) if d in MASKLESS}
        assert all(abs(v["pi"] - want) < 1e-9 for v in pi[t].values())
        assert set(mf[t]) == set(sources_for(t)) and all(v["pi"] == 0.30 for v in mf[t].values())


def test_mvfa_det_loss_matches_mvfa_formula():
    """Same arithmetic as train_zero.py:125-131, including BCEWithLogits on a probability."""
    from train import mvfa_det_loss
    scores = [torch.randn(2, 289, 2) * 3 for _ in range(4)]
    labels = torch.tensor([0.0, 1.0])
    want = 0
    for s in scores:
        p = torch.softmax(s, -1)[..., 1].mean(-1)
        want += torch.nn.BCEWithLogitsLoss()(p, labels)
    assert torch.allclose(mvfa_det_loss(scores, labels), want)
