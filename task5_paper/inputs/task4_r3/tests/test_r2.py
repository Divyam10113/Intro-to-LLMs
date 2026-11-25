"""Round 2 pre-run checks (plan Step 2). CPU only."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import common2
from common2 import FOLDS, gate_from_bank, greedy_coreset, held_out, sources, train_lists, valid_lists
import train2
from train2 import ARMS, build_epoch, gated_idx, gated_mil_loss, mask_budget_set, mil_loss

R2 = Path(common2.__file__).parent


def _lists(fold, cap=None):
    out = {}
    for d in sources(fold):
        n, a = train_lists(d)
        out[d] = (n[:cap], a[:cap]) if cap else (n, a)
    return out


# ---------------------------------------------------------------- gate (idea #1)
def test_gate_zero_for_patch_identical_to_bank_entry():
    torch.manual_seed(0)
    bank = torch.nn.functional.normalize(torch.randn(500, 64), dim=1)
    f = torch.cat([bank[[3, 77]], torch.nn.functional.normalize(torch.randn(5, 64), dim=1)])
    g = gate_from_bank(f, bank, chunk=128)
    assert torch.allclose(g[:2], torch.zeros(2), atol=1e-6)
    assert (g[2:] > 0.1).all()


def test_selection_indices_carry_no_gradient_and_only_selected_patches_get_gradient():
    z = torch.randn(289, requires_grad=True)
    gate = torch.rand(289)
    idx = gated_idx(torch.sigmoid(z), gate, 1)
    assert not idx.requires_grad and idx.dtype == torch.long and len(idx) == 3
    assert set(idx.tolist()) == set((torch.sigmoid(z) * gate).topk(3).indices.tolist())
    gated_mil_loss(z, 1, gate).backward()
    nz = set(torch.nonzero(z.grad).flatten().tolist())
    assert nz == set(idx.tolist())


def test_normal_images_ignore_gate_and_weigh_2():
    z = torch.randn(289)
    gate = torch.zeros(289)          # a normal training image's own patches are in the bank (G1-12)
    assert set(gated_idx(torch.sigmoid(z), gate, 0).tolist()) == set(torch.sigmoid(z).topk(3).indices.tolist())
    assert torch.isclose(gated_mil_loss(z, 0, gate), 2 * mil_loss(z, 0))


def test_gate_changes_selection_on_abnormal_images():
    z = torch.zeros(289); z[:3] = 5.0; z[10:13] = 4.0     # top-3 by p = 0..2
    gate = torch.ones(289); gate[:3] = 0.0                 # 0..2 look like the normal bank
    assert set(gated_idx(torch.sigmoid(z), gate, 1).tolist()) == {10, 11, 12}


def test_coreset_selects_distinct_points_and_covers_clusters():
    torch.manual_seed(0)
    centres = torch.randn(10, 32) * 10
    x = torch.cat([c + 0.01 * torch.randn(100, 32) for c in centres])
    idx = greedy_coreset(x, 10, seed=0, proj_dim=16)
    assert len(set(idx.tolist())) == 10
    assert len({int(i) // 100 for i in idx}) == 10       # one point from every cluster


# ---------------------------------------------------------------- PU (round 1's checks, unchanged code)
def test_round1_pu_checks_still_pass():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:anyio",
                        str(common2.R1 / "tests" / "test_pu.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


# ---------------------------------------------------------------- masks
@pytest.mark.parametrize("arm", ["mil", "g", "pu"])
@pytest.mark.parametrize("fold", list(FOLDS))
def test_mask_free_arms_schedule_no_mask(arm, fold):
    items = build_epoch(arm, fold, 0, 0, _lists(fold), set())
    assert items and all(k != "mask" for k, *_ in items)


def test_mask_item_in_mask_free_arm_raises():
    ds = train2._Items([("mask", "Liver", "Liver_AD/test/Ungood/img/x.png", 1)], allow_masks=False, gates={})
    with pytest.raises(train2.MaskLeakError):
        ds[0]


@pytest.mark.parametrize("fold", list(FOLDS))
def test_mvfa_and_m100_masks_only_on_masked_sources(fold):
    for arm in ("mvfa", "pu_m100", "g_m100"):
        items = build_epoch(arm, fold, 0, 0, _lists(fold), set())
        for k, d, *_ in items:
            assert (k == "mask") == common2.has_masks(d), (arm, k, d)
    pu = build_epoch("pu_m100", fold, 0, 0, _lists(fold), set())
    assert all((k == "pair") == (not common2.has_masks(d)) for k, d, *_ in pu)


def test_budget_arm_masks_a_fixed_share_of_abnormal_masked_images():
    L = _lists("Chest")
    s = mask_budget_set("g_m10", "Chest", 0, L)
    n_ab = sum(len(L[d][1]) for d in L if common2.has_masks(d))
    assert abs(len(s) - 0.1 * n_ab) <= 3
    items = build_epoch("g_m10", "Chest", 0, 0, L, s)
    assert {p for k, d, p, y in items if k == "mask"} == s
    assert s == mask_budget_set("g_m10", "Chest", 0, L)


# ---------------------------------------------------------------- held-out data never used
@pytest.mark.parametrize("fold", list(FOLDS))
def test_no_heldout_file_in_training_bank_or_validation(fold):
    used = []
    for d in sources(fold):
        n, a = train_lists(d)
        used += n + a + [p for p, _ in valid_lists(d, 0)]
    common2.assert_no_heldout(used, fold)
    assert not set(sources(fold)) & set(held_out(fold))
    # the bank of dataset d is built from d's own normal subsample only
    for d in sources(fold):
        assert all(p.startswith(f"{d}_AD/") for p in train_lists(d)[0])


def test_no_training_file_is_a_test_file_of_a_heldout_dataset():
    from common import list_test_set
    for fold in FOLDS:
        test = {p for h in held_out(fold) for p, _ in list_test_set(h)}
        used = {p for d in sources(fold) for p in sum(train_lists(d), [])}
        assert not used & test


def test_valid_excludes_files_identical_to_test_files():
    v = json.load(open(R2 / "splits" / "valid.json"))
    ex = json.load(open(R2 / "splits" / "valid_excluded.json"))
    assert len(ex) == 11
    allv = {p for d in v.values() for k in d.values() for p in k}
    assert not allv & set(ex)


def test_valid_cap_and_seeded_draw():
    a, b = valid_lists("Chest", 0), valid_lists("Chest", 1)
    assert sum(y for _, y in a) == 250 and sum(1 - y for _, y in a) == 70
    assert a != b and valid_lists("Chest", 0) == a


def test_only_evaluate_target_reads_test_sets():
    for f in R2.glob("*.py"):
        if f.name == "evaluate_target.py":
            continue
        src = f.read_text()
        assert "list_test_set" not in src, f.name
        assert "/test/" not in src.replace("Ungood/img/x.png", ""), f.name


# ---------------------------------------------------------------- schedules
def test_mvfa_mil_g_see_same_images_in_same_order():
    L = _lists("Brain")
    ref = [(d, p, y) for _, d, p, y in build_epoch("mvfa", "Brain", 1, 2, L, set())]
    for arm in ("mil", "g", "g_m100"):
        assert [(d, p, y) for _, d, p, y in build_epoch(arm, "Brain", 1, 2, L, set())] == ref
    assert ref != [(d, p, y) for _, d, p, y in build_epoch("mvfa", "Brain", 1, 3, L, set())]
    assert ref != [(d, p, y) for _, d, p, y in build_epoch("mvfa", "Brain", 2, 2, L, set())]


def test_epoch_covers_every_subsample_image_once():
    L = _lists("Liver")
    items = build_epoch("mil", "Liver", 0, 0, L, set())
    assert sorted(p for _, _, p, _ in items) == sorted(p for d in L for p in L[d][0] + L[d][1])
    pu = build_epoch("pu", "Liver", 0, 0, L, set())
    seen = {p for _, _, a, b in pu for p in (a, b)}
    assert seen == {p for d in L for p in L[d][0] + L[d][1]}
    assert all(a.split("/")[0] == b.split("/")[0] == d + "_AD" and "/good/" in a and "/Ungood/" in b
               for _, d, a, b in pu)


def test_mvfa_det_only_items_on_maskless_sources():
    items = build_epoch("mvfa", "Liver", 0, 0, _lists("Liver"), set())
    assert {k for k, d, *_ in items if not common2.has_masks(d)} == {"det"}
