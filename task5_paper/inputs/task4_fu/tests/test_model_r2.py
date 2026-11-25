"""Seeds set the initial weights (G1-10). Loads CLIP on CPU (~1 min)."""
import torch

from train2 import build_r2_model


def test_seed_sets_initial_weights():
    w = {}
    for s in (0, 1, 0):
        m, _ = build_r2_model(s, "cpu")
        w.setdefault(s, []).append(torch.cat([p.flatten() for p in
                                              list(m.seg_adapters.parameters()) + list(m.det_adapters.parameters())]))
        del m
    assert not torch.equal(w[0][0], w[1][0])
    assert torch.equal(w[0][0], w[0][1])
