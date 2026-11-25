"""Our CPU-safe forward must equal MVFA's original forward, and only the
segmentation adapters may change during training. Loads CLIP (~30 s on CPU)."""
import types

import pytest
import torch

from common import build_model  # adds MVFA-AD to sys.path
import CLIP.adapter as adapter_mod  # noqa: E402


@pytest.fixture(scope="module")
def r0():
    torch.manual_seed(0)
    return build_model("Liver", "cpu")


def test_forward_identical_to_mvfa(r0):
    # MVFA's original forward only works at batch size 1: it loops
    # range(len(attn)) (= batch size) over a 1-element list -> IndexError at B > 1.
    model, _ = r0
    x = torch.rand(1, 3, 240, 240)

    class _Z:  # makes MVFA's `torch.zeros(...).to('cuda')` a CPU no-op
        def __init__(self, t): self.t = t
        def to(self, *_): return self.t
    proxy = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    # only the attention-map buffer is built as torch.zeros([H, H]).to("cuda")
    proxy.zeros = lambda *a, **k: _Z(torch.zeros(*a, **k)) if isinstance(a[0], list) else torch.zeros(*a, **k)
    old = adapter_mod.torch
    adapter_mod.torch = proxy
    try:
        with torch.no_grad():
            _, seg_o, det_o = adapter_mod.CLIP_Inplanted.forward(model, x)
            _, seg_n, det_n = model(x)
    finally:
        adapter_mod.torch = old
    for a, b in zip(seg_o + det_o, seg_n + det_n):
        assert torch.equal(a, b)


def test_r0_weights_loaded_and_reinit_differs(r0):
    model, _ = r0
    from common import R0_CKPT_DIR
    ck = torch.load(R0_CKPT_DIR / "Liver.pth", map_location="cpu")
    for k, v in ck["seg_adapters"].items():
        assert torch.equal(model.seg_adapters.state_dict()[k], v)
    m2, _ = build_model("Liver", "cpu", reinit_seg=True)
    m3, _ = build_model("Liver", "cpu", reinit_seg=True)
    k0 = next(iter(ck["seg_adapters"]))
    assert not torch.equal(m2.seg_adapters.state_dict()[k0], ck["seg_adapters"][k0])
    assert torch.equal(m2.seg_adapters.state_dict()[k0], m3.seg_adapters.state_dict()[k0])  # fixed init seed
    for k, v in ck["det_adapters"].items():                                                  # det kept from R0
        assert torch.equal(m2.det_adapters.state_dict()[k], v)


def test_batch_of_two_equals_two_singles(r0):
    """Training forwards the PU pair (normal, abnormal) as one batch of 2. Every
    op is per-sample (LayerNorm, attention within an image), so this must equal
    two batch-1 forwards, i.e. the plan's gradient accumulation over the pair."""
    model, _ = r0
    x = torch.rand(2, 3, 240, 240)
    with torch.no_grad():
        _, seg2, det2 = model(x)
        _, seg_a, det_a = model(x[:1])
        _, seg_b, det_b = model(x[1:])
    for t2, ta, tb in zip(seg2 + det2, seg_a + det_a, seg_b + det_b):
        # float32 batched-matmul rounding only (observed max 3.6e-5 on values ~9)
        assert torch.allclose(t2, torch.cat([ta, tb]), rtol=1e-5, atol=1e-4)


def test_only_seg_adapters_get_gradients(r0):
    from common import level_scores, text_features
    from train import compute_loss
    model, clip_model = build_model("Liver", "cpu")
    for p in model.seg_adapters.parameters():
        p.requires_grad_(True)
    tf = text_features(clip_model, "cpu")["Chest"]
    _, seg, _ = model(torch.rand(2, 3, 240, 240))
    compute_loss("mh", "pair", level_scores(seg, tf), pi=0.1).backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad and all(n.startswith("seg_adapters.") for n in with_grad)
    # The last adapter's fc2 output only enters the residual stream after block
    # 24, which feeds nothing but the unused pooled output, so it never gets a
    # gradient (same in MVFA: dead weights). Every other seg-adapter weight must.
    dead = {"seg_adapters.3.fc2.0.weight"}
    for n, p in model.named_parameters():
        if n.startswith("seg_adapters."):
            if n in dead:
                assert p.grad is None
            else:
                assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_fresh_det_adapters_mask_free_training(r0):
    """DECISIONS D3: m/b0 re-initialise both adapter sets; the det loss (image
    labels) reaches the detection adapters; seg init identical to seg-only reinit."""
    from common import R0_CKPT_DIR, level_scores, text_features
    from train import compute_loss, mvfa_det_loss
    ck = torch.load(R0_CKPT_DIR / "Liver.pth", map_location="cpu")
    m_seg, _ = build_model("Liver", "cpu", reinit_seg=True)
    m_both, cm = build_model("Liver", "cpu", reinit_seg=True, reinit_det=True)
    k = next(iter(ck["det_adapters"]))
    assert not torch.equal(m_both.det_adapters.state_dict()[k], ck["det_adapters"][k])
    ks = next(iter(ck["seg_adapters"]))
    assert torch.equal(m_both.seg_adapters.state_dict()[ks], m_seg.seg_adapters.state_dict()[ks])
    assert not torch.equal(m_both.seg_adapters.state_dict()[ks], m_both.det_adapters.state_dict()[ks])
    for p in m_both.parameters():
        p.requires_grad_(False)
    for p in list(m_both.seg_adapters.parameters()) + list(m_both.det_adapters.parameters()):
        p.requires_grad_(True)
    tf = text_features(cm, "cpu")["Brain"]
    _, seg, det = m_both(torch.rand(2, 3, 240, 240))
    loss = compute_loss("m", "pair", level_scores(seg, tf), pi=0.3) + \
        mvfa_det_loss(level_scores(det, tf), torch.tensor([0.0, 1.0]))
    loss.backward()
    with_grad = {n for n, p in m_both.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert all(n.startswith(("seg_adapters.", "det_adapters.")) for n in with_grad)
    assert any(n.startswith("det_adapters.") for n in with_grad)
