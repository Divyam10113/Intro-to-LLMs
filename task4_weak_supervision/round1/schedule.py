"""Deterministic training schedules (the run seed controls only this order).

Two independent random streams, so that M-h and M-h-ctrl see the SAME masked
images in the SAME order:
  * mask stream (rng seed*1000 + 1): shuffled pool of all masked-source train
    images (normal + abnormal), reshuffled each pass, like MVFA's loader;
  * pair stream (rng seed*1000 + 2): per PU/MIL step, a dataset chosen
    uniformly from the stream's datasets, then the next normal and the next
    abnormal image of that dataset (per-dataset shuffled cycles).

Configs (S = steps = iterations):
  mh       iterations alternate mask, pair, mask, pair, ...    (S/2 + S/2)
  mh_ctrl  the same iterations with the pair items removed     (S/2 mask updates)
           -> lambda = 0 without ever calling optimizer.step() on a zero
              gradient, which with Adam is NOT a no-op (momentum keeps moving)
  m, b0    S pair items drawn from ALL sources; never a mask
"""
import numpy as np

from common import has_masks, load_splits, sources_for, MASKLESS


class _Cycle:
    def __init__(self, items, rng):
        self.items, self.rng, self.order, self.i = list(items), rng, [], 0

    def next(self):
        if self.i >= len(self.order):
            self.order = self.rng.permutation(len(self.items)).tolist()
            self.i = 0
        x = self.items[self.order[self.i]]
        self.i += 1
        return x


def build_schedule(config, target, seed, steps, splits=None):
    """Returns a list of items:
       ("mask", dataset, img_rel, label)            one masked-source image
       ("pair", dataset, normal_rel, abnormal_rel)  one normal + one abnormal image
    """
    assert config in ("mh", "mh_ctrl", "m", "b0"), config
    srcs = sources_for(target)
    splits = splits or {d: load_splits(d) for d in srcs}

    if config in ("mh", "mh_ctrl"):
        pair_ds = [d for d in srcs if d in MASKLESS]
    else:
        pair_ds = list(srcs)
    masked_ds = [d for d in srcs if has_masks(d)]

    rng_mask = np.random.default_rng(seed * 1000 + 1)
    rng_pair = np.random.default_rng(seed * 1000 + 2)
    mask_pool = sorted((d, p, y) for d in masked_ds
                       for y, k in ((0, "train_normal"), (1, "train_abnormal")) for p in splits[d][k])
    mask_cyc = _Cycle(mask_pool, rng_mask)
    norm_cyc = {d: _Cycle(splits[d]["train_normal"], np.random.default_rng([seed, i, 0]))
                for i, d in enumerate(pair_ds)}
    abn_cyc = {d: _Cycle(splits[d]["train_abnormal"], np.random.default_rng([seed, i, 1]))
               for i, d in enumerate(pair_ds)}

    def pair():
        d = pair_ds[int(rng_pair.integers(len(pair_ds)))]
        return ("pair", d, norm_cyc[d].next(), abn_cyc[d].next())

    sched = []
    if config in ("mh", "mh_ctrl"):
        for it in range(steps):
            if it % 2 == 0:
                d, p, y = mask_cyc.next()
                sched.append(("mask", d, p, y))
            else:
                item = pair()
                if config == "mh":
                    sched.append(item)
    else:
        sched = [pair() for _ in range(steps)]
    return sched
