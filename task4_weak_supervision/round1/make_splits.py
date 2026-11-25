"""Step 1.4: fixed subsample lists per source dataset (seed 0).

MVFA's zero-shot training reads each SOURCE dataset's test/ folder (there is
no separate training split), so the lists are drawn from test/good and
test/Ungood. Per dataset: at most 500 normal + 500 abnormal images, then 10%
of each class (rounded up) is held out for pi estimation.

Output: splits/<dataset>.json with train_normal, train_abnormal, pi_normal,
pi_abnormal (paths relative to DATA_DIR), and splits/manifest.json with counts
and md5 of each list.

    python make_splits.py            # write splits
    python make_splits.py --check    # leakage check only (also run by tests)
"""
import argparse
import hashlib
import json
import math

import numpy as np

from common import DATA_DIR, DATASETS, SPLITS_DIR, TARGETS, list_test_set, load_splits, sources_for, write_json

MAX_PER_CLASS = 500
PI_FRAC = 0.10
SEED = 0


def make(dataset, rng):
    items = list_test_set(dataset)
    out = {}
    for y, name in ((0, "normal"), (1, "abnormal")):
        files = sorted(p for p, lab in items if lab == y)
        pick = sorted(rng.choice(len(files), size=min(MAX_PER_CLASS, len(files)), replace=False).tolist())
        chosen = [files[i] for i in pick]
        n_pi = math.ceil(PI_FRAC * len(chosen))
        perm = rng.permutation(len(chosen)).tolist()
        out[f"pi_{name}"] = sorted(chosen[i] for i in perm[:n_pi])
        out[f"train_{name}"] = sorted(chosen[i] for i in perm[n_pi:])
    return out


def file_md5(rel):
    h = hashlib.md5()
    with open(DATA_DIR / rel, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def leakage_check(content_hash=True):
    """For each target: no file used for training or pi (i.e. from any source
    dataset's lists) may be a target test file, by path or (optionally) by
    content. Also checks train and pi lists are disjoint. Raises on failure."""
    problems = []
    splits = {d: load_splits(d) for d in DATASETS}
    for d, s in splits.items():
        tr = set(s["train_normal"]) | set(s["train_abnormal"])
        pi = set(s["pi_normal"]) | set(s["pi_abnormal"])
        if tr & pi:
            problems.append(f"{d}: {len(tr & pi)} files in both train and pi lists")
    hashes = {}
    for t in TARGETS:
        test = {p for p, _ in list_test_set(t)}
        used = set()
        for d in sources_for(t):
            s = splits[d]
            used |= set(s["train_normal"] + s["train_abnormal"] + s["pi_normal"] + s["pi_abnormal"])
        if used & test:
            problems.append(f"target {t}: {len(used & test)} target test files in source lists")
        if any(p.startswith(f"{t}_AD/") for p in used):
            problems.append(f"target {t}: a source list points into {t}_AD/")
        if content_hash:
            for p in used | test:
                if p not in hashes:
                    hashes[p] = file_md5(p)
            th = {hashes[p] for p in test}
            dup = [p for p in used if hashes[p] in th]
            if dup:
                problems.append(f"target {t}: {len(dup)} source files are byte-identical to target test "
                                f"files, e.g. {dup[:3]}")
    if problems:
        raise AssertionError("LEAKAGE:\n  " + "\n  ".join(problems))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no_hash", action="store_true")
    a = ap.parse_args()
    if not a.check:
        rng = np.random.default_rng(SEED)
        manifest = {}
        for d in DATASETS:          # fixed order -> reproducible
            s = make(d, rng)
            write_json(SPLITS_DIR / f"{d}.json", s)
            manifest[d] = {k: len(v) for k, v in s.items()}
            manifest[d]["md5"] = hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()
        write_json(SPLITS_DIR / "manifest.json", manifest)
        print(json.dumps(manifest, indent=1))
    leakage_check(content_hash=not a.no_hash)
    print("leakage check passed")


if __name__ == "__main__":
    main()
