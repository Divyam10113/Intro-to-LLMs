"""Source-validation lists (G1-5): each dataset's BMAD valid/good + valid/Ungood,
minus every valid image that is byte-identical (MD5) to ANY test/ image of any
dataset (the sources train on test/, round 1 B1). Writes splits/valid.json and
splits/valid_excluded.json.

    python make_valid.py
"""
import hashlib
import os
from collections import defaultdict

from common2 import DATA_DIR, DATASETS, SPLITS2, write_json


def listing(ds, split):
    out = []
    for sub, y in (("good", 0), ("Ungood", 1)):
        d = DATA_DIR / f"{ds}_AD" / split / sub / "img"
        if d.is_dir():
            out += [(f"{ds}_AD/{split}/{sub}/img/{f}", y) for f in sorted(os.listdir(d))]
    return out


def md5(rel):
    return hashlib.md5(open(DATA_DIR / rel, "rb").read()).hexdigest()


def main():
    test_hash = defaultdict(list)
    for ds in DATASETS:
        for p, _ in listing(ds, "test"):
            test_hash[md5(p)].append(p)
    valid, excluded = {}, {}
    for ds in DATASETS:
        keep = {"normal": [], "abnormal": []}
        for p, y in listing(ds, "valid"):
            h = md5(p)
            if h in test_hash:
                excluded[p] = test_hash[h]
            else:
                keep["normal" if y == 0 else "abnormal"].append(p)
        valid[ds] = keep
    write_json(SPLITS2 / "valid.json", valid)
    write_json(SPLITS2 / "valid_excluded.json", excluded)
    for ds, k in valid.items():
        print(ds, len(k["normal"]), len(k["abnormal"]))
    print("excluded", len(excluded))


if __name__ == "__main__":
    main()
