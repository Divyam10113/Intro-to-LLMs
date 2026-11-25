"""Summarise `./run_all.sh bench` and choose jobs per GPU.

    python bench_summary.py row <k> <smi.csv>        -> one TSV row (appended by run_all.sh)
    python bench_summary.py choose <bench.tsv> <out> -> table + writes chosen k to <out>

Running k jobs on one GPU makes each job slower; it only helps if one job
leaves the GPU partly idle. The choice is the smallest k whose TOTAL
throughput (sum of updates/s over the k jobs) is within 5% of the best, so
k = 1 whenever a single job already saturates the card.
"""
import json
import sys

import numpy as np

from common import RUNS_DIR


def row(k, smi):
    spu = [json.load(open(RUNS_DIR / f"mh_Liver_s{j}_bench{k}_{j}" / "run.json"))["seconds_per_update"]
           for j in range(1, k + 1)]
    rows = np.array([[float(x) for x in l.split(",")] for l in open(smi) if l.strip()])
    busy = rows[rows[:, 1] > rows[:, 1].min() + 500] if len(rows) else rows  # samples with models loaded
    util = float(np.median(busy[:, 0])) if len(busy) else float("nan")
    vram = float(rows[:, 1].max()) if len(rows) else float("nan")
    print(f"{k}\t{sum(1 / s for s in spu):.2f}\t{np.mean(spu):.4f}\t{util:.0f}\t{vram:.0f}")


def choose(tsv, out):
    rows = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
    best = max(float(r[1]) for r in rows)
    print("k   total updates/s   s/update per job   GPU util %   VRAM MiB")
    for r in rows:
        print(f"{r[0]:<4}{float(r[1]):>12.2f}{float(r[2]):>18.4f}{r[3]:>13}{r[4]:>11}")
    k = min(int(r[0]) for r in rows if float(r[1]) >= 0.95 * best)
    open(out, "w").write(str(k))
    print(f"-> JOBS_PER_GPU={k} saved to {out} (smallest k within 5% of the best total throughput)")


if __name__ == "__main__":
    if sys.argv[1] == "row":
        row(int(sys.argv[2]), sys.argv[3])
    else:
        choose(sys.argv[2], sys.argv[3])
