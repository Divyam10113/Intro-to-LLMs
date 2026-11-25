"""Round 3 GPU checks (plan §4 steps 3-4), run by go.sh after the GPU jobs.

    python gpu_checks.py smoke    # verify ./run_all.sh smoke: every arm ran, finite losses, outputs written
    python gpu_checks.py timing   # steady s/step per arm from ./run_all.sh timing -> projection -> cuts

Both append to report/gpu_checks.md. `timing` writes logs/cuts.txt when the projection
exceeds the plan's cap (4x round 2's compute) and applies the plan's cut order:
(a) MIL+25 %, then (b) the two extra Brain seeds. Tier 1 blocks 1-3, MIL+100 % and
MIL+10 % are never cut. Reads no target test set.
"""
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common2 import HDD3, R2 as R3, sources, train_lists
from train2 import build_epoch, mask_budget_set

ARMS = ["mvfa", "mil", "pu_m100", "mil_m100", "mil_m10", "mil_m25"]
EPOCHS = 20
CAP_RUN_EPOCHS = 1400                          # plan §4 step 4 / §7: 4x round 2 = 1,400 round-2-equivalent run-epochs
PLANNED_RUN_EPOCHS = 1160                      # plan §7: 58 runs x 20 epochs (3.4x round 2's 345)
SLOTS = 12
REPORT = R3 / "report" / "gpu_checks.md"
LOGS = R3 / "logs"
ISSUES = R3 / "report" / "ISSUES_LOG.md"


def say(lines):
    REPORT.parent.mkdir(exist_ok=True)
    new = not REPORT.exists()
    with open(REPORT, "a") as f:
        if new:
            f.write("# Round 3 GPU checks (written by gpu_checks.py)\n\n")
        f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- smoke
def smoke():
    ok, lines = True, [f"## GPU smoke test ({time.strftime('%F %T')})", "",
                       "Each round 3 arm ran 50 steps on the Brain fold (seed 3, 1 epoch), with validation, the "
                       "diagnostic and a 16-image target evaluation. Smoke runs are never written to results.csv.", "",
                       "| arm | exit | steps | loss first → last | finite | val AUROC | eval rows | s/step |",
                       "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        d = HDD3 / "smoke_runs" / f"{a}_Brain_s3"
        try:
            meta = json.load(open(d / "run.json"))
            loss = [float(r["loss"]) for r in csv.DictReader(open(d / "loss.csv"))]
            fin = all(math.isfinite(x) for x in loss)
            good = fin and len(loss) == 50 and len(meta["results"]) == 1 and (d / "adapters.pth").exists()
            lines.append(f"| {a} | 0 | {len(loss)} | {loss[0]:.3f} → {loss[-1]:.3f} | {fin} | {meta['val_curve'][0]:.3f} "
                         f"| {len(meta['results'])} | {meta['seconds_per_update']:.3f} |")
        except Exception as e:                    # noqa: BLE001
            good = False
            lines.append(f"| {a} | **missing/failed** ({type(e).__name__}: {e}) | | | | | | |")
        ok &= good
    log = LOGS / "smoke_console.log"
    if log.exists() and "FAILED" in log.read_text():
        ok = False
        lines.append("- smoke_console.log contains FAILED")
    lines.append(f"\n**Smoke test: {'PASS' if ok else 'FAIL'}**")
    say(lines)
    return ok


# ----------------------------------------------------------------------------- timing
STEP = re.compile(r"\] ep 0 step (\d+)/\d+ .*\(([\d.]+)s/step\)")


def steady(log):
    """Steady-state s/step between steps 50 and 200 from the cumulative averages the log prints."""
    t = {int(m.group(1)): float(m.group(2)) for m in STEP.finditer(Path(log).read_text())}
    if 50 not in t or 200 not in t:
        raise ValueError(f"{log}: steps 50/200 not logged")
    return (t[200] * 201 - t[50] * 51) / 150


def steps_per_epoch(arm, fold, seed=3):
    lists = {d: train_lists(d) for d in sources(fold)}
    return len(build_epoch(arm, fold, seed, 0, lists, mask_budget_set(arm, fold, seed, lists)))


def r2_overheads():
    """From round 2's records: per-epoch validation minutes per fold and the fixed per-run minutes
    (load + diagnostic + evaluation), and round 2's total job-minutes (the cap's reference)."""
    rows = {r["run_id"]: r for r in csv.DictReader(open(R3.parent / "task4r2" / "results" / "results.csv"))}
    ep_s = {}                                    # "[run_id] epoch k done in Xs" = training time without validation
    for log in (R3.parent / "task4r2" / "logs").glob("train2.py_*.log"):
        for rid, k, x in re.findall(r"\[(\S+)\] epoch (\d+) done in (\d+)s", log.read_text()):
            ep_s.setdefault(rid, {})[int(k)] = float(x)
    val, fixed, total = {}, {}, 0.0
    for rid, r in rows.items():
        total += float(r["total_minutes"])
        if len(ep_s.get(rid, {})) == int(r["epochs"]):
            tr = sum(ep_s[rid].values()) / 60
            val.setdefault(r["fold"], []).append((float(r["train_minutes"]) - tr) / int(r["epochs"]))
        fixed.setdefault(r["fold"], []).append(float(r["total_minutes"]) - float(r["train_minutes"]))
    r2_sps = {}                                  # pure training s/step, Brain fold, seeds 0-2 (arms run in both rounds)
    for a in ("mvfa", "mil", "pu_m100"):
        v = [sum(ep_s[f"{a}_Brain_s{s}"].values()) / (len(ep_s[f"{a}_Brain_s{s}"]) * steps_per_epoch(a, "Brain", s))
             for s in (0, 1, 2) if f"{a}_Brain_s{s}" in ep_s]
        r2_sps[a] = float(np.mean(v))
    return ({f: float(np.median(v)) for f, v in val.items()}, {f: float(np.median(v)) for f, v in fixed.items()},
            total, len(rows), r2_sps)


def jobs():
    import os
    env = dict(os.environ, CUTS_FILE=str(LOGS / "cuts.txt"))
    out = subprocess.run(["bash", str(R3 / "run_all.sh"), "list", "r3"], capture_output=True, text=True,
                         check=True, env=env).stdout.split("\n")
    return [dict(re.findall(r"--(\w+) (\S+)", j)) for j in out if j.strip()]


def project(sps, val, fixed, cuts):
    (LOGS / "cuts.txt").write_text("".join(c + "\n" for c in cuts))
    js = jobs()
    mins = [EPOCHS * (steps_per_epoch(j["arm"], j["fold"]) * sps[j["arm"]] / 60 + val[j["fold"]]) + fixed[j["fold"]]
            for j in js]
    free = [0.0] * SLOTS                         # the dynamic queue: each job goes to the first free slot
    for m in mins:
        i = int(np.argmin(free))
        free[i] += m
    return js, mins, max(free)


def timing():
    sps, lines = {}, [f"## Timing check ({time.strftime('%F %T')})", "",
                      "First 201 steps of each arm, 2 copies each, 12 concurrent jobs on 2 GPUs (round 2's G2-5 "
                      "method). Steady s/step = the average over steps 50–200.", "", "| arm | s/step (2 runs) | mean |",
                      "|---|---|---|"]
    for a in ARMS:
        v = [steady(p) for p in sorted(LOGS.glob(f"train2.py_--arm_{a}_--fold_Brain_*timing*.log"))
             if not p.name.endswith(".retry.log")]
        if not v:
            raise SystemExit(f"no timing log for {a}")
        sps[a] = float(np.mean(v))
        lines.append(f"| {a} | {', '.join(f'{x:.4f}' for x in v)} | {sps[a]:.4f} |")
    mem = LOGS / "gpu_mem_timing.csv"
    if mem.exists():
        m = [float(r.split(",")[2]) for r in mem.read_text().split("\n")[1:] if r.count(",") == 2]
        if m:
            lines.append(f"\nGPU memory per job (sampled during timing): max {max(m):.0f} MiB, median {np.median(m):.0f} MiB.")
    val, fixed, r2_total, r2_n, r2_sps = r2_overheads()
    slow = float(np.mean([sps[a] / r2_sps[a] for a in r2_sps]))
    lines += ["", "**Projection rule (fixed before any run, R3-4):** round-2-equivalent run-epochs = run-epochs × slowdown, "
              "where slowdown = mean over MVFA, MIL and PU+100 % of (measured s/step ÷ round 2's training s/step on the "
              f"same Brain fold). The plan's 58 runs are {PLANNED_RUN_EPOCHS} run-epochs at round 2's speed; the cap is "
              f"{CAP_RUN_EPOCHS} (4× round 2's 345, ≈ 6½ h).", "",
              "Slowdown: " + ", ".join(f"{a} {sps[a]:.4f} / {r2_sps[a]:.4f} = {sps[a] / r2_sps[a]:.2f}" for a in r2_sps) +
              f"; mean **{slow:.2f}**.",
              "Per-epoch validation and fixed per-run minutes (for the wall-clock estimate) are round 2's medians per fold: " +
              ", ".join(f"{f} val {val[f]:.2f} / fixed {fixed[f]:.2f}" for f in ("Brain", "Liver", "RESC_OCT")), ""]
    for cuts in ([], ["mil_m25"], ["mil_m25", "brain_extra"]):
        js, mins, wall = project(sps, val, fixed, cuts)
        eq = len(js) * EPOCHS * slow
        lines.append(f"- Cuts {cuts or 'none'}: {len(js)} runs = {len(js) * EPOCHS} run-epochs × {slow:.2f} = **{eq:.0f}** "
                     f"round-2-equivalent run-epochs ({eq / 345:.2f}× round 2); {sum(mins):.0f} job-minutes "
                     f"({sum(mins) / r2_total:.2f}× round 2's {r2_total:.0f}); wall-clock on {SLOTS} slots ≈ {wall / 60:.1f} h")
        if eq <= CAP_RUN_EPOCHS:
            break
    else:
        lines.append("- Still above the cap after both cuts: the remaining runs are never cut (plan §4 step 4); "
                     "running them anyway and logging the overrun.")
    lines.append(f"\n**Decision: {'no cut' if not cuts else 'cut ' + ' + '.join(cuts)}.** "
                 f"{len(js)} runs, ≈ {wall / 60:.1f} h wall-clock if both GPUs stay free.")
    say(lines)
    json.dump({"s_per_step": sps, "r2_s_per_step": r2_sps, "slowdown": slow, "cuts": cuts, "runs": len(js), "job_minutes": sum(mins), "wall_hours": wall / 60,
               "per_job_minutes": dict(zip([f"{j['arm']}_{j['fold']}_s{j['seed']}" for j in js], mins)),
               "when": time.strftime("%F %T")}, open(LOGS / "projection.json", "w"), indent=1)
    if cuts:
        with open(ISSUES, "a") as f:
            f.write(f"| R3-auto | **Compute cut applied by the timing check ({time.strftime('%F')}).** Projection above "
                    f"{CAP_RUN_EPOCHS} round-2-equivalent run-epochs. | `gpu_checks.py timing` (report/gpu_checks.md) | — (pre-set rule, plan §4 step 4) | "
                    f"Cut {', '.join(cuts)} in the plan's order; {len(js)} runs remain. |\n")
    return True


if __name__ == "__main__":
    ok = {"smoke": smoke, "timing": timing}[sys.argv[1]]()
    sys.exit(0 if ok else 1)
