#!/usr/bin/env python3
"""Build paper/inputs/results.json from one or more Task 4 rounds (the only source of numbers for the paper).

    python build_results.py --round r1=../task4_r1 --pair r1=mh:mh_ctrl --retention r1=m:b0 --out results.json
    # later, when round 2 arrives (same folder layout):
    python build_results.py --round r1=../task4_r1 --round r2=../task4_r2 \
        --pair r1=mh:mh_ctrl --pair r2=g:mil --retention r1=m:b0 --out results.json

Each round folder must contain results/results.csv (columns: run_id, config, target, seed, steps, lambda,
pi_json, img_auroc, pix_auroc, aupro, gpu_minutes, notes). Optional: results/paper_numbers.csv,
results/pi.json, results/pi_maskfree.json.

Conventions
- All metrics are written in percent, rounded to 2 decimals. std is the sample std (ddof = 1); null if n = 1.
- A run is a MAIN run if run_id == "<config>_<target>_s<seed>" (or "R0_<target>"); every other run
  (tags such as pi0.30, r0det, 2S) goes to "secondary" and never enters a mean.
- Rounds are NEVER pooled: round 1 and round 2 use different protocols. Keep them apart in the paper.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = ("img_auroc", "pix_auroc", "aupro")


def pct(x):
    """Percent, UNROUNDED: round only when writing, so means match the report's rounding."""
    return 100.0 * float(x)


def stats(values):
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": round(mean, 2), "std": None if std is None else round(std, 2),
            "values": [round(v, 2) for v in values]}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def is_main(r):
    if r["config"] == "R0":
        return r["run_id"] == f"R0_{r['target']}"
    return r["run_id"] == f"{r['config']}_{r['target']}_s{r['seed']}"


def build_round(folder, pairs, retention):
    folder = Path(folder)
    rows = read_csv(folder / "results" / "results.csv")
    main, secondary, r0 = defaultdict(lambda: defaultdict(list)), [], {}
    for r in rows:
        if r["config"] == "R0":
            r0[r["target"]] = {m: round(pct(r[m]), 2) for m in METRICS}
            continue
        if is_main(r):
            main[r["config"]][r["target"]].append(r)
        else:
            secondary.append({"run_id": r["run_id"], "config": r["config"], "target": r["target"],
                              "seed": int(r["seed"]), "steps": int(r["steps"]),
                              **{m: round(pct(r[m]), 2) for m in METRICS}, "notes": r["notes"]})
    out = {"r0_reproduced": r0, "configs": {}, "secondary": secondary}
    pub = Path(folder / "results" / "paper_numbers.csv")
    if pub.exists():
        out["r0_published"] = {r["target"]: {"img_auroc": float(r["img_auroc"]), "pix_auroc": float(r["pix_auroc"])}
                               for r in read_csv(pub)}
    for cfg, per_target in main.items():
        out["configs"][cfg] = {}
        for t, rs in per_target.items():
            rs = sorted(rs, key=lambda r: int(r["seed"]))
            out["configs"][cfg][t] = {
                "seeds": [int(r["seed"]) for r in rs],
                "steps": sorted({int(r["steps"]) for r in rs}),
                **{m: stats([pct(r[m]) for r in rs]) for m in METRICS},
            }
    # paired differences: A - B, matched by (target, seed)
    out["paired"] = {}
    for a, b in pairs:
        key, rows_a, rows_b = f"{a}-{b}", main[a], main[b]
        res = {"per_pair": [], "per_target_mean": {}}
        for t in rows_a:
            ra = {int(r["seed"]): r for r in rows_a[t]}
            rb = {int(r["seed"]): r for r in rows_b.get(t, [])}
            for s in sorted(set(ra) & set(rb)):
                res["per_pair"].append({"target": t, "seed": s,
                                        **{m: pct(ra[s][m]) - pct(rb[s][m]) for m in METRICS}})
            ds = [p for p in res["per_pair"] if p["target"] == t]
            if ds:
                res["per_target_mean"][t] = {m: round(sum(p[m] for p in ds) / len(ds), 2) for m in METRICS}
        n = len(res["per_pair"])
        res["n_pairs"] = n
        res["positive"] = {m: sum(p[m] > 0 for p in res["per_pair"]) for m in METRICS}
        for p in res["per_pair"]:
            for m in METRICS:
                p[m] = round(p[m], 2)
        # one-sided sign test for AUPRO: P(X >= k), X ~ Bin(n, 0.5)
        k = res["positive"]["aupro"]
        res["sign_test_p_aupro"] = round(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n, 4) if n else None
        out["paired"][key] = res
    # retention vs the reproduced R0: (pix - 50) / (R0 pix - 50); AUPRO ratio
    out["retention"] = {}
    for cfg in retention:
        out["retention"][cfg] = {}
        for t, d in out["configs"].get(cfg, {}).items():
            if t not in r0:
                continue
            base = r0[t]["pix_auroc"] - 50.0
            seed0 = d["seeds"].index(0) if 0 in d["seeds"] else None
            out["retention"][cfg][t] = {
                "mean_over_seeds": round((d["pix_auroc"]["mean"] - 50.0) / base, 3),
                "seed0": None if seed0 is None else round((d["pix_auroc"]["values"][seed0] - 50.0) / base, 3),
                "aupro_ratio_mean": round(d["aupro"]["mean"] / r0[t]["aupro"], 3),
                "aupro_ratio_seed0": None if seed0 is None else round(d["aupro"]["values"][seed0] / r0[t]["aupro"], 3),
            }
        vals = [v["seed0"] for v in out["retention"][cfg].values() if v["seed0"] is not None]
        out["retention"][cfg]["_mean_seed0_over_targets"] = round(sum(vals) / len(vals), 3) if vals else None
    # Preserve every supplied pi*.json file, including the estimator's raw
    # segmentation/detection outputs and validation diagnostics. Those are
    # required to report why the estimator failed, while remaining separate
    # from the priors actually used for training.
    for path in sorted((folder / "results").glob("pi*.json")):
        data = read_json(path)
        if data is not None:
            out.setdefault("pi", {})[path.name] = data
    return out


# ----------------------------------------------------------------------------------------------
# Rounds 2, 3 and the follow-up (train2.py logger: one row per (run, evaluated dataset)).
# Added 2026-11-25 for the paper's final update. Conventions follow analyze2.py / analyze3.py:
#   * a MAIN run has run_id == f"{arm}_{fold}_s{seed}" (no tag); tagged runs go to "secondary";
#   * means/stds over the round's main seeds (--main-seeds); extra seeds (e.g. Brain 6-7 in round 3)
#     are kept in "configs_all_seeds" and enter only the paired tests that use them;
#   * retention = (mean pix(arm) - 50) / (mean pix(baseline) - 50), per target, over the main seeds;
#   * sign tests are one-sided P(X >= k), X ~ Bin(n, 0.5); Holm within the tiers given by --holm.
MASKED_TARGETS = ("Brain", "Liver", "Retina_RESC")


def sign_p(k, n):
    return round(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n, 4) if n else None


def holm(ps):
    """Holm-adjusted p-values, returned in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = round(running, 4)
    return adj


def build_round_runs(folder, pairs, retention, main_seeds, baseline, r0_ref, holm_tiers, be_exclude=()):
    folder = Path(folder)
    rows = read_csv(folder / "results" / "results.csv")
    main, secondary, runs = defaultdict(lambda: defaultdict(dict)), [], {}
    for r in rows:
        v = {m: pct(r[m]) for m in METRICS if r.get(m) not in ("", None)}
        if r["run_id"] == f"{r['arm']}_{r['fold']}_s{r['seed']}":
            main[r["arm"]][r["eval_dataset"]][int(r["seed"])] = {**v, "best_epoch": int(r["best_epoch"])}
        else:
            secondary.append({"run_id": r["run_id"], "arm": r["arm"], "target": r["eval_dataset"],
                              "seed": int(r["seed"]), "epochs": int(r["epochs"]), "pi": r.get("pi") or None,
                              **{m: round(x, 2) for m, x in v.items()}})
        if r["eval_dataset"] in MASKED_TARGETS:
            runs[r["run_id"]] = r
    ms = set(main_seeds)

    def stats_for(arm, t, seeds):
        cell = main[arm][t]
        ss = sorted(s for s in cell if s in seeds)
        o = {"seeds": ss, "best_epoch": [cell[s]["best_epoch"] for s in ss]}
        for m in METRICS:
            vals = [cell[s][m] for s in ss if m in cell[s]]
            if vals:
                o[m] = stats(vals)
        return o

    out = {"format": "train2 logger", "main_seeds": sorted(ms), "baseline": baseline,
           "epochs": sorted({int(r["epochs"]) for r in rows}), "configs": {}, "configs_all_seeds": {},
           "secondary": secondary}
    for arm, per_t in main.items():
        out["configs"][arm] = {t: stats_for(arm, t, ms) for t in per_t if set(per_t[t]) & ms}
        extra = {t: stats_for(arm, t, set(per_t[t])) for t in per_t if set(per_t[t]) - ms}
        if extra:
            out["configs_all_seeds"][arm] = extra
    out["paired"] = {}
    for a, b in pairs:
        res = {"per_pair": [], "per_target": {}, "main_seeds": {}}
        for t in MASKED_TARGETS:
            ca, cb = main[a].get(t, {}), main[b].get(t, {})
            common = sorted(set(ca) & set(cb))
            if not common:
                continue
            ds = [{"target": t, "seed": s, **{m: ca[s][m] - cb[s][m] for m in METRICS if m in ca[s] and m in cb[s]}}
                  for s in common]
            res["per_pair"] += ds
            pos = sum(d["aupro"] > 0 for d in ds)
            res["per_target"][t] = {"seeds": common, "n": len(ds), "aupro_positive": pos,
                                    "sign_test_p_aupro": sign_p(pos, len(ds)),
                                    **{f"mean_delta_{m}": round(sum(d[m] for d in ds) / len(ds), 2) for m in METRICS}}
        sel = [d for d in res["per_pair"] if d["seed"] in ms]
        if sel:
            pos = {m: sum(d[m] > 0 for d in sel) for m in METRICS}
            by_t = {}
            for t in MASKED_TARGETS:
                dt = [d for d in sel if d["target"] == t]
                if dt:
                    by_t[t] = {m: round(sum(d[m] for d in dt) / len(dt), 2) for m in METRICS}
            res["main_seeds"] = {"n_pairs": len(sel), "positive": pos, "sign_test_p_aupro": sign_p(pos["aupro"], len(sel)),
                                 "mean_delta_by_target": by_t,
                                 "pix_higher_targets": sum(v["pix_auroc"] > 0 for v in by_t.values())}
        for d in res["per_pair"]:
            for m in METRICS:
                if m in d:
                    d[m] = round(d[m], 2)
        out["paired"][f"{a}-{b}"] = res
    out["retention"], out["retention_vs_r0"] = {}, {}
    base = out["configs"].get(baseline, {})

    def raw_pix(arm, t):          # unrounded mean over the main seeds, as analyze2/3.py compute it
        v = [main[arm][t][s]["pix_auroc"] for s in main[arm][t] if s in ms]
        return sum(v) / len(v)
    for arm in retention:
        cfg = out["configs"].get(arm, {})
        rv, r0v = {}, {}
        for t in MASKED_TARGETS:
            if t in cfg and t in base:
                rv[t] = round((raw_pix(arm, t) - 50) / (raw_pix(baseline, t) - 50), 3)
                if r0_ref and t in r0_ref:
                    r0v[t] = round((raw_pix(arm, t) - 50) / (r0_ref[t]["pix_auroc"] - 50), 3)
        if rv:
            vals = list(rv.values())
            rv["_mean"], rv["_min"] = round(sum(vals) / len(vals), 3), round(min(vals), 3)
            out["retention"][arm] = rv
        if r0v:
            r0v["_mean"] = round(sum(r0v.values()) / len(r0v), 3)
            out["retention_vs_r0"][arm] = r0v
    out["inmask"] = {}
    per = defaultdict(lambda: defaultdict(list))
    for rid, r in runs.items():
        if rid == f"{r['arm']}_{r['fold']}_s{r['seed']}" and int(r["seed"]) in ms and r.get("diag_inmask_p"):
            per[r["arm"]][r["fold"]].append(float(r["diag_inmask_p"]))
    for arm, folds in per.items():
        pf = {f: round(sum(v) / len(v), 3) for f, v in folds.items()}
        out["inmask"][arm] = {"per_fold": pf, "mean_of_folds": round(sum(pf.values()) / len(pf), 3)}
    be = [(int(r["best_epoch"]), int(r["epochs"])) for r in runs.values()]
    out["best_epoch"] = {"n_runs": len(be), "last": sum(b == e - 1 for b, e in be),
                         "last_two": sum(b >= e - 2 for b, e in be),
                         "histogram": {str(k): sum(b == k for b, _ in be) for k in sorted({b for b, _ in be})}}
    if be_exclude:                # e.g. round 2's report counts its 57 main masked-target runs without the budget arms
        be2 = [(int(r["best_epoch"]), int(r["epochs"])) for r in runs.values() if r["arm"] not in be_exclude]
        out["best_epoch"]["excluding"] = {"arms": list(be_exclude), "n_runs": len(be2),
                                          "last": sum(b == e - 1 for b, e in be2),
                                          "last_two": sum(b >= e - 2 for b, e in be2)}
    vals = [float(r["val_img_auroc"]) for r in runs.values() if r.get("val_img_auroc")]
    out["source_val_img_auroc"] = ({"mean": round(100 * sum(vals) / len(vals), 2), "max": round(100 * max(vals), 2)}
                                   if vals else None)
    out["n_runs"] = len({r["run_id"] for r in rows})
    out["n_runs_masked_targets"] = len(runs)
    out["tests"] = {}
    for tier, items in holm_tiers.items():
        ps, labels = [], []
        for label, pair, target in items:
            pr = out["paired"].get(pair, {})
            p = pr["per_target"][target]["sign_test_p_aupro"] if target else pr.get("main_seeds", {}).get("sign_test_p_aupro")
            ps.append(p)
            labels.append(label)
        for label, p, adj in zip(labels, ps, holm(ps)):
            out["tests"][label] = {"tier": tier, "raw_p": p, "holm_p": adj}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--pair", action="append", default=[], metavar="NAME=A:B", help="paired difference A - B")
    ap.add_argument("--retention", action="append", default=[], metavar="NAME=CFG[:CFG]")
    ap.add_argument("--main-seeds", action="append", default=[], metavar="NAME=S,S,S")
    ap.add_argument("--baseline", action="append", default=[], metavar="NAME=ARM")
    ap.add_argument("--holm", action="append", default=[], metavar="NAME=TIER:LABEL:A-B[@TARGET],...")
    ap.add_argument("--r0-from", default="r1")
    ap.add_argument("--best-epoch-exclude", action="append", default=[], metavar="NAME=ARM,ARM")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs, ret = defaultdict(list), defaultdict(list)
    for p in a.pair:
        name, ab = p.split("=")
        pairs[name].append(tuple(ab.split(":")))
    for p in a.retention:
        name, cfgs = p.split("=")
        ret[name] += cfgs.split(":")
    result = {
        "units": "percent (x100), 2 decimals; std = sample std (ddof=1)",
        "protocol": {
            "benchmark": "BMAD",
            "dataset_count": 6,
            "masked_source_count": 3,
            "maskless_source_count": 3,
            "adapter_levels": 4,
            "targets": ["Brain", "Liver", "Retina_RESC"],
            "source_datasets": ["Brain", "Liver", "Retina_RESC", "Retina_OCT2017", "Chest", "Histopathology"],
            "maskless_sources": ["Retina_OCT2017", "Chest", "Histopathology"],
            "max_train_images_per_class_per_source": 500,
            "pi_holdout_fraction": 0.10,
            "frozen_backbone": "CLIP ViT-L/14",
            "backbone_layer_count": 14,
            "aUpro_false_positive_limit": 0.30,
            "steps_per_run": 2000,
            "mil_topk_fraction": 0.01,
            "mil_topk_min_patches": 3,
            "pixel_auroc_chance_floor": 50.0,
            "source_plan": "paper/inputs/plans/TASK4_PLAN_RECOMMENDED.md",
            "source_decisions": "paper/inputs/task4_r1/report/DECISIONS.md"
        },
        "rounds": {}
    }
    seeds = {k: [int(x) for x in v.split(",")] for k, v in (y.split("=") for y in a.main_seeds)}
    bases = dict(x.split("=") for x in a.baseline)
    be_ex = dict(x.split("=") for x in a.best_epoch_exclude)
    tiers = defaultdict(lambda: defaultdict(list))
    for x in a.holm:
        name, spec = x.split("=")
        for item in spec.split(","):
            tier, label, pt = item.split(":")
            pair, _, target = pt.partition("@")
            tiers[name][tier].append((label, pair, target or None))
    # Rounds 2, 3 and the follow-up share Task 3's protocol (from scratch; five folds with RESC and OCT17
    # held out together; source-validation checkpoints). Constants as logged in task4r2/report/ISSUES_LOG.md.
    result["protocol_runs"] = {
        "folds": 5, "train_images_per_class_per_source": 500, "oct17_normal_train_images": 242,
        "max_train_images_per_source": 1000, "validation_images_per_class_cap": 250,
        "valid_test_duplicates_removed": 11, "valid_test_duplicates_brain_abnormal": 10,
        "patches_per_image": 289, "patch_grid": 17, "image_size": 240, "topk_patches": 3,
        "pu_pi_main": 0.01, "pu_pi_sensitivity": 0.05, "normal_bank_size": 20000, "gate_layer": 24,
        "adam_lr": 1e-4, "batch_size": 1, "mask_budgets_percent": [0, 10, 25, 100]}
    rounds = [r.split("=") for r in a.round]
    for name, folder in rounds:          # legacy (round 1) format first, so R0 exists for later rounds
        if "arm" not in read_csv(Path(folder) / "results" / "results.csv")[0]:
            result["rounds"][name] = build_round(folder, pairs[name], ret[name])
    r0_ref = result["rounds"].get(a.r0_from, {}).get("r0_reproduced")
    for name, folder in rounds:
        if name not in result["rounds"]:
            result["rounds"][name] = build_round_runs(folder, pairs[name], ret[name], seeds.get(name, [0, 1, 2]),
                                                      bases.get(name, "mvfa"), r0_ref, tiers.get(name, {}),
                                                      tuple(be_ex.get(name, "").split(",")) if be_ex.get(name) else ())
    result["rounds"] = {name: result["rounds"][name] for name, _ in rounds}
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}: rounds {list(result['rounds'])}")


if __name__ == "__main__":
    main()
