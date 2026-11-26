"""Round 3 pre-run checks (plan §3 "New tests" and §4 step 2). CPU only."""
import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

import common2
from common2 import FOLDS, MASKED_FOLDS, sources, train_lists
import train2
from train2 import ARMS, MASK_BUDGET, WEAK, build_epoch, mask_budget_set

R3 = Path(common2.__file__).parent
NEW = ["mil_m100", "mil_m10", "mil_m25"]


def _lists(fold):
    return {d: train_lists(d) for d in sources(fold)}


# ---------------------------------------------------------------- the code change (plan §3)
def test_new_arms_registered_exactly_as_plan():
    assert all(a in ARMS for a in NEW)
    assert {a: MASK_BUDGET[a] for a in NEW} == {"mil_m100": 1.0, "mil_m10": 0.10, "mil_m25": 0.25}
    assert {a: WEAK[a] for a in NEW} == {a: "mil" for a in NEW}
    # round 2's arms are unchanged
    assert ARMS[:8] == ["mvfa", "mil", "g", "pu", "pu_m100", "g_m100", "g_m10", "g_m25"]


@pytest.mark.parametrize("fold", list(FOLDS))
def test_mil_m100_masks_only_on_masked_sources(fold):
    items = build_epoch("mil_m100", fold, 3, 0, _lists(fold), set())
    for k, d, *_ in items:
        assert k == ("mask" if common2.has_masks(d) else "weak"), (k, d)


@pytest.mark.parametrize("fold", MASKED_FOLDS)
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_budget_arms_mask_only_their_seeded_set_and_25_contains_10(fold, seed):
    L = _lists(fold)
    s10, s25 = mask_budget_set("mil_m10", fold, seed, L), mask_budget_set("mil_m25", fold, seed, L)
    assert s10 < s25
    for b, s in ((0.10, s10), (0.25, s25)):
        for d in L:
            if common2.has_masks(d):
                mine = {p for p in s if p in set(L[d][1])}
                assert len(mine) == int(round(b * len(L[d][1])))
        assert all("/Ungood/" in p for p in s)                      # abnormal images only
    for arm, s in (("mil_m10", s10), ("mil_m25", s25)):
        items = build_epoch(arm, fold, seed, 0, L, s)
        assert {p for k, d, p, y in items if k == "mask"} == s
        assert {k for k, d, p, y in items if p not in s} == {"weak"}
    assert s10 == mask_budget_set("mil_m10", fold, seed, L)          # fixed for the seed
    assert s10 != mask_budget_set("mil_m10", fold, seed + 1, L)


def test_budget_set_is_independent_of_the_arm_family():
    L = _lists("Brain")
    assert mask_budget_set("mil_m10", "Brain", 3, L) == mask_budget_set("g_m10", "Brain", 3, L)


@pytest.mark.parametrize("fold", MASKED_FOLDS)
def test_mil_mvfa_and_new_arms_see_same_images_in_same_order(fold):
    L = _lists(fold)
    ref = [(d, p, y) for _, d, p, y in build_epoch("mvfa", fold, 4, 7, L, set())]
    for arm in ["mil"] + NEW:
        got = [(d, p, y) for _, d, p, y in build_epoch(arm, fold, 4, 7, L, mask_budget_set(arm, fold, 4, L))]
        assert got == ref, arm


def test_new_arms_get_no_gate_and_plain_mil():
    # train2.run passes gates to the loader only for weak == "g"; the MIL branch is used for weak == "mil"
    src = (R3 / "train2.py").read_text()
    assert 'gates if weak == "g" else {}' in src and 'if weak == "mil":' in src


# ---------------------------------------------------------------- outputs and queue
def test_outputs_point_at_round3_and_banks_at_round2():
    if "TASK4_RUNS" not in os.environ:
        assert common2.RUNS2 == Path("/mnt/HDD1/divyam/task4/r3/runs")
    if "TASK4R2_HDD" not in os.environ:
        assert common2.BANKS == Path("/mnt/HDD1/divyam/task4/r2/banks")
    assert common2.RESULTS2.parent == R3


def _jobs():
    r = subprocess.run(["bash", str(R3 / "run_all.sh"), "list", "r3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()


def test_r3_queue_is_the_plan_order():
    if (R3 / "logs" / "cuts.txt").exists():
        pytest.skip("cuts applied by the timing check")
    jobs = _jobs()
    assert len(jobs) == 58 and len(set(jobs)) == 58
    parse = [dict(re.findall(r"--(\w+) (\S+)", j)) for j in jobs]
    assert all(p["epochs"] == "20" for p in parse)
    arms = [p["arm"] for p in parse]
    assert arms[:27] == ["mvfa"] * 9 + ["mil"] * 9 + ["pu_m100"] * 9
    assert arms[27:31] == ["mvfa", "mil", "mvfa", "mil"] and all(p["fold"] == "Brain" for p in parse[27:31])
    assert [p["seed"] for p in parse[27:31]] == ["6", "6", "7", "7"]
    assert arms[31:] == ["mil_m100"] * 9 + ["mil_m10"] * 9 + ["mil_m25"] * 9
    for p in parse[:27] + parse[31:]:
        assert p["seed"] in "345" and p["fold"] in MASKED_FOLDS
    blk = [p["fold"] for p in parse[:9]]
    assert blk == ["Brain"] * 3 + ["Liver"] * 3 + ["RESC_OCT"] * 3          # longest fold first
    maps = [j for j in jobs if "--save_maps" in j]
    assert len(maps) == 6 and all("--fold Liver --seed 3" in j for j in maps)


# ---------------------------------------------------------------- rules frozen and correct
def test_rules_files_match_frozen_hashes():
    frozen = R3 / "report" / "FROZEN.sha256"
    if not frozen.exists():
        pytest.skip("rules not frozen yet")
    for line in frozen.read_text().split("\n"):
        if line.strip():
            h, name = line.split()
            assert hashlib.sha256((R3 / name).read_bytes()).hexdigest() == h, f"{name} changed after freezing"


def _r2_as_r3(tmp_path):
    """Round 2's results with seeds 0-2 relabelled 3-5: analyze3 must reproduce round 2's numbers."""
    rows = list(csv.DictReader(open(R3.parent / "task4r2" / "results" / "results.csv")))
    out = tmp_path / "r.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            s = int(r["seed"])
            r = dict(r, seed=str(s + 3), run_id=r["run_id"].replace(f"_s{s}", f"_s{s + 3}"))
            w.writerow(r)
    return out


def test_analyze3_reproduces_round2_numbers(tmp_path):
    import analyze3
    out = analyze3.main(_r2_as_r3(tmp_path), tmp_path)
    md = (tmp_path / "analysis.md").read_text()
    assert "AUPRO higher than MVFA in 7/9 pairs" in md                 # PU+100% in round 2
    assert out["H3"] == "suggestive again"
    ret, m, o1, _ = analyze3.h1(analyze3.load(_r2_as_r3(tmp_path)))
    assert abs(m - 0.930) < 0.0015 and o1.startswith("KEEPS MOST")    # round 2: MIL retention 0.930
    assert out["H2"].startswith("holds on most seeds") and "3/3" in out["H2"]   # 3 Brain seeds only
    assert out["H4"] == "not run" and out["H5"] == "not run"
    assert (tmp_path / "per_seed.md").exists()


def _synthetic(tmp_path, delta, n_pos=9, pix=0.5, img=0.0, brain_pos=5, arms=("pu_m100",)):
    """MVFA fixed; each arm in `arms` gets AUPRO +delta on the first n_pos pairs, -delta on the rest."""
    F = train2.FIELDS
    rows, k = [], 0

    def row(arm, t, s, a, pxa, im):
        fold = "RESC_OCT" if t.startswith("Retina") else t
        rows.append(dict({f: "" for f in F}, run_id=f"{arm}_{fold}_s{s}", arm=arm, fold=fold, eval_dataset=t,
                         seed=s, epochs=20, best_epoch=10, img_auroc=im, pix_auroc=pxa, aupro=a,
                         diag_inmask_p=0.5, diag_inmask_pg=0.5, notes=""))
    for t in analyze3_T():
        for s in (3, 4, 5):
            row("mvfa", t, s, 0.60, 0.90, 0.80)
            for arm in arms:
                sign = 1 if k < n_pos else -1
                row(arm, t, s, 0.60 + sign * delta, 0.90 + pix / 100, 0.80 + img / 100)
            k += 1
    for i, s in enumerate((3, 4, 5, 6, 7)):
        if s > 5:
            row("mvfa", "Brain", s, 0.60, 0.90, 0.80)
        row("mil", "Brain", s, 0.60 + (0.05 if i < brain_pos else -0.05), 0.91, 0.80)
    for t in ("Liver", "Retina_RESC"):
        for s in (3, 4, 5):
            row("mil", t, s, 0.55, 0.86, 0.80)
    p = tmp_path / "syn.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F)
        w.writeheader()
        w.writerows(rows)
    return p


def analyze3_T():
    import analyze3
    return analyze3.T


@pytest.mark.parametrize("n_pos,pix,img,expect", [
    (9, 0.5, 0.0, "CONFIRMED"), (8, 0.5, 0.0, "CONFIRMED"), (7, 0.5, 0.0, "suggestive again"),
    (6, 0.5, 0.0, "did not replicate"), (9, -0.5, 0.0, "suggestive again"), (9, 0.5, -0.6, "suggestive again")])
def test_h3_rule(tmp_path, n_pos, pix, img, expect):
    import analyze3
    assert analyze3.main(_synthetic(tmp_path, 0.02, n_pos, pix, img), tmp_path)["H3"] == expect


@pytest.mark.parametrize("brain_pos,h3_pos,expect", [
    (5, 9, "CONFIRMED"), (5, 8, "CONFIRMED"), (5, 6, "holds on most seeds (not confirmed) (5/5 seeds, p = 0.031 uncorrected)"),
    (4, 9, "holds on most seeds (not confirmed)"), (3, 9, "did not replicate")])
def test_h2_rule_and_holm(tmp_path, brain_pos, h3_pos, expect):
    import analyze3
    # H2 confirmable only through Holm's second step, i.e. when H3's p also passes (plan §5)
    assert analyze3.main(_synthetic(tmp_path, 0.02, h3_pos, brain_pos=brain_pos), tmp_path)["H2"] == expect


def test_holm():
    import analyze3
    a = analyze3.holm({"a": 0.031, "b": 0.002})
    assert abs(a["b"] - 0.004) < 1e-12 and abs(a["a"] - 0.031) < 1e-12
    a = analyze3.holm({"a": 0.031, "b": 0.09})
    assert abs(a["a"] - 0.062) < 1e-12 and abs(a["b"] - 0.09) < 1e-12


@pytest.mark.parametrize("n_pos,pix,expect4,expect5", [
    (9, 0.5, "SUPPORTED (beats MVFA)", "BEATS MVFA"), (5, -0.9, "MATCHES", "SUPPORTED (matches MVFA)"),
    (5, -1.2, "TRAILS", "TRAILS")])
def test_h4_h5_rules(tmp_path, n_pos, pix, expect4, expect5):
    import analyze3
    out = analyze3.main(_synthetic(tmp_path, 0.02, n_pos, pix, arms=("pu_m100", "mil_m100", "mil_m10")), tmp_path)
    assert out["H4"] == expect4 and out["H5"] == expect5
