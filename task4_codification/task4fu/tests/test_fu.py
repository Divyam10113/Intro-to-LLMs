"""Follow-up pre-run checks (task4fu/README.md). CPU only."""
import csv
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

import common2
from common2 import sources, train_lists
from train2 import build_epoch, mask_budget_set

FU = Path(common2.__file__).parent
SEEDS = (8, 9, 10, 11, 12)


def _lists(fold):
    return {d: train_lists(d) for d in sources(fold)}


# ---------------------------------------------------------------- what runs, and where it writes
def test_queue_is_exactly_the_10_preregistered_runs():
    out = subprocess.run(["bash", str(FU / "run_all.sh"), "list"], capture_output=True, text=True, check=True).stdout
    jobs = [j for j in out.split("\n") if j.strip()]
    want = [f"train2.py --arm {a} --fold Brain --seed {s} --epochs 20" for s in SEEDS for a in ("mvfa", "mil_m10")]
    assert jobs == want


def test_outputs_never_go_to_round_3():
    src = (FU / "common2.py").read_text()
    assert '"/mnt/HDD1/divyam/task4/fu"' in src and '"/mnt/HDD1/divyam/task4/r3"' not in src
    assert "TASK4R3_HDD=${TASK4R3_HDD:-/mnt/HDD1/divyam/task4/fu}" in (FU / "run_all.sh").read_text()
    assert common2.RESULTS2 == FU / "results" or os.environ.get("TASK4R2_RESULTS")


def test_code_is_round_3s_apart_from_the_output_folder():
    r3 = FU.parent / "task4r3"
    for f in ("train2.py", "evaluate_target.py", "build_normal_bank.py", "make_valid.py",
              "splits/valid.json", "splits/valid_excluded.json"):
        assert (FU / f).read_bytes() == (r3 / f).read_bytes(), f


# ---------------------------------------------------------------- the budget arm on the new seeds
@pytest.mark.parametrize("seed", SEEDS)
def test_mil_m10_masks_a_seeded_10pct_set_on_brain(seed):
    L = _lists("Brain")
    s10 = mask_budget_set("mil_m10", "Brain", seed, L)
    for d in L:
        if common2.has_masks(d):
            assert len({p for p in s10 if p in set(L[d][1])}) == int(round(0.10 * len(L[d][1])))
    assert all("/Ungood/" in p for p in s10)
    assert s10 < mask_budget_set("mil_m25", "Brain", seed, L)          # nested, as in round 3
    items = build_epoch("mil_m10", "Brain", seed, 0, L, s10)
    assert {p for k, d, p, y in items if k == "mask"} == s10
    assert s10 != mask_budget_set("mil_m10", "Brain", seed + 1, L)


@pytest.mark.parametrize("seed", (8, 12))
def test_mvfa_and_mil_m10_see_the_same_images_in_the_same_order(seed):
    L = _lists("Brain")
    ref = [(d, p, y) for _, d, p, y in build_epoch("mvfa", "Brain", seed, 3, L, set())]
    got = [(d, p, y) for _, d, p, y in build_epoch("mil_m10", "Brain", seed, 3, L,
                                                    mask_budget_set("mil_m10", "Brain", seed, L))]
    assert got == ref


# ---------------------------------------------------------------- the frozen rule
def _csv(tmp_path, rows):
    p = tmp_path / "results.csv"
    keys = ["run_id", "arm", "fold", "eval_dataset", "seed", "img_auroc", "pix_auroc", "aupro", "best_epoch",
            "diag_inmask_p", "notes"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for arm, s, img, pix, au in rows:
            w.writerow(dict(run_id=f"{arm}_Brain_s{s}", arm=arm, fold="Brain", eval_dataset="Brain", seed=s,
                            img_auroc=img / 100, pix_auroc=pix / 100, aupro=au / 100, best_epoch=10,
                            diag_inmask_p=0.5, notes=""))
    return p


def _run(tmp_path, d_au, d_pix):
    import analyze_fu
    rows = []
    for s, a, p in zip(SEEDS, d_au, d_pix):
        rows += [("mvfa", s, 55, 88, 60), ("mil_m10", s, 60, 88 + p, 60 + a)]
    return analyze_fu.main(_csv(tmp_path, rows), tmp_path / "a.md")


@pytest.mark.parametrize("d_au,d_pix,want", [
    ([5, 5, 5, 5, 5], [1, 1, 1, 1, 1], "CONFIRMED"),
    ([5, 5, 5, 5, 5], [-1, -1, -1, -1, 1], "did not replicate"),      # 5/5 but mean pix Δ <= 0
    ([5, 5, 5, 5, -5], [1, 1, 1, 1, 1], "holds on most seeds (not confirmed)"),
    ([5, 5, 5, 0, 5], [1, 1, 1, 1, 1], "holds on most seeds (not confirmed)"),  # a tie counts against
    ([5, 5, 5, -1, -1], [1, 1, 1, 1, 1], "did not replicate"),
])
def test_rule_outcomes(tmp_path, d_au, d_pix, want):
    assert _run(tmp_path, d_au, d_pix) == want


def test_rule_incomplete_and_smoke_rows_ignored(tmp_path):
    import analyze_fu
    rows = []
    for s in SEEDS[:4]:
        rows += [("mvfa", s, 55, 88, 60), ("mil_m10", s, 60, 90, 70)]
    p = _csv(tmp_path, rows)
    with open(p, "a", newline="") as f:      # a smoke row for seed 12 must not complete the set
        f.write("mil_m10_Brain_s12,mil_m10,Brain,Brain,12,0.6,0.9,0.7,0,0.5,SMOKE\n"
                "mvfa_Brain_s12,mvfa,Brain,Brain,12,0.5,0.8,0.6,0,0.5,SMOKE\n")
    out = analyze_fu.main(p, tmp_path / "a.md")
    assert out.startswith("incomplete (4/5")


def test_round_3_observation_is_reproduced_from_round_3_results():
    import analyze_fu
    d3 = analyze_fu.load(FU.parent / "task4r3" / "results" / "results.csv")
    t = analyze_fu.test(d3, (3, 4, 5))
    assert t["n"] == 3 and t["pos"] == 3 and t["mpix"] > 0          # the post-hoc 3/3 this run tests


def test_rules_files_match_frozen_hashes():
    frozen = FU / "report" / "FROZEN.sha256"
    if not frozen.exists():
        pytest.skip("rules not frozen yet")
    for line in frozen.read_text().split("\n"):
        if line.strip():
            h, name = line.split()
            assert hashlib.sha256((FU / name).read_bytes()).hexdigest() == h, f"{name} changed after freezing"
