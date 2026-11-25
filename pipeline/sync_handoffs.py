#!/usr/bin/env python3
"""Copy each stage's output forward to the stage that reads it, and record sha256 hashes.

    python pipeline/sync_handoffs.py            # copy every handoff whose copy differs, rewrite the manifest
    python pipeline/sync_handoffs.py --dry-run  # say what would be copied; change nothing
    python pipeline/sync_handoffs.py --check    # exit 1 if any copy or the manifest has drifted

The handoffs (source -> copy the next stage reads):

  task1.v6_to_handoff   Task 1 run v6 export            -> Task 1 task3_handoff/ (the published hand-off)
  task1_to_task3        Task 1 task3_handoff/           -> Task 3 data/task1/
  task2_to_task3        Task 2 outputs/full_nim + gaps  -> Task 3 data/task2/   (also named in data/inputs_manifest.json)
  task3_to_task4        Task 3 A1 evolution + ranking   -> Task 4 plans/        (the A1 problem Task 4 tests)

The manifest (pipeline/handoffs_manifest.json) lists each pair with the sha256 of source and copy.
`--check` fails when a copy differs from its source, when a file is missing, or when the manifest's
hashes are not the files' current hashes (someone edited a source without re-syncing). Only the
standard library is used. Copies are byte-for-byte (shutil.copyfile); nothing is regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "pipeline" / "handoffs_manifest.json"

T1, T2, T3, T4 = "task1_literature_review", "task2_limitations", "task3_ideation", "task4_weak_supervision"
HANDOFFS = {
    "task1.v6_to_handoff": [
        (f"{T1}/runs/v6_2026-11-16/final_ref_set.json", f"{T1}/task3_handoff/final_ref_set.json"),
        (f"{T1}/runs/v6_2026-11-16/intermediates/final_ref_set_excluded.json",
         f"{T1}/task3_handoff/final_ref_set_excluded.json"),
    ],
    "task1_to_task3": [
        (f"{T1}/task3_handoff/final_ref_set.json", f"{T3}/data/task1/final_ref_set.json"),
        (f"{T1}/task3_handoff/final_ref_set_excluded.json", f"{T3}/data/task1/final_ref_set_excluded.json"),
        (f"{T1}/task3_handoff/README.md", f"{T3}/data/task1/README_task1_output.md"),
    ],
    "task2_to_task3": [
        (f"{T2}/outputs/full_nim/final_limitations.json", f"{T3}/data/task2/final_limitations.json"),
        (f"{T2}/deliverables/task2/task2_research_gaps.md", f"{T3}/data/task2/task2_research_gaps.md"),
    ],
    "task3_to_task4": [
        (f"{T3}/outputs/stage3_evolution_A1.json", f"{T4}/plans/stage3_evolution_A1.json"),
        (f"{T3}/outputs/stage4_ranking_A1.json", f"{T4}/plans/stage4_ranking_A1.json"),
    ],
}
NOTES = {
    "task1.v6_to_handoff": "Task 1's own export step; the hand-off folder is what other tasks are told to read.",
    "task1_to_task3": "Task 3 run 3 reads data/task1/ (Task 1 run v6: 594 candidate papers).",
    "task2_to_task3": "Task 3 stage 0 reads data/task2/final_limitations.json (39 limitations, MiniMax via NIM). "
                      "Task 2 was built on Task 1 run v4, not v6: see PIPELINE.md.",
    "task3_to_task4": "Task 4 tests idea family A1 (image-label MIL for MVFA's segmentation adapters); "
                      "the plans in plans/ cite these two files.",
}


def sha256(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def build_manifest() -> dict:
    out = {"_about": "Written by pipeline/sync_handoffs.py; checked by `make check`. Paths are relative to the repo root.",
           "handoffs": {}}
    for name, pairs in HANDOFFS.items():
        out["handoffs"][name] = {"note": NOTES[name], "files": [
            {"source": s, "copy": d, "sha256": sha256(ROOT / s)} for s, d in pairs]}
    return out


def check() -> int:
    problems = []
    for name, pairs in HANDOFFS.items():
        for s, d in pairs:
            hs, hd = sha256(ROOT / s), sha256(ROOT / d)
            if hs is None:
                problems.append(f"{name}: source missing: {s}")
            elif hd is None:
                problems.append(f"{name}: copy missing: {d}")
            elif hs != hd:
                problems.append(f"{name}: DRIFT {d} differs from {s}")
    if not MANIFEST.exists():
        problems.append(f"manifest missing: {MANIFEST.relative_to(ROOT)} (run make sync)")
    else:
        recorded = json.loads(MANIFEST.read_text())
        if recorded.get("handoffs") != build_manifest()["handoffs"]:
            problems.append("manifest is stale: recorded hashes or pairs differ from the files (run make sync)")
    n = sum(len(p) for p in HANDOFFS.values())
    if problems:
        print("\n".join(problems))
        print(f"handoffs: {len(problems)} problem(s) in {n} file pairs")
        return 1
    print(f"handoffs: OK, {n} file pairs byte-identical and match {MANIFEST.relative_to(ROOT)}")
    return 0


def sync(dry: bool) -> int:
    changed = 0
    for name, pairs in HANDOFFS.items():
        for s, d in pairs:
            src, dst = ROOT / s, ROOT / d
            if not src.exists():
                print(f"[missing] {s}")
                return 1
            if sha256(src) == sha256(dst):
                print(f"[same]    {d}")
                continue
            changed += 1
            print(f"[{'would copy' if dry else 'copy'}] {s} -> {d}")
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    if dry:
        print(f"dry run: {changed} file(s) would be copied; manifest not written")
        return 0
    MANIFEST.write_text(json.dumps(build_manifest(), indent=1) + "\n")
    print(f"{changed} file(s) copied; manifest written: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="fail on drift; change nothing")
    g.add_argument("--dry-run", action="store_true", help="show what would be copied; change nothing")
    a = ap.parse_args()
    sys.exit(check() if a.check else sync(a.dry_run))
