#!/usr/bin/env python3
"""Repository checks run by `make check` (standard library only, no network, no GPU).

1. Task 4's frozen files match each report/FROZEN.sha256 (round 3 and the follow-up).
2. The compatibility symlinks that the frozen analysers rely on resolve
   (task4_weak_supervision/task4r3 -> round3, etc.; analyze_fu.py reads FU.parent/"task4r3"; Plans -> plans
   keeps the README links ../Plans/... valid).
3. No model weights or datasets are tracked: no *.pth/*.pt/*.ckpt/*.npz/*.safetensors/*.onnx,
   and no tracked file over 20 MB.
4. No hard-coded server paths are left in code (the defaults live in shared/config.*).
5. The git user's e-mail address does not appear in any tracked file (read from git config at
   run time and never printed or written).
6. Task 5's copies of Task 4's data (task5_paper/inputs/task4_*) match Task 4, line endings aside.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T4 = ROOT / "task4_weak_supervision"
WEIGHTS = (".pth", ".pt", ".ckpt", ".npz", ".safetensors", ".onnx")
MAX_MB = 20
# Server paths that may appear only as config defaults / documentation of those defaults.
SERVER_PATHS = ("/home/f20241442/", "/mnt/HDD1/")
ALLOWED_SERVER_PATH_FILES = {
    "shared/config.sh", "shared/config.py", ".env.example", "pipeline/check_repo.py", "pipeline/smoke.py",
    # env-var defaults kept verbatim: the tests check these exact strings
    "task4_weak_supervision/round1/common.py", "task4_weak_supervision/round2/common2.py",
    "task4_weak_supervision/round3/common2.py", "task4_weak_supervision/followup/common2.py",
    "task4_weak_supervision/round3/run_all.sh", "task4_weak_supervision/followup/run_all.sh",
    "task4_weak_supervision/followup/go.sh",
    "task4_weak_supervision/round3/tests/test_r3.py", "task4_weak_supervision/followup/tests/test_fu.py",
}
# Snapshots the paper was built from (copies of Task 4's code as it ran); kept verbatim, never edited.
ALLOWED_SERVER_PATH_PREFIXES = ("task5_paper/inputs/",)
CODE_SUFFIXES = (".py", ".sh")
# Task 5 keeps its own copy of each Task 4 round (task5_paper/inputs/task4_*). Its data files must say the same
# as Task 4's. They are compared with line endings normalised: the copies went through Windows (CRLF).
TASK4_TO_TASK5 = {"task4_r1": "round1", "task4_r2": "round2", "task4_r3": "round3", "task4_fu": "followup"}
DATA_SUFFIXES = (".csv", ".json", ".md")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def check_frozen(errs):
    for d in ("round3", "followup"):
        f = T4 / d / "report" / "FROZEN.sha256"
        for line in f.read_text().split("\n"):
            if line.strip():
                h, name = line.split()
                got = hashlib.sha256((T4 / d / name).read_bytes()).hexdigest()
                if got != h:
                    errs.append(f"frozen file changed: {d}/{name}")
                else:
                    print(f"  frozen OK  {d}/{name}")


def check_symlinks(errs):
    for link, target in (("task4", "round1"), ("task4r2", "round2"), ("task4r3", "round3"), ("task4fu", "followup"), ("Plans", "plans")):
        p = T4 / link
        if not (p.is_symlink() and p.resolve() == (T4 / target).resolve()):
            errs.append(f"symlink {p.relative_to(ROOT)} must point to {target}")
    if not (T4 / "followup").resolve().parent.joinpath("task4r3", "results", "results.csv").exists():
        errs.append("analyze_fu.py's FU.parent/'task4r3'/results/results.csv does not resolve")


def check_tracked(errs):
    files = [f for f in git("ls-files", "-z").split("\0") if f]
    for f in files:
        p = ROOT / f
        if f.lower().endswith(WEIGHTS):
            errs.append(f"weights tracked: {f}")
        if p.is_file() and not p.is_symlink() and p.stat().st_size > MAX_MB * 2**20:
            errs.append(f"file over {MAX_MB} MB tracked: {f}")
        if (f.endswith(CODE_SUFFIXES) and f not in ALLOWED_SERVER_PATH_FILES and not f.startswith(ALLOWED_SERVER_PATH_PREFIXES)
                and p.is_file() and "/logs/" not in f):
            text = p.read_text(errors="ignore")
            for sp in SERVER_PATHS:
                for i, line in enumerate(text.splitlines(), 1):
                    s = line.strip()
                    if sp in line and not s.startswith("#") and "python evaluate_target.py --run_dir" not in s:
                        errs.append(f"hard-coded server path in {f}:{i}")
    print(f"  {len(files)} tracked files scanned for weights, size and server paths")
    email = git("config", "user.email").strip()
    if email:
        hits = git("grep", "-I", "-l", "-F", email).split()
        if hits:
            errs.append(f"the git user's e-mail address appears in {len(hits)} tracked file(s): {', '.join(hits)}")
        else:
            print("  git user's e-mail address: not in any tracked file")


def check_task5_inputs(errs):
    """Task 5's copies of Task 4's results and report data say the same as Task 4 (line endings aside)."""
    n = 0
    for copy, rnd in TASK4_TO_TASK5.items():
        base = ROOT / "task5_paper" / "inputs" / copy
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if not (p.is_file() and p.suffix in DATA_SUFFIXES and rel.parts[0] in ("results", "report", "splits")):
                continue
            src = T4 / rnd / rel
            if not src.is_file():
                errs.append(f"Task 5 input has no source in Task 4: task5_paper/inputs/{copy}/{rel}")
            elif p.read_bytes().replace(b"\r\n", b"\n") != src.read_bytes().replace(b"\r\n", b"\n"):
                errs.append(f"Task 5 input differs from Task 4: task5_paper/inputs/{copy}/{rel} vs {rnd}/{rel}")
            n += 1
    print(f"  {n} Task 4 data files copied into task5_paper/inputs/ match Task 4 (line endings aside)")


if __name__ == "__main__":
    errs: list[str] = []
    check_frozen(errs)
    check_symlinks(errs)
    check_task5_inputs(errs)
    check_tracked(errs)
    if errs:
        print("\n".join("FAIL " + e for e in errs))
        sys.exit(1)
    print("repo checks: OK")
