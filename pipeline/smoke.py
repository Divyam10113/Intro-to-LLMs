#!/usr/bin/env python3
"""Offline smoke runs, one per stage: no paid API call, no GPU, no change to tracked files.

    python pipeline/smoke.py task0|task1|task2|task3|task4|all [--keep]

Every stage runs in a scratch copy (a temporary folder, or $SMOKE_DIR) and the repository is
left as it was. For the run, API keys are removed from the environment, LLM backends are set to
the offline "file" mode, network access through HTTP(S) clients is pointed at a dead proxy
(127.0.0.1:9), and CUDA_VISIBLE_DEVICES is empty. What each stage does:

  task0  course starter: re-runs generate_prompt.py and extract_related.py (pandas / text only)
         and compares their outputs with the committed ones; the scripts that need the network,
         an embedding model or an LLM are only byte-compiled.
  task1  QUAL-SG: byte-compiles everything, imports every qual_sg module, prints the driver's
         stage list, and validates the hand-off final_ref_set.json (594 papers, 12 non-empty fields).
         A real run needs OpenAlex + an LLM and the caches that are not committed.
  task2  limitations: byte-compiles, imports every module, checks run_pipeline.py --help, and
         validates outputs/full_nim/final_limitations.json (39 limitations). A real run needs an
         NVIDIA NIM or AgentRouter key; the LLM cache (cache/) was never committed.
  task3  ideation: replays the whole pipeline (run_all.py) from logs/cache with the file backend,
         which raises instead of calling a model when an answer is missing, then compares every
         regenerated output with the committed one.
  task4  weak supervision: checks the frozen hashes, re-runs the frozen analysers (analyze3.py,
         analyze_fu.py) on the committed results.csv and compares with the committed analyses,
         parses every shell script, and lists the round 3 and follow-up GPU queues
         (run_all.sh list) without running them.

Exit code 0 means every check passed; differences in regenerated outputs are failures.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared import config  # noqa: E402  (loads .env)

KEYS = ["LLM_API_KEY", "JUDGE_API_KEY", "NVIDIA_API_KEY", "AGENTROUTER_API_KEY", "GROQ_API_KEY",
        "OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
DEAD = "http://127.0.0.1:9"


def offline_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in KEYS and not k.startswith(("LLM_", "JUDGE_"))}
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env[k] = DEAD
    env.pop("NO_PROXY", None); env.pop("no_proxy", None)
    env.update(LLM_BACKEND="file", QUAL_SG_BACKEND="file", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg")
    env.update(extra)
    return env


def python() -> str:
    """Python for Tasks 0-3: $PYTHON, else the repo's .venv-pipeline (make venv), else this interpreter.
    Needs 3.12+ (Task 3 uses PEP 701 f-strings) with numpy, pandas, scipy and scikit-learn."""
    if os.environ.get("PYTHON"):
        return os.environ["PYTHON"]
    venv = ROOT / ".venv-pipeline" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


class Stage:
    def __init__(self, name, scratch: Path):
        self.name, self.scratch, self.fails, self.notes = name, scratch / name, [], []
        self.scratch.mkdir(parents=True, exist_ok=True)

    def run(self, cmd, cwd, env=None, check=True, quiet=False):
        print(f"  $ {' '.join(map(str, cmd))}   (in {cwd})")
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env or offline_env(), capture_output=True, text=True)
        if not quiet and r.stdout.strip():
            print("    " + "\n    ".join(r.stdout.strip().splitlines()[-12:]))
        if r.returncode != 0 and check:
            self.fail(f"exit {r.returncode}: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
        return r

    def ok(self, cond, msg):
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
        if not cond:
            self.fails.append(msg)

    def fail(self, msg):
        print(f"  FAIL {msg}")
        self.fails.append(msg.splitlines()[0])

    def same(self, a: Path, b: Path, label):
        self.ok(a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False), f"identical to committed: {label}")

    def compile_all(self, folder: Path, skip=("logs", "mvfa_code", "runs")):
        n = 0
        for p in folder.rglob("*.py"):
            if any(s in p.parts for s in skip):
                continue
            try:
                py_compile.compile(str(p), cfile=str(self.scratch / "pyc" / (str(n) + ".pyc")), doraise=True)
                n += 1
            except py_compile.PyCompileError as e:
                self.fail(f"syntax error: {p}: {e}")
        self.ok(True, f"{n} Python files byte-compile")

    def copy(self, rel, ignore=()):
        dst = self.scratch / Path(rel).name
        shutil.copytree(ROOT / rel, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", *ignore))
        return dst


# ----------------------------------------------------------------------------- stages
def task0(st: Stage):
    src = ROOT / "task0_starter"
    w = st.scratch / "work"
    w.mkdir()
    for f in ("ranked_papers.csv", "generate_prompt.py", "extract_related.py"):
        shutil.copy(src / f, w)
    (w / "base_paper").symlink_to(ROOT / "base_paper")
    st.compile_all(src)
    py = python()
    st.run([py, "generate_prompt.py"], w)
    st.same(w / "llm_prompt.txt", src / "llm_prompt.txt", "llm_prompt.txt (from the committed ranked_papers.csv)")
    st.run([py, "extract_related.py"], w)
    st.same(w / "base_paper_related_titles.txt", src / "base_paper_related_titles.txt",
            "base_paper_related_titles.txt (from base_paper/)")
    st.notes.append("not run: scholar_scraper.py (arXiv network), rank_papers.py (downloads the all-MiniLM-L6-v2 "
                    "embedding model; sentence-transformers is not in the server environment), generate_review.py "
                    "(AgentRouter LLM call); ranked_output.txt, lit_review.md and final_report.md have no generating "
                    "script in the repo")


def task1(st: Stage):
    t1 = st.copy("task1_literature_review")
    st.compile_all(t1, skip=("runs",))
    run = t1 / "runs" / "v6_2026-11-16"
    env = offline_env(QUAL_SG_RUN_DIR=str(run), QUAL_SG_TOPIC_CSV=str(run / "topic.csv"))
    mods = sorted(p.stem for p in (t1 / "qual_sg").glob("*.py") if p.stem != "__init__")
    code = ("import importlib,sys\nbad=[]\nfor m in sys.argv[1:]:\n    try: importlib.import_module('qual_sg.'+m)\n"
            "    except Exception as e: bad.append(f'{m}: {type(e).__name__}: {e}')\n"
            "print(f'{len(sys.argv)-1-len(bad)}/{len(sys.argv)-1} qual_sg modules import'); print(*bad, sep='\\n')\n"
            "sys.exit(1 if bad else 0)")
    st.run([python(), "-c", code, *mods], t1, env=env)
    r = st.run([python(), "-c", "from qual_sg.run_pipeline import STAGES; print(' -> '.join(s for s, _ in STAGES))"],
               t1, env=env)
    st.ok(r.returncode == 0 and "final_ref_set" in r.stdout, "driver stage list loads (run_pipeline.STAGES)")
    data = json.loads((ROOT / "task1_literature_review/task3_handoff/final_ref_set.json").read_text())
    recs = data if isinstance(data, list) else data.get("papers") or data.get("records") or data.get("items") or []
    fields = ["doi", "title", "abstract", "year", "venue", "citation_count", "keywords", "topics",
              "relevance_score", "composite_score", "origin", "selected"]
    st.ok(len(recs) == 594, f"final_ref_set.json has {len(recs)} papers (expected 594)")
    empty = sum(1 for r in recs for f in fields if r.get(f) in (None, "", []))
    st.ok(all(set(fields) <= set(r) for r in recs) and empty == 0, f"every paper has the 12 fields, none empty ({empty} empty)")
    st.notes.append("not run: the QUAL-SG stages themselves (OpenAlex/arXiv/Crossref network, LLM judge; HTTP and "
                    "LLM caches are git-ignored, so a re-run starts from retrieval)")


def task2(st: Stage):
    t2 = st.copy("task2_limitations", ignore=("cache", "pdfs", "http_cache"))
    st.compile_all(t2)
    env = offline_env(LLM_PROVIDER="nim")
    code = ("import importlib\nfor m in ['common','retrieval','agents','evaluate','build_corpus','run_pipeline','check_models']:\n"
            "    importlib.import_module(m)\nprint('7/7 Task 2 modules import')")
    st.run([python(), "-c", code], t2, env=env)
    r = st.run([python(), "run_pipeline.py", "--help"], t2, env=env)
    st.ok(r.returncode == 0 and "--no_citation" in r.stdout, "run_pipeline.py --help")
    fl = json.loads((ROOT / "task2_limitations/outputs/full_nim/final_limitations.json").read_text())
    items = fl if isinstance(fl, list) else fl.get("limitations", [])
    st.ok(len(items) == 39, f"outputs/full_nim/final_limitations.json has {len(items)} limitations (expected 39)")
    papers = json.loads((ROOT / "task2_limitations/data/papers.json").read_text())
    n = len(papers) if isinstance(papers, (list, dict)) else 0
    st.ok(n == 223, f"data/papers.json has {n} papers (expected 223)")
    st.notes.append("not run: build_corpus.py (OpenAlex/arXiv network), run_pipeline.py and evaluate.py (NVIDIA NIM "
                    "or AgentRouter LLM calls; Task 2's LLM cache/ was never committed, so nothing can be replayed)")


def task3(st: Stage):
    t3 = st.copy("task3_ideation", ignore=("pending",))
    st.compile_all(t3)
    # delete what the replay must regenerate, so "identical" below means "rebuilt identically"
    # (run_all.py's own outputs; hypercheck_*, stage2b_* and rejudge_* come from the GPU / Gemini checks
    # and template.html is an input, so those stay)
    keep = ("hypercheck_", "stage2b_", "rejudge_", "template.html")
    removed = [q for sub in ("outputs", "explorer") for q in (t3 / sub).rglob("*")
               if q.is_file() and not q.name.startswith(keep)]
    for q in removed:
        q.unlink()
    print(f"  removed {len(removed)} files that run_all.py writes from the scratch copy: "
          + ", ".join(q.name for q in removed))
    env = offline_env()
    before = len(list((t3 / "logs" / "cache").glob("*.json")))
    r = st.run([python(), "run_all.py"], t3, env=env, quiet=True)
    print("    " + "\n    ".join(l for l in r.stdout.splitlines() if l.startswith("=====") or "stopped" in l))
    st.ok(r.returncode == 0, "run_all.py replayed every stage from logs/cache (file backend: no model call possible)")
    after = len(list((t3 / "logs" / "cache").glob("*.json")))
    pend = list((t3 / "logs" / "pending").glob("*.json")) if (t3 / "logs" / "pending").exists() else []
    st.ok(after == before and not pend, f"no new cache entries ({before} -> {after}) and no pending prompts ({len(pend)})")
    diffs, n = [], 0
    for sub in ("outputs", "explorer"):
        for p in sorted((ROOT / "task3_ideation" / sub).rglob("*")):
            if p.is_file():
                n += 1
                q = t3 / p.relative_to(ROOT / "task3_ideation")
                if not q.exists() or not filecmp.cmp(p, q, shallow=False):
                    diffs.append(str(p.relative_to(ROOT)))
    new = [str(q.relative_to(t3)) for sub in ("outputs", "explorer") for q in (t3 / sub).rglob("*")
           if q.is_file() and not (ROOT / "task3_ideation" / q.relative_to(t3)).exists()]
    st.ok(not diffs, f"{n - len(diffs)}/{n} files in outputs/ and explorer/ identical after the replay"
          + (": differ: " + ", ".join(diffs) if diffs else ""))
    if new:
        st.notes.append("replay also wrote files that are not committed: " + ", ".join(new))
    st.notes.append("not run: make_reports.py (xelatex), rejudge.py (Gemini API), fetch_citations.py / "
                    "novelty_search.py (Semantic Scholar network), colab/ (GPU)")


def task4(st: Stage):
    t4 = ROOT / "task4_weak_supervision"
    py = str(config.TASK4_PYTHON)
    if not Path(py).exists():
        st.fail(f"TASK4_PYTHON not found: {py} (set it in .env)")
        return
    st.run([sys.executable, str(ROOT / "pipeline" / "check_repo.py")], ROOT)
    env = offline_env(TASK4R2_RESULTS=str(st.scratch / "unused"))   # never write into results/
    out3 = st.scratch / "r3"
    st.run([py, "analyze3.py", str(t4 / "round3/results/results.csv"), str(out3)], t4 / "round3", env=env)
    for f in ("analysis.md", "per_seed.md"):
        st.same(out3 / f, t4 / "round3/results" / f, f"round3/results/{f} (frozen analyze3.py)")
    outfu = st.scratch / "analysis_fu.md"
    st.run([py, "analyze_fu.py", str(t4 / "followup/results/results.csv"), str(outfu)], t4 / "followup", env=env)
    st.same(outfu, t4 / "followup/results/analysis_fu.md", "followup/results/analysis_fu.md (frozen analyze_fu.py)")
    shells = sorted(t4.glob("[rf]*/*.sh")) + [t4 / "download_data.sh"]
    bad = [str(f.relative_to(t4)) for f in shells if subprocess.run(["bash", "-n", str(f)]).returncode]
    st.ok(not bad, f"{len(shells)} shell scripts parse (bash -n)" + (": " + ", ".join(bad) if bad else ""))
    for d, args, want in (("round3", ["list", "r3"], 58), ("followup", ["list"], 10)):
        r = st.run(["bash", "run_all.sh", *args], t4 / d, env=env, quiet=True)
        jobs = [j for j in r.stdout.splitlines() if j.strip()]
        st.ok(r.returncode == 0 and (want is None or len(jobs) == want),
              f"{d}/run_all.sh {' '.join(args)}: {len(jobs)} GPU jobs listed, none started")
    st.notes.append("not run: training and evaluation (GPU, and data/weights on $TASK4_HDD). `make test` runs the "
                    "CPU test suites; `make task4-smoke-cpu` runs round 3's CPU end-to-end smoke (a few minutes)")


STAGES = {"task0": task0, "task1": task1, "task2": task2, "task3": task3, "task4": task4}


def git_status():
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                          capture_output=True, text=True).stdout


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="offline smoke runs (no API calls, no GPU)")
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder")
    a = ap.parse_args()
    scratch = Path(os.environ["SMOKE_DIR"]) if os.environ.get("SMOKE_DIR") else Path(tempfile.mkdtemp(prefix="smoke_"))
    scratch.mkdir(parents=True, exist_ok=True)
    status0 = git_status()
    results = {}
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"\n===== smoke {name}  (scratch: {scratch / name})")
        st = Stage(name, scratch)
        try:
            STAGES[name](st)
        except Exception as e:  # noqa: BLE001
            st.fail(f"{type(e).__name__}: {e}")
        for n in st.notes:
            print(f"  note {n}")
        results[name] = st.fails
    status1 = git_status()
    if status1 != status0:
        print("FAIL the smoke run changed the working tree:\n" + status1)
        results["working tree"] = ["changed"]
    if not a.keep and not os.environ.get("SMOKE_DIR"):
        shutil.rmtree(scratch, ignore_errors=True)
    print("\n===== summary")
    for name, fails in results.items():
        print(f"  {name:6s} {'PASS' if not fails else 'FAIL (' + str(len(fails)) + ')'}")
    sys.exit(1 if any(results.values()) else 0)
