"""Resumable driver for a documented pipeline run (run v5 and later).

Runs every stage as ``python3 -m qual_sg.<module>`` with the current
environment, writes each stage's full output to ``<run>/intermediates/logs/
stage_<name>.log``, and records status, timing and settings in
``<run>/run_state.json`` and ``<run>/run_manifest.json``.  Completed stages are
skipped on the next invocation, so the run can stop and resume:

* exit 10 - a stage is waiting for file-backend LLM answers
  (``python3 -m qual_sg.llm_exchange status``); answer them and re-run;
* exit 3  - the OpenAlex daily budget is too low (see the preflight report);
  re-run after ``resets_in_seconds``;
* exit 1  - a stage failed for another reason (see its log).

    QUAL_SG_RUN_DIR=runs/v5_... python3 -m qual_sg.run_pipeline [--only stage,...] [--redo stage,...]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

from . import common, config

STAGES: List[Tuple[str, List[str]]] = [
    ("preflight", ["preflight"]),
    ("step1_retrieve", ["step1_retrieve"]),
    ("step2_quality", ["step2_quality"]),
    ("fill_abstracts", ["fill_abstracts"]),
    ("anchor_flags", ["anchor_merge", "--flag-only"]),
    ("step3_rerank", ["step3_rerank"]),
    ("export_refs", ["export_refs"]),
    ("bib_overlap", ["bib_eval", "--pdf", "../2403.12570v1.pdf", "--selected", "--pool",
                     "--out", str(config.INTERMEDIATES_DIR / "bib_overlap_v5.json")]),
    ("bibliography", ["bibliography"]),
    ("step4_outline", ["step4_outline"]),
    ("step5_generate", ["step5_generate"]),
    ("short_review", ["short_review", "--source", "qual_sg"]),
    ("metrics", ["run_metrics"]),
]

STATE = config.OUT_DIR / "run_state.json"
MANIFEST = config.OUT_DIR / "run_manifest.json"


def pending_count() -> int:
    d = config.LLM_EXCHANGE_DIR / "pending"
    return len(list(d.glob("*.json"))) if d.exists() else 0


def manifest() -> Dict[str, Any]:
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("QUAL_SG_")}
    versions = {}
    for mod in ("pandas", "numpy", "sklearn", "sentence_transformers", "torch", "requests", "pypdf"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    return {
        "run_dir": str(config.OUT_DIR),
        "started_or_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "environment": env,
        "openalex_key_configured": bool(config.openalex_api_key()),
        "llm_backend": config.LLM_BACKEND,
        "models_requested": {s: config.model_for(s) for s in ("judge", "outline", "writer", "eval")},
        "embedding_model": config.EMBED_MODEL_NAME,
        "stages": [name for name, _ in STAGES],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="documented, resumable pipeline run")
    ap.add_argument("--only", default="", help="comma-separated stage names to run")
    ap.add_argument("--redo", default="", help="comma-separated completed stages to run again")
    args = ap.parse_args()
    config.ensure_dirs()
    common.write_json(MANIFEST, manifest())
    state: Dict[str, Any] = common.read_json(STATE) if STATE.exists() else {"stages": {}}
    only = {s for s in args.only.split(",") if s}
    redo = {s for s in args.redo.split(",") if s}

    for name, cmd in STAGES:
        if only and name not in only:
            continue
        done = state["stages"].get(name, {}).get("status") == "done"
        if done and name not in redo:
            print(f"[skip] {name} (done)")
            continue
        log_path = config.LOG_DIR / f"stage_{name}.log"
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== {name} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log.flush()
            proc = subprocess.run([sys.executable, "-m", f"qual_sg.{cmd[0]}", *cmd[1:]],
                                  cwd=str(config.BASE_DIR), stdout=log, stderr=subprocess.STDOUT)
        entry = {"exit_code": proc.returncode, "seconds": round(time.time() - t0, 1),
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": str(log_path)}
        if proc.returncode == 0:
            entry["status"] = "done"
        elif name == "preflight" and proc.returncode == 3:
            entry["status"] = "openalex_budget_low"
        elif pending_count():
            entry["status"] = "awaiting_llm"
            entry["pending_prompts"] = pending_count()
        else:
            entry["status"] = "failed"
        state["stages"][name] = entry
        common.write_json(STATE, state)
        print(f"[{entry['status']}] {name} ({entry['seconds']}s) -> {log_path}")
        if entry["status"] == "openalex_budget_low":
            return 3
        if entry["status"] == "awaiting_llm":
            print(f"{entry['pending_prompts']} prompt(s) pending: python3 -m qual_sg.llm_exchange status")
            return 10
        if entry["status"] == "failed":
            return 1
    print(json.dumps({k: v.get("status") for k, v in state["stages"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
