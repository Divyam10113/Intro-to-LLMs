"""Manage the file-backend LLM exchange (``QUAL_SG_BACKEND=file``).

Pending prompts live in ``<run>/intermediates/llm_exchange/pending/<key>.json``.
This tool exports them in batches for whoever answers them (in this project:
Claude models run as Claude Code subagents) and ingests the answers, recording
the exact model id that produced each one.

    python3 -m qual_sg.llm_exchange status
    python3 -m qual_sg.llm_exchange export --stage judge --size 60 --out <dir>
    python3 -m qual_sg.llm_exchange ingest <answers.jsonl> --model claude-sonnet-5-5 \
        --answered-by "Claude Code subagent (Claude Sonnet 5.5)"

An answers file holds one JSON object per line: ``{"key": ..., "text": ...}``.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from . import common, config

PENDING = config.LLM_EXCHANGE_DIR / "pending"
ANSWERS = config.LLM_EXCHANGE_DIR / "answers"


def pending_items() -> List[Dict[str, Any]]:
    return [common.read_json(p) for p in sorted(PENDING.glob("*.json"))] if PENDING.exists() else []


def status() -> Dict[str, Any]:
    items = pending_items()
    by = Counter((i.get("stage"), i.get("tag"), i.get("model_requested")) for i in items)
    report = {
        "pending": len(items),
        "answered": len(list(ANSWERS.glob("*.json"))) if ANSWERS.exists() else 0,
        "by_stage_tag_model": [{"stage": s, "tag": t, "model": m, "n": n} for (s, t, m), n in sorted(by.items(), key=str)],
    }
    print(json.dumps(report, indent=2))
    return report


def export(stage: str, size: int, out: Path, tag: str = "") -> List[Path]:
    items = [i for i in pending_items() if i.get("stage") == stage and (not tag or i.get("tag") == tag)]
    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for n, start in enumerate(range(0, len(items), size), start=1):
        path = out / f"{stage}{('_' + tag) if tag else ''}_batch{n:02d}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for item in items[start:start + size]:
                fh.write(json.dumps({"key": item["key"], "json_mode": item.get("json_mode", False),
                                     "messages": item["messages"]}, ensure_ascii=False) + "\n")
        paths.append(path)
    print(json.dumps({"stage": stage, "tag": tag, "exported": len(items), "batches": [str(p) for p in paths]}, indent=2))
    return paths


def ingest(answers_file: Path, model: str, answered_by: str) -> Dict[str, Any]:
    pending = {i["key"]: i for i in pending_items()}
    written, unknown, empty = 0, [], []
    for line in answers_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key, text = row.get("key", ""), row.get("text")
        if key not in pending:
            unknown.append(key)
            continue
        if text is None or (isinstance(text, str) and not text.strip()):
            empty.append(key)
            continue
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        common.write_json(ANSWERS / f"{key}.json", {
            "text": text, "model": model, "answered_by": answered_by,
            "stage": pending[key].get("stage"), "tag": pending[key].get("tag"),
            "model_requested": pending[key].get("model_requested"), "ingested": time.time(),
        })
        written += 1
    report = {"file": str(answers_file), "written": written, "unknown_keys": unknown, "empty": empty}
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="file-backend LLM exchange")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    ex = sub.add_parser("export")
    ex.add_argument("--stage", required=True)
    ex.add_argument("--tag", default="")
    ex.add_argument("--size", type=int, default=60)
    ex.add_argument("--out", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("answers")
    ing.add_argument("--model", required=True)
    ing.add_argument("--answered-by", default="")
    args = ap.parse_args()
    if args.cmd == "status":
        status()
    elif args.cmd == "export":
        export(args.stage, args.size, Path(args.out), args.tag)
    else:
        ingest(Path(args.answers), args.model, args.answered_by)


if __name__ == "__main__":
    main()
