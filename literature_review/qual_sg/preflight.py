"""Pre-run check: are the literature services reachable, and is there OpenAlex budget?

Calls each service once with a known paper (MVFA) and reads OpenAlex's
``/rate-limit`` endpoint.  Exit codes:

    0  everything required is up and the OpenAlex budget suffices
    2  a required service is down, or no OpenAlex key is configured
    3  OpenAlex daily budget too low; ``resets_in_seconds`` says when it refills

    python3 -m qual_sg.preflight [--min-budget 0.25]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

from . import common, config

MVFA_DOI = "10.1109/cvpr52733.2024.01081"
MVFA_ARXIV = "2403.12570"


def check(name: str, required: bool, fn) -> Dict[str, Any]:
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    return {"service": name, "required": required, "ok": bool(ok), "detail": detail,
            "seconds": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="literature-service preflight")
    ap.add_argument("--min-budget", type=float, default=0.25, help="minimum OpenAlex USD left today")
    args = ap.parse_args()
    config.ensure_dirs()
    # Fresh responses: a preflight must not be answered from the response cache.
    config.HTTP_CACHE = False
    client = common.HttpClient(logger=common.get_logger("preflight"))
    key_present = bool(config.openalex_api_key())

    budget: Dict[str, Any] = {}

    def openalex():
        if not key_present:
            return False, "no OPENALEX_API_KEY in .env"
        data = client.get_json(f"{config.OPENALEX_BASE}/rate-limit") or {}
        budget.update(data.get("rate_limit") or {})
        work = client.get_json(f"{config.OPENALEX_BASE}/works/https://doi.org/{MVFA_DOI}") or {}
        return bool(work.get("id")), {k: budget.get(k) for k in ("daily_budget_usd", "daily_remaining_usd", "resets_in_seconds")}

    def crossref():
        data = client.get_json(f"https://api.crossref.org/works/{MVFA_DOI}", params={"mailto": config.OPENALEX_MAILTO}) or {}
        msg = data.get("message") or {}
        return bool(msg.get("title")), {"references": len(msg.get("reference") or [])}

    def opencitations():
        data = client.get_json(f"{config.OPENCITATIONS_BASE}/references/doi:{MVFA_DOI}")
        return isinstance(data, list) and len(data) > 0, {"references": len(data) if isinstance(data, list) else 0}

    def europepmc():
        data = client.get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                               params={"query": "medical anomaly detection CLIP", "format": "json", "pageSize": 5}) or {}
        return int(data.get("hitCount") or 0) > 0, {"hitCount": data.get("hitCount")}

    def arxiv():
        text = client.get_text("https://export.arxiv.org/api/query", params={"id_list": MVFA_ARXIV}) or ""
        return "<entry>" in text, {}

    results = [
        check("openalex", True, openalex),
        check("crossref", True, crossref),
        check("arxiv", True, arxiv),
        check("europepmc", True, europepmc),
        check("opencitations", False, opencitations),
    ]
    remaining = float(budget.get("daily_remaining_usd") or 0.0)
    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "openalex_key_configured": key_present,
        "openalex_budget": {k: budget.get(k) for k in ("daily_budget_usd", "daily_used_usd", "daily_remaining_usd",
                                                         "resets_at", "resets_in_seconds", "endpoint_costs_usd")},
        "services": results,
    }
    status = 0
    if any(r["required"] and not r["ok"] for r in results):
        status = 2
    elif key_present and remaining < args.min_budget:
        status = 3
    report["exit_code"] = status
    common.write_json(config.INTERMEDIATES_DIR / f"preflight_{int(time.time())}.json", report)
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
