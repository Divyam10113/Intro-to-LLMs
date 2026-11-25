#!/usr/bin/env python3
"""
novelty_search.py — re-run the Stage 4 novelty queries against the Semantic Scholar search API (run on your machine).

The novelty evidence used in this run (data/novelty_search.json) was collected with a web search tool because the
Semantic Scholar API was not reachable from the run environment. This script re-runs the same queries on
Semantic Scholar and writes data/novelty_search_s2.json with the top hits, so the evidence can be reproduced or
replaced. Standard library only.

    python3 novelty_search.py            # writes data/novelty_search_s2.json
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://api.semanticscholar.org/graph/v1/paper/search"


def search(q, limit=8):
    url = f"{API}?{urllib.parse.urlencode({'query': q, 'limit': limit, 'fields': 'title,year,url,abstract,citationCount'})}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "task3"}), timeout=60) as r:
                return json.loads(r.read().decode()).get("data", [])
        except Exception as e:  # rate limits
            time.sleep(3 * 2 ** attempt)
    return []


def main():
    src = json.loads((ROOT / "data" / "novelty_search.json").read_text())
    out = {}
    for cid, v in src.items():
        if cid.startswith("_"):
            continue
        hits = search(v["query"])
        out[cid] = dict(query=v["query"], hits=[dict(title=h["title"], year=h.get("year"), url=h.get("url"),
                                                     citations=h.get("citationCount")) for h in hits])
        print(cid, len(hits))
        time.sleep(1.2)
    (ROOT / "data" / "novelty_search_s2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
