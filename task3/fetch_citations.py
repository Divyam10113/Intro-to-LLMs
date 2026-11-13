#!/usr/bin/env python3
"""
fetch_citations.py — forward-citation supplement for the HypER stage (run this on your own machine).

What it does (mirrors Step 2 of HypER, "Citation Graph Retrieval using the Semantic Scholar API"):
  1. Resolves the base paper (MVFA, arXiv:2403.12570) on Semantic Scholar.
  2. Hop 1: every paper that cites MVFA, with year, abstract, venue, citation count, and the
     citation *contexts* (the sentences in which MVFA is cited) and *intents* (background /
     methodology / result). Contexts are very useful evidence for HypER's 0/1/2 link scoring.
  3. Hop 2: for the most-cited hop-1 papers, the papers that cite *them* (same fields), so chains
     can be longer than one step: MVFA -> A -> B.
  4. MVFA's own references (backward), so the source paper has context too.
  5. Publication years for the 341 papers of our SurveyGen corpus (final_ref_set.json), which
     OpenAlex did not include.

Only the Python standard library is used, so no install is needed. No API key is required
(an optional key can be given with S2_API_KEY to get higher rate limits).

Usage (from the folder containing this script and final_ref_set.json):
    python3 fetch_citations.py
    python3 fetch_citations.py --hop2-top 40 --corpus final_ref_set.json --out s2_citations.json

It takes roughly 2-5 minutes because the public API is rate limited (~1 request/second).
Upload the resulting s2_citations.json back to the chat.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1"
BASE_ID = "arXiv:2403.12570"  # MVFA: Adapting Visual-Language Models for Generalizable AD in Medical Images
PAPER_FIELDS = "paperId,title,year,publicationDate,abstract,venue,citationCount,externalIds,url"
EDGE_FIELDS = "contexts,intents,isInfluential"
HEADERS = {"User-Agent": "task3-research-ideation/1.0"}
if os.environ.get("S2_API_KEY"):
    HEADERS["x-api-key"] = os.environ["S2_API_KEY"]

_last = [0.0]


def request(url: str, data: dict | None = None, retries: int = 8):
    """GET (or POST JSON) with polite pacing and exponential backoff on 429/5xx."""
    delay = 3.0
    for attempt in range(retries):
        wait = 1.1 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        body = json.dumps(data).encode() if data is not None else None
        hdrs = dict(HEADERS, **({"Content-Type": "application/json"} if body else {}))
        req = urllib.request.Request(url, data=body, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                print(f"  HTTP {e.code}, retrying in {delay:.0f}s ...", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                print(f"  network error {e}, retrying in {delay:.0f}s ...", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            raise


def citations_of(paper_id: str, cap: int = 2000) -> list[dict]:
    out, offset = [], 0
    while offset < cap:
        q = urllib.parse.urlencode({"fields": f"{EDGE_FIELDS},{PAPER_FIELDS}", "offset": offset, "limit": 500})
        res = request(f"{API}/paper/{paper_id}/citations?{q}")
        if not res or not res.get("data"):
            break
        for item in res["data"]:
            p = item.get("citingPaper") or {}
            if not p.get("paperId"):
                continue
            p["citation_contexts"] = item.get("contexts") or []
            p["citation_intents"] = item.get("intents") or []
            p["is_influential"] = item.get("isInfluential")
            out.append(p)
        if res.get("next") is None:
            break
        offset = res["next"]
    return out


def references_of(paper_id: str) -> list[dict]:
    q = urllib.parse.urlencode({"fields": PAPER_FIELDS, "limit": 1000})
    res = request(f"{API}/paper/{paper_id}/references?{q}") or {}
    return [i["citedPaper"] for i in res.get("data", []) if (i.get("citedPaper") or {}).get("paperId")]


def corpus_ids(corpus_path: str) -> list[tuple[str, str]]:
    """Map each SurveyGen record to a Semantic Scholar lookup id (arXiv id or DOI)."""
    recs = json.load(open(corpus_path))
    ids = []
    for r in recs:
        doi = (r.get("doi") or "").replace("https://doi.org/", "").strip()
        if not doi:
            continue
        if doi.lower().startswith("10.48550/arxiv."):
            ids.append((doi, "ARXIV:" + doi.split("arxiv.", 1)[1]))
        else:
            ids.append((doi, "DOI:" + doi))
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hop2-top", type=int, default=40, help="expand this many most-cited hop-1 papers (default 40)")
    ap.add_argument("--corpus", default="final_ref_set.json", help="SurveyGen corpus file, for publication years")
    ap.add_argument("--out", default="s2_citations.json")
    args = ap.parse_args()

    print(f"[1/5] resolving base paper {BASE_ID}")
    base = request(f"{API}/paper/{BASE_ID}?fields={PAPER_FIELDS}")
    if not base:
        sys.exit("could not resolve the base paper on Semantic Scholar")
    print(f"      {base['title']} ({base.get('year')}), cited by {base.get('citationCount')}")

    print("[2/5] hop 1: papers citing the base paper")
    hop1 = citations_of(base["paperId"])
    print(f"      {len(hop1)} citing papers")

    print(f"[3/5] hop 2: papers citing the {args.hop2_top} most-cited hop-1 papers")
    hop2 = {}
    for i, p in enumerate(sorted(hop1, key=lambda x: x.get("citationCount") or 0, reverse=True)[: args.hop2_top], 1):
        hop2[p["paperId"]] = citations_of(p["paperId"], cap=1000)
        print(f"      {i:2d}/{args.hop2_top}  {len(hop2[p['paperId']]):4d} citing  <- {p['title'][:70]}")

    print("[4/5] references of the base paper")
    refs = references_of(base["paperId"])
    print(f"      {len(refs)} references")

    years = {}
    if os.path.exists(args.corpus):
        ids = corpus_ids(args.corpus)
        print(f"[5/5] publication years for {len(ids)} SurveyGen corpus papers")
        for s in range(0, len(ids), 400):
            chunk = ids[s:s + 400]
            res = request(f"{API}/paper/batch?fields=paperId,year,publicationDate,citationCount",
                          data={"ids": [c[1] for c in chunk]}) or []
            for (doi, _), r in zip(chunk, res):
                if r:
                    years[doi] = {"year": r.get("year"), "publicationDate": r.get("publicationDate"),
                                  "s2_paperId": r.get("paperId"), "s2_citationCount": r.get("citationCount")}
        print(f"      found {len(years)} of {len(ids)}")
    else:
        print(f"[5/5] skipped: {args.corpus} not found (put final_ref_set.json next to this script)")

    out = dict(retrieved=time.strftime("%Y-%m-%d %H:%M:%S"), source="Semantic Scholar Academic Graph API",
               base=base, hop1=hop1, hop2=hop2, references=refs, corpus_years=years,
               settings=dict(hop2_top=args.hop2_top))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    n2 = sum(len(v) for v in hop2.values())
    print(f"\nDone. Wrote {args.out}: {len(hop1)} hop-1, {n2} hop-2 records, {len(refs)} references, {len(years)} corpus years.")
    print("Please upload this file to the chat.")


if __name__ == "__main__":
    main()
