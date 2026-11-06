"""Import an external agent's reference selection into the pipeline's ranked CSV.

Deep-research agents (ChatGPT Deep Research, Gemini, Perplexity, "Argon", …)
produce prose, Markdown tables or loose JSON — not the pipeline's
``paper_after_reranking.csv``.  This module converts whatever they returned into
that exact 28-column schema:

* rows are joined back to the candidate pool (``final_ref_set.json``) by DOI and
  by normalised title, so citation counts, author/venue h-index and provenance
  come from the pipeline rather than from the agent;
* relevance scores are taken from the cached judge output
  (``step3_relevance_scores.jsonl``) when present, otherwise from the agent's own
  column if it provided one;
* the binning, ``impact_score``, ranks, ``composite_score`` and columns that need
  the paper embeddings are recomputed with the pipeline's own functions, using
  ``step3_embeddings.npz`` when available.

    python3 -m qual_sg.import_ranking --input argon_output.md --out paper_after_reranking.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import common, config
from .step3_rerank import (
    CSV_COLUMNS,
    map_bin_q1_best,
    minmax01,
    percentile_bin_q1_best,
)

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s|,;)\]\"'<>]+")


# --------------------------------------------------------------------------- #
# Parsing the agent's output
# --------------------------------------------------------------------------- #
def _clean_cell(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # markdown links
    return " ".join(text.split()).strip(" |\t")


def _row_from_text(text: str) -> Optional[Dict[str, str]]:
    text = _clean_cell(text)
    if len(text) < 8:
        return None
    doi = ""
    m = DOI_RE.search(text)
    if m:
        doi = common.norm_doi(m.group(0))
        text = text.replace(m.group(0), " ")
    # drop leading numbering / bullets / years / marker columns
    text = re.sub(r"^\s*[-*\u2022]?\s*\d{1,3}[.)]\s*", "", text)
    text = re.sub(r"^\s*[-*\u2022]+\s*", "", text)
    cells = [c for c in (_clean_cell(c) for c in text.split("|")) if c]
    numeric = re.compile(r"^[\d.,%\s-]+$")
    # drop a leading row-number cell so it is not mistaken for a score
    if len(cells) >= 3 and numeric.match(cells[0]) and len(cells[0]) <= 4:
        cells = cells[1:]
    title_bits = [c for c in cells if not numeric.match(c) and len(c) > 8]
    title = max(title_bits, key=len) if title_bits else ""

    # agent-supplied relevance: the last small integer 1..5 (never a 4-digit year)
    relevance = None
    for c in reversed(cells):
        if not numeric.match(c):
            continue
        for tok in reversed(re.findall(r"\d+(?:\.\d+)?", c)):
            val = float(tok)
            if val.is_integer() and 1 <= val <= 5:
                relevance = int(val)
                break
        if relevance is not None:
            break

    if not doi and not title:
        return None
    tag = "selected" if re.search(r"\bselected\b", text, re.IGNORECASE) else None
    return {"title": _clean_cell(title), "doi": doi, "relevance": relevance, "tag": tag}


def parse_input(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("references") or data.get("papers") or data.get("rows") or []
        rows = []
        for item in data:
            if isinstance(item, dict):
                rows.append({"title": _clean_cell(str(item.get("title", ""))),
                             "doi": common.norm_doi(item.get("doi")),
                             "relevance": item.get("relevance_llm_1to5") or item.get("relevance") or item.get("score")})
            elif isinstance(item, str):
                r = _row_from_text(item)
                if r:
                    rows.append(r)
        return rows

    if suffix == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("title") or next((c for c in df.columns if "title" in c.lower()), None)
        dcol = cols.get("doi") or next((c for c in df.columns if "doi" in c.lower()), None)
        rcol = next((c for c in df.columns if "relevan" in c.lower() or c.lower() == "score"), None)
        gcol = next((c for c in df.columns if c.lower() == "tag"), None)
        rows = []
        for _, r in df.iterrows():
            rows.append({"title": _clean_cell(str(r.get(tcol, ""))) if tcol else "",
                         "doi": common.norm_doi(r.get(dcol)) if dcol else "",
                         "relevance": r.get(rcol) if rcol else None,
                         "tag": (str(r.get(gcol)).strip() if gcol and pd.notna(r.get(gcol)) else None)})
        return [r for r in rows if r["title"] or r["doi"]]

    # markdown / plain text
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if not stripped or set(stripped) <= set("-|: "):              # table rule
            continue
        if stripped.startswith("#"):                                  # markdown heading
            continue
        if "title" in line.lower() and "doi" in line.lower():          # header row
            continue
        r = _row_from_text(line)
        if r:
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# Joining + scoring
# --------------------------------------------------------------------------- #
def load_pool() -> Dict[str, Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    if not config.CANONICAL_REF_SET.exists():
        return pool
    for rec in common.read_json(config.CANONICAL_REF_SET):
        for key in (common.norm_doi(rec.get("doi")), common.normalize_text(rec.get("title", ""))):
            if key:
                pool[key] = rec
    return pool


def load_scores() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in common.read_jsonl(config.STEP3_SCORES):
        key = str(row.get("key") or common.norm_doi(row.get("doi")))
        if key:
            out[key] = row.get("score")
    return out


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    pool, scores = load_pool(), load_scores()
    records = []
    for i, row in enumerate(rows, start=1):
        doi = common.norm_doi(row.get("doi"))
        title = row.get("title", "")
        rec = pool.get(doi) or pool.get(common.normalize_text(title)) or {}
        score = scores.get(doi) or scores.get("title:" + common.sha256_hex(str(rec.get("title") or title))[:16])
        agent_rel = _as_float(row.get("relevance"))
        relevance = score if score is not None else (agent_rel if not np.isnan(agent_rel) else np.nan)
        records.append({
            "title": rec.get("title") or title,
            "doi": doi or common.norm_doi(rec.get("doi")),
            "abstract": rec.get("abstract", ""),
            "origin": rec.get("origin", "external"),
            "cited_by_anchor": bool(rec.get("cited_by_anchor", False)),
            "co_cited_count": rec.get("co_cited_count", 0),
            "cited_by_count": rec.get("cited_by_count"),
            "publication_date": rec.get("publication_date", ""),
            "first_author_hindex": rec.get("first_author_hindex"),
            "last_author_hindex": rec.get("last_author_hindex"),
            "journal_hindex": rec.get("journal_hindex"),
            "relevance_llm_1to5": relevance,
            "_agent_tag": row.get("tag"),
            "_order": i,
            "_in_pool": bool(rec),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["pub_year"] = pd.to_datetime(df["publication_date"], errors="coerce", format="mixed").dt.year
    age = (pd.Timestamp.utcnow().year - df["pub_year"]).clip(lower=0) + 1
    df["citations_per_year"] = pd.to_numeric(df["cited_by_count"], errors="coerce") / age
    df["author_hindex_avg"] = df[["first_author_hindex", "last_author_hindex"]].apply(
        lambda r: np.nanmean([v for v in r if pd.notna(v)]) if any(pd.notna(v) for v in r) else np.nan, axis=1)

    df["cited_by_bin"] = percentile_bin_q1_best(df["citations_per_year"] if config.CITE_AGE_NORMALIZE else df["cited_by_count"])
    df["author_h_bin"] = percentile_bin_q1_best(df["author_hindex_avg"])
    df["journal_h_bin"] = percentile_bin_q1_best(pd.to_numeric(df["journal_hindex"], errors="coerce"))
    df["relevance_bin"] = percentile_bin_q1_best(pd.to_numeric(df["relevance_llm_1to5"], errors="coerce"))

    cited = minmax01(map_bin_q1_best(df["cited_by_bin"]))
    author = minmax01(map_bin_q1_best(df["author_h_bin"]))
    venue = minmax01(map_bin_q1_best(df["journal_h_bin"]))
    w = config.W_CITED + config.W_AUTHOR + config.W_VENUE
    df["impact_score"] = (config.W_CITED * cited + config.W_AUTHOR * author + config.W_VENUE * venue) / w

    rel_norm = minmax01(pd.to_numeric(df["relevance_llm_1to5"], errors="coerce"))
    div_norm = pd.Series(np.nan, index=df.index)
    if config.STEP3_EMBEDDINGS.exists():
        try:
            npz = np.load(config.STEP3_EMBEDDINGS, allow_pickle=True)
            emb_dois = [common.norm_doi(d) for d in npz["dois"]]
            emb = npz["embeddings"].astype(np.float32)
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
            idx = {d: k for k, d in enumerate(emb_dois)}
            keep = [idx[d] for d in df["doi"] if d in idx]
            if len(keep) > 1:
                sub = emb[keep]
                sim = sub @ sub.T
                avg_sim = [(sim[i].sum() - 1.0) / (len(sub) - 1) for i in range(len(sub))]
                col = pd.Series(np.nan, index=df.index)
                col.loc[[i for i, d in zip(df.index, df["doi"]) if d in idx]] = avg_sim
                df["avg_cos_sim_to_others"] = col
                df["avg_cos_dist_to_others"] = 1.0 - col
                div_norm = 1.0 - minmax01(col)
        except Exception:  # noqa: BLE001 - diversity stays empty if embeddings are unavailable
            pass
    for col in ("avg_cos_sim_to_others", "avg_cos_dist_to_others"):
        if col not in df.columns:
            df[col] = np.nan

    df["rank_relevance"] = pd.to_numeric(df["relevance_llm_1to5"], errors="coerce").rank(ascending=False, method="average")
    df["rank_impact"] = df["impact_score"].rank(ascending=False, method="average")
    df["rank_diversity"] = df["avg_cos_dist_to_others"].rank(ascending=False, method="average")
    df["avg_rank"] = df[["rank_relevance", "rank_impact", "rank_diversity"]].mean(axis=1, skipna=True)

    weights = np.array([config.W_CITED, config.W_AUTHOR, config.W_VENUE, config.W_RELEVANCE, config.W_DIVERSITY])
    parts = np.column_stack([
        cited.to_numpy(dtype=float), author.to_numpy(dtype=float), venue.to_numpy(dtype=float),
        rel_norm.to_numpy(dtype=float), div_norm.to_numpy(dtype=float),
    ])

    def row_score(vals: np.ndarray) -> float:
        mask = ~np.isnan(vals)
        if not mask.any():
            return np.nan
        return float((vals[mask] * weights[mask]).sum() / weights[mask].sum())

    df["composite_score"] = [row_score(v) for v in parts]
    df["diversity_score"] = div_norm
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert an agent's reference list into the pipeline's ranked CSV")
    ap.add_argument("--input", required=True, help="agent output (.md/.txt/.csv/.json)")
    ap.add_argument("--out", default=None, help="output CSV (default: paper_after_reranking_imported.csv)")
    ap.add_argument("--no-select", action="store_true", help="leave the tag column empty instead of marking every row selected")
    ap.add_argument("--limit", type=int, default=None, help="mark only the top-N rows as selected")
    args = ap.parse_args()

    config.ensure_dirs()
    rows = parse_input(Path(args.input))
    if not rows:
        raise SystemExit(f"no candidate rows found in {args.input}")
    df = build(rows)
    out = Path(args.out) if args.out else (config.BASE_DIR / "paper_after_reranking_imported.csv")

    df["tag"] = None
    agent_tags = df["_agent_tag"].astype("string")
    if agent_tags.notna().any() and agent_tags.str.strip().ne("").any():
        # the agent marked its own selection -> preserve it verbatim
        df["tag"] = agent_tags.where(agent_tags.str.strip().ne(""), None)
        n_sel = int((df["tag"].astype("string").str.lower() == "selected").sum())
    else:
        n_sel = len(df) if not args.no_select else 0
        if args.limit:
            n_sel = min(args.limit, len(df))
        if n_sel:
            df.loc[df["avg_rank"].sort_values(kind="mergesort").index[:n_sel], "tag"] = "selected"

    printable = df.reindex(columns=list(CSV_COLUMNS)).astype(object)
    printable = printable.where(pd.notna(printable), "")
    printable.to_csv(out, index=False)
    matched = int(df["_in_pool"].sum())
    print(f"parsed {len(rows)} rows -> {out}")
    print(f"  matched to the candidate pool: {matched}/{len(df)}")
    print(f"  selected (tag=selected): {n_sel}")
    print(f"  columns: {len(CSV_COLUMNS)} (pipeline schema)")


if __name__ == "__main__":
    main()
