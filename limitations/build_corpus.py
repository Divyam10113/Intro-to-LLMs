"""Step 1 - build the evidence corpus for the base paper (MVFA, arXiv:2403.12570).

Sources (following the Citation Agent of arXiv:2601.11578 and the RAG indices of BAGELS):
  * Cited-by : papers citing MVFA (OpenAlex), full text from arXiv when available
  * Cited-in : papers MVFA cites (OpenAlex referenced_works), full text from arXiv
  * Task 1   : the literature-review pipeline's selected papers (abstracts)

Citing papers that explicitly discuss MVFA are split in two (fixed seed):
  * evidence half -> goes into the RAG index
  * held-out half -> NOT indexed; their MVFA passages become the reference set for evaluation
This replaces OpenReview reviews, which a CVPR paper does not have.

Outputs in limitations/data/:
  papers.json, chunks.jsonl (evidence index), heldout_mentions.jsonl, mvfa_mentions.jsonl
No LLM calls happen here.
"""
import argparse
import csv
import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

from common import (BASE_OPENALEX_ID, BASE_TITLE, DATA_DIR, TASK1_DIR, base_paper_text, norm,
                    title_key, write_json, write_jsonl)

OPENALEX = "https://api.openalex.org"
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)
HTTP_CACHE = DATA_DIR / "http_cache"
HTTP_CACHE.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "intro-to-llms-course-project (limitation extraction)"


# -----------------------------------------------------------------------------
# HTTP with on-disk cache (OpenAlex and arXiv are both rate limited)
# -----------------------------------------------------------------------------
def get_json(url, params=None, pause=0.0):
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    cp = HTTP_CACHE / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    for attempt in range(4):
        r = SESSION.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        cp.write_text(r.text)
        time.sleep(pause)
        return r.json()
    raise RuntimeError(f"rate limited: {url}")


def get_text(url, pause=3.0):
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    cp = HTTP_CACHE / f"{key}.txt"
    if cp.exists():
        return cp.read_text()
    for attempt in range(4):
        r = SESSION.get(url, timeout=60)
        if r.status_code in (429, 503):
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        cp.write_text(r.text)
        time.sleep(pause)  # arXiv API asks for >= 3 s between calls
        return r.text
    raise RuntimeError(f"rate limited: {url}")


# -----------------------------------------------------------------------------
# OpenAlex
# -----------------------------------------------------------------------------
FIELDS = "id,title,publication_year,doi,ids,locations,abstract_inverted_index,type"


def abstract_from_index(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def arxiv_from_locations(work):
    for loc in work.get("locations") or []:
        for u in (loc.get("landing_page_url") or "", loc.get("pdf_url") or ""):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
            if m:
                return m.group(1)
    doi = (work.get("doi") or "").lower()
    m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi)
    return m.group(1) if m else None


def to_paper(work, group):
    return {
        "openalex_id": work["id"].rsplit("/", 1)[-1],
        "title": work.get("title") or "",
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "arxiv_id": arxiv_from_locations(work),
        "abstract": abstract_from_index(work.get("abstract_inverted_index")),
        "oa_pdfs": [l["pdf_url"] for l in work.get("locations") or [] if l.get("pdf_url")],
        "group": group,
    }


def fetch_citing():
    out, cursor = [], "*"
    while cursor:
        d = get_json(f"{OPENALEX}/works", {"filter": f"cites:{BASE_OPENALEX_ID}", "per_page": 200,
                                           "cursor": cursor, "select": FIELDS})
        out += [to_paper(w, "cited_by") for w in d["results"]]
        cursor = d["meta"].get("next_cursor") if d["results"] else None
    return out


def fetch_cited():
    refs = get_json(f"{OPENALEX}/works/{BASE_OPENALEX_ID}", {"select": "referenced_works"})["referenced_works"]
    ids = [r.rsplit("/", 1)[-1] for r in refs]
    out = []
    for i in range(0, len(ids), 50):
        d = get_json(f"{OPENALEX}/works", {"filter": "openalex_id:" + "|".join(ids[i:i + 50]),
                                           "per_page": 50, "select": FIELDS})
        out += [to_paper(w, "cited_in") for w in d["results"]]
    return out


def load_task1():
    p = TASK1_DIR / "item1_selected_papers.csv"
    if not p.exists():
        print(f"[!] Task 1 file missing: {p}")
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = (re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", row.get("arxiv_url") or "") or
                 re.search(r"arxiv\.(\d{4}\.\d{4,5})", (row.get("doi") or "").lower()))
            out.append({"openalex_id": None, "title": row["title"], "year": _int(row.get("year")),
                        "doi": row.get("doi"), "arxiv_id": m.group(1) if m else None,
                        "abstract": row.get("abstract") or "", "group": "task1"})
    return out


def _int(x):
    try:
        return int(str(x)[:4])
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# arXiv resolution + full text
# -----------------------------------------------------------------------------
def resolve_arxiv(title):
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 2][:10]
    if not words:
        return None
    query = " AND ".join(f"ti:{w}" for w in words)
    xml = get_text("http://export.arxiv.org/api/query?search_query=" + requests.utils.quote(query) +
                   "&max_results=5")
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.search(r"<title>(.*?)</title>", entry, re.S)
        i = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5})", entry)
        if t and i and SequenceMatcher(None, norm(t.group(1)), norm(title)).ratio() > 0.9:
            return i.group(1)
    return None


def download_pdf(arxiv_id=None, url=None, name=None):
    p = PDF_DIR / f"{arxiv_id or name}.pdf"
    if p.exists() and p.stat().st_size > 10_000:
        return p
    try:
        r = SESSION.get(url or f"https://arxiv.org/pdf/{arxiv_id}", timeout=120)
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return None
    p.write_bytes(r.content)
    time.sleep(3)
    return p


def pdf_text(path):
    import fitz  # PyMuPDF
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


HEADING = re.compile(
    r"^\s*(?:(\d{1,2}(?:\.\d{1,2})?)\.?\s+([A-Z][A-Za-z0-9 ,:&\-/()]{2,80})|"
    r"(Abstract|Introduction|Related Work[s]?|Background|Method(?:s|ology)?|Approach|Experiments?|"
    r"Results|Discussion|Conclusions?|Limitations?|References|Bibliography|Acknowledg(?:e)?ments?))\s*$",
    re.I)


def split_sections(text):
    """Split PDF text into (heading, body) by numbered / well-known headings. Stops at References."""
    sections, cur, buf = [], "Front matter", []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m and len(line.strip()) < 90:
            name = (m.group(2) or m.group(3) or "").strip()
            if sections or buf:
                sections.append((cur, "\n".join(buf)))
            if re.match(r"references|bibliography", name, re.I):
                return sections, text[text.find(line):]
            cur, buf = (f"{m.group(1)} {name}" if m.group(1) else name), []
        else:
            buf.append(line)
    sections.append((cur, "\n".join(buf)))
    return sections, ""


def clean(t):
    t = re.sub(r"-\n(?=[a-z])", "", t)            # de-hyphenate
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def chunk_words(text, max_words=450, overlap=60):
    w = text.split()
    if len(w) <= max_words:
        return [text] if w else []
    out, i = [], 0
    while i < len(w):
        out.append(" ".join(w[i:i + max_words]))
        i += max_words - overlap
    return out


# -----------------------------------------------------------------------------
# Finding where a citing paper discusses MVFA
# -----------------------------------------------------------------------------
MVFA_NAME = re.compile(r"\bMVFA(?:-AD)?\b|multi-level (?:visual )?feature adapt|Huang et al\.?,? (?:\(?2024|\[)", re.I)


def mvfa_ref_number(refs_text):
    """Numeric label of MVFA in the citing paper's bibliography, e.g. '[23]' -> '23'."""
    flat = clean(refs_text)
    # "[12] Huang, C. ..." (IEEE/CVPR) or "12. Huang, C. ..." (LNCS/Elsevier)
    for pattern in (r"\[(\d{1,3})\]", r"(?:^|\s)(\d{1,3})\.\s+(?=[A-Z])"):
        parts = re.split(pattern, flat)
        for label, entry in zip(parts[1::2], parts[2::2]):
            if "generalizable anomaly detection in medical" in norm(entry):
                return label
    return None


def find_mentions(sections, refs_text):
    num = mvfa_ref_number(refs_text)
    num_pat = re.compile(r"\[[^\]]*\b" + num + r"\b[^\]]*\]") if num else None
    hits = []
    for name, body in sections:
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", clean(body))
        for k, s in enumerate(sents):
            if MVFA_NAME.search(s) or (num_pat and num_pat.search(s)):
                window = " ".join(sents[max(0, k - 1): k + 3])
                hits.append({"section": name, "text": window})
    # merge overlapping windows
    merged, seen = [], set()
    for h in hits:
        key = h["text"][:200]
        if key not in seen:
            seen.add(key)
            merged.append(h)
    return merged


# -----------------------------------------------------------------------------
# Relevance pre-filter (cheap, no LLM): TF-IDF cosine to the base paper abstract+intro
# -----------------------------------------------------------------------------
def relevance(papers, base_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [base_text] + [(p["title"] + ". " + p["abstract"]) for p in papers]
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    sims = (X[1:] @ X[0].T).toarray().ravel()
    for p, s in zip(papers, sims):
        p["relevance"] = round(float(s), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cited_by", type=int, default=70, help="citing papers to fetch full text for")
    ap.add_argument("--max_cited_in", type=int, default=25, help="cited papers to fetch full text for")
    ap.add_argument("--heldout_frac", type=float, default=0.5)
    ap.add_argument("--seed", default="mvfa")
    args = ap.parse_args()

    print("[1] OpenAlex: citing + cited papers")
    cited_by, cited_in = fetch_citing(), fetch_cited()
    task1 = load_task1()
    print(f"    cited_by={len(cited_by)} cited_in={len(cited_in)} task1={len(task1)}")

    # de-duplicate by title (a Task-1 paper is often also a citing paper)
    papers, by_title = [], {}
    for p in cited_by + cited_in + task1:
        k = title_key(p["title"])
        if not k or k == title_key(BASE_TITLE):
            continue
        if k in by_title:
            q = by_title[k]
            q["groups"] = sorted(set(q["groups"]) | {p["group"]})
            q["arxiv_id"] = q["arxiv_id"] or p["arxiv_id"]
            q["abstract"] = q["abstract"] or p["abstract"]
            continue
        p["groups"] = [p.pop("group")]
        by_title[k] = p
        papers.append(p)

    base_head = base_paper_text(max_chars=6000)
    relevance(papers, base_head)
    for i, p in enumerate(papers):
        p["pid"] = f"P{i:03d}"

    def pick(group, n):
        cand = [p for p in papers if group in p["groups"]]
        return sorted(cand, key=lambda p: -p["relevance"])[:n]

    full = {p["pid"]: p for p in pick("cited_by", args.max_cited_by) + pick("cited_in", args.max_cited_in)}
    # every Task-1 paper that cites MVFA is worth reading in full too
    for p in papers:
        if "task1" in p["groups"] and "cited_by" in p["groups"]:
            full[p["pid"]] = p

    print(f"[2] resolving + downloading full text for {len(full)} papers (arXiv, cached)")
    chunks, mentions = [], []
    for n, p in enumerate(full.values(), 1):
        if not p["arxiv_id"]:
            try:
                p["arxiv_id"] = resolve_arxiv(p["title"])
            except Exception as e:  # noqa: BLE001
                print(f"    arXiv lookup failed for {p['title'][:60]}: {e}")
        pdf = download_pdf(p["arxiv_id"]) if p["arxiv_id"] else None
        for url in ([] if pdf else p.get("oa_pdfs", [])):   # open-access copy from OpenAlex
            pdf = download_pdf(url=url, name=p["openalex_id"] or p["pid"])
            if pdf:
                break
        if not pdf:
            continue
        try:
            sections, refs = split_sections(pdf_text(pdf))
        except Exception as e:  # noqa: BLE001
            print(f"    could not parse {pdf.name}: {e}")
            continue
        p["full_text"] = True
        if "cited_by" in p["groups"]:
            for m in find_mentions(sections, refs):
                mentions.append({"pid": p["pid"], "title": p["title"], "year": p["year"], **m})
        for name, body in sections:
            for j, piece in enumerate(chunk_words(clean(body))):
                if len(piece.split()) >= 40:
                    chunks.append({"pid": p["pid"], "section": name, "part": j, "text": piece})
        print(f"    ({n}/{len(full)}) {p['arxiv_id'] or pdf.name}  {p['title'][:70]}", flush=True)

    # abstracts for everything (cheap context for papers without full text)
    for p in papers:
        if p["abstract"]:
            chunks.append({"pid": p["pid"], "section": "Abstract", "part": -1, "text": p["abstract"]})

    # ---- held-out split over citing papers that actually discuss MVFA ----
    discussing = sorted({m["pid"] for m in mentions})
    by_hash = sorted(discussing, key=lambda pid: hashlib.sha256((args.seed + pid).encode()).hexdigest())
    heldout = set(by_hash[:round(args.heldout_frac * len(by_hash))])
    for p in papers:
        p["split"] = "heldout" if p["pid"] in heldout else "evidence"
        p["discusses_mvfa"] = p["pid"] in discussing

    evidence_chunks = [c for c in chunks if c["pid"] not in heldout]
    for i, c in enumerate(evidence_chunks):
        c["cid"] = f"C{i:04d}"
    pmap = {p["pid"]: p for p in papers}
    for c in evidence_chunks:
        p = pmap[c["pid"]]
        c["title"], c["year"], c["groups"] = p["title"], p["year"], p["groups"]

    write_json(DATA_DIR / "papers.json", papers)
    write_jsonl(DATA_DIR / "chunks.jsonl", evidence_chunks)
    write_jsonl(DATA_DIR / "mvfa_mentions.jsonl", [m for m in mentions if m["pid"] not in heldout])
    write_jsonl(DATA_DIR / "heldout_mentions.jsonl", [m for m in mentions if m["pid"] in heldout])

    print("\n[done]")
    print(f"    papers: {len(papers)} | full text: {sum(1 for p in papers if p.get('full_text'))}")
    print(f"    citing papers discussing MVFA: {len(discussing)} "
          f"-> evidence {len(discussing) - len(heldout)}, held-out {len(heldout)}")
    print(f"    evidence chunks: {len(evidence_chunks)} | MVFA passages (evidence / held-out): "
          f"{sum(m['pid'] not in heldout for m in mentions)} / {sum(m['pid'] in heldout for m in mentions)}")


if __name__ == "__main__":
    main()
