"""
corpus.py — loads the literature used by all stages.

Sources
* data/paper_after_reranking.csv — the final Task 1 corpus from the QUAL-SG pipeline (259 papers with
  abstracts, years, LLM relevance and composite scores; 60 tagged 'selected'), and
  data/pipeline_ref.json — the 60 references of the submitted review (run 2).
  Run 1 used data/final_ref_set.json (SurveyGen/OpenAlex, 341 papers); see runs/run1_surveygen341/.
* data/s2_citations.json (optional)  — forward-citation supplement fetched with
  fetch_citations.py from the Semantic Scholar API (papers citing MVFA, hop 1 and hop 2,
  with years, abstracts and citation contexts) plus publication years for the SurveyGen corpus.

The papers of the submitted literature review are the "related papers" of ResearchAgent; the entity
store is built over the whole corpus.
"""
from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Entities that carry no information for idea generation (appear in most CS papers).
GENERIC = {
    "computer science", "artificial intelligence", "machine learning", "pattern recognition (psychology)",
    "data science", "mathematics", "engineering", "physics", "biology", "medicine", "psychology",
    "programming language", "operating system", "philosophy", "linguistics", "art", "geography",
    "economics", "law", "statistics", "algorithm", "key (lock)", "focus (optics)", "computer vision",
    "natural language processing", "deep learning", "artificial neural network", "computer security",
    "world wide web", "information retrieval", "epistemology", "chemistry", "quantum mechanics",
    "biochemistry", "geology", "astronomy", "computer network", "theoretical computer science",
    "set (abstract data type)", "task (project management)", "management", "process (computing)",
    "materials science", "embedding", "image (mathematics)", "construct (python library)",
    # wrong-sense links produced by OpenAlex's concept tagger (e.g. "generative" -> linguistics)
    "cartography", "generative grammar", "human–computer interaction", "remote sensing", "geodesy",
}
# OpenAlex disambiguates concepts with a parenthetical qualifier; almost all non-CS qualifiers in this
# corpus are wrong-sense links ("Pascal (unit)", "Context (archaeology)", "Field (mathematics)").
# We keep a parenthetical entity only if its qualifier is one of these CS/ML senses.
KEEP_QUALIFIERS = {"machine learning", "neural networks", "learning theory", "information retrieval",
                   "computer graphics", "vision", "xai", "computer science", "video"}
MIN_COUNT = 2  # an entity must occur in >= 2 papers to be retrievable (removes one-off tagging noise)


def entity_ok(e: str) -> bool:
    el = e.lower().strip()
    if not el or el in GENERIC:
        return False
    m = re.search(r"\(([^)]*)\)", el)
    return not m or m.group(1) in KEEP_QUALIFIERS


def _list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return []


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def load_surveygen() -> list[dict]:
    """Corpus records.
    Run 2 (default): data/paper_after_reranking.csv from the QUAL-SG pipeline (final version, 259 papers
    with abstracts and publication years). Entities are extracted from the abstracts (keyphrases()).
    Run 1: data/final_ref_set.json (OpenAlex records with keywords/topics) + the reranking CSV."""
    rank = pd.read_csv(DATA / "paper_after_reranking.csv")
    rank = rank[rank["title"].notna()]
    rank_by_title = {norm_title(r.title): r for r in rank.itertuples()}
    if not (DATA / "final_ref_set.json").exists():
        papers = []
        for r in rank.itertuples():
            doi = str(r.doi).replace("https://doi.org/", "") if isinstance(r.doi, str) else ""
            ab = r.abstract if isinstance(r.abstract, str) else ""
            papers.append(dict(
                id=doi or norm_title(r.title), title=r.title, abstract=ab,
                year=int(r.pub_year) if not pd.isna(r.pub_year) else None,
                keywords=[], topics=[], phrases=keyphrases(r.title + ". " + ab),
                relevance=float(r.relevance_llm_1to5) if not pd.isna(r.relevance_llm_1to5) else None,
                composite=float(r.composite_score) if not pd.isna(r.composite_score) else None,
                cited_by=int(r.cited_by_count) if not pd.isna(r.cited_by_count) else 0,
                selected=(r.tag == "selected"), source="qualsg"))
        return papers
    recs = json.loads((DATA / "final_ref_set.json").read_text())
    years = {}
    s2 = DATA / "s2_citations.json"
    if s2.exists():
        years = json.loads(s2.read_text()).get("corpus_years", {})
    papers = []
    for r in recs:
        doi = (r.get("doi") or "").replace("https://doi.org/", "")
        rr = rank_by_title.get(norm_title(r["title"]))
        y = (years.get(doi) or {}).get("year")
        if y is None and doi.lower().startswith("10.48550/arxiv."):  # arXiv id encodes yymm
            m = re.match(r"(\d{2})(\d{2})\.", doi.split("arxiv.", 1)[1])
            y = 2000 + int(m.group(1)) if m else None
        papers.append(dict(
            id=doi or norm_title(r["title"]), title=r["title"], abstract=r.get("abstract") or "", year=y,
            keywords=[k for k in _list(r.get("keywords"))], topics=[t for t in _list(r.get("topics"))],
            relevance=float(rr.relevance_llm_1to5) if rr is not None and not pd.isna(rr.relevance_llm_1to5) else None,
            composite=float(rr.composite_score) if rr is not None and not pd.isna(rr.composite_score) else None,
            cited_by=int(float(r.get("cited_by_count") or 0)), source="surveygen"))
    return papers


# --------------------------------------------------------------------------- keyphrase entities (run 2)
# Without OpenAlex concepts, entities are candidate phrases from titles and abstracts, RAKE-style
# (Rose et al., 2010): text is split at stop words and punctuation, and the remaining 1-4 word
# phrases are normalised (lower case, simple singularisation). Phrases built only from generic
# research vocabulary are dropped. The co-occurrence store and Eq. 2 then work as before.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKSTOP

_STOP = set(_SKSTOP) | {
    "propose", "proposed", "proposes", "present", "presents", "introduce", "introduces", "introduced", "paper",
    "work", "study", "approach", "approaches", "method", "methods", "framework", "results", "result", "show",
    "shows", "shown", "demonstrate", "demonstrates", "achieve", "achieves", "achieved", "performance", "existing",
    "state", "art", "novel", "new", "based", "using", "use", "used", "uses", "different", "various", "extensive",
    "experiments", "experimental", "significant", "significantly", "effective", "effectively", "able", "via",
    "however", "furthermore", "moreover", "specifically", "particularly", "respectively", "compared", "outperforms",
    "outperform", "superior", "previous", "recent", "recently", "challenging", "challenge", "challenges", "task",
    "tasks", "problem", "problems", "model", "models", "data", "datasets", "dataset", "code", "available", "github",
    "com", "https", "http", "www", "make", "makes", "leverage", "leverages", "leveraging", "enable", "enables",
    "address", "addresses", "improve", "improves", "improved", "improving", "including", "high", "low", "large",
    "small", "better", "best", "key", "main", "important", "crucial", "critical", "remains", "remain", "limited",
    "lack", "lacks", "provide", "provides", "potential", "furthermore", "thereby", "thus", "ability", "capability",
    "capabilities", "even", "well", "across", "within", "without", "e.g", "i.e", "etc", "named", "called", "termed",
    "first", "second", "simple", "simultaneously", "strong", "robust", "consistently", "substantially", "notably",
    "single", "multiple", "specific", "general", "set", "sets", "number", "way", "ways", "terms", "level", "levels",
    "time", "times", "real", "world", "learn", "learns", "learned", "learning", "help", "helps", "allow", "allows",
    "rely", "relies", "relying", "struggle", "struggles", "future", "achieving", "competitive", "detecting", "extracted", "extracting",
    "comprehensive", "conducted", "conduct", "evaluated", "evaluate", "evaluating", "obtained", "proposing", "direction", "directions", "empirical", "often", "typically", "usually", "mainly", "highly", "largely", "fully", "directly",
}
_SPLIT = re.compile(r"[^a-z0-9\-\+/ ]")
_KEEP_SHORT = {"ct", "mri", "oct", "vlm", "vlms", "llm", "clip", "sam", "vit", "auroc", "auc"}


def _sing(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _named(tok: str) -> bool:
    """Acronym or internally capitalised name in the original text (CLIP, BMAD, MVTec, BraTS, WinCLIP)."""
    core = tok.strip("-+/")
    return len(core) >= 2 and sum(ch.isupper() for ch in core[1:]) >= 1 and not core.isdigit()


def keyphrases(text: str, max_len: int = 4) -> list[str]:
    """Entity candidates: multi-word phrases (2-4 words) between stop words, plus single tokens only when
    they are names or acronyms in the original casing. Single common words ('training', 'image') are
    not entities in the entity-linking sense and are dropped."""
    out = []
    for chunk in re.split(r"[^A-Za-z0-9\-\+/ ]", text or ""):
        cur = []
        for w in chunk.split() + [None]:
            lw = w.lower() if w else None
            if w is None or lw in _STOP or lw.isdigit() or "/" in lw or (len(lw) < 3 and lw not in _KEEP_SHORT):
                for i in range(0, len(cur), max_len):   # long runs are cut into <= max_len pieces
                    piece = cur[i:i + max_len]
                    if len(piece) >= 2:
                        out.append(" ".join(_sing(x.lower()) for x in piece))
                    elif len(piece) == 1 and _named(piece[0]):
                        out.append(piece[0].lower())
                cur = []
            else:
                cur.append(w)
    return out


def review_set(papers: list[dict]) -> list[dict]:
    """The papers of the submitted Task 1 literature review.
    Run 2: the 30 references selected by the QUAL-SG pipeline (data/pipeline_ref.json), in its order,
    matched to corpus records by DOI or title (abstracts come from the corpus when the list has none).
    Run 1 (no pipeline_ref.json): the 16 SurveyGen papers with relevance >= 3, by composite score."""
    ref = DATA / "pipeline_ref.json"
    if ref.exists():
        by_doi = {p["id"].lower(): p for p in papers}
        by_title = {norm_title(p["title"]): p for p in papers}
        out = []
        for r in json.loads(ref.read_text())["references"]:
            p = by_doi.get((r.get("doi") or "").lower()) or by_title.get(norm_title(r["title"]))
            if p is None:
                p = dict(id=r.get("doi") or norm_title(r["title"]), title=r["title"], abstract=r.get("abstract") or "",
                         year=r.get("year"), keywords=[], topics=[], relevance=r.get("relevance_llm_1to5"),
                         composite=r.get("composite_score"), cited_by=r.get("citations") or 0, source="pipeline_ref")
            elif not p["abstract"] and r.get("abstract"):
                p = dict(p, abstract=r["abstract"])
            out.append(p)
        return out
    sel = [p for p in papers if (p["relevance"] or 0) >= 3]
    return sorted(sel, key=lambda p: -(p["composite"] or 0))


def base_paper(papers: list[dict]) -> dict:
    return next(p for p in papers if p["title"].startswith("Adapting Visual-Language Models for Generalizable"))


# --------------------------------------------------------------------------- entity store
class EntityStore:
    """ResearchAgent's entity-centric knowledge store: a sparse co-occurrence matrix K over entities,
    built from all corpus papers. Entities here are the OpenAlex keywords + topics of each paper
    (ResearchAgent used the BLINK entity linker on titles/abstracts; OpenAlex already provides
    linked concepts, so no extra model is needed)."""

    def __init__(self, papers: list[dict]):
        self.count = Counter()
        self.co = defaultdict(Counter)
        for p in papers:
            ents = self.entities(p)
            self.count.update(ents)
            for a, b in combinations(sorted(ents), 2):
                self.co[a][b] += 1
                self.co[b][a] += 1
        self.total = sum(self.count.values())

    @staticmethod
    def entities(p: dict) -> set[str]:
        ents = {e.strip() for e in p.get("keywords", []) + p.get("topics", []) if e and e.strip()}
        if "phrases" in p:                      # run 2: abstract keyphrases
            ents |= set(p["phrases"])
        elif not ents and p.get("abstract"):    # papers outside the corpus (e.g. citing papers)
            ents |= set(keyphrases(p.get("title", "") + ". " + p["abstract"]))
        return {e for e in ents if entity_ok(e)}

    def retrieve(self, context_papers: list[dict], k: int = 10, smoothing: float = 0.1):
        """Eq. (2) of ResearchAgent: argmax_{|I|=k} prod_{e_j in E_ctx} P(e_j|e_i) * P(e_i),
        over entities e_i NOT already in the context. Computed in log space with add-smoothing.
        Because entities are scored independently, the top-k set is the k best single scores."""
        ctx = set().union(*(self.entities(p) for p in context_papers))
        ctx = {e for e in ctx if e in self.count}
        V = len(self.count)
        scores = {}
        for ei, n_i in self.count.items():
            if ei in ctx or n_i < MIN_COUNT:
                continue
            s = math.log(n_i / self.total)
            for ej in ctx:
                s += math.log((self.co[ei][ej] + smoothing) / (n_i + smoothing * V))
            scores[ei] = s
        best = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [e for e, _ in best], sorted(ctx)
