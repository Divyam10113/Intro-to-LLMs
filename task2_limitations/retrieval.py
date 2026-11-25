"""Hybrid retrieval + LLM re-ranking over the evidence chunks.

Follows arXiv:2601.11578 / BAGELS: sparse (BM25) and dense scores weighted 50/50,
top-20 candidates, then an LLM re-ranker keeps only chunks scored >= 8/10.
Dense = sentence-transformers if installed, otherwise TF-IDF + LSA (no extra installs).
"""
import math
import re
from collections import Counter

import numpy as np

from common import DATA_DIR, chat_json, read_jsonl

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of and or to in for on with by is are was were be this that from as at it its we our "
           "their they these those which can not but also such than then into via using based".split())


def tokens(t):
    return [w for w in TOKEN.findall(t.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tokens(query)
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            denom_base = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + denom_base)
                         for w in q if w in tf)
        return out


class Dense:
    def __init__(self, docs):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.emb = self.model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            self.kind = "minilm"
        except ImportError:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
            X = self.vec.fit_transform(docs)
            self.svd = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=0)
            self.emb = normalize(self.svd.fit_transform(X))
            self.kind = "tfidf-lsa"

    def scores(self, query):
        if self.kind == "minilm":
            q = self.model.encode([query], normalize_embeddings=True)[0]
        else:
            from sklearn.preprocessing import normalize
            q = normalize(self.svd.transform(self.vec.transform([query])))[0]
        return self.emb @ q


def _minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else read_jsonl(DATA_DIR / "chunks.jsonl")
        texts = [f"{c['title']} | {c['section']} | {c['text']}" for c in self.chunks]
        self.bm25, self.dense = BM25(texts), Dense(texts)
        print(f"[retriever] {len(self.chunks)} chunks, dense={self.dense.kind}")

    def search(self, query, k=20):
        s = 0.5 * _minmax(self.bm25.scores(query)) + 0.5 * _minmax(self.dense.scores(query))
        return [self.chunks[i] for i in np.argsort(-s)[:k]]


RERANK_SYS = (
    "You are a re-ranker for a system that identifies limitations of a research paper (the INPUT PAPER). "
    "For each candidate passage from related literature, rate 1-10 how useful it is for identifying or "
    "substantiating a limitation of the INPUT PAPER: e.g. it states a weakness of the input paper, shows a "
    "setting/capability the input paper lacks, offers a stronger baseline, protocol or evaluation practice, or "
    "contradicts an assumption of the input paper. Generic topical similarity alone deserves <= 5. "
    'Return JSON: [{"cid": "...", "score": int}] for every candidate.'
)


def rerank(paper_summary, query, candidates, threshold=8, tag="rerank"):
    if not candidates:
        return []
    listing = "\n\n".join(f"[{c['cid']}] ({c['title']}, {c['year']}; {c['section']})\n{c['text'][:1800]}"
                          for c in candidates)
    user = (f"INPUT PAPER (summary):\n{paper_summary}\n\nRETRIEVAL QUERY: {query}\n\n"
            f"CANDIDATES:\n{listing}")
    scores = chat_json(RERANK_SYS, user, temperature=0, tag=tag)
    by_id = {str(s.get("cid")): int(s.get("score", 0)) for s in scores if isinstance(s, dict)}
    kept = [dict(c, rerank=by_id.get(c["cid"], 0)) for c in candidates if by_id.get(c["cid"], 0) >= threshold]
    return sorted(kept, key=lambda c: -c["rerank"])
