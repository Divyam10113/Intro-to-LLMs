# QUAL-SG — implementation notes, approach and every intermediate result

Reproduction of **SurveyGen: Quality-Aware Scientific Survey Generation with Large
Language Models** (Bao, Naeem, Rafiei, Zhang — EMNLP 2025, [arXiv:2508.17647](https://arxiv.org/abs/2508.17647)),
implemented from scratch in `Paper implementation/qual_sg/`.

> **Naming.** *SurveyGen* is the paper's **dataset/benchmark** (4 205 human-written
> surveys, 242 143 references). **QUAL-SG** is the **framework** that the paper
> contributes on top of it. "Running the paper" therefore means running the
> QUAL-SG pipeline; the SurveyGen dataset itself is only needed to obtain a
> *human gold standard* for the evaluation stage, and is not required to generate
> a survey. This run implements and executes the full QUAL-SG pipeline **plus**
> the paper's two baselines (Fully-LLMGen = Task 1, Naive-RAG = Task 2) so that
> the Table-3 comparison can be reproduced on a new topic.

Topic (from `example/topic.csv`):

| field | value |
|---|---|
| title | Generalizable Vision-Language Models for Zero-Shot / Few-Shot Anomaly Detection in Medical Images |
| year | 2024 |
| keywords | Vision-Language Models; Medical Images; Anomaly Detection; Zero-Shot; CLIP |
| max_refs (K) | 30 |

---

## 1. Deliverables (what is on disk)

| File / dir | What it is |
|---|---|
| `qual_sg/` | the from-scratch implementation (11 modules, see §3) |
| `final_ref_set.json` | **updated** — 349 quality-augmented candidate papers (step 2 output) |
| `paper_after_reranking.csv` | **updated** — 349 ranked rows, 30 tagged `selected` (step 3 output) |
| `literature_review_2page.md` / `.tex` / `.json` | **the deliverable**: 2-page QUAL-SG literature review (1 119 words, 5 sections, 30 refs) |
| `baselines/naive_rag/`, `baselines/fully_llm/` | the two baselines' 2-page reviews (same generator, different reference sets) |
| `comparison_report.json` | quantitative comparison of the three reviews |
| `survey_outline.json`, `sections/`, `survey_content.json` (11/28 subsections) | full-survey artefacts produced before the switch to a 2-page review |
| `evaluation_report.json`, `qual_sg/intermediates/step6_metrics.json` | evaluation (step 6) |
| `qual_sg/intermediates/` | **every raw intermediate artefact + logs + LLM cache/audit** |

---

## 2. Environment and LLM backend

* Python 3.10, `requests`, `pandas`, `numpy`, `scikit-learn`, `rouge-score`,
  `sentence-transformers` (CPU).
* **LLM backend:** all generation/judging calls go through the coding assistant's
  own runtime model (`qual_sg/harness_backend.py`, harness model
  `river/deepseek-ai/DeepSeek-V4.1-Flash`) — no external API key required.
  The same code runs unchanged against Groq's free open-weights endpoint by
  setting `GROQ_API_KEY` (default models `openai/gpt-oss-120b`,
  fallback `llama-3.3-70b-versatile`).
* **Token/cost accounting:** the two bulk judge runs cost **$0.086** (344 items,
  earlier pool) and **$0.130** (337 items, final pool); every raw response is
  appended to `qual_sg/intermediates/cache/llm_audit.jsonl`.
* **Embeddings:** the paper uses `bge-large-en-v1.5` (Appendix C, fn. 13). The
  1.3 GB checkpoint could not be downloaded reliably here, so the **same family
  small variant** `bge-small-en-v1.5` (133 MB, 384-dim, vendored under
  `qual_sg/models/`) is used for Eq. (1), Eq. (4) and the evaluation metrics.
  Switch back with `QUAL_SG_EMBED_MODEL=BAAI/bge-large-en-v1.5`.

Run order: `python3 -m qual_sg.run_all` (resumable; `QUAL_SG_FORCE=1` to redo).

---

## 3. Stage-by-stage approach and results

### 3.1 Step 1 — literature retrieval (`step1_retrieve.py`)

Paper: retrieve the *n* most relevant papers (Eq. 1) from an external database.
The authors use Semantic Scholar; it rate-limits anonymous traffic (HTTP 429) from
this host, so retrieval runs on **OpenAlex (primary) + arXiv (secondary)** and the
paper's embedding similarity is then applied explicitly.

| Source | query | raw | usable |
|---|---|---|---|
| Semantic Scholar | `topic.query` | — | disabled (`QUAL_SG_USE_S2=0`); returns 429 |
| OpenAlex | keywords `Vision-Language Models Medical Images Anomaly Detection Zero-Shot CLIP`, relevance sort, 2000–2024, DOI required | 620 | 559 |
| arXiv | 5 facet queries (`"anomaly detection" AND "medical images"`, `AND "vision-language"`, `medical AND CLIP`, `"zero-shot anomaly detection"`, `"few-shot anomaly detection"`) | 155 | 155 |

* Deduplication on DOI **and** normalised title (arXiv/OpenAlex register the same
  paper under different DOIs): **48 duplicates removed** → 666 unique.
* **Eq. (1) filter**: `cos(embed(topic), embed(title + abstract))` with
  `bge-small-en-v1.5`; kept the **top 300 of 666** (max similarity 0.910, minimum
  kept 0.721). This mirrors the paper's "top-n with the highest embedding
  similarity" and is why the final pool is far more on-topic than a raw
  citation-sorted OpenAlex query (the earlier citation-sorted run put "Diffusion
  Models: A Comprehensive Survey" at rank 1).
* Artefacts: `intermediates/step1_candidates.json`,
  `intermediates/step1_retrieval_log.json`. Elapsed 142 s.

### 3.2 Step 2 — quality signals + co-citation expansion (`step2_quality.py`)

Paper §2.3/§2.4: enrich each candidate with citation performance, author influence
and venue reputation, then expand `D` with co-cited papers (`cited by ≥ 2` of `D`,
top 50).

| quantity | value |
|---|---|
| candidates in | 300 |
| matched in OpenAlex by DOI | 244 |
| unmatched (kept, impact = *unknown* → NaN, not 0) | 56 |
| referenced works cited by ≥ 2 candidates | 1 061 |
| co-cited papers requested / kept | 50 / 49 (37 with abstract) |
| author h-index lookups resolved | 488 / 488 |
| venue h-index lookups resolved | 63 / 63 |
| **`final_ref_set.json` total** | **349 records** (300 retrieved + 49 co-cited) |

Per-record fields: `doi, title, abstract, topics, keywords, referenced_works,
referenced_works_count, cited_by_count, first/last_author_id + h-index,
journal_id/name + h-index, origin ∈ {retrieved, co_cited}, co_cited_count`.
Artefacts: `final_ref_set.json`, `intermediates/step2_cocitation_stats.json`.

### 3.3 Step 3 — quality-aware re-ranking (`step3_rerank.py`)

Implements Eq. (2) `S_t` (LLM judge, 1–5), Eq. (3) `S_a = αC + βA + γV` over
quartile bins, Eq. (4) `S_d` (mean cosine distance), and selects the top-K by
**mean rank** across the three (paper §2.4, final paragraph). Weights follow the
authors' released code: citation 0.165, author 0.067, venue 0.10, relevance 0.33,
diversity 0.33 (the paper text quotes 0.5/0.3/0.2 for C/V/A — see §5).

| intermediate result | value |
|---|---|
| rows with abstracts / dropped | 337 / 12 |
| `S_t` source | bulk LLM judge (`judge_batch`), 337/337 scored, 87.5 s, **$0.130** |
| `S_t` distribution | 1: 107, 2: 168, 3: 42, 4: 8, 5: 11 |
| judge sanity check | 3 known on-topic papers → 5; 3 known off-topic → 1 (calibrated) |
| `S_d` backend | `bge-small-en-v1.5` (384-d), cosine distance (Eq. 4) |
| selected K | 30 |
| selected composition | 22 retrieved + 8 co-cited; relevance {5:3, 4:3, 3:5, 2:18, 1:1} |
| selected citation profile | median 52, mean 6 344 (skewed by *Attention Is All You Need*) |
| CSV columns | 25, incl. `rank_relevance/rank_impact/rank_diversity/avg_rank/impact_score/diversity_score/composite_score/tag` |
| artefacts | `paper_after_reranking.csv`, `intermediates/step3_rankings.json`, `step3_embeddings.npz`, `step3_relevance_scores.jsonl` |

Top of the final selection (full list in the CSV):

| # | title | origin | S_t | cites | avg_rank |
|---|---|---|---|---|---|
| 1 | A self-supervised framework for abnormality detection from brain MRI | retrieved | 5 | 0 | 48.5 |
| 2 | CPLIP: Zero-Shot Learning for Histopathology… | retrieved | 3 | 27 | 54.8 |
| 3 | Position-Guided Prompt Learning for Anomaly Detection in Chest X-Rays | retrieved | 5 | – | 60.0 |
| 4 | Fine-grained Abnormality Prompt Learning for Zero-shot AD | retrieved | 5 | – | 70.0 |
| 5 | Mammo-CLIP: CLIP for Enhanced Breast Cancer Diagnosis | retrieved | 4 | 4 | 74.3 |
| 6 | Attention Is All You Need | co_cited | 2 | 78 007 | 74.8 |
| 7 | Discriminative Feature Learning Framework with Gradient Preference | retrieved | 3 | – | 75.2 |
| 8 | An Image is Worth 16x16 Words (ViT) | co_cited | 2 | 21 319 | 75.7 |
| 9 | Large Language Models for Anomaly Detection in Computational Workflows | retrieved | 2 | – | 77.8 |
| 10 | SEM-CLIP: Few-Shot Learning for Nanoscale Defect Detection | retrieved | 2 | 8 | 82.8 |

### 3.4 Step 4 — outline generation (`step4_outline.py`) *(full-survey artefact)*

Verbatim authors' outline prompt, harness model, 1 attempt, 13.6 s:

* `survey_outline.json`: **6 sections, 28 subsections**
  (Foundations and Motivation 4 · Core VLM Architectures 4 · Zero-Shot Methods 4 ·
  Few-Shot & Cold-Start 4 · Domain Applications 6 · Challenges & Future 6).

### 3.5 Step 5 — section-wise survey generation (`step5_generate.py`) *(superseded)*

Verbatim authors' section prompt (≥ 300 words, `ref [n]` citations, strict JSON),
resumable per subsection. **11 of 28 subsections** were generated (≈ 5 479 words)
before the deliverable was changed to a 2-page review; artefacts are kept in
`sections/` and `survey_content.json`.

### 3.6 Short 2-page literature review (`short_review.py`) — **the deliverable**

Same reference set as QUAL-SG step 3, condensed into a 2-page review
(5 thematic paragraphs, ~1 100 words, inline `ref [n]` citations).

| source | words | sections | refs | review title |
|---|---|---|---|---|
| **qual_sg** | **1 119** | 5 | 30 | Generalizable VLMs for Zero-Shot and Few-Shot Anomaly Detection in Medical Images |
| naive_rag | 1 129 | 5 | 30 | (same) |
| fully_llm | 1 313 | 5 | 0 | (same) |

Files: `literature_review_2page.{md,tex,json}`; baselines under `baselines/<src>/`.

### 3.7 Step 6 — evaluation (`step6_evaluate.py`)

Implements the paper's metrics: citation P/R/F (title `difflib ≥ 0.95`),
ROUGE + semantic similarity, structural overlap (semantic threshold 0.80) and the
LLM 5-point structural rubric. **There is no human gold-standard survey for this
topic** (the repo ships one for "Deep Learning Applications in Computer Vision"),
so step 6 detects the mismatch (topic overlap 0.25 < 0.34 threshold) and reports
gold-free intrinsic metrics instead; the gold-dependent numbers are computed but
flagged non-comparable.

Intrinsic metrics of the full-survey artefacts: sections 6, subsections 28 (11 with
content), distinct refs cited 28/30 (**coverage 0.933**), section semantic coherence
0.869, 5 479 words (mean 498/subsection). `evaluation_report.json` written.

### 3.8 Baselines + comparison (`baselines.py`, `compare.py`)

| metric | QUAL-SG | Naive-RAG | Fully-LLM |
|---|---|---|---|
| words | 1 037 | 1 048 | 1 243 |
| citation markers | 46 | 37 | 0 |
| distinct refs cited (of 30) | 28 (**0.933**) | 23 (0.767) | – |
| out-of-range citations | 0 | 0 | – |
| topic similarity (review ↔ topic) | 0.908 | **0.916** | 0.907 |
| reference mean relevance (S_t) | 2.63 | **3.47** | – |
| refs with S_t ≥ 3 | 0.367 | **0.833** | – |
| median citations of refs | **52** | 5 | – |
| reference-set overlap (QUAL-SG ∩ Naive-RAG) | 1 of 30 (Jaccard 0.017) | | |

---

## 4. Findings

1. **The Eq. (1) filter is essential.** OpenAlex's raw relevance ranking for this
   narrow topic is noisy; embedding-filtering 666 → 300 raised the share of
   on-topic papers from 6 % to 18 % under the LLM judge, and the top 30 of the
   filtered pool are all judged 3–5.
2. **QUAL-SG and Naive-RAG select almost disjoint reference sets** (1 shared ref).
   QUAL-SG's selected set has ~10× the median citation count (52 vs 5): the
   academic-impact and diversity terms promote seminal/foundational works.
3. **On a narrow topic, that trade-off costs topical relevance.** The LLM judge
   prefers the Naive-RAG set (mean 3.47 vs 2.63). This is exactly the tension the
   paper describes: QUAL-SG optimises *overlap with human reference selection*
   (which includes seminal works) rather than raw abstract similarity — and that
   can only be measured with a human gold standard, which this topic lacks
   (SurveyGen's gold file is for a different topic).
4. **The diversity term can over-reward off-topic papers.** `S_d` (Eq. 4) is a
   distance to other candidates, so a *dissimilar* paper scores highly regardless
   of relevance — e.g. *Attention Is All You Need* enters at rank 6 (relevance 2,
   diversity rank 4). Restricting Eq. (4) to the relevance-filtered subset, or
   gating on `S_t ≥ 2`, would mitigate this; both are documented options, not
   applied, to stay faithful to the paper.
5. **Citation support is high regardless**: QUAL-SG cites 28/30 available refs with
   0 out-of-range markers (93 % coverage), i.e. the generated prose is fully
   grounded in the retrieved pool.

---

## 5. Deviations from the paper (all deliberate, all documented)

| # | Paper / authors' code | This run | Why |
|---|---|---|---|
| 1 | Semantic Scholar retrieval | OpenAlex + arXiv (+ optional S2) | S2 returns 429 anonymously; arXiv adds strong CS/ML coverage |
| 2 | `bge-large-en-v1.5` embeddings | `bge-small-en-v1.5` (same family) | 1.3 GB download stalls; 133 MB local copy is reproducible |
| 3 | GPT-4o backbone; GPT-4o judge | harness model `DeepSeek-V4.1-Flash` (Groq path available) | no external key required; judge sanity-checked |
| 4 | impact weights 0.5/0.3/0.2 (Appendix C text; internally inconsistent with Eq. 3's α/β/γ) | authors' code values 0.165/0.067/0.10 + 0.33 relevance + 0.33 diversity | the code's values are the reproducible ones; paper's own §F ablation uses the code |
| 5 | selection by *average rank* (paper) vs weighted composite (code) | **average rank** primary (paper-faithful), `composite_score` also exported | both are in the CSV for comparison |
| 6 | citation counts normalised by years since publication (fn. 15) | raw `cited_by_count` | OpenAlex lacks a normalised field; note recorded |
| 7 | structural overlap `2·|S_H∩S_G| / (S_H+S_G)` | code's `matched/(gen+gold)` | matches the authors' `evaluation.py` |
| 8 | gold-standard evaluation on 120 SurveyGen surveys | unavailable for this topic → gold-free intrinsic metrics + baseline comparison | the shipped gold file is for a different topic (flagged automatically) |

---

## 6. Reproducibility / audit trail

* `qual_sg/intermediates/step*/` — raw outputs of every stage (candidate pool,
  co-citation statistics, rankings, embeddings, metrics).
* `qual_sg/intermediates/logs/*.log` — per-stage logs.
* `qual_sg/intermediates/cache/llm_cache.jsonl` + `llm_audit.jsonl` — every LLM
  request/response, model name and token usage (deterministic replay).
* Everything is idempotent and resumable (`QUAL_SG_FORCE=1` to redo a stage).
* Previous ad-hoc outputs are preserved in `legacy/pre_rerun_outputs/`.

---

## 7. Pipeline v2 — making the reference list match the field (paper-agnostic)

**Motivation.** For the topic's own reference paper (*Adapting Visual-Language
Models for Generalizable Anomaly Detection in Medical Images*, arXiv:2403.12570,
68 references) the v1 pipeline rediscovered only **3 of 68** references
(recall 0.044, precision 0.100) in its top-30. That is the failure mode the user
identified: the pipeline was topically sensible but blind to the citation
context that defines a field.

### 7.1 Diagnosis, one change at a time

| # | Root cause | v2 change (generic) | Where |
|---|---|---|---|
| 1 | Retrieval used the literal topic keywords only | **LLM query expansion**: 6 facet queries (sub-tasks, method families, benchmark names) generated from the topic/anchor, merged with topic-derived fallbacks | `step1.expand_queries`, `fallback_queries` |
| 2 | Publication window cut at the topic year, silently truncating the pool | **`QUAL_SG_YEAR_MODE=now`** (default) cites work published after the topic year; `topic` restores old behaviour | `step1.year_end` |
| 3 | Co-citation kept only works cited by ≥2 pool papers, so seminal methods/benchmark datasets cited once were invisible | **Backward snowballing**: the reference lists of the 30 most similar retrieved papers are fetched (≤12 each) and added as `origin="snowball"` | `step2` §2b, `EXPAND_FROM_TOP` |
| 4 | One provider = single point of failure (OpenAlex 429s with a ~27 000 s `Retry-After`) | **Multi-source retrieval + per-source clients + automatic block guard**: OpenAlex (query budgeted), **Crossref**, **arXiv**, optional Semantic Scholar; a 429 with a long `Retry-After` disables only that provider | `step1`, `crossref.py`, `common.HttpClient` |
| 5 | Raw citation counts sent every slot to decade-old megacited papers (e.g. *Attention Is All You Need*, 78 k) | **Age-normalised citations** (`citations_per_year`), as the paper's Appendix C fn. 15 prescribes | `step3.add_impact_components` |
| 6 | Mean-rank selection + diversity-as-distance actively rewarded off-topic-but-distinct papers | **Relevance gate (≥2, relaxed if needed) + Maximal Marginal Relevance** (λ=0.72) — relevance-first, diversity as a tie-breaker | `step3.select_references` |
| 7 | The relevance judge scored against the one-line topic title — a weak scope statement | **Judge against the anchor's abstract** (a human-written statement of scope) when an anchor is supplied | `step3.judge_topic_text` |
| 8 | No way to use the citation context of a specific paper | **Generic anchor mode**: any PDF/DOI supplies (a) the scope text, (b) its bibliography resolved via Crossref → DataCite → arXiv → OpenAlex, and (c) a `cited_by_anchor` provenance flag | `anchor.py`, `anchor_merge.py` |
| 9 | No way to measure alignment | **Bibliography-overlap metric** (recall / precision / F1 of our selection against any paper's reference list) | `bib_eval.py` |

Anchor handling is paper-agnostic: nothing keys off a specific title or DOI. The
same commands work for any PDF/DOI the user supplies:

```bash
# 1. tell the pipeline which paper defines the field
export QUAL_SG_ANCHOR_PDF="../any_paper.pdf"        # or QUAL_SG_ANCHOR_DOI=10.xxxx/yyy
# 2. merge its bibliography into the candidate pool (Crossref/DataCite/arXiv, no OpenAlex needed)
python3 -m qual_sg.anchor_merge --pdf "../any_paper.pdf"
# 3. score + re-rank (gate + MMR; anchor-cited papers get the provenance prior)
python3 -m qual_sg.step3_rerank
# 4. measure the alignment
python3 -m qual_sg.bib_eval --pdf "../any_paper.pdf" --selected
```

`QUAL_SG_ANCHOR_PRIORITY` controls how hard the anchor is pushed:
`off` (ignore provenance) · `boost` (default: gate-exempt + MMR prior) ·
`strict` (fill the quota from the anchor's bibliography first, then MMR).

### 7.2 Measured effect (same topic, same 30-reference budget)

| configuration | refs matched | recall | precision | F1 | of our 30 refs, in the paper's list |
|---|---|---|---|---|---|
| v1 (mean-rank, topic-title judge, no anchor) | 3/68 | 0.044 | 0.100 | 0.061 | 3 (10 %) |
| v2 `boost` | 20/68 | 0.294 | 0.667 | 0.408 | 20 (67 %) |
| v2 `strict` (**canonical output**) | **28/68** | **0.412** | **0.933** | **0.571** | **30 (100 %)** |

Candidate-pool coverage of the paper's bibliography rose from **21/68 (31 %)** to
**41/68 (60 %)** via anchor-bibliography resolution + strict title/reference
matching + snowballing (Crossref → DataCite → arXiv → OpenAlex chain; 3
false-positive resolutions were detected and dropped by the
`accept_match` guard, e.g. *Deep Semi-Supervised Anomaly Detection* having been
matched to a paper literally called "Uller").

**Reference budget.** With K = 30 the recall ceiling is 0.44 (a 68-entry
bibliography cannot be fully cited on two pages); `strict` reaches 0.412 of it.
Raise `max_refs` in `example/topic.csv` for full bibliography coverage.

### 7.3 Trade-offs (documented, not hidden)

* `strict` maximises bibliography alignment but admits papers the topical judge
  rates 1–2 (benchmark datasets, seminal architectures). They are cited by the
  field's own paper, which is exactly why they belong in the review, but they are
  off-topic by abstract. `boost` keeps 7 score-5 papers in the selection.
* Anchor-bibliography entries resolved without an abstract are judged on title
  alone (`title_only` flag) — noted per record.
* `V2` review artefacts live in `v2/` so the v1 outputs remain comparable.

### 7.4 v2 outputs

| file | content |
|---|---|
| `paper_after_reranking.csv` (**canonical**) | 367 ranked rows, 30 `selected`, `strict` anchor prioritisation — **all 30 selected refs are in the reference paper's bibliography** |
| `pipeline_ref.json` | **the pipeline's own reference set** — the 30 selected references in citation order with `refNo, title, doi, year, venue, citations, citations_per_year, relevance_llm_1to5, cited_by_anchor, anchor_ref_num, origin, avg_rank, composite_score, abstract`, plus the pipeline config and the overlap metrics |
| `v2/literature_review_2page.{md,tex,json}` | 1070-word, 5-section review over the aligned 30 references (61 in-text `ref [n]` markers, 25 distinct refs cited, 0 out-of-range) |
| `qual_sg/intermediates/bib_overlap_before.json` / `_after_boost.json` / `_after_strict.json` / `_final.json` | the overlap measurements above |
| `qual_sg/intermediates/final_ref_set_v1.json` | the pre-v2 pool, kept for before/after comparison |

---

## 8. v3 — 60 high-quality references, strict relevance gate, no missing abstracts

Requested changes: (a) every selected reference must have an abstract, (b) only
papers the judge scores **3, 4 or 5** may be selected, (c) K = 60.

### 8.1 What was wrong with v2's `pipeline_ref.json`

* **16 of 30 selected references had no abstract** - they had entered the pool
  from the anchor bibliography via Crossref, which carries no abstract for
  conference papers, and the soft gate let them in.
* Those same entries were judged on title alone, which is why the selection
  contained papers the judge considers *irrelevant* (`S_t = 1-2`: dataset papers,
  generic segmentation work).

### 8.2 Changes

| # | Change | Where |
|---|---|---|
| 1 | **Hard relevance gate**: only `S_t >= 3` is selectable; no relaxation when the pool is small, and anchor provenance no longer bypasses the gate (it only reorders *within* the gate) | `config.RELEVANCE_GATE=3`, `RELEVANCE_GATE_HARD=1`, `step3.select_references` |
| 2 | **Abstract required** for selection (`REQUIRE_ABSTRACT=1`) | `step3.select_references` |
| 3 | **Abstract enrichment** (`fill_abstracts.py`): OpenAlex -> arXiv -> Semantic Scholar -> Crossref -> DataCite, per missing record; failures flagged `abstract_missing` and excluded | `qual_sg/fill_abstracts.py` |
| 4 | **Pool expansion** so that 60 papers can clear the gate: 15 LLM query variants x (OpenAlex + Crossref + arXiv), snowballing from the 80 most relevant papers, co-citation top-80 | `step1`, `step2` |
| 5 | **Full re-judge** of the enlarged pool against the anchor's scope; the 22 records whose abstracts were filled were re-judged (they had previously been judged on title only) | `judge_batch` |
| 6 | Reference list handed to the writers is no longer capped at 40 (`GENERATION_MAX_REFS=0` = all selected) | `config`, `short_review`, `step5` |

### 8.3 Intermediate results

| stage | result |
|---|---|
| retrieval | 1 203 raw hits -> 597 kept after DOI/title dedup + Eq. (1) filter |
| enrichment | 597 retrieved + 57 co-cited + **349 snowballed** = **1 003** pool records |
| abstracts | 118 missing -> 49 filled (arXiv 11+, OpenAlex, S2, Crossref, DataCite); **69 excluded** as unresolvable |
| judging | **934** scored against the anchor scope, $0.67, 314 s |
| score distribution | `{1: 386, 2: 451, 3: 38, 4: 29, 5: 30}` -> **97 papers clear the gate** |
| selection | hard gate `S_t >= 3`, anchor prior `boost`, MMR lambda 0.72, K = 60 -> **60/60 selected** |
| selected score profile | `{3: 2, 4: 28, 5: 30}` - **no paper below 3** |
| selected abstracts | **0 missing** |

### 8.4 Trade-off, stated explicitly

The strict gate kills bibliography alignment: of the 57 pool papers cited by the
reference paper, the judge scores 18 as 1, 33 as 2 and only **2** as >= 3, so the
selection overlaps the reference paper's bibliography by **2/68** (v2 with the soft
gate reached 28/68). Dataset/benchmark/seminal papers are cited by a survey because
they are *used*, not because their abstracts are on-topic, and the topical judge
cannot express that.

Both behaviours are now one setting apart:

```bash
# high-quality / relevance-first (current default, 60 refs, all S_t >= 3)
QUAL_SG_RELEVANCE_GATE=3 QUAL_SG_ANCHOR_PRIORITY=boost  python3 -m qual_sg.step3_rerank

# bibliography-aligned (admits anchor-cited papers below the gate)
QUAL_SG_RELEVANCE_GATE_HARD=0 QUAL_SG_ANCHOR_PRIORITY=strict python3 -m qual_sg.step3_rerank
```

### 8.5 v3 outputs

| file | content |
|---|---|
| `pipeline_ref.json` | **the 60 selected references**, `refNo/title/doi/year/venue/citations/citations_per_year/relevance_llm_1to5/cited_by_anchor/origin/avg_rank/composite_score/abstract` + pipeline config + overlap |
| `paper_after_reranking.csv` | 1 003 ranked rows, 60 tagged `selected` |
| `final_ref_set.json` | 1 003-record candidate pool (934 with abstracts) |
| `v3/literature_review_2page.{md,tex,json}` | 1 110-word review over all 60 references (67 in-text citations, 43 distinct refs cited, 0 out-of-range) |
| `qual_sg/intermediates/bib_overlap_v3.json` / `bib_overlap_final.json` | overlap measurements (2/68 under the strict gate; 28/68 for v2, kept as `bib_overlap_v2_strict.json`) |
| `qual_sg/intermediates/final_ref_set_367.json` | the previous pool, kept for comparison |

---

## 9. Run v4 — the re-specified topic, 60 references, `paper_after_reranking_final.csv`

### 9.1 Parameters used (as specified)

| parameter | value |
|---|---|
| ANCHOR | *Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images* (CVPR 2024; arXiv:2403.12570; DOI 10.1109/cvpr52733.2024.01081 / 10.48550/arXiv.2403.12570) |
| TOPIC | Generalizable Medical Image Anomaly Detection via Vision-Language Model Adaptation — keywords: vision-language models; CLIP; medical anomaly detection; zero-shot; few-shot; multi-level feature adaptation; anomaly segmentation |
| K | 60 |
| YEAR_RANGE | 2014–2026 (`QUAL_SG_EARLIEST_YEAR=2014`, `YEAR_MODE=now`) |
| POOL_TARGET | 76 fully evaluated candidates (`QUAL_SG_EVAL_POOL_TARGET=76`) |

### 9.2 Changes made for this run

| # | Change | Why |
|---|---|---|
| 1 | `QUAL_SG_EARLIEST_YEAR` (default 2000) — the lower bound was hard-coded | the spec asks for 2014–2026 |
| 2 | **Evaluation shortlist** `EVAL_POOL_TARGET`: only the top-N candidates by Eq. (1) similarity are fully evaluated (judged); if fewer than K clear the gate the shortlist is widened automatically and the event is logged | implements "POOL_TARGET = 76 fully evaluated" without capping the achievable K |
| 3 | **Europe PMC source** (`europepmc.py`) added to retrieval **and** abstract filling | OpenAlex was hard-throttled (`Retry-After ≈ 60 000 s`); Europe PMC is key-less, generous and the best source for medical/clinical work |
| 4 | **Per-source HTTP clients in step 1** (bug fix) | OpenAlex's 429 was disabling the *shared* client, which silently zeroed Crossref and arXiv after the first request |
| 5 | Judge scope = **topic (title + keywords) + anchor title/abstract** | the topic statement is now broader than one method paper; the anchor still fixes the method line |

### 9.3 Intermediate results

| stage | result |
|---|---|
| retrieval | 1 998 raw hits (Crossref + arXiv + Europe PMC + anchor bibliography; OpenAlex disabled) -> **1 251 unique** after DOI/title dedup -> **264** kept by the Eq. (1) filter (cosine 0.776–0.949) -> **297** candidates with the anchor's bibliography |
| enrichment | 297 retrieved; **0 co-cited / 0 snowballed** — OpenAlex is the only source with `referenced_works` and it was blocked, so citation-graph expansion could not run this time |
| abstracts | 120 missing -> **82 filled** (Europe PMC / arXiv / Semantic Scholar / Crossref / DataCite), 38 unresolvable -> excluded |
| judging | **259** scored against the new scope, 72.5 s, $0.13 |
| score distribution | `{1: 22, 2: 135, 3: 60, 4: 17, 5: 25}` -> **102 papers clear the gate** |
| shortlist | top 76 by Eq. (1) yielded only **57** ≥ 3 -> auto-widened to the full 259 (logged) |
| selection | gate ≥3, MMR (λ 0.72), anchor prior `boost`, K = 60 -> **60/60 selected** |
| selected profile | `{3: 18, 4: 17, 5: 25}` — **0 below 3, 0 missing abstracts** |
| review | `v4/literature_review_2page.*` — 1 188 words, 74 in-text citations, 52 distinct refs, 0 out-of-range |
| bibliography overlap | **7/68** (recall 0.103, precision 0.117) — the anchor's bibliography is dominated by dataset/benchmark and pre-2020 generic-AD papers that the topical judge scores 1–2, so the strict gate excludes them (same trade-off as §8.4) |

### 9.4 Outputs

| file | content |
|---|---|
| **`paper_after_reranking_final.csv`** | 297 ranked rows, **60 tagged `selected`**, 28-column schema |
| **`pipeline_ref.json`** | the **60 selected references**, ordered **by relevance score descending** (`relevance_llm_1to5`, ties broken by `composite_score` then `citations_per_year`); each entry carries `refNo` (position in this file), `refNoInReview` (the number used by `ref [n]` in the review), `title/doi/year/venue/citations/citations_per_year/relevance_llm_1to5/cited_by_anchor/origin/avg_rank/composite_score/abstract`, plus the pipeline config and the overlap block. `--sort rank` restores citation order. |
| `v4/literature_review_2page.{md,tex,json}` | the 2-page review written from those 60 references |
| `final_ref_set.json` | the 297-record candidate pool |
| `qual_sg/intermediates/bib_overlap_v4.json` | the overlap measurement for this run |

Nothing about the pipe is specific to this paper: the same commands with a
different `example/topic.csv` and `QUAL_SG_ANCHOR_PDF` reproduce the run for any
topic/anchor.
