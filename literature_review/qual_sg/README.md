# QUAL-SG — from-scratch implementation

Implementation of **SurveyGen: Quality-Aware Scientific Survey Generation with Large
Language Models** (Bao, Nayeem, Rafiei, Zhang — EMNLP 2025, [arXiv:2508.17647](https://arxiv.org/abs/2508.17647)).

This package reproduces the QUAL-SG pipeline end to end and writes every
intermediate artefact to disk.  It replaces the ad-hoc scripts that previously
sat in `Paper implementation/`; the old copies are kept under `legacy/` for
reference.

## Pipeline

```mermaid
flowchart LR
  T[topic.csv] --> S1[step1 retrieval]
  S1 --> S2[step2 quality signals + co-citation]
  S2 --> S3[step3 re-ranking]
  S3 --> S4[step4 outline]
  S4 --> S5[step5 survey generation]
  S5 --> S6[step6 evaluation]
  S2 -. final_ref_set.json .-> S3
  S3 -. paper_after_reranking.csv .-> S4
```

| Step | Module | Paper | Output (canonical) |
|---|---|---|---|
| 1 | `step1_retrieve` | §2.4 retrieval, Eq. 1 | `qual_sg/intermediates/step1_candidates.json` |
| 2 | `step2_quality` | §2.3 + §2.4 co-citation / Eq. 3 inputs | `final_ref_set.json` |
| 3 | `step3_rerank` | §2.4 Eqs. 2-4 | `paper_after_reranking.csv`, `intermediates/step3_*` |
| 4 | `step4_outline` | §2.1 Task 2 | `survey_outline.json` |
| 5 | `step5_generate` | §2.1 Task 2 | `survey_content.json`, `sections/`, `literature_review.tex` |
| 6 | `step6_evaluate` | §3.2 / Appendix D | `evaluation_report.json` |
| + | `short_review` | 2-page deliverable | `literature_review_2page.{md,tex,json}` |
| + | `anchor` / `anchor_merge` | generic anchor paper (any PDF/DOI): scope text + bibliography ingestion + `cited_by_anchor` provenance | `qual_sg/intermediates/anchor.json` |
| + | `crossref` / `datacite` | extra key-less sources (keep running when OpenAlex throttles) | — |
| + | `export_refs` | exports **the pipeline's selected references** (the consumable artefact) | `pipeline_ref.json` |
| + | `import_ranking` | converts an external agent's Markdown/CSV/JSON selection into the pipeline's 28-column ranked CSV (joins the pool, recomputes scores) | `paper_after_reranking_imported.csv` |
| + | `bib_eval` | bibliography-overlap metric (recall/precision/F1 vs any paper) | `intermediates/bib_overlap_*.json` |
| + | `baselines` / `compare` | Fully-LLMGen + Naive-RAG, and their comparison | `baselines/`, `comparison_report.json` |

## Install

```bash
cd "Paper implementation"
python3 -m pip install openai requests pandas numpy scikit-learn rouge-score sentence-transformers
```

## Configure

Copy `qual_sg/.env.example` to `Paper implementation/.env` and set **one** of:

* `GROQ_API_KEY=...` — standalone runs against Groq's free tier.  Models default
  to the strongest open-weights model available there (`openai/gpt-oss-120b`),
  with `llama-3.3-70b-versatile` as fallback.  Free key: <https://console.groq.com/keys>.
* `QUAL_SG_BACKEND=harness` — route all LLM calls through the coding assistant's
  own runtime model instead of an external API (no key required; used for the
  recorded run in `IMPLEMENTATION_NOTES.md`).

Everything else (candidate pool size, co-citation threshold, weights, models,
thresholds) is documented in `qual_sg/config.py`.

## Run

```bash
cd "Paper implementation"
python3 -m qual_sg.run_all                 # steps 1-6, resumable
python3 -m qual_sg.run_all --steps 1,2     # retrieval + quality signals only
python3 -m qual_sg.step3_rerank            # a single step
```

All stages are **idempotent and resumable**: LLM responses are cached by payload
hash in `qual_sg/intermediates/cache/llm_cache.jsonl`, relevance scores are
replayed from `step3_relevance_scores.jsonl`, and generated sections are skipped
when the file already exists.  Set `QUAL_SG_FORCE=1` to redo a stage.

## Reproducibility / audit trail

* `qual_sg/intermediates/cache/llm_audit.jsonl` — every raw model response, model
  name, token usage and timestamp.
* `qual_sg/intermediates/logs/*.log` — per-stage logs.
* `qual_sg/intermediates/step*` — raw step artefacts (candidate pool, co-citation
  statistics, rankings, embeddings, metrics).
* `IMPLEMENTATION_NOTES.md` — narrative record of the approaches and results.
