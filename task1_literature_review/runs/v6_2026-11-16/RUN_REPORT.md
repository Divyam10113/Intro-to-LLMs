# Run v6 (16 November 2026): Task 1 output for Task 3

**Status: complete.** v6 is v5 with five changes requested so that Task 1's output feeds Task 3 directly. The output is now **one file with a fixed format, `final_ref_set.json`**: 594 candidate papers, each with every field filled; 59 are marked `selected`.

The review is unchanged. Selection is pinned to v5's 60 references (`QUAL_SG_FIXED_SELECTION`), so v5's survey, 2-page review, bibliography and metrics (copied here) still hold. Only the candidate pool, the scores and the export are new.

Rerun from `Paper implementation/` with `bash runs/v6_2026-11-16/run_env.sh`. Completed stages are skipped, and API responses and LLM answers are replayed from the cache.

## 1. The requested changes

| # | Request | Done in v6 |
|---|---|---|
| 1 | OpenAlex keywords and topics for every paper | Stage `enrich_metadata` resolves every record in OpenAlex (by id, then DOI, then exact title) and keeps its `keywords` and `topics`. One record OpenAlex had not tagged got them from OpenAlex's own `/text` tagger (same vocabulary). On average a paper has 7.2 keywords and 3.0 topics. |
| 2 | An abstract for every paper | Abstracts are filled from OpenAlex, Europe PMC, arXiv, Semantic Scholar, Crossref and DataCite. 68 records have no open abstract anywhere (mostly Elsevier and Springer papers that cite MVFA). They cannot be judged, so they are left out of the file and listed in `intermediates/final_ref_set_excluded.json`. |
| 3 | A "who cites MVFA" step | New stage `cites_anchor`: OpenAlex `cites:W4402703054` (148 citing works) plus Semantic Scholar (154). 164 papers were new to the pool; 50 were already in it and are now tagged. In the final file **167 papers carry `cites_mvfa`**, 95 of them with relevance ≥ 3. |
| 4 | Years, no empty rows, checked IDs | Every paper has an integer `year`. No field in any row is empty. Every DOI was checked to resolve (OpenAlex, Crossref or DataCite) **to a paper with the same title**. See section 2. |
| 5 | One output file with a fixed format | New stage `final_ref_set` writes `final_ref_set.json` (section 3). The CSV and `pipeline_ref.json` remain pipeline artefacts and are not needed downstream. |

**Prompt-based list.** `final_ref_set.json` is built only from the pipeline. `Prompted_ref.json` (prompt-based, 20 Oct) is not used. The same DOI check run on it would have caught the problem it had: `10.48550/arXiv.2407.12345`, given for MedSAD-CLIP, resolves in DataCite to *"VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions"*.

## 2. Identifier check

| Result | Records |
|---|---|
| DOI resolves to the same title (OpenAlex 668, Crossref 6, DataCite 1) | 676 of 678 |
| DOI pointed to a different paper and was corrected | 1: *Segment Any Anomaly without Training…* carried `10.1109/tcyb.2025.3536165`, which is *"Personalizing Vision-Language Models With Hybrid Prompts…"*; it was replaced by `10.48550/arxiv.2305.10724` |
| Could not be verified (excluded) | 2: *Defect-aware Hybrid Prompt Optimization via Progressive Tuning…* (its DOI is registered under a different title, so it is probably a retitled version; no exact match was found) and *Visualizing Data using t-SNE* (JMLR, no DOI) |

## 3. `final_ref_set.json`

A JSON array, one object per paper, always with these 12 fields in this order:

| Field | Type | Content |
|---|---|---|
| `doi` | string | Verified DOI. arXiv-only papers use `10.48550/arxiv.<id>`. |
| `title` | string | |
| `abstract` | string | |
| `year` | int | Publication year |
| `venue` | string | Journal or proceedings (OpenAlex primary location, else Crossref); `"arXiv"` for 130 preprints |
| `citation_count` | int | OpenAlex `cited_by_count`, fetched 16 Nov 2026 |
| `keywords` | [string] | OpenAlex keywords |
| `topics` | [string] | OpenAlex topics |
| `relevance_score` | int 1–5 | LLM judge (Claude Sonnet 5.5), same rubric and prompt as v5 |
| `composite_score` | float 0–1 | Step-3 weighted composite (citations, author and venue h-index, relevance, diversity), recomputed over the v6 pool |
| `origin` | [string] | How the paper entered the pool, primary route first: `search`, `mvfa_reference` (MVFA's bibliography), `co_citation`, `snowball`, `cites_mvfa`; a second tag records a second route; `anchor` marks MVFA itself |
| `selected` | bool | `true` for the papers the review cites |

Order: the selected papers first (in the review's numbering), then the rest by relevance and composite score.

**Counts:** 594 papers. Primary origin: 234 search, 182 snowball, 120 cites_mvfa, 29 mvfa_reference, 28 co_citation, 1 anchor. Relevance scores 1/2/3/4/5: 114 / 179 / 196 / 85 / 20.

**Why 59 selected, not 60:** v5's reference list contains BMAD twice, once as the CVPRW 2024 paper and once as its arXiv version. The file has one row per paper, so the two are merged into one selected row with the published DOI. Of the 59, 18 also cite MVFA.

**Left out (84, with reasons, in `intermediates/final_ref_set_excluded.json`):** 68 without any open abstract, 13 duplicates (published vs arXiv versions, or the same paper reached twice; the selected or better-scored record is kept and the routes are merged into its `origin`), 2 with unverifiable DOIs (above), and 1 proceedings volume (not a paper).

## 4. Stages run in v6

| Stage | Result |
|---|---|
| `cites_anchor` (new) | 148 OpenAlex and 154 Semantic Scholar citing works → +164 new records, 50 existing records tagged; pool 514 → 678 |
| `fill_abstracts` | Filled 13 of 81 missing abstracts (arXiv 8, Europe PMC 3, Semantic Scholar 2) |
| `anchor_flags` | MVFA's bibliography re-flagged on the larger pool |
| `enrich_metadata` (new) | 673 of 678 resolved in OpenAlex; keywords, topics, year, venue and citation counts filled; DOI check (section 2) |
| `step3_rerank` | 610 records scored (489 v5 scores replayed, 121 new); selection pinned to v5's 60 (60/60 matched) |
| `final_ref_set` (new) | `final_ref_set.json`: 594 papers |

Stages `export_refs` through `metrics` were not rerun. Their outputs are v5's and still match the pinned selection.

**LLM calls:** 120 new relevance judgments, answered by Claude Sonnet 5.5 subagents through the file exchange (the same model and prompt as v5). The batches and raw answers are in `llm_batches/round01_relevance_new_candidates/`. For one batch, the subagent's first pass read only the start of 48 abstracts. Those 48 were re-scored from the full text before the answers were ingested, and 8 scores changed.

## 5. Files

| File | Content |
|---|---|
| **`final_ref_set.json`** | **The output (section 3)** |
| `candidate_pool.json` | Full candidate pool with all provenance fields (called `final_ref_set.json` up to v5) |
| `paper_after_reranking.csv`, `pipeline_ref.json` | Step-3 ranking (now with a `cites_anchor` column) and the 60 selected references (v5's, unchanged) |
| `intermediates/cites_anchor_stats.json`, `enrich_metadata_stats.json`, `final_ref_set_excluded.json` | Statistics of the new stages and the excluded records with reasons |
| `run_env.sh`, `run_state.json`, `run_manifest.json` | Settings, stage status, environment |
| Survey, 2-page review, bibliography, metrics | Copied from v5, still valid for the pinned selection; see `../v5_2026-11-06/RUN_REPORT.md` |

Code changes, in `qual_sg/`: new `cites_anchor.py`, `enrich_metadata.py` and `export_final_set.py`; `config.py` gains `QUAL_SG_POOL_JSON` and `QUAL_SG_FIXED_SELECTION`; `step3_rerank.py` gains pinned selection and a `cites_anchor` CSV column; `run_pipeline.py` registers the three new stages. The code used for v5 is kept in `legacy/qual_sg_v5_2026-11-06/`.
