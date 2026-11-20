# Task 1 — Literature review pipeline (QUAL-SG)

This is a reimplementation of QUAL-SG from *SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models* (Bao et al., EMNLP 2025, arXiv:2508.17647). We applied it to MVFA (arXiv:2403.12570), the base paper.

This folder is `SurveyGen/Paper implementation/` from the working copy. Comments that say "Paper implementation/" refer to it.

## Output for Task 3

**`task3_handoff/final_ref_set.json`** is the file Task 3 uses (from run v6, 16 Nov 2026). It lists 594 candidate papers, each with the same 12 fields, none empty: `doi, title, abstract, year, venue, citation_count, keywords, topics, relevance_score, composite_score, origin, selected`. `task3_handoff/README.md` describes the fields and how each Task 3 request is met. The papers that are left out are listed in `final_ref_set_excluded.json` with the reason.

Task 2 reads the review's reference list (`pipeline_ref.json`) and the survey, not this file.

## Contents

| Path | What it is |
|---|---|
| `qual_sg/` | The pipeline. Its steps: retrieval; quality signals with co-citation and snowball expansion; **who cites MVFA** (v6); abstract filling; **OpenAlex keywords, topics and metadata, and a DOI check** (v6); an LLM relevance judge with MMR selection; **the `final_ref_set.json` export** (v6); the outline; section-wise generation with a claim check; the bibliography; metrics. See `qual_sg/README.md`. |
| `task3_handoff/` | The Task 3 output (above) |
| `runs/v6_2026-11-16/` | Current run: settings (`run_env.sh`, `topic.csv`), `final_ref_set.json`, the excluded list and `RUN_REPORT.md`. v6 keeps v5's 60 selected papers, so v5's review stays valid. |
| `runs/v5_2026-11-06/` | Run that wrote the review: settings and its reference list (`pipeline_ref.json`, which v6 keeps as the selection) |
| `IMPLEMENTATION_NOTES.md` | Approach, design decisions and intermediate results. |
| `*.py` (top level) | Earlier scripts and helpers. These include modified copies of the SurveyGen authors' code, plus build scripts for the final documents and for Task 2's `gap_inputs` (`build_gap_inputs.py`). They use the pre-v6 meaning of `final_ref_set.json` (the candidate pool) and are not part of the current pipeline. |
| `example/` | Topic file. |

## Run

```bash
cd literature_review
cp qual_sg/.env.example .env              # add OPENALEX_API_KEY and an LLM backend; .env is git-ignored
bash runs/v6_2026-11-16/run_env.sh        # runs or resumes every stage; writes runs/v6_2026-11-16/final_ref_set.json
bash runs/v6_2026-11-16/run_env.sh export_final_set   # one stage with the run's settings
```

`python3 -m qual_sg.run_pipeline` is the driver (`--only` / `--redo <stage>`). The older `qual_sg.run_all` runs only the six paper steps and does not write `final_ref_set.json`.

Not committed (see `.gitignore`):
- `.env`;
- the local embedding model (`qual_sg/models/`, `bge-small-en-v1.5`, about 130 MB, downloaded on first use);
- HTTP/LLM caches and the other run outputs (candidate pool, ranking CSV, survey). Without them, a run starts again from retrieval.

`create_ref_json.py`, `extract_prompt_csv.py` and `list_scores.py` are one-off helpers with local paths hard-coded.

The SurveyGen code is licensed under Apache 2.0; see `LICENSE-SurveyGen`.
