# Task 1 — Literature review pipeline (QUAL-SG)

This is a reimplementation of QUAL-SG from *SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models* (Bao et al., EMNLP 2025, arXiv:2508.17647). We applied it to MVFA (arXiv:2403.12570), the base paper.

This folder is `SurveyGen/Paper implementation/` from the working copy. Comments that say "Paper implementation/" refer to it.

| Path | What it is |
|---|---|
| `qual_sg/` | The pipeline used for the final run (v5). Its steps: retrieval, quality signals with co-citation and snowball expansion, an LLM relevance judge with MMR selection, the outline, section-wise generation with a claim check, the bibliography, and metrics. See `qual_sg/README.md`. |
| `IMPLEMENTATION_NOTES.md` | Approach, design decisions and intermediate results. |
| `runs/v5_2026-11-06/` | Settings for the final run (`run_env.sh`, `topic.csv`). Run outputs are not committed. |
| `*.py` (top level) | Earlier scripts and helpers. These include modified copies of the SurveyGen authors' code, plus build scripts for the final documents and for Task 2's `gap_inputs` (`build_gap_inputs.py`). |
| `example/` | Topic file. |

## Run

```bash
cd literature_review
cp qual_sg/.env.example .env        # add OPENALEX_API_KEY and an LLM backend; .env is git-ignored
source runs/v5_2026-11-06/run_env.sh
python3 -m qual_sg.run_pipeline     # resumable; --only / --redo <stage>
```

Not committed (see `.gitignore`):
- `.env`;
- the local embedding model (`qual_sg/models/`, `bge-small-en-v1.5`, about 130 MB, downloaded on first use);
- HTTP/LLM caches and run outputs.

`create_ref_json.py`, `extract_prompt_csv.py` and `list_scores.py` are one-off helpers with local paths hard-coded.

The SurveyGen code is licensed under Apache 2.0; see `LICENSE-SurveyGen`.
