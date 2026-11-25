#!/usr/bin/env bash
# Run v6 (2026-11-16): v5 plus the Task 1 -> Task 3 integration changes
#   - who cites MVFA (stage cites_anchor), metadata completion and DOI check (enrich_metadata),
#   - one fixed-format output, final_ref_set.json (stage final_ref_set).
# The selected set is pinned to v5's 60 references, so v5's survey and 2-page review stay valid.
# Usage, from "Paper implementation/":
#   bash runs/v6_2026-11-16/run_env.sh            # runs/resumes the pipeline
#   bash runs/v6_2026-11-16/run_env.sh <module>   # runs one qual_sg module with the same settings
export QUAL_SG_RUN_DIR=runs/v6_2026-11-16
export QUAL_SG_TOPIC_CSV=runs/v6_2026-11-16/topic.csv
# candidate pool (called final_ref_set.json up to v5); final_ref_set.json is now the export
export QUAL_SG_POOL_JSON=runs/v6_2026-11-16/candidate_pool.json
# --- v6 additions ---
export QUAL_SG_ANCHOR_WORKS=10.1109/cvpr52733.2024.01081,2403.12570   # MVFA (CVPR 2024 + arXiv)
export QUAL_SG_ANCHOR_LABEL=mvfa
export QUAL_SG_FIXED_SELECTION=runs/v5_2026-11-06/pipeline_ref.json
# --- v5 settings (unchanged) ---
export QUAL_SG_EARLIEST_YEAR=2014
export QUAL_SG_YEAR_MODE=now
export QUAL_SG_EVAL_POOL_TARGET=0
export QUAL_SG_ANCHOR_PDF=../2403.12570v1.pdf
export QUAL_SG_ANCHOR_PRIORITY=boost
export QUAL_SG_BACKEND=file
export QUAL_SG_JUDGE_MODEL=claude-sonnet-5-5     # relevance judging, query expansion
export QUAL_SG_EVAL_MODEL=claude-sonnet-5-5
export QUAL_SG_OUTLINE_MODEL=claude-opus-5-5
export QUAL_SG_WRITER_MODEL=claude-opus-5-5
export QUAL_SG_SURVEY_WORDS=4800
export QUAL_SG_SURVEY_SECTIONS=4-6
export QUAL_SG_SURVEY_SUBSECTIONS=12-15
export QUAL_SG_REFS_PER_SUBSECTION=10
export QUAL_SG_CLAIM_CHECK=1
export QUAL_SG_HTTP_CACHE=1
if [ $# -gt 0 ]; then python3 -m "qual_sg.$1" "${@:2}"; else python3 -m qual_sg.run_pipeline $QUAL_SG_PIPELINE_ARGS; fi
