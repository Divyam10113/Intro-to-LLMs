#!/usr/bin/env bash
# Run v5 (2026-11-06): the v4 configuration plus the approved changes
# (retrieval F1, F2; writing 1-9, 11, 13). Usage, from "Paper implementation/":
#   bash runs/v5_2026-11-06/run_env.sh            # runs/resumes the pipeline
#   bash runs/v5_2026-11-06/run_env.sh <module>   # runs one qual_sg module with the same settings
export QUAL_SG_RUN_DIR=runs/v5_2026-11-06
export QUAL_SG_TOPIC_CSV=runs/v5_2026-11-06/topic.csv
# --- v4 settings (IMPLEMENTATION_NOTES.md section 9) ---
export QUAL_SG_EARLIEST_YEAR=2014
export QUAL_SG_YEAR_MODE=now
# v4 judged a 76-paper shortlist; expansion papers have no similarity score and were
# never judged, so the final set held 0 MVFA references. Decision (6 Nov): judge the whole pool.
export QUAL_SG_EVAL_POOL_TARGET=0
export QUAL_SG_ANCHOR_PDF=../2403.12570v1.pdf
export QUAL_SG_ANCHOR_PRIORITY=boost
# --- LLM: Claude models answering through the file exchange ---
export QUAL_SG_BACKEND=file
export QUAL_SG_JUDGE_MODEL=claude-sonnet-5-5     # relevance judging, query expansion
export QUAL_SG_EVAL_MODEL=claude-sonnet-5-5      # claim check (differs from the writer)
export QUAL_SG_OUTLINE_MODEL=claude-opus-5-5
export QUAL_SG_WRITER_MODEL=claude-opus-5-5
# --- approved writing changes 4-6 (defaults made explicit) ---
export QUAL_SG_SURVEY_WORDS=4800
export QUAL_SG_SURVEY_SECTIONS=4-6
export QUAL_SG_SURVEY_SUBSECTIONS=12-15
export QUAL_SG_REFS_PER_SUBSECTION=10
export QUAL_SG_CLAIM_CHECK=1
export QUAL_SG_HTTP_CACHE=1
if [ $# -gt 0 ]; then python3 -m "qual_sg.$1" "${@:2}"; else python3 -m qual_sg.run_pipeline; fi
