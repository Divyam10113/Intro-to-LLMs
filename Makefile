# One command per stage. `make help` lists them.
#
# Offline targets (no paid API call, no GPU, no change to tracked files):
#   smoke / taskN-dry-run, check, test, sync-dry-run
# Real runs need keys or GPUs (see README.md "What needs keys or GPUs"): taskN, all.
#
# Settings come from the environment, then .env (copy .env.example), then shared/config.sh defaults.

SHELL := /bin/bash
.DEFAULT_GOAL := help
CONFIG := . shared/config.sh
# Python for Tasks 0-3: 3.12+ with requirements-smoke.txt (smoke) or requirements.txt (real runs).
PYTHON ?= $(if $(wildcard .venv-pipeline/bin/python),.venv-pipeline/bin/python,python3)
PY_ABS := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
TASK4_SUITES := round1 round2 round3 followup

.PHONY: help venv config sync sync-dry-run check test test-fast smoke \
        task0-dry-run task1-dry-run task2-dry-run task3-dry-run task4-dry-run task4-smoke-cpu task5-dry-run \
        task0 task1 task2 task3 task4 task5 all

help:  ## list the commands
	@echo "Usage: make <target>   (settings: environment > .env > shared/config.sh defaults)"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------------- setup
venv:  ## create .venv-pipeline (Python 3.12+) with requirements-smoke.txt for Tasks 0-3
	@test -x .venv-pipeline/bin/python || $${PYTHON312:-python3} -m venv .venv-pipeline
	.venv-pipeline/bin/pip install -q -r requirements-smoke.txt

config:  ## print the paths and which API keys are set (never the keys)
	@$(PYTHON) -m shared.config

# ----------------------------------------------------------------------------- handoffs and checks
sync:  ## copy every handoff forward (Task 1 -> 3, Task 2 -> 3, Task 3 -> 4) and rewrite the sha256 manifest
	$(PYTHON) pipeline/sync_handoffs.py

sync-dry-run:  ## show which handoff copies would change; change nothing
	$(PYTHON) pipeline/sync_handoffs.py --dry-run

check:  ## fail on handoff drift, changed frozen files, broken symlinks, tracked weights or server paths
	$(PYTHON) pipeline/sync_handoffs.py --check
	$(PYTHON) pipeline/check_repo.py

test:  ## Task 4 test suites on CPU (round1 27, round2 41, round3 80, followup 59; round1 takes 2-4 min)
	@$(CONFIG); set -e; for d in $(TASK4_SUITES); do echo "== task4 $$d"; \
	  (cd task4_weak_supervision/$$d && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
	   "$$TASK4_PYTHON" -m pytest -q -p no:cacheprovider tests); done

test-fast:  ## the same without round 1 (its model test loads CLIP on CPU)
	@$(MAKE) --no-print-directory test TASK4_SUITES="round2 round3 followup"

# ----------------------------------------------------------------------------- offline smoke runs
smoke:  ## every stage's dry run (task0..task5), then check
	$(PYTHON) pipeline/smoke.py all
	@$(MAKE) --no-print-directory check

task0-dry-run:  ## course starter: regenerate llm_prompt.txt etc. in a scratch copy and compare
	$(PYTHON) pipeline/smoke.py task0
task1-dry-run:  ## QUAL-SG: compile + import every module, validate the 594-paper hand-off
	$(PYTHON) pipeline/smoke.py task1
task2-dry-run:  ## limitations: compile + import, validate the 39 limitations
	$(PYTHON) pipeline/smoke.py task2
task3-dry-run:  ## ideation: replay stages 0-4 from logs/cache (no model call) and compare every output
	$(PYTHON) pipeline/smoke.py task3
task4-dry-run:  ## Task 4: frozen hashes, re-run frozen analysers on committed results, list GPU queues
	$(PYTHON) pipeline/smoke.py task4

task5-dry-run:  ## Task 5 (the paper): placeholder, not added yet
	$(PYTHON) pipeline/smoke.py task5

task4-smoke-cpu:  ## Task 4 round 3 end-to-end CPU smoke (tiny training on CPU; needs the data on TASK4_HDD)
	@$(CONFIG); OUT=$${OUT:-$$(mktemp -d)/task4r3_smoke} bash task4_weak_supervision/round3/smoke_cpu.sh

# ----------------------------------------------------------------------------- real runs (keys / GPU)
task0:  ## starter scripts for real: arXiv scrape, rank (embedding model), prompt, review (AGENTROUTER_API_KEY)
	cd task0_starter && $(PY_ABS) scholar_scraper.py && $(PY_ABS) rank_papers.py && \
	  $(PY_ABS) generate_prompt.py && $(PY_ABS) extract_related.py && $(PY_ABS) generate_review.py

task1:  ## QUAL-SG run v6 (OPENALEX_API_KEY + LLM; resumable, exits 10 when file-backend prompts wait)
	cd task1_literature_review && PATH="$(dir $(PY_ABS)):$$PATH" bash runs/v6_2026-11-16/run_env.sh

task2:  ## limitation pipeline, main run (LLM_PROVIDER=nim, NVIDIA_API_KEY); see task2_limitations/README.md
	cd task2_limitations && LLM_PROVIDER=$${LLM_PROVIDER:-nim} $(PY_ABS) run_pipeline.py

task3:  ## ideation, all stages (replays logs/cache; set LLM_BACKEND=openai + keys to call a model)
	cd task3_ideation && $(PY_ABS) run_all.py

task4:  ## Task 4 follow-up end to end on the GPUs (go.sh: tests, GPU wait, smoke, 10 runs, analysis)
	@echo "Task 4 trains on shared GPUs. Rounds: task4_weak_supervision/{round1,round2,round3}/run_all.sh;"
	@echo "the pre-registered follow-up: cd task4_weak_supervision/followup && ./go.sh  (see its README)."
	@test "$${CONFIRM_GPU:-0}" = 1 || { echo "Refusing to start GPU jobs: re-run with CONFIRM_GPU=1"; exit 2; }
	cd task4_weak_supervision/followup && ./go.sh

task5:  ## Task 5 (the paper): not added yet; prints a note and exits 0
	@echo "Task 5 (the paper) is not added yet: see task5_paper/README.md."

all:  ## sync, then task1 -> task2 -> task3 -> task4 -> task5 for real (keys and CONFIRM_GPU=1 needed)
	@$(MAKE) --no-print-directory sync task1 task2 task3 task4 task5
