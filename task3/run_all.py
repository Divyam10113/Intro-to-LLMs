#!/usr/bin/env python3
"""
run_all.py — runs the whole Task 3 pipeline in order.

    python3 run_all.py                 # replay from the cache (no API needed) and rebuild all outputs
    LLM_BACKEND=openai ... python3 run_all.py   # run with a real model (see README)

With the file backend, a stage stops when it needs answers that are not cached (see llm_client.py); with an API
backend every stage runs end to end.
"""
import importlib
import sys

STAGES = ["stage0_gaps", "stage0b_verify", "stage1_problems", "stage2_hyper", "stage3_evolve", "stage4_rank",
          "export_explorer_data"]

if __name__ == "__main__":
    for s in STAGES:
        print(f"\n===== {s}")
        mod = importlib.import_module(s)
        try:
            mod.main()
        except Exception as e:  # PendingCalls in file mode
            print(f"{s} stopped: {e}")
            sys.exit(1)
