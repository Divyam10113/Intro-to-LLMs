"""Run the whole QUAL-SG pipeline (steps 1-6) in order.

    cd "Paper implementation"
    python3 -m qual_sg.run_all                # all steps, resumable
    python3 -m qual_sg.run_all --steps 1,2    # a subset
    python3 -m qual_sg.run_all --steps 3 --force

Each step is also runnable on its own, e.g. ``python3 -m qual_sg.step3_rerank``.
Steps 3-6 need an LLM backend: either ``GROQ_API_KEY`` (standalone) or the
in-session harness backend (``qual_sg/harness_backend.py``).
"""
from __future__ import annotations

import argparse
import importlib
import time
import traceback

from . import config, common

STEP_MODULES = {
    1: "qual_sg.step1_retrieve",
    2: "qual_sg.step2_quality",
    3: "qual_sg.step3_rerank",
    4: "qual_sg.step4_outline",
    5: "qual_sg.step5_generate",
    6: "qual_sg.step6_evaluate",
}
STEP_NAMES = {
    1: "literature retrieval",
    2: "quality-signal augmentation + co-citation expansion",
    3: "quality-aware re-ranking",
    4: "outline generation",
    5: "section-wise survey generation",
    6: "evaluation",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="QUAL-SG end-to-end runner")
    parser.add_argument("--steps", default="1,2,3,4,5,6", help="comma-separated step numbers")
    parser.add_argument("--force", action="store_true", help="ignore existing outputs")
    args = parser.parse_args()
    if args.force:
        import os

        os.environ["QUAL_SG_FORCE"] = "1"

    config.ensure_dirs()
    logger = common.get_logger("run_all")
    logger.info("QUAL-SG run: topic=%s", common.load_topic().title)
    totals = {}
    for step in [int(s) for s in args.steps.split(",") if s.strip()]:
        module_name = STEP_MODULES.get(step)
        if not module_name:
            raise SystemExit(f"unknown step {step}")
        logger.info("=" * 78)
        logger.info("STEP %d - %s (%s)", step, STEP_NAMES[step], module_name)
        logger.info("=" * 78)
        t0 = time.time()
        module = importlib.import_module(module_name)
        try:
            module.main()
        except Exception:  # noqa: BLE001
            logger.error("step %d failed:\n%s", step, traceback.format_exc())
            raise
        totals[step] = round(time.time() - t0, 1)
        logger.info("step %d done in %.1fs", step, totals[step])
    logger.info("All done. Per-step seconds: %s", totals)


if __name__ == "__main__":
    main()
