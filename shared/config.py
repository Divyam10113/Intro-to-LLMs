"""Central configuration for the pipeline scripts in pipeline/ (and for anyone who wants it).

Same precedence as shared/config.sh: the environment, then <repo>/.env (KEY=value lines,
git-ignored; see .env.example), then defaults that reproduce the course server's layout
(this repo at proj/Intro-to-LLMs/, with proj/MVFA-AD/ and proj/.venv-task4/ next to it).

Task code reads its own environment variables (listed in .env.example); this module only
gathers them in one place and loads .env, so `python -m shared.config` prints what is in effect.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Set KEY=value lines from .env for keys not already in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


load_dotenv()

PROJ_ROOT = Path(os.environ.get("PROJ_ROOT", REPO_ROOT.parent))
TASK4_PYTHON = Path(os.environ.get("TASK4_PYTHON", PROJ_ROOT / ".venv-task4" / "bin" / "python"))
MVFA_DIR = Path(os.environ.get("MVFA_DIR", PROJ_ROOT / "MVFA-AD"))
TASK4_HDD = Path(os.environ.get("TASK4_HDD", "/mnt/HDD1/divyam/task4"))

TASK0 = REPO_ROOT / "task0_starter"
TASK1 = REPO_ROOT / "task1_literature_review"
TASK2 = REPO_ROOT / "task2_limitations"
TASK3 = REPO_ROOT / "task3_ideation"
TASK4 = REPO_ROOT / "task4_weak_supervision"

# API keys, by stage (names only; values come from the environment or .env)
API_KEYS = {
    "task0 (generate_review.py)": ["AGENTROUTER_API_KEY"],
    "task1": ["OPENALEX_API_KEY", "GROQ_API_KEY"],
    "task2": ["NVIDIA_API_KEY", "AGENTROUTER_API_KEY"],
    "task3": ["LLM_API_KEY", "JUDGE_API_KEY"],
    "task5 (PaperOrchestra only)": ["GEMINI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"],
}

if __name__ == "__main__":
    for k in ("REPO_ROOT", "PROJ_ROOT", "TASK4_PYTHON", "MVFA_DIR", "TASK4_HDD"):
        print(f"{k:13s} {globals()[k]}")
    for stage, keys in API_KEYS.items():
        print(f"{stage:27s} " + ", ".join(f"{k}={'set' if os.environ.get(k) else 'unset'}" for k in keys))
