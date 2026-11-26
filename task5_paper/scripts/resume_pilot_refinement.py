"""Resume a saved pilot draft at PaperOrchestra's local refinement stage."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCHESTRA = ROOT / "paper-orchestra"
sys.path.insert(0, str(ORCHESTRA))

from methods.agents.content_refinement_agent import ContentRefinementAgent

run = ROOT / "runs" / "pilot_dummy"
raw = run / "latex_writeup" / "raw_draft_paper.tex"
text = raw.read_text(encoding="utf-8")
if r"\usepackage{cvpr}" not in text:
    text, n = re.subn(
        r"(\\documentclass[^\n]*\n)",
        lambda m: m.group(1) + r"\usepackage{cvpr}" + "\n",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit("Could not restore the required CVPR style package")
raw.write_text(text, encoding="utf-8")

# The generated BibTeX came from the saved pilot before the source fix below.
pilot_bib = run / "content_refinement_workdir" / "references.bib"
bib_text = pilot_bib.read_text(encoding="utf-8")
bib_text = re.sub(r"(?<!\\)&", lambda _m: r"\&", bib_text)
pilot_bib.write_text(bib_text, encoding="utf-8")

agent = ContentRefinementAgent(
    experimental_log_path=str(run / "raw_materials" / "experiment_log.md"),
    citation_map_path=str(run / "literature_agent_output" / "citation_map.json"),
    guidelines_path=str(run / "latex_writeup" / "guidelines.md"),
    model_name="gemini-3.5-flash",
    max_reflections=1,
    work_dir=str(run / "content_refinement_workdir"),
)
result = agent.run(texfile_path=str(raw))
print("PILOT_REFINEMENT_RESULT=" + str(result))
