"""Resume a saved PaperOrchestra draft at ContentRefinementAgent."""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper-orchestra"))
from methods.agents.content_refinement_agent import ContentRefinementAgent

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--model", default="gemini-3.5-flash")
args = parser.parse_args()
run = args.run_dir.resolve()
raw = run / "latex_writeup" / "raw_draft_paper.tex"
work = run / "content_refinement_workdir"
text = raw.read_text(encoding="utf-8")
if r"\usepackage{cvpr}" not in text:
    text, count = re.subn(
        r"(\\documentclass[^\n]*\n)",
        lambda m: m.group(1) + r"\usepackage{cvpr}" + "\n",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("Could not restore required CVPR style package")
raw.write_text(text, encoding="utf-8")
for name in ("figures/fig1_method.pdf", "figures/fig2_liver_heatmaps.pdf"):
    source = run / "raw_materials" / name
    if not source.is_file():
        raise SystemExit(f"Missing supplied figure: {source}")
    for target_dir in (run / "latex_writeup" / "figures", work / "figures"):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / Path(name).name).write_bytes(source.read_bytes())
if not (work / "references.bib").is_file():
    (work / "references.bib").write_bytes((run / "latex_writeup" / "references.bib").read_bytes())
bib_path = work / "references.bib"
bib = bib_path.read_text(encoding="utf-8")
bib_path.write_text(re.sub(r"(?<!\\)&", lambda _m: r"\&", bib), encoding="utf-8")

agent = ContentRefinementAgent(
    experimental_log_path=str(run / "raw_materials" / "experiment_log.md"),
    citation_map_path=str(run / "literature_agent_output" / "citation_map.json"),
    guidelines_path=str(run / "latex_writeup" / "guidelines.md"),
    model_name=args.model,
    max_reflections=3,
    work_dir=str(work),
)
result = agent.run(texfile_path=str(raw))
print("REFINEMENT_RESULT=" + str(result))
