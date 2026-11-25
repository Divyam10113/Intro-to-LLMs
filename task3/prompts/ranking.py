"""
Prompts for Stage 4 (final multi-criteria ranking).

* Review: ResearchAgent's ReviewingAgent template (Tables 9-11) and human-induced rubrics (Tables 13-15). The five
  criteria requested by the assignment map to ResearchAgent rubrics as follows:
      Novelty     -> Method / Innovativeness rubric (informed by the novelty-search evidence, as in HypER's use of
                     Lu et al. 2024 novelty checking)
      Feasibility -> Experiment / Feasibility rubric
      Clarity     -> Method / Clarity rubric
      Impact      -> Problem / Significance rubric
      Validity    -> Method / Validity rubric
  ADAPTED: ResearchAgent queries one criterion per call; here one call scores all five criteria for an idea
  (5x fewer calls). Each criterion is still scored against its own full rubric.
* Families: one call per problem groups candidates into method families for the diversity constraint.
"""
import json
from pathlib import Path

RUBRICS = json.loads((Path(__file__).parent / "researchagent_rubrics.json").read_text())
CRITERIA = {
    "Novelty": ("method", "Innovativeness"),
    "Feasibility": ("experiment", "Feasibility"),
    "Clarity": ("method", "Clarity"),
    "Impact": ("problem", "Significance"),
    "Validity": ("method", "Validity"),
}

REVIEW_SYSTEM = (
    "You are an AI assistant whose primary goal is to assess the quality and soundness of scientific research ideas "
    "across diverse dimensions, in order to aid researchers in refining their ideas based on your evaluations and "
    "feedback, thereby enhancing the impact and reach of their work."
)

REVIEW_USER = """You are going to evaluate a research idea on five criteria: Novelty, Feasibility, Clarity, Impact and Validity.

As part of your evaluation, you can refer to the research problem, the target paper, and the closest prior work found by a literature search for this idea.
Research problem: {problem}
Target paper title: {title}
Target paper abstract: {abstract}
Verified facts about the target paper's code: {code_map}
Resources: {resources}

Closest prior work found by the novelty search (query: "{query}"):
{prior}

Now, proceed with your evaluation approach that should be systematic:
- Start by thoroughly reading the research idea, keeping in mind the research problem and the prior work.
- For each criterion, write a short review that is constructive, helpful and concise, and provide a score on a 5-point Likert scale, with 1 being the lowest, ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified. For Novelty, judge the idea against the prior work listed above and against your knowledge of the literature.

{criteria}

The research idea:
{idea}

Return ONLY valid JSON:
{{"Novelty": {{"review": "...", "score": 1-5}}, "Feasibility": {{"review": "...", "score": 1-5}}, "Clarity": {{"review": "...", "score": 1-5}}, "Impact": {{"review": "...", "score": 1-5}}, "Validity": {{"review": "...", "score": 1-5}}}}"""

FAMILY_SYSTEM = "You are an expert in machine-learning research who organises research ideas into method families."
FAMILY_USER = """Research problem: {problem}

Below are {n} candidate research ideas for this problem. Group them into 4-7 method families such that ideas in the same family share their core mechanism (not merely their goal). Give each family a short name.

{ideas}

Return ONLY valid JSON: {{"families": {{"<family name>": ["<idea id>", ...], ...}}}}"""


def criteria_block():
    out = []
    for name, (kind, metric) in CRITERIA.items():
        out.append(f"{name} (rubric: {kind} {metric}):\n" + "\n".join(RUBRICS[kind][metric]))
    return "\n\n".join(out)
