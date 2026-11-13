"""
Prompt templates for the ResearchAgent stages, adapted from Baek et al. (2025),
"ResearchAgent: Iterative Research Idea Generation over Scientific Literature with LLMs",
Tables 6-11 (generation and ReviewingAgent prompts) and Tables 12-15 (criteria and rubrics).

Adaptations (marked '# ADAPTED' in comments below; the prompt text itself carries no markers):
 1. The problem-identification prompt additionally receives a group of limitations of the target
    paper produced by our Task 2 pipeline, and the shared evaluation protocol (gap group C).
    ResearchAgent discovers problems from literature alone; our assignment requires the problems to
    come from the previously identified gaps.
 2. A feasibility constraint describes the resources available for the next assignment.
 3. The paper describes iterative refinement from reviews but does not print the refinement prompt;
    REFINE_* below is written in the same style as Tables 6-8.
"""
import json
from pathlib import Path

RUBRICS = json.loads((Path(__file__).parent / "researchagent_rubrics.json").read_text())

SHORT = {  # Table 12
    "problem": {
        "Clarity": "It assesses whether the problem is defined in a clear, precise, and understandable manner.",
        "Relevance": "It measures whether the problem is pertinent and applicable to the current field or context of study.",
        "Originality": "It evaluates whether the problem presents a novel challenge or unique perspective that has not been extensively explored before.",
        "Feasibility": "It examines whether the problem can realistically be investigated or solved with the available resources and within reasonable constraints.",
        "Significance": "It assesses the importance and potential impact of solving the problem, including its contribution to the field or its broader implications.",
    },
}

RESOURCES = (   # ADAPTED (2)
    "The problem will be investigated by a student team in the next few weeks by modifying the public code of the "
    "target paper (https://github.com/MediaBrain-SJTU/MVFA-AD) and evaluating on the public BMAD benchmark "
    "(brain MRI, liver CT, retinal OCT: RESC and OCT17, chest X-ray, histopathology), using a single Google Colab Pro "
    "GPU (A100 40GB or L4 24GB). No new clinical data can be collected and no new expert annotations can be made."
)

PROTOCOL_C = (  # ADAPTED (1): gap group C (G2, G10, G12, G16), attached to every problem
    "Shared evaluation protocol (from limitation group C: G2, G10, G12, G16): checkpoints and hyper-parameters are "
    "selected on a held-out validation split of the SOURCE modalities and never on the target test set, and the MVFA "
    "baseline is re-run under the same rule; results are reported under a modality-disjoint leave-one-modality-out split "
    "(RESC and OCT17 are held out together, never one with the other in training), with patient-level separation where "
    "identifiers exist, over at least 3 random seeds and 3 few-shot support draws with mean and standard deviation, and "
    "against the missing baselines named in G12 where relevant."
)

# --------------------------------------------------------------------------------- Table 6
PROBLEM_SYSTEM = (
    "You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on "
    "existing scientific literature, in order to aid researchers in discovering novel and significant research "
    "opportunities that can advance the field."
)

PROBLEM_USER = """You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the target paper, those of {n_ref} related papers in the existing literature, {n_ent} entities potentially connected to the research area, and a group of limitations of the target paper identified in a previous critical appraisal.

Understanding of the target paper, related papers, entities, and limitations is essential:
- The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem.
- The related papers are studies that are directly relevant to the primary research topic (the literature review set of the target paper and, where available, studies that have cited the target paper), providing additional context and insights that are essential for understanding and expanding upon the target paper.
- The entities can include topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, serving as auxiliary sources of inspiration or information that may be instrumental in formulating the research problem.
- The limitations are weaknesses of the target paper that the research problem must address.

Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the target paper to understand its core focus.
- Next, read the limitations, which define what the research problem must address.
- Then, proceed to read the titles and abstracts of the related papers to gain a broader perspective and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the target paper, limitations, related papers, and entities, as follows:
Target paper title: {title}
Target paper abstract: {abstract}

Limitations of the target paper to address (group "{group_name}"; shared root cause: {root_cause}):
{limitations}

Related papers:
{related}

Entities: {entities}

Constraints:
- {resources}
- {protocol}
- The problem must be fine-grained and to the point: one well-delimited question about one component of the target method, not a combination of several independent contributions. Name the exact component, setting, and measurable outcome.
{diversity}
With the provided target paper, limitations, related papers, and entities, your objective now is to formulate a research problem that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title and abstract of the target paper, to ensure it remains the focal point of your research problem identification process.

Target paper title: {title}
Target paper abstract: {abstract}

Then, following your review of the above content, please proceed to generate one research problem with the rationale, in the format of
Problem:
Rationale:"""

DIVERSITY = (
    "- A different research problem has already been proposed for these limitations:\n  \"{previous}\"\n"
    "  Your problem must target a clearly different question (a different mechanism, setting, or outcome).\n"
)

# --------------------------------------------------------------------------------- Table 9
REVIEW_PROBLEM_SYSTEM = (
    "You are an AI assistant whose primary goal is to assess the quality and validity of scientific problems across "
    "diverse dimensions, in order to aid researchers in refining their problems based on your evaluations and feedback, "
    "thereby enhancing the impact and reach of their work."
)

REVIEW_PROBLEM_USER = """You are going to evaluate a research problem for its {metric}, focusing on how well it is defined in a clear, precise, and understandable manner.

As part of your evaluation, you can refer to the existing studies that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem.

The existing studies (target paper & related papers) are as follows:
Target paper title: {title}
Target paper abstract: {abstract}
Related papers:
{related}

The resources available to investigate the problem are: {resources}

Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the {metric} of the problem.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified:
{metric}: {short}
{criteria}

I am going to provide the research problem with its rationale, as follows:
Research problem: {problem}
Rationale: {rationale}

After your evaluation of the above content, please provide your review, feedback, and rating, in the format of
Review:
Feedback:
Rating (1-5):"""

# ------------------------------------------------------------------------- refinement (ADAPTED 3)
REFINE_USER = """You previously formulated the following research problem for the target paper.

Research problem: {problem}
Rationale: {rationale}

ReviewingAgents evaluated it on five criteria. Their reviews, feedback, and ratings are:
{reviews}

Revise the research problem so that it addresses the feedback while keeping what the reviewers rated highly. Keep it a single, fine-grained, to-the-point problem that still addresses the same limitations, respects the constraints below, and stays centred on the target paper.

Target paper title: {title}
Target paper abstract: {abstract}
Limitations to address:
{limitations}
Constraints:
- {resources}
- {protocol}

Please provide the revised research problem with its rationale, in the format of
Problem:
Rationale:"""


def criteria_text(kind: str, metric: str) -> str:
    return "\n".join(RUBRICS[kind][metric])
