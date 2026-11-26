"""
Prompt templates for the HypER stage (Vasu et al., EMNLP 2025).

* Relevancy scoring: the authors' released prompt (code/chain_construction/prompts/relevancy_scoring_prompt_002.py,
  identical in substance to Listing 1 of the paper) is imported unchanged from
  prompts/hyper_relevancy_scoring_prompt_002_original.py, together with the authors' two few-shot examples
  (data/hyper_fewshot/, built exactly as in RAG_temporal_extension_llama.py). The examples come from clinical-trial
  literature; they illustrate the 0/1/2 scale and output format.
  ADAPTED (R1): each paper in the list additionally carries the sentence(s) in which it cites the source paper
  (Semantic Scholar citation contexts), when available.

* Hypothesis generation: Listing 4 of the paper ("adapted from Baek et al., 2024a"), used at inference on a
  literature chain. ADAPTED (G1): the research problem selected in Stage 1 is given as the focus; (G2) a short,
  code-verified map of the target paper's public code is given so that the method is codable; (G3) the output adds
  TITLE/SUMMARY (HypoEvolve's hypothesis format) and Method / Experiment fields (ResearchAgent's p, m, d) and a
  provenance list, because these seeds become the initial population of HypoEvolve.
"""
import json
from pathlib import Path

from prompts.hyper_relevancy_scoring_prompt_002_original import relevancy_scoring_prompt

ROOT = Path(__file__).resolve().parent.parent
RELEVANCY_SYSTEM = "You are a helpful assistant designed to evaluate scientific literature."  # Listing 1


def few_shot_prompt() -> str:
    """Identical construction to HypER's RAG_temporal_extension_llama.py main()."""
    data = json.loads((ROOT / "data" / "hyper_fewshot" / "output.json").read_text())
    evaluations = json.loads((ROOT / "data" / "hyper_fewshot" / "gpt4_output").read_text())
    few_shot_papers = data[1:3]
    few_shot_evaluations = list(evaluations.items())[1:3]
    s = "Examples:\n\n"
    for idx, paper in enumerate(few_shot_papers):
        s += f"{idx + 1}. Title: {paper['Title']} Abstract: {paper['Abstract']}; ({paper['Year']})\n"
    s += "\nExample evaluations in JSON format:\n```json\n{\n"
    for idx, (eval_title, d) in enumerate(few_shot_evaluations):
        s += f'    "{eval_title}": {{\n        "explanation": "{d["explanation"]}",\n        "relevance": {d["relevance"]}\n    }}'
        if idx < len(few_shot_evaluations) - 1:
            s += ",\n"
    s += "\n}\n```"
    return s


def relevancy_prompt(source, year, papers) -> str:
    """HypER utils.relevancy_prompt, plus citation contexts (ADAPTED R1)."""
    paper_list = ""
    for idx, p in enumerate(papers):
        paper_list += f"\n\n{idx + 1}. Title: {p['title']} Abstract: {p.get('abstract') or 'No abstract available'}"
        ctx = [c for c in (p.get("citation_contexts") or []) if c][:3]
        if ctx:
            paper_list += " Citation context(s) in which this paper cites the source paper: " + " | ".join(ctx)
    return relevancy_scoring_prompt().format(few_shot_prompt=few_shot_prompt(), source_title=source["title"],
                                             source_abstract=source["abstract"], year=year, paper_list=paper_list)


# ---------------------------------------------------------------------------------------------- Listing 4 (adapted)
GEN_SYSTEM = (  # Listing 4, verbatim
    "You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on "
    "existing scientific literature, in order to aid researchers in discovering novel and significant research "
    "opportunities that can advance the field."
)

CODE_MAP = """Code-verified facts about the public code of the source paper (github.com/MediaBrain-SJTU/MVFA-AD):
- CLIP/adapter.py: CLIP_Inplanted wraps a frozen CLIP ViT-L/14-336 (input 240x240); ClipAdapter bottleneck adapters are inserted after layers [6, 12, 18, 24]; there are two adapter sets, seg_adapters (segmentation branch) and det_adapters (detection branch), mixed residually with ratio 0.1.
- prompt.py / utils.py: the text side is two frozen text anchors (normal, abnormal) averaged over 35 templates.
- train_zero.py (zero-shot, leave-one-dataset-out): det branch trained with BCE on the MEAN-pooled patch anomaly map for every image; seg branch trained with Focal + Dice on pixel masks only for BrainMRI, LiverCT, RESC; batch size 1; the checkpoint is selected on the target test set.
- train_few.py: few-shot fine-tuning on K normal + K abnormal (masked) images from BMAD's valid/ split, one fixed draw (dataset/fewshot_seed); the checkpoint is selected on the target test set.
- test_few.py: zero-shot map (text similarity) and few-shot map (1 - max cosine to a memory bank of support patch features, per level) are each min-max normalised over the whole test set and added with weight 0.5; image score = max of fused pixel map (masked datasets) or mean-pooled det score (others).
- BMAD: BrainMRI, LiverCT, RESC have masks; OCT17, ChestXray, HIS have image labels only; each has train/valid/test splits."""

GEN_USER = """You are going to generate a research idea that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {n_rel} related papers in the existing literature, and a research problem that the idea must address.

Understanding of the target paper, and the related papers is essential:
- The source paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem.
- The related papers are arranged in temporal order of citation, such that paper 1 cites the source paper, paper 2 cites paper 1 and so on. The relevant papers provide additional context and insights that are essential for understanding and expanding upon the target paper. However, all the papers in the list may not be relevant to the primary research you are focusing on. Identify the most relevant papers from the list in your analysis and only use those for research idea generation.

Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list. Identify the papers that form a logical reasoning chain starting from the source paper.
- Use only these papers to gain a broader perspective about the progression of the primary research topic over time.

Source paper title: {title}
Source paper abstract: {abstract}
Source paper year of publication: {year}
Related papers: {related}

Research problem the idea must address (selected in an earlier step):
{problem}

{code_map}

Required evaluation protocol: {protocol}

With the provided source paper and the related papers, your objective now is to formulate a research idea that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant, and that is a concrete solution to the research problem above. Before crafting the idea, revisit the title and abstract of the source paper, to ensure it remains the focal point. Your research idea will be scored for clarity. It should contain a short description of the general research idea and its impact followed by more details on all the variables and how they will be measured. State clearly how the outcome could potentially be measured.

Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a relationship between two variables like independent or dependent variables or left group and right group in a given context. Your hypothesis should contain the key variable or variables from your research idea and how they will be measured. Your hypothesis will be scored on clarity and novelty.

Then, following your review of the above content, please proceed to analyze the progression of the research topic. For analysis, output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly (say so in their entry).

The idea must be complete enough to be implemented directly in the source paper's public code in the next few weeks on one Colab GPU: name the files and functions that change, the new components (with equations or pseudo-code where needed), the training and inference procedure, and all hyper-parameters.

Output a JSON object in the following format:
```json
{{
"Analysis": {{"<paper title>": "<how it builds on the previous papers, or why it is ignored>", ...}},
"Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
"Title": "A concise, descriptive title",
"Summary": "Single-sentence summary",
"Research idea": "Delineate an elaborate research idea here including the key variables.",
"Hypothesis": "Provide a concrete testable hypothesis that follows from the above research idea here",
"Method": "Step-by-step method: components, equations or pseudo-code, files/functions to change, training and inference, hyper-parameters.",
"Experiment": "Datasets and splits, baselines and ablations, metrics, seeds/draws, model selection, and the result that would falsify the hypothesis.",
"Provenance": ["<title of each related paper the idea actually builds on>"]
}}
```
This JSON will be automatically parsed, so ensure the format is precise. DO NOT leave any field empty. If you cannot generate a specific part, provide a best guess."""
