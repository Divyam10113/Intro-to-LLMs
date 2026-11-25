"""
Prompt templates for the HypoEvolve stage (Liu et al., 2026), Appendix D.

Verbatim structure, adapted from drug repurposing to research ideas for a machine-learning method:
* Pairwise Comparison Prompt (D.2): same structure (compare two, allow a tie, REASONING / CONFIDENCE / WINNER).
  The three judging criteria are mapped from (specificity to the cancer, target evidence, testability) to
  (specificity to the research problem and the target method, grounding in evidence, testability), and the
  "Explicitly DO NOT reward" list is mapped to the failure modes of research ideas.
* Combination Crossover (D.3) and Out-of-Box Mutation (D.3): verbatim instructions.
* Inspiration crossover and drug-substitution mutation are described in Section 3.1 but their prompts are not
  printed; INSPIRATION and SUBSTITUTION below are written in the same style. "Drug substitution" (change the
  compound, keep the mechanistic argument) becomes "component substitution" (change the core technical component,
  keep the problem framing, hypothesis target and evaluation).
* The output format extends D.1 (TITLE / SUMMARY / HYPOTHESIS / RATIONALE) with the Method / Experiment fields
  needed for codable ideas, matching the Stage 2 seed format.
"""

PREFERENCES = (
    "A strong research idea (i) directly solves the stated research problem for the target method, (ii) is grounded in "
    "evidence from the literature and in verified facts about the target method's code, (iii) makes a concrete, "
    "falsifiable prediction with named variables and metrics, (iv) is new relative to the cited literature, and (v) can be "
    "implemented in the target method's public code and evaluated on BMAD with one Colab GPU in a few weeks."
)

OUTPUT_FORMAT = """Required Output Format (valid JSON, all fields non-empty):
```json
{
"Title": "A concise, descriptive title",
"Summary": "Single-sentence summary",
"Hypothesis": "Clear, testable statement in 2-3 sentences with the key variables and how they are measured",
"Rationale": "Detailed explanation including key mechanisms, evidence from the literature and the code, and testability",
"Method": "Step-by-step method: components, equations or pseudo-code, files/functions to change, training and inference, hyper-parameters",
"Experiment": "Datasets and splits, baselines and ablations, metrics, seeds/draws, model selection, and the result that would falsify the hypothesis",
"Provenance": ["titles of the papers the idea builds on"]
}
```"""

# ------------------------------------------------------------------------------------------------ D.2 (adapted)
COMPARE_SYSTEM = "You are an expert reviewer of machine-learning research ideas."
COMPARE_USER = """Compare two research ideas for the SAME research problem and pick the stronger one.

Research problem: {problem}

A stronger idea is one whose proposed change acts on the SPECIFIC mechanism of the target method that causes the problem (here: the released MVFA code and its verified behaviour), rather than a generic improvement that would apply to any anomaly detector.

Judge on:
1. Specificity. Would this idea plausibly solve THIS problem in THIS method better than a generic alternative? A mechanism that applies equally to any model and any problem is WEAKER, not stronger, because it does not explain why it fixes this problem.
2. Evidence. Is the proposed mechanism supported by the cited literature or by verified properties of the code and the benchmark?
3. Testability. Does the idea make a concrete, falsifiable prediction that can be checked on BMAD with one Colab GPU?

Explicitly DO NOT reward: length or polish of the write-up, stacking many modules, vague "add an attention/adapter module" proposals, or ideas that need data, labels or compute that are not available.

Idea A:
{a}

Idea B:
{b}

Respond in exactly this format:
REASONING: <one to three sentences explaining your choice>
CONFIDENCE: <HIGH | MEDIUM | LOW>
WINNER=<A | B | TIE>"""

# ------------------------------------------------------------------------------------------------ D.3
EVOLVE_SYSTEM = "You are an expert machine-learning researcher developing research ideas."

COMBINATION = """You are synthesizing a unified hypothesis from multiple parent hypotheses.
Goal: {goal}
Parent hypotheses: {hypotheses}
Review feedback: {reviews}
Instructions: Integrate the strongest aspects from each parent into a coherent unified hypothesis. Preserve beneficial mechanisms while addressing identified weaknesses. The offspring should be superior to any individual parent.
Criteria for a strong hypothesis: {preferences}
{output_format}"""

INSPIRATION = """You are generating a new hypothesis that uses the ideas of the parent hypotheses as starting points for a different explanation aligned with the goal.
Goal: {goal}
Parent hypotheses (starting points): {hypotheses}
Review feedback: {reviews}
Instructions:
1. Identify what makes each parent promising and what limits it
2. Develop a different mechanism or explanation, aligned with the goal, that is suggested by the parents but is not a merge of them
3. Keep the result concrete and testable
Criteria for a strong hypothesis: {preferences}
{output_format}"""

SUBSTITUTION = """You are revising a single hypothesis by substituting its core technical component.
Goal: {goal}
Hypothesis to revise: {hypotheses}
Review feedback: {reviews}
Instructions: Replace the central technical component of the hypothesis (the specific technique that implements the intervention) with a different, well-motivated technique, while keeping its problem framing, the quantity it targets, and its evaluation. Explain why the substitute should work at least as well.
Criteria for a strong hypothesis: {preferences}
{output_format}"""

OUT_OF_BOX = """You are generating a novel hypothesis inspired by but distinct from provided concepts.
Goal: {goal}
Inspiration (use analogy, not replication): {hypotheses}
Instructions:
1. Identify promising avenues for exploration
2. Develop a detailed, original hypothesis leveraging analogous principles
3. This should not be a mere aggregation of existing methods; think out-of-the-box
Criteria for a strong hypothesis: {preferences}
{output_format}"""
