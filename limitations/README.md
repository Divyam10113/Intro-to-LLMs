# Task 2 — Research gaps and problem statement for MVFA (arXiv:2403.12570)

An evidence-grounded, multi-agent pipeline that extracts the limitations of the base paper and
turns them into a research problem. It builds on the three assigned papers and adds three components of our own.

| Component | From |
|---|---|
| Extractor / Analyzer / Reviewer / **Citation** agents, Judge with one regeneration round (score < 8/10), Master with provenance tags | Multi-Agent LLMs for Generating Research Limitations (arXiv:2601.11578) |
| Citation-agent RAG: indexes of **cited-in** and **cited-by** papers, chunked by section; hybrid BM25 + dense retrieval (50/50) → top-20 → LLM re-rank, keep ≥ 8/10 | BAGELS (arXiv:2505.18207), 2601.11578 |
| Aspect taxonomy (Methodology, Experimental Design, Results & Analysis, Literature); Faithfulness / Soundness / Importance rubric | LimitGen (arXiv:2507.02694) |
| Pointwise LLM-as-judge **coverage** (C_GT, C_LLM, F1) | BAGELS, 2601.11578 |
| **Verifier**: each limitation must quote its evidence verbatim (checked by string match), then an LLM checks that the evidence supports the claim | ours |
| **Clinical & Deployment** aspect specific to medical anomaly detection | ours |
| **Gap status**: each limitation is checked against post-2024 papers and the Task 1 review, and labelled open / partially addressed / addressed; the research problem is built from the open ones | ours |
| **Reference set without OpenReview**: citing papers that discuss MVFA are split in half. One half is evidence for retrieval; the other half is never indexed, and its critiques of MVFA become the evaluation reference | ours |

## Pipeline

```
build_corpus.py   OpenAlex: 125 citing + 62 cited papers  ─┐
                  Task 1 selected papers (abstracts)        ├─► arXiv full text ─► section chunks
                  find passages that discuss MVFA          ─┘     │
                  split MVFA-discussing citing papers: evidence │ held-out
run_pipeline.py   Extractor ─┐                                   │
                  Analyzer  ─┤                                   │
                  Reviewer  ─┼─► Judge (≤1 regen) ─► Verifier ─► Master ─► Gap status ─► Problem statement
                  Citation  ─┘   (queries → hybrid retrieval → LLM re-rank ≥ 8 + MVFA-mention passages)
evaluate.py       reference (held-out critiques) · baselines (zero-shot, old pipeline) · scoring
```

## Run

```bash
pip install openai requests numpy scikit-learn pymupdf      # sentence-transformers optional (better dense retrieval)
cd limitations
echo 'AGENTROUTER_API_KEY=...' > .env   # gitignored; or export it in your shell
python build_corpus.py                 # no LLM; ~10 min first time (arXiv rate limit), cached afterwards
python run_pipeline.py                 # full system      -> outputs/full/research_gaps.md
python run_pipeline.py --no_citation   # ablation         -> outputs/full_nocite/
python evaluate.py reference           # reference set from held-out citing papers
python evaluate.py baselines           # zero-shot + old pipeline, same models
python evaluate.py score full full_nocite zero_shot old_pipeline   # -> outputs/evaluation.md
```

Models (AgentRouter): generation uses `gpt-6-astra`, falling back to `gpt-6.1-sol` only if a call fails.
The evaluator starts with `gpt-6.1-sol`, so the system isn't graded by the model that generated it. If the router has no backend for a model, that model is skipped for the rest of the run.
GPT models are called through the Responses API (`/v1/responses`); the router does not serve them on `/v1/chat/completions`.
Override with `GEN_MODELS=... EVAL_MODELS=...`. All LLM calls are cached in `cache/`, so re-runs are free.

## Outputs

- `outputs/full/research_gaps.md` — **the deliverable**: bulleted gaps, a research-problem paragraph, and an appendix
  listing every limitation with its evidence quotes, provenance, importance and status in later work.
- `outputs/full/final_limitations.json`, `verified.json`, `workers.json`, `citation_passages.json` — full trace.
- `outputs/evaluation.md` — comparison table.

## Caveats
- The reported runs use `gpt-6-astra`, with `gpt-6.1-sol` as a fallback for calls where astra was unavailable. In the
  full run, the Reviewer and Master calls were answered by `gpt-6.1-sol`. Each answer's model is recorded in `cache/`.
- The evaluation covers a single paper. The reference set is critiques that later papers wrote about MVFA, so it is incomplete
  by construction. For that reason we report C_GT (recall) as the main number (as 2601.11578 does) and treat C_LLM as indicative only.
- Full text is only available for papers on arXiv or with an open-access PDF (35 of 223). Other papers contribute their abstracts only.
- The `old_pipeline` baseline reads the Task 1 review, which summarises some of the held-out papers. If anything, that gives
  the baseline a slight advantage on coverage.
