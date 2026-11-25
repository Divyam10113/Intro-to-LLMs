# Deep-research agent prompt — quality-aware literature review (paper-agnostic)

Copy everything inside the fenced block into any deep-research agent (ChatGPT Deep
Research, Gemini Deep Research, Perplexity, Elicit, Claude with web search, …).
Fill only the four placeholders. **Nothing in the prompt is specific to a
particular paper** — it works for any anchor or topic you supply.

```text
ROLE
You are a research librarian and survey author. You produce a short, fully
referenced literature review using an explicit, auditable retrieval and
selection procedure. You never guess a citation: every reference you list must
be a real publication you actually retrieved, with a verifiable identifier.

INPUTS (fill these in)
- ANCHOR: "<paper title / DOI / URL, or NONE>"
- TOPIC: "<one-line topic title; keywords separated by ';'>"
- K: <number of references the review must cite, e.g. 30>
- YEAR_RANGE: <start year>–<end year or "current">

If ANCHOR is not NONE, treat it as the authoritative statement of scope: read its
title, abstract, introduction and bibliography before anything else. If ANCHOR is
NONE, treat TOPIC (+ keywords) as the scope and skip every step that mentions the
anchor.

PROCEDURE — follow the phases in order and report the intermediate counts.

Phase 1 — Scope definition
1. Extract from ANCHOR (or TOPIC) a 3–5 sentence SCOPE STATEMENT covering: the
   problem, the method families involved, the application domain, and the
   evaluation setting. Use it verbatim as the relevance criterion later.
2. Expand the scope into 6–10 short search queries (3–8 words) that together
   cover sub-tasks, method families, model/architecture names, dataset names and
   benchmark names. Do not use only the literal topic wording.

Phase 2 — Candidate retrieval (recall first, precision later)
3. Backward snowballing: if ANCHOR exists, resolve EVERY entry of its
   bibliography to a real record and add it to the candidate pool. Report how
   many resolved.
4. Forward snowballing: add papers that cite ANCHOR (and, for the 5 most relevant
   retrieved papers, papers that cite them), newest first.
5. Database search: for each Phase-1 query, retrieve the top ≈100 results from at
   least two independent sources (e.g. OpenAlex, Semantic Scholar, arXiv, Crossref,
   PubMed as appropriate). Record DOI, title, abstract, year, venue, citation count.
6. De-duplicate on DOI and on normalised title (the same paper is often indexed
   twice under a preprint and a journal DOI). Report the raw and de-duplicated counts.
7. Keep the candidates with the highest cosine similarity between the SCOPE
   STATEMENT and (title + abstract); retain ≈10× K. Report the similarity range.

Phase 3 — Quality signals (for every retained candidate)
8. Collect, with sources: citation count, citations per year (citation count ÷
   years since publication — never raw counts alone, they favour old megacited
   papers), first- and last-author h-index, venue h-index or CORE rank, and
   whether the paper is a dataset/benchmark paper, a method paper, a survey, or
   an application paper.
9. Group each numeric signal into quartiles across the candidate pool and score
   Q1=4 … Q4=1.

Phase 4 — Relevance judging and selection
10. Score every candidate's topical relevance 1–5 against the SCOPE STATEMENT,
    using the criteria: 1 = unrelated; 2 = loose background; 3 = one sub-aspect;
    4 = substantially addresses the scope; 5 = entirely focused / foundational.
    Judge title + abstract, not the venue.
11. Apply a relevance gate: prefer candidates scoring ≥3; include score-2
    candidates only to fill remaining slots. Never include score-1 candidates.
12. Rank by the average of three ranks — relevance, academic impact (weighted
    citations-per-year, author h-index, venue reputation), and content diversity
    (mean cosine distance to the other candidates) — and select the top K with a
    maximal-marginal-relevance step so the final set is not redundant.
13. Report the final K as a table: title, year, venue, citations/year, relevance
    score, one-line reason for inclusion. Explicitly flag any entry that entered
    because of impact or diversity despite a relevance score ≤2.

Phase 5 — Write the review
14. Write a ~1000–1200 word review (about 2 pages) organised into 5 thematic
    sections: background/motivation → core architectures or methods → zero-/few-shot
    or main-line approaches → domain-specific generalisation → open challenges.
15. Every claim must be supported by an in-text citation in the form ref [n],
    using only the K selected references; number them in order of first mention.

RULES
- NEVER invent a reference, DOI, venue or number. If a value is unknown, write
  "unknown".
- Prefer primary sources; state the database each record came from.
- Separate what you retrieved from what you inferred; label inferences.
- The procedure must not change if the ANCHOR changes: no paper-specific heuristics.
- If coverage of ANCHOR's bibliography is low, say so and explain which
  categories (datasets, seminal methods, prior surveys) are missing.

OUTPUT
1. Scope statement.
2. Retrieval log: per-source counts, de-duplication, similarity range, expansion counts.
3. Selection table (Step 13) + the reference list with DOIs.
4. The review text (Step 14).
5. Coverage report: how many of ANCHOR's own references were rediscovered
   (recall), how many of your K are in ANCHOR's bibliography (precision), and the
   list of notable misses with the reason (not retrieved / no abstract / scored low).
```

## Why these knobs (mapping to the QUAL-SG paper)

| Prompt element | Paper / this repo |
|---|---|
| 3-query expansion + multi-source retrieval | `step1_retrieve.py` (`expand_queries`, OpenAlex + arXiv) |
| Backward/forward snowballing | `anchor.py` + co-citation expansion (`CO_CITATION_MIN`) |
| cosine-similarity pre-filter | Eq. (1) `Sim(q, d_i)`; `semantic_filter` |
| quartile impact bins, citations-per-year | Eq. (3) `S_a`; `CITE_AGE_NORMALIZE` |
| relevance 1–5 rubric | Eq. (2) `S_t` (LLM-as-judge) |
| average rank + diversity | Eq. (4) `S_d`; `SELECTION=mmr` |
| coverage report (recall/precision vs anchor bibliography) | `bib_eval.py` |

## Variations

* **No anchor, straight topic:** set `ANCHOR = NONE`; Phases 3–5 are unchanged, and
  the coverage report in the output becomes optional.
* **Strict bibliography alignment:** tell the agent "prioritise the anchor's
  bibliography: include every resolved entry that scores ≥2, then fill remaining
  slots by Phase 4"— this is the anchor-constrained mode.
* **Longer output:** raise the word budget and section count in Step 14 only;
  keep the retrieval procedure identical so results stay comparable.

---

## Addendum A — force the ranked-reference CSV (if the agent must emit the pipeline schema)

The main prompt produces a *selection table* in prose. Deep-research agents cannot
write `paper_after_reranking.csv` unless you specify the exact output contract and
the numeric weights, because those live in the pipeline, not in the prompt. Paste
**both** the main prompt and this addendum to get a CSV the pipeline can consume.

```text
OUTPUT CONTRACT — ranked reference table
In addition to the review, emit the ranked candidate table as CSV with EXACTLY
these 28 columns, in this order, one row per candidate you considered (mark the
ones you selected with tag=selected, the rest leave tag empty):

title,doi,abstract,origin,cited_by_anchor,co_cited_count,cited_by_count,pub_year,
citations_per_year,cited_by_bin,first_author_hindex,last_author_hindex,
author_hindex_avg,author_h_bin,journal_hindex,journal_h_bin,relevance_llm_1to5,
relevance_bin,avg_cos_sim_to_others,avg_cos_dist_to_others,rank_relevance,
rank_impact,rank_diversity,avg_rank,impact_score,diversity_score,composite_score,tag

Column definitions and weights (use these values; do not invent alternatives):
- origin: retrieved | co_cited | snowball | anchor_ref
- cited_by_anchor: true if the paper appears in ANCHOR's bibliography, else false
- citations_per_year: cited_by_count / (current_year - pub_year + 1)   [age normalisation]
- cited_by_bin, author_h_bin, journal_h_bin, relevance_bin: quartile bin over the
  candidate pool, Q1..Q4 where Q1 is the highest value
- impact_score = (0.165*C + 0.067*A + 0.10*V) / (0.165+0.067+0.10)
  where C, A, V are the Q1..Q4 bins mapped to 4..1 and min-max normalised to [0,1]
  (C = citations_per_year, A = mean of first- and last-author h-index, V = venue h-index)
- relevance_llm_1to5: your 1-5 relevance judgement against the SCOPE STATEMENT
- avg_cos_sim_to_others: mean cosine similarity of this candidate's (title+abstract)
  embedding to every other candidate; avg_cos_dist_to_others = 1 - that value
- rank_relevance / rank_impact / rank_diversity: rank (1 = best) of the candidate on
  relevance_llm_1to5 / impact_score / avg_cos_dist_to_others over the pool
- avg_rank: mean of the three ranks above
- diversity_score: min-max normalised avg_cos_dist_to_others
- composite_score = (0.165*cited_bin_norm + 0.067*author_bin_norm + 0.10*venue_bin_norm
  + 0.33*relevance_norm + 0.33*diversity_score) / (sum of the weights present on that row)
- tag: "selected" for the K references the review cites, otherwise empty
- Use an empty cell for an unknown value; never write "nan", "N/A" or "unknown".

Sort the rows by avg_rank ascending. If you cannot compute a numeric column, still
emit the column with empty cells — do not drop the column.
```

If the agent only returns Markdown, convert it locally:

```bash
cd "Paper implementation"
python3 -m qual_sg.import_ranking --input argon_output.md --out paper_after_reranking.csv
```

The converter parses Markdown tables / CSV / JSON, joins the rows against the
candidate pool (`final_ref_set.json`) and the cached relevance scores
(`step3_relevance_scores.jsonl`), recomputes the binning, `impact_score`,
`avg_rank` and `composite_score` with the pipeline's own functions, and writes the
canonical 28-column CSV.
