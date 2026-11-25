# Prompt: full candidate reranking CSV from a deep-research agent

Use this when you want an agent (ChatGPT Deep Research, Gemini Deep Research,
Perplexity, Argon, …) to return **every article it retrieved and scored** as a
ranked CSV — not a prose summary and not just the handful it decided to cite.

Two companions:

* `DEEP_RESEARCH_PROMPT.md` — the review-writing prompt;
* this file — the *dump the whole candidate pool, ranked* prompt.

The CSV follows the pipeline's 28-column schema, so it can be dropped straight
into `python3 -m qual_sg.import_ranking --input <file>` (which re-joins the pool
and fills any column the agent could not compute).

---

```text
ROLE
You are a research librarian building a ranked candidate pool for a literature
review. You must return ONE CSV table containing EVERY article you retrieved and
evaluated - not a summary and not only the articles you would cite.

INPUTS (fill these in)
- ANCHOR: "<paper title / DOI / URL, or NONE>"
- TOPIC: "<one-line topic title; keywords separated by ';'>"
- K: <how many articles the review will actually cite, e.g. 30>
- YEAR_RANGE: <start year> - <end year or "current">
- POOL_TARGET: <how many candidate articles to include, e.g. 200>

STEP 1 - Build the candidate pool (record every article, do not filter yet)
1. If ANCHOR is not NONE, read its title, abstract, introduction and bibliography,
   and convert its scope into a 3-5 sentence SCOPE STATEMENT. Use that sentence as
   the relevance criterion for every judgement below.
2. Expand the scope into 6-10 short search queries covering sub-tasks, method
   families, model names and benchmark datasets.
3. Run every query against at least two independent databases (OpenAlex, Semantic
   Scholar, arXiv, Crossref, PubMed - whichever fit the field) and keep the top
   ~100 results per query.
4. Add backward snowballing: if ANCHOR exists, resolve every entry of its
   bibliography; also add the reference lists of the 20-30 most relevant articles
   you found. Add forward snowballing: articles citing ANCHOR, newest first.
5. De-duplicate on DOI and on normalised title (preprints appear twice).
6. Keep the POOL_TARGET articles with the highest similarity between the SCOPE
   STATEMENT and (title + abstract). Every one of these becomes one CSV row.

STEP 2 - Signals (use real values; write an empty cell if you cannot obtain one)
- cited_by_count: total citations (state the source in your notes, not in the CSV)
- pub_year: publication year
- citations_per_year = cited_by_count / (current_year - pub_year + 1)
- first_author_hindex, last_author_hindex, journal_hindex: from the author/venue
  profiles; empty if unavailable
- origin: retrieved | co_cited | snowball | anchor_ref
- cited_by_anchor: true if the article appears in ANCHOR's bibliography, else false
- co_cited_count: how many pool articles cite it (0 if unknown)

STEP 3 - Judgements and scores
- relevance_llm_1to5: your 1-5 relevance judgement against the SCOPE STATEMENT
  (1 = unrelated, 2 = loose background, 3 = one sub-aspect, 4 = substantially
  addresses the scope, 5 = entirely focused / foundational).
- Quartile-bin cited_by_bin, author_h_bin, journal_h_bin, relevance_bin over the
  whole pool: Q1..Q4, where Q1 is the highest value.
- impact_score = (0.165*C + 0.067*A + 0.10*V) / (0.165 + 0.067 + 0.10)
  where C, A, V are those bins mapped Q1->4 ... Q4->1 and min-max normalised to
  [0,1] across the pool (C = citations_per_year, A = mean of first- and
  last-author h-index, V = journal_hindex).
- rank_relevance / rank_impact / rank_diversity: rank 1 = best over the pool.
  (For diversity use mean pairwise semantic distance; see the empty-cell rule.)
- avg_rank = mean(rank_relevance, rank_impact, rank_diversity)
- composite_score = weighted mean over the values present on that row:
  0.165*cited_bin_norm + 0.067*author_bin_norm + 0.10*venue_bin_norm
  + 0.33*relevance_norm + 0.33*diversity_score, divided by the sum of those weights.
- tag = "selected" for the K articles the review cites, empty for all the others.
- If you cannot embed text, leave avg_cos_sim_to_others, avg_cos_dist_to_others
  and diversity_score EMPTY; the local converter recomputes them.

OUTPUT - one CSV, fenced as ```csv ... ```
Header, exactly these 28 columns in this order:
title,doi,abstract,origin,cited_by_anchor,co_cited_count,cited_by_count,pub_year,citations_per_year,cited_by_bin,first_author_hindex,last_author_hindex,author_hindex_avg,author_h_bin,journal_hindex,journal_h_bin,relevance_llm_1to5,relevance_bin,avg_cos_sim_to_others,avg_cos_dist_to_others,rank_relevance,rank_impact,rank_diversity,avg_rank,impact_score,diversity_score,composite_score,tag

RULES
- ONE ROW PER CANDIDATE. Include articles you did not cite (tag empty). Never
  summarise "and 40 similar papers" - list them.
- Sort rows by avg_rank ascending.
- Empty cell = unknown. Never write "nan", "N/A", "unknown" or 0 as a placeholder.
- Quote every field containing a comma, quote or newline; keep the abstract to one
  line (replace newlines with spaces).
- Do not invent DOIs, citation counts, h-indices or venues.
- After the CSV, print exactly one line: "<rows> rows emitted, <selected> selected,
  <databases> databases, pool target <POOL_TARGET>".
- If your output limit is reached before every row is emitted, stop mid-list, end
  the message with "CONTINUE", and repeat the header when you resume. Do not
  compress or drop rows to fit.
```

## What to do with the CSV

```bash
cd "Paper implementation"
python3 -m qual_sg.import_ranking --input agent_pool.csv --out paper_after_reranking.csv
python3 -m qual_sg.bib_eval --pdf "../your_paper.pdf" --selected
```

`import_ranking` joins each row to the pipeline's candidate pool by DOI/title,
pulls the cached judge score where available, recomputes the bins, `impact_score`,
ranks, `composite_score` and (when `step3_embeddings.npz` covers the paper) the
diversity columns, then writes the canonical 28-column file. Rows the pipeline has
never seen are kept with empty score cells instead of being dropped.

## Practical notes

* Agents truncate aggressively. `POOL_TARGET` of 150-250 rows is realistic for one
  message; above that, accept the `CONTINUE` protocol and concatenate the parts
  (drop repeated headers before feeding the file to `import_ranking`).
* If the agent refuses the numeric columns, drop STEP 3 from the prompt and keep
  only `title,doi,abstract,pub_year,cited_by_count,relevance_llm_1to5,tag` — the
  converter recomputes everything else locally.
* Ask for CSV, not a Markdown table: Markdown columns get lost during copying.
