# Task 1 output for Task 3

**`final_ref_set.json`** is the one output file. It is a JSON array with 594 candidate papers for MVFA
(arXiv 2403.12570), in UTF-8. It comes from Task 1 pipeline run v6 (16 Nov 2026).

## Fields

Every entry has exactly these 12 fields, in this order, and none is empty.

| Field | Type | Content |
|---|---|---|
| `doi` | string | Checked DOI. arXiv-only papers use `10.48550/arxiv.<id>`. Use it as the key. |
| `title` | string | |
| `abstract` | string | |
| `year` | int | Publication year (2003–2026) |
| `venue` | string | Journal or proceedings; `"arXiv"` for preprints |
| `citation_count` | int | OpenAlex citation count, fetched 16 Nov 2026 |
| `keywords` | [string] | OpenAlex keywords (7 per paper on average) |
| `topics` | [string] | OpenAlex topics (3 per paper on average) |
| `relevance_score` | int 1–5 | Topical relevance to MVFA's topic, scored by an LLM judge (Claude Sonnet 5.5) |
| `composite_score` | float 0–1 | Ranking score: citations, author and venue h-index, relevance, diversity |
| `origin` | [string] | How the paper entered the pool; the first entry is the main route. Values: `search`, `mvfa_reference` (in MVFA's bibliography), `co_citation`, `snowball`, `cites_mvfa` (cites MVFA), `anchor` (MVFA itself) |
| `selected` | bool | `true` for the 59 papers cited in the current literature review |

Order: the selected papers first (in the review's numbering), then the rest by relevance and composite score.

## The five requests

| # | Request | Status |
|---|---|---|
| 1 | OpenAlex keywords and topics for every paper | Done: every entry has both |
| 2 | An abstract for every paper | Done: every entry has one. 68 papers have no open abstract anywhere, so they are left out (see below) |
| 3 | A "who cites MVFA" step | Done: 167 papers have `cites_mvfa` in `origin` (OpenAlex and Semantic Scholar) |
| 4 | Years, no empty rows, checked IDs | Done: every entry has a year; no field is empty; every DOI was checked to lead to a paper with the same title (one wrong DOI was corrected). `Prompted_ref.json` is not used |
| 5 | One file with a fixed format | Done: this file, replacing the CSV and the other JSON files |

## Also in this folder

`final_ref_set_excluded.json` lists the 84 candidates that are not in the main file, each with the reason:

- 68 have no abstract available;
- 13 are duplicates, for example the published and arXiv versions of the same paper (their routes are merged into the kept entry's `origin`);
- 2 have DOIs that could not be checked;
- 1 is a proceedings volume rather than a paper.

## Notes

- **59 selected, not 60.** The review's reference list has BMAD twice (the CVPRW paper and its arXiv version), so here it is one entry.
- **`selected` may change.** If the review is regenerated with newer papers (run v7), about 17 `selected` flags would change. The papers, fields and format stay the same, so key on `doi` and a new version can simply replace this file.
- `composite_score` is not "should have been selected". Selection also favours variety, so some unselected papers have higher composite scores than selected ones.
