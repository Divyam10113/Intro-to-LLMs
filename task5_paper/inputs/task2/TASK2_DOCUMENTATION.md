# Task 2 — Documentation

This document explains what we did in Task 2, why we did it, how to run it again, and what later tasks can reuse.
It is written in plain language. Technical words are explained in the [glossary](#2-glossary) the first time you need them.

**Contents**
1. [Task 2 in one minute](#1-task-2-in-one-minute)
2. [Glossary](#2-glossary)
3. [What went in: the inputs](#3-what-went-in-the-inputs)
4. [How the pipeline works, step by step](#4-how-the-pipeline-works-step-by-step)
5. [Where everything is](#5-where-everything-is)
6. [How to run it again](#6-how-to-run-it-again)
7. [What we ran](#7-what-we-ran)
8. [Results and what they mean](#8-results-and-what-they-mean)
9. [What we checked by hand](#9-what-we-checked-by-hand)
10. [Problems we hit and how we fixed them](#10-problems-we-hit-and-how-we-fixed-them)
11. [Known weaknesses](#11-known-weaknesses)
12. [For the next tasks](#12-for-the-next-tasks)
13. [FAQ](#13-faq)

---

## 1. Task 2 in one minute

**The task:** find the limitations and research gaps of our base paper. Give them as bullet points, then write one paragraph that defines a new research problem continuing from those gaps.

**The base paper:** MVFA, *"Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images"*, CVPR 2024, arXiv:2403.12570.
In short: MVFA takes CLIP, a model trained on everyday photos and captions, and adds small trainable layers called adapters so that it can find diseases (anomalies) in medical images. It works with zero or only a few example images from the new dataset.

**What we built:** a pipeline in which several LLM "agents" read MVFA and the papers around it, list its weaknesses, check each weakness against real quotes, merge the lists, check whether later papers have already fixed each weakness, and finally write a research problem.
It follows the methods of the three papers the assignment pointed to, and adds a few ideas of our own.

**The answer (main result, run on MiniMax):**
- **15 research gaps.** The most important themes:
  - results come from one random draw of example images, with no error bars;
  - training needs hand-drawn masks of the diseased area, which are expensive;
  - the few abnormal example images are used only in training, never as comparisons at test time;
  - the two scores are mixed 50/50 with no calibration;
  - only ROC-AUC is reported;
  - the "unseen modality" test is leaky: OCT17 is tested while RESC, another retinal OCT dataset, is used in training.
- **Research problem:** build a version of MVFA that
  - needs no pixel masks;
  - uses both normal *and* abnormal example images at test time;
  - stays stable whatever mix of example images it gets;
  - gives one calibrated score.

**Where the answer is:** [`task2_research_gaps.md`](task2_research_gaps.md). The main result is at the top; the second run, on gpt-6-astra, is further down for comparison.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **Anomaly detection (AD)** | Finding images (or parts of images) that are not normal, e.g. a tumour. |
| **AC / AS** | *Anomaly classification* (is this image abnormal: yes or no) and *anomaly segmentation* (which pixels are abnormal). |
| **Zero-shot** | The model is tested on a dataset it never trained on. In MVFA, it trains on the other 5 datasets and is tested on the 6th. |
| **Few-shot (K-shot)** | The model gets K labelled example images from the test dataset (K = 2, 4, 8 or 16). |
| **CLIP** | A model that links images and text. MVFA keeps it frozen and adds small trainable parts. |
| **Adapter** | A small trainable layer added inside a big frozen model. |
| **Memory bank** | A store of features from normal example images. At test time, an image region that is far from everything in the bank is suspicious. |
| **ROC-AUC** | A score from 50 to 100 for how well a model ranks abnormal above normal. It does not show how accurately the lesion outline is drawn. |
| **LLM agent** | One LLM call with a specific role and instructions, e.g. "act as a peer reviewer". |
| **RAG** | Retrieval-augmented generation: first search a collection of documents, then give the best passages to the LLM. |
| **BM25 / dense retrieval** | Two ways to search text. BM25 matches keywords; dense retrieval matches meaning. We use 50% of each ("hybrid"). |
| **Re-rank** | After searching, an LLM scores each passage from 0 to 10 for usefulness, and we keep only scores of 8 or more. |
| **LLM-as-judge** | Using an LLM to grade another LLM's output. |
| **Provenance** | Where a limitation came from: *Author-stated*, *Inferred*, *Peer-review* or *Literature*. Defined in section 4, step 6. |
| **Held-out set** | Data the pipeline is never shown, kept aside to test it fairly. |
| **Cache** | Saved LLM answers. If the same question is asked again, the saved answer is reused for free. |

---

## 3. What went in: the inputs

| Input | Where it lives | Used for |
|---|---|---|
| MVFA paper (LaTeX source, about 53k characters; 36.5k without the supplementary section) | `base_paper/` | Every agent reads it. |
| Papers that **cite** MVFA (125) and papers MVFA **cites** (62) | Downloaded from OpenAlex into `limitations/data/papers.json` | The Citation agent's search index, and the gap-status check. |
| Task 1's selected papers (59) | `DELIVERABLES/pipeline/gap_inputs/item1_selected_papers.csv` | Added to the corpus. |
| Task 1's literature review | `DELIVERABLES/pipeline/gap_inputs/item3_lit_review.md` | The gap-status check, and the old-pipeline baseline. |

In total there are 223 papers. Full text was available for 35 of them: those on arXiv or with an open-access PDF. The others contribute only their abstracts.
The text was split by section into **828 searchable chunks**, of about 450 words each.

**The held-out answer key.** 14 papers that cite MVFA actually discuss it in their text. We split them 7/7:
- **Evidence half:** 7 papers (52 passages). These can be found by the Citation agent.
- **Held-out half:** 7 papers (45 passages). These are never indexed. The critiques of MVFA in them became the **answer key** for evaluation: 3 reference limitations, in `limitations/outputs/reference_limitations.json`.

We did this because the assigned papers evaluate against OpenReview reviews, and a CVPR paper has no public OpenReview page.

---

## 4. How the pipeline works, step by step

```
build_corpus.py  ─►  collect papers → download PDFs → split into chunks → find passages that discuss MVFA → split 7/7
                                                                                         │
run_pipeline.py                                                                          ▼
  ┌─ Extractor ─┐
  ├─ Analyzer  ─┤                                            
  ├─ Reviewer  ─┼─► Judge (rewrite once if score < 8/10) ─► Verifier ─► Master ─► Gap status ─► Research problem
  └─ Citation  ─┘   (Citation agent: 6 search queries → hybrid search → LLM re-rank ≥ 8 → passages)
                                                                                         │
evaluate.py      ─►  answer key from held-out papers · baselines · scoring ◄─────────────┘
```

| # | Step | What it does, simply | Idea from | Code |
|---|---|---|---|---|
| 1 | **Extractor** | Lists the limitations the authors admit themselves. | arXiv:2601.11578 | `agents.py: extractor` |
| 2 | **Analyzer** | Looks for flaws in the method and experiments. | arXiv:2601.11578 | `agents.py: analyzer` |
| 3 | **Reviewer** | Writes a critique like a peer reviewer. | arXiv:2601.11578 | `agents.py: reviewer` |
| 4 | **Citation agent** | Writes 6 search queries, searches the corpus (BM25 + meaning-based search), has an LLM keep only passages rated 8 or more, adds passages where later papers discuss MVFA, and lists the limitations those passages reveal. Up to 30 passages. | BAGELS (arXiv:2505.18207), 2601.11578 | `run_pipeline.py: citation_passages`, `retrieval.py`, `agents.py: citation` |
| 5 | **Judge** | Scores each agent's list on 7 criteria out of 10. If the average is below 8, the agent rewrites its list once, using the feedback. | arXiv:2601.11578 | `agents.py: judge`, `run_pipeline.py: run_worker` |
| 6 | **Verifier** (ours) | (a) Each limitation must quote evidence. The code checks that the quote really appears in the source text. (b) An LLM checks that the quote supports the claim. (c) If the Extractor says "the authors admit this" but they don't, the item is relabelled *Inferred*. | ours | `agents.py: check_quotes, verify_support, _provenance` |
| 7 | **Master** | Merges duplicates across agents, rates importance from 1 to 5, and keeps the provenance tags. | arXiv:2601.11578 | `agents.py: master` |
| 8 | **Gap status** (ours) | For each limitation, searches papers from 2024 onward and the Task 1 review, and labels it *open*, *partially addressed* or *addressed*. | ours | `agents.py: gap_status` |
| 9 | **Research problem** | Picks 3–5 important gaps that are still open or only partially addressed, and writes one research-problem paragraph of 180–260 words: setting, assumptions, what the method must do, why existing work falls short, and how success is measured. | ours | `agents.py: formulate_problem` |
| 10 | **Render** | Writes `research_gaps.md`: bullet points, the problem paragraph, and an appendix with every limitation and its evidence. | – | `run_pipeline.py: render` |

**The aspect labels** come from LimitGen (arXiv:2507.02694): Methodology, Experimental Design, Results & Analysis, Literature. We added **Clinical & Deployment** because this is a medical paper.

---

## 5. Where everything is

```
project_intro_to_llms/
├── base_paper/                         MVFA LaTeX source (input)
├── DELIVERABLES/
│   ├── pipeline/gap_inputs/            Task 1 outputs that Task 2 reads
│   └── task2/
│       ├── task2_research_gaps.md      ★ THE DELIVERABLE (MiniMax main + astra comparison + evaluation)
│       ├── minimax_full_with_evidence.md   all 39 MiniMax limitations with quotes
│       ├── astra_full_with_evidence.md     all 27 astra limitations with quotes
│       └── TASK2_DOCUMENTATION.md      this file
├── limitations/                        ★ THE TASK 2 CODE
│   ├── common.py         settings, model list, LLM calling with retries, fallback and cache; loads the base paper
│   ├── build_corpus.py   collects papers from OpenAlex/arXiv, splits text into chunks, finds MVFA mentions, makes the 7/7 split
│   ├── retrieval.py      BM25 + meaning-based search, and LLM re-ranking
│   ├── agents.py         all agent prompts: extractor, analyzer, reviewer, citation, judge, verifier, master, gap status, problem
│   ├── run_pipeline.py   runs the whole pipeline end to end
│   ├── evaluate.py       builds the answer key, runs the baselines, scores the systems
│   ├── check_models.py   quick test of which models respond and how fast
│   ├── README.md         short technical README
│   ├── .env              API keys (never commit or share)
│   ├── data/             corpus: papers.json, chunks.jsonl, mvfa_mentions.jsonl, heldout_mentions.jsonl, pdfs/
│   ├── cache/            saved LLM answers (234 files)
│   └── outputs/          one folder per run, plus evaluation.md/json and run logs
└── Intro-to-LLMs/multi_agent_limitations.py   the OLD pipeline we started from (used as a baseline)
```

**Inside each run folder** (`outputs/<run>/`), in the order the files are produced:

| File | Contents |
|---|---|
| `citation_passages.json` | The passages the Citation agent used. |
| `workers.json` | Raw lists from the 4 agents, with the Judge's scores. |
| `verified.json` | Each item with its quote-check result and support verdict. |
| `final_limitations.json` | The merged list, ready for reuse (fields in section 12). |
| `research_gaps.md` | Human-readable output. |

---

## 6. How to run it again

**Setup (once):**
```bash
pip install openai requests numpy scikit-learn pymupdf
```
Put the keys in `limitations/.env`: `AGENTROUTER_API_KEY=...` and/or `NVIDIA_API_KEY=...`.

**Run order:**
```bash
cd limitations
python build_corpus.py                  # no LLM; ~10 min the first time, cached afterwards
python run_pipeline.py                  # full pipeline  -> outputs/full/
python run_pipeline.py --no_citation    # without the Citation agent -> outputs/full_nocite/
python evaluate.py reference            # builds the answer key from the held-out papers
python evaluate.py baselines            # zero-shot prompt + old pipeline, same models
python evaluate.py score full full_nocite zero_shot old_pipeline   # -> outputs/evaluation.md
```

**Choosing models and providers** (environment variables):

| Variable | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `agentrouter` or `nim` (NVIDIA) for generation | `agentrouter` |
| `EVAL_PROVIDER` | Provider for the judge | same as `LLM_PROVIDER` |
| `GEN_MODELS` | AgentRouter models to try, in order | `gpt-6-astra,gpt-6.1-sol` |
| `EVAL_MODELS` | AgentRouter judge models | `gpt-6.1-sol,gpt-6-astra` |
| `NIM_GEN_MODELS` / `NIM_EVAL_MODELS` | NVIDIA models | MiniMax M3.1 Flash, GLM-5.3 / Nemotron 3 Ultra, GLM-5.3 Flash |
| `RUN_SUFFIX` | Added to baseline folder names, e.g. `_sol` | empty, or `_nim` on NIM |
| `MAX_ATTEMPTS` | Retries per model on temporary errors | 5 |

**Flags for `run_pipeline.py`:**

| Flag | Effect |
|---|---|
| `--name X` | Output folder name. |
| `--no_citation` | Skip the Citation agent. |
| `--no_verify` | Skip the Verifier. |
| `--workers N` | How many LLM calls run in parallel (default 4). Use 1–2 when the provider is unstable. |
| `--no_supp` | Leave out the paper's supplementary section (shorter input). |

**Examples:**
```bash
LLM_PROVIDER=nim python run_pipeline.py                     # MiniMax run -> outputs/full_nim/
EVAL_PROVIDER=nim python evaluate.py score full full_nim    # Nemotron as judge
```

**Important things to know:**
- **Crashes can be resumed.** Every finished LLM call is cached. If a run crashes, re-run *exactly the same command*; finished steps are skipped and it continues from where it stopped.
- **Changing the model list resets the cache.** The cache key includes the model list, so setting `GEN_MODELS` differently means every call is made again.
- **Fallback.** If the first model fails 5 times, the next model in the list answers. Each cached answer records which model actually answered.
- **Expected run time.** A full run is about 100–150 LLM calls. That's about 1–2 hours on a stable provider, plus waiting time when the provider has outages. Scoring 5 systems with Nemotron takes about 10–15 minutes.

---

## 7. What we ran

| Run folder | What | Model | Status |
|---|---|---|---|
| `full_nim` | **New pipeline, main result** | MiniMax M3.1 Flash (NVIDIA NIM); GLM-5.3 answered about 4 calls | ✅ complete, 39 limitations |
| `full` | New pipeline | gpt-6-astra (AgentRouter); Reviewer and Master fell back to gpt-6.1-sol | ✅ complete, 27 limitations |
| `full_nocite` | New pipeline without the Citation agent | gpt-6-astra | ✅ complete, 25 limitations |
| `old_pipeline` | Old pipeline (`Intro-to-LLMs/multi_agent_limitations.py`) re-run on the same model | gpt-6-astra | ✅ complete, 11 limitations |
| `zero_shot` | One plain prompt: "list the limitations" | gpt-6-astra | ✅ complete, 15 limitations |
| `full_sol`, `full_nocite_sol` | Planned second-model comparison | gpt-6.1-sol | ❌ abandoned partway (provider errors); not used |
| `zero_shot_sol`, `old_pipeline_sol` | Baselines for the sol comparison | gpt-6.1-sol | Complete but not scored |

**Judge for all scoring:** Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`) on NVIDIA NIM. We chose a model different from the generators so that no system is graded by the model that wrote it.

**Note:** the old pipeline's code is set to `deepseek-v4-flash`. For a fair comparison we re-ran it on gpt-6-astra, the same model as our new pipeline. The team's original September output, probably made with DeepSeek, is `Intro-to-LLMs/final_research_gaps.md`. It was not scored.

---

## 8. Results and what they mean

| System | Model | # limitations | Answer-key items caught (of 3) | Importance (1–5) |
|---|---|---|---|---|
| **New pipeline (main)** | MiniMax | 39 | **3** | **4.62** |
| New pipeline | gpt-6-astra | 27 | 2 | 3.74 |
| New pipeline without the Citation agent | gpt-6-astra | 25 | 1 | 3.72 |
| Old pipeline | gpt-6-astra | 11 | 2 | 4.00 |
| Zero-shot prompt | gpt-6-astra | 15 | 1 | 4.60 |

The 3 answer-key items:
1. MVFA uses fixed, hand-written text prompts.
2. Its segmentation masks are noisy.
3. Its memory bank is expensive.

**Proof check.** This is plain code, with no LLM involved:
- MiniMax: 259 of 259 quotes from the MVFA paper were found word-for-word, and 32 of 32 quotes from other papers were verified.
- astra: 118 of 118 quotes were verified.

**What we can honestly claim:**
- ✅ Every limitation is backed by a real quote that anyone can check.
- ✅ The Citation agent helps: removing it lost one answer-key item.
- ✅ The MiniMax run was the only system to catch all 3 answer-key items.
- ⚠️ With only 3 answer-key items, one item more or less is a 33-point jump, so these are hints, not proof.
- ⚠️ The judge gave **every** system 5/5 for faithfulness and soundness. Those scores are useless and are left out of the table.
- ⚠️ The old pipeline gets a higher F1 only because it lists fewer items (F1 penalises items not in the answer key).
- ⚠️ Read side by side, the old and new pipelines cover **mostly the same topics**. The new pipeline's real advantage is that its output is grounded, checkable, organised and marked with status in later work. It is not that it finds many new issues.

The full numbers, including C_GT, C_LLM and F1, are in `limitations/outputs/evaluation.md`.

---

## 9. What we checked by hand

Every claim in the 15 MiniMax gap bullets and the problem paragraph was checked against the MVFA paper and the corpus.

**Correct:**
- all the numbers:
  - HIS accuracy 85.10 → 82.62 from K=8 to K=16;
  - the dual adapter is +1.29 AC;
  - the headline gain is +7.33;
  - OCT17 at K=2: 97.98 vs APRIL-GAN's 99.21;
  - the 50 training epochs;
  - the 35 prompt templates;
  - the 115 test WSIs (whole-slide images) for HIS;
- the dataset descriptions.

**Fixed by hand:**
1. **Wrong fact.** It said patient-level splits were documented "for HIS and RESC only". BMAD also documents patient-ID splits for BrainMRI.
2. **Wrong timeline.** BMAD's 5-seed repetition and validation checkpointing were called "later work". BMAD is the benchmark MVFA is built on.
3. **Duplicate.** TAFT was listed twice under two names.
4. **Overstatements softened:**
   - "contradicts" the paper's use of abnormal samples;
   - "unsupported" became "not statistically supported";
   - "never" became "the later work we found";
   - "unreproducible" was removed.

**Lesson:** the automatic Verifier catches made-up quotes, but not wrong conclusions drawn from real quotes. A human or LLM fact-check pass on the final text is still needed.

---

## 10. Problems we hit and how we fixed them

| Problem | Fix | Where |
|---|---|---|
| Only 26 of 125 citing papers had arXiv IDs. | Searched arXiv by title; fell back to open-access PDFs. | `build_corpus.py` |
| A regex took a DOI fragment for an arXiv ID. | Stricter patterns. | `build_corpus.py` |
| Some papers cite MVFA as `12.` instead of `[12]`. | Added a second citation-label pattern. | `build_corpus.py` |
| The random split put only 4 of 14 papers in the held-out half. | Exact sorted 50/50 split. | `build_corpus.py` |
| One paper filled all the Citation agent's passages. | Round-robin, at most 3 passages per paper. | `run_pipeline.py` |
| AgentRouter: Claude had no backend (503). | The model is skipped for the rest of the run. | `common.py` |
| AgentRouter: GPT models need `/v1/responses`. | Switch to the Responses API automatically. | `common.py` |
| Long calls dropped (502s, connection errors). | Streaming, retries with waits of 20–80 s, `--workers`, `--no_supp`. | `common.py` |
| "Upstream service unavailable" and "token rate limit" errors. | Treated as temporary and retried. | `common.py` |
| NVIDIA NIM: some models return 404; timeouts. | Unavailable models are skipped; timeouts are retried. | `common.py` |
| The JSON parser grabbed an inner list from an object. | Parse whichever of `[` or `{` comes first. | `common.py` |
| The Extractor labelled too many items "author-stated". | The Verifier double-checks and relabels them. | `agents.py` |
| The research problem was too long and read like an audit. | Tighter prompt: 180–260 words, 3–5 gaps, evaluation issues only in the success measures. | `agents.py` |

---

## 11. Known weaknesses

1. **Tiny answer key.** Only 3 items, so the evaluation cannot separate systems reliably.
2. **The judge is too lenient.** Nemotron gave 5/5 to everything for faithfulness and soundness.
3. **The Verifier is lenient.** Its LLM check uses the same model as the generator, so it rarely rejects anything (115 of 116 passed).
4. **Gap status rarely discriminates.** 35 of the 39 MiniMax limitations are "partially addressed".
5. **The importance ranking is uneven.** For example, the fixed-prompts issue, which real researchers raised, was rated only 3/5 in the astra run.
6. **Many gaps are reporting gaps.** "The paper doesn't say X" is less interesting than a real method weakness.
7. **The old-pipeline baseline reads the Task 1 review,** which summarises some held-out papers. That is a small advantage for it.
8. **Full text for only 35 of 223 papers.** The rest contribute abstracts only.
9. **Model mixing.** The astra run is "astra with sol fallback"; MiniMax had a few GLM fallbacks.

**Ideas to improve:**
- a pairwise judge (show two lists side by side and pick the better one);
- a stricter or different-family model for the Verifier;
- a bigger answer key, e.g. more citing papers or a second base paper;
- an LLM fact-check pass on the final text.

---

## 12. For the next tasks

**Main reusable output:** `limitations/outputs/full_nim/final_limitations.json`, a list of 39 objects with these fields:

| Field | Meaning |
|---|---|
| `lid` | ID, e.g. `L5`. |
| `limitation` | One-sentence claim. |
| `explanation` | Details. |
| `aspect` | Methodology / Experimental Design / Results & Analysis / Literature / Clinical & Deployment. |
| `provenance` | List: Author-stated / Inferred / Peer-review / Literature. |
| `importance` | 1–5. |
| `suggested_fix` | How to address it. |
| `evidence` | List of `{source, quote, quote_verified}`. `source` is `BASE` (the MVFA paper) or a corpus chunk ID such as `C0257`. |
| `merged_from` | Which agent items were merged into it. |
| `status` | `open` / `partially_addressed` / `addressed`. |
| `addressed_by` | Papers that partly or fully address it. |
| `remaining_gap` | What is still missing. |

Note: the hand corrections in section 9 were made in the deliverable markdown files only. This JSON still has the original pipeline wording.

**If the next task is to work on the research problem** (designing or implementing a method):
- **The problem:** a mask-free, support-composition-robust, calibrated few-shot detector on top of MVFA. It builds on gaps L5, L6, L10 and L18.
- **Datasets:** the 6 BMAD benchmarks MVFA uses: BrainMRI, LiverCT, RESC, OCT17, ChestXray, HIS.
- **Baselines named in the problem:** WinCLIP, APRIL-GAN, AnomalyCLIP, MadCLIP, UniVAD, TAFT. All are in our corpus (`limitations/data/papers.json`).
- **Success measures already defined:**
  - repeated K draws;
  - AUROC/AUPRC;
  - ECE/Brier calibration;
  - sensitivity at fixed specificity;
  - PRO/Dice;
  - per-anomaly-type results.
- **Other gaps that could become their own projects:**
  - full 3D volumes instead of slices (the astra run's problem);
  - proper statistical evaluation;
  - leakage-free splits.

**Reusable components:**

| Component | Reuse it for |
|---|---|
| `data/papers.json` (223 papers with abstracts, years, groups) | Any later literature work. |
| `data/chunks.jsonl` (828 section chunks) with `retrieval.py` | Searching the literature. |
| `common.chat()` / `chat_json()` | LLM calls with fallback, retries and cache, on AgentRouter or NIM. |
| `agents.quote_found()` | Checking that any LLM quote really appears in a source text. |

**Analysing a different paper:**
1. Put its LaTeX in `base_paper/`.
2. Update `BASE_TITLE` and `BASE_OPENALEX_ID` in `common.py`. Also update the section file names in `load_base_paper()`: they are hard-coded to MVFA's files in `base_paper/sec/`, such as `3_method`.
3. Run `build_corpus.py`, then `run_pipeline.py`.

---

## 13. FAQ

**Is our output good if the tests can't prove it?**
It is proven to be *accurate*: every quote is real and the numbers were checked by hand. Whether the points are the *most important* ones is a judgement call. The small test can't settle that, so a human read is the best check.

**Why did the old pipeline score a higher F1?**
F1 penalises items that are not in the 3-item answer key. The old pipeline has 11 items and ours has 27–39, so ours gets penalised more, even though most of the extra items are real weaknesses.

**Why MiniMax as the main result and not astra?**
MiniMax caught all 3 answer-key items, astra 2. The judge rated MiniMax's points as more important. Its research problem also targets weaknesses closer to the heart of MVFA.

**Why was Nemotron the judge?**
So that the judge is a different model from the ones that generated the lists. It runs on NVIDIA NIM.

**Why AgentRouter for generation and not only NVIDIA?**
NVIDIA NIM was slow and timed out on long calls; AgentRouter was faster when it was up. The MiniMax run on NIM did finish, which is why we have both.

**A run crashed. Do I start over?**
No. Re-run the same command and the cache skips finished steps. Don't change `GEN_MODELS` between attempts, or the cache won't match.
