"""
make_reports.py — builds the two PDF documents from the pipeline outputs.

  reports/Task3_Submission.pdf      the assignment submission (problems, ranked ideas, top-5 per problem in full)
  reports/Task3_Documentation.pdf   full documentation: every stage, every decision, deviations, limitations, future work

All tables, scores and idea texts are read from outputs/*.json, so the documents always match the logged run.
Requires xelatex.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
O = ROOT / "outputs"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
D = json.loads((O / "explorer_data.json").read_text())
CRIT = ["Novelty", "Feasibility", "Clarity", "Impact", "Validity"]
PCRIT = ["Clarity", "Relevance", "Originality", "Feasibility", "Significance"]
ORDER = ["A1", "B1"]  # Problem 1 = A1 (highest Stage 1 score in run 2), Problem 2 = B1
PNUM = {"A1": 1, "B1": 2}
PSHORT = {"A1": "Mask-free localisation for MVFA's segmentation adapters", "B1": "Inductive per-image scoring for few-shot MVFA"}
RUN1 = json.loads((ROOT / "runs" / "run1_surveygen341" / "outputs" / "explorer_data.json").read_text())


def corpus_stats():
    import sys
    sys.path.insert(0, str(ROOT))
    from corpus import load_surveygen, review_set, EntityStore, MIN_COUNT
    P = load_surveygen()
    st = EntityStore(P)
    return dict(papers=len(P), review=len(review_set(P)), entities=len(st.count),
                retrievable=sum(1 for c in st.count.values() if c >= MIN_COUNT))


CS = corpus_stats()


def tex(s):
    s = str(s if s is not None else "")
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
    s = "".join(rep.get(c, c) for c in s).replace("--", "-{}-")
    s = s.replace("×", r"$\times$").replace(r"\_", r"\_\allowbreak{}")
    for u in URLS:
        s = s.replace(tex_plain(u), r"\url{" + u + "}")
    return s


URLS = ["https://api.groq.com/openai/v1", "github.com/rosnikv/HypER"]


def tex_plain(u):
    return u.replace("_", r"\_")


def para(s):
    """Paragraph text; turns '1) ... 2) ...' step lists into an enumerate."""
    s = str(s or "").strip()
    parts = re.split(r"(?:(?<=\s)|^)(\d{1,2})\)\s", s)
    if len(parts) >= 5 and parts[1] == "1":
        head = tex(parts[0].strip())
        items = [tex(parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
        return (head + "\n" if head else "") + "\\begin{enumerate}[leftmargin=1.6em,itemsep=1pt,topsep=2pt]\n" + \
            "\n".join(f"\\item {it}" for it in items) + "\n\\end{enumerate}"
    return tex(s)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Heros}
\setmonofont[Scale=0.85]{DejaVu Sans Mono}
\usepackage{microtype}
\usepackage{xcolor}
\definecolor{ink}{HTML}{172230}\definecolor{accent}{HTML}{2C55C7}\definecolor{heat}{HTML}{C86A12}\definecolor{muted}{HTML}{5A6878}\definecolor{soft}{HTML}{EEF2FB}
\usepackage[colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent]{hyperref}
\usepackage{amssymb}
\usepackage{booktabs,longtable,tabularx,array,enumitem,parskip}
\usepackage{titlesec}
\titleformat{\section}{\sffamily\Large\bfseries\color{ink}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\sffamily\large\bfseries\color{ink}}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries\color{ink}}{\thesubsubsection}{0.6em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\footnotesize\color{muted}\sffamily __FOOT__}\fancyfoot[R]{\footnotesize\color{muted}\sffamily\thepage}
\usepackage[most]{tcolorbox}
\newtcolorbox{ideabox}[1]{colback=soft,colframe=accent,boxrule=0.6pt,arc=2pt,left=8pt,right=8pt,top=6pt,bottom=6pt,breakable,title={\sffamily\bfseries #1},fonttitle=\small,coltitle=white}
\newtcolorbox{note}{colback=white,colframe=heat,boxrule=0.6pt,arc=2pt,left=8pt,right=8pt,top=4pt,bottom=4pt,breakable}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\lbl}[1]{{\sffamily\footnotesize\bfseries\color{muted}\MakeUppercase{#1}}\par}
\emergencystretch=3em\setlength{\LTpre}{4pt}\setlength{\LTpost}{6pt}
\begin{document}
"""


def compile_tex(name, body, foot):
    path = OUT / f"{name}.tex"
    path.write_text(PREAMBLE.replace("__FOOT__", foot) + body + "\n\\end{document}\n")
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", path.name], cwd=OUT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit(f"xelatex failed for {name}")
    for ext in ("aux", "log", "out", "toc"):
        (OUT / f"{name}.{ext}").unlink(missing_ok=True)
    print("built", OUT / f"{name}.pdf")


def fit_latest(c):
    return c["fitness"][max(c["fitness"], key=int)]


def ranking_table(pid):
    E = D["evolution"][pid]["candidates"]
    R = D["ranking"][pid]
    rows = sorted([E[i] for i in R["shortlist"]], key=lambda c: c["rank"])
    s = ("{\\small\\setlength{\\tabcolsep}{4pt}\\begin{longtable}{@{}r >{\\raggedright\\arraybackslash}p{8.4cm} ccccc c r r@{}}\n\\toprule\n"
         "\\# & Idea & Nov & Fea & Cla & Imp & Val & Mean & Fit. & Final\\\\\\midrule\\endhead\n")
    for c in rows:
        mark = "$\\bigstar$ " if c["id"] in R["top5"] else ""
        skip = " {\\footnotesize\\color{muted}(not selected: family limit)}" if any(x["id"] == c["id"] for x in R["skipped"]) else ""
        s += (f"{c['rank']} & {mark}{tex(c['title'])}{skip} {{\\footnotesize\\color{{muted}}[{tex(c['id'])}]}} & "
              + " & ".join(str(c["scores"][k]) for k in CRIT)
              + f" & {c['review_mean']:.1f} & {fit_latest(c):.1f} & {c['final']:.3f}\\\\\n")
    s += "\\bottomrule\n\\end{longtable}}\n"
    return s


def lineage(pid, c):
    E = D["evolution"][pid]["candidates"]
    if not c["parents"]:
        return f"HypER seed (chain {c.get('chain')})"
    return f"{' + '.join(c['operators'])} of " + "; ".join(f"{p} ({E[p]['title']})" for p in c["parents"])


def idea_block(pid, rank_pos, c):
    nv = c.get("novelty") or {}
    prior = "\n".join(f"\\item \\href{{{p['url']}}}{{{tex(p['title'])}}} — {tex(p['relation'])}" for p in nv.get("closest", []))
    s = f"\\begin{{ideabox}}{{Idea {PNUM[pid]}.{rank_pos} — {tex(c['title'])}}}\n"
    s += ("{\\small " + " \\quad ".join(f"{k}: \\textbf{{{c['scores'][k]}}}/5" for k in CRIT)
          + f" \\quad Final score: \\textbf{{{c['final']:.3f}}} (rank {c['rank']} of 12) \\quad Family: {tex(c['family'])}}}\\par\\smallskip\n")
    s += f"\\lbl{{Summary}}{para(c['summary'])}\n\n"
    s += f"\\lbl{{Hypothesis}}{para(c['hypothesis'])}\n\n"
    s += f"\\lbl{{Rationale}}{para(c['rationale'])}\n\n"
    s += f"\\lbl{{Method}}{para(c['method'])}\n\n"
    s += f"\\lbl{{Experiment design}}{para(c['experiment'])}\n\n"
    s += "\\lbl{Reviewer assessment}\\begin{itemize}[leftmargin=1.2em,itemsep=1pt,topsep=2pt]\n" + \
         "\n".join(f"\\item \\textbf{{{k} ({c['scores'][k]})}}: {tex(c['reviews'][k])}" for k in CRIT) + "\n\\end{itemize}\n"
    s += f"\\lbl{{Closest prior work found (novelty search)}}\\begin{{itemize}}[leftmargin=1.2em,itemsep=1pt,topsep=2pt]\n{prior}\n\\end{{itemize}}\n"
    prov = "; ".join(tex(p) for p in (c.get("provenance") or []))
    s += f"\\lbl{{Lineage and provenance}}{{\\small {tex(lineage(pid, c))}. Builds on: {prov or 'the target paper only'}.}}\n"
    s += "\\end{ideabox}\n"
    return s


def problem_scores_table():
    s = "\\begin{tabularx}{\\linewidth}{@{}l Y ccccc c c@{}}\\toprule\nID & Source group & " + \
        " & ".join(k[:4] + "." for k in PCRIT) + " & Rounds (mean) & Selected\\\\\\midrule\n"
    for p in sorted(D["problems"].values(), key=lambda p: -p["trajectory"][-1]):
        s += f"{p['id']} & {tex(p['group'])} & " + " & ".join(str(p["scores"][k]) for k in PCRIT) + \
             f" & {' / '.join(str(x) for x in p['trajectory'])} & {'yes' if p['selected'] else ''}\\\\\n"
    return s + "\\bottomrule\\end{tabularx}\n"


GAP_TITLES = {g["id"]: g for g in D["gaps"]}


def gaps_of(pid):
    grp = {"B1": "Score calibration and robust aggregation", "A1": "Label-efficient adaptation"}[pid]
    g = next(x for x in D["groups"] if x["name"] == grp)
    return g


# =============================================================================================== SUBMISSION
def submission():
    s = r"""
\begin{center}
{\sffamily\bfseries\huge Research Problem Discovery and Ideation}\\[4pt]
{\sffamily\large Task 3 — continuation of the limitations of \emph{Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images} (MVFA, CVPR 2024)}\\[6pt]
{\color{muted}Course: Introduction to Large Language Models \quad·\quad Methods: ResearchAgent, HypER, HypoEvolve}
\end{center}
\vspace{4pt}
"""
    tops = {pid: [D["evolution"][pid]["candidates"][i] for i in D["ranking"][pid]["top5"]] for pid in ORDER}
    s += r"\section*{Summary}" + "\n"
    s += tex("Starting from the 15 limitations identified in our previous assignment, we formulated two fine-grained research problems with ResearchAgent and discovered ideas for each with a pipeline that grounds seed ideas in citation chains (HypER), evolves them with a generational genetic algorithm (HypoEvolve), and ranks them on Novelty, Feasibility, Clarity, Impact and Validity. Before formulating problems, every implementation-level limitation was checked against MVFA's public code; this corrected four limitations and revealed two new ones, including model selection on the target test set.") + "\n\n"
    for pid in ORDER:
        s += f"\\textbf{{Problem {PNUM[pid]} ({tex(PSHORT[pid])}).}} Top-5 ideas: " + "; ".join(
            f"({i}) {tex(c['title'])}" for i, c in enumerate(tops[pid], 1)) + ".\n\n"
    s += tex(f"The literature comes from our final Task 1 review (QUAL-SG pipeline: {CS['papers']}-paper corpus, {CS['review']}-paper review set) and the papers citing MVFA. In total the pipeline made {D['total_calls'] + sum(D['reused_calls'].values())} logged LLM calls; all prompts, answers, code and an interactive explorer of every intermediate result accompany this submission.") + "\n"

    s += r"\section{Approach}" + "\n"
    s += tex("The pipeline implements the three prescribed methods and connects them so that each one does the job it was designed for.") + "\n\n"
    s += r"""\begin{tabularx}{\linewidth}{@{}l l Y@{}}\toprule
Stage & Method & What it does here\\\midrule
0 & Pre-processing & The 15 Task 2 limitations are parsed and grouped into five groups that a single problem could address.\\
0b & Code verification & Every implementation claim is checked against MVFA's public code (file and line evidence); gaps are corrected.\\
1 & ResearchAgent & Two candidate problems per group from the target paper, 69 related papers (our Task 1 review and papers citing MVFA) and 20 retrieved entities; ReviewingAgents score five criteria with human-induced rubrics; two refinement rounds; top-2 problems selected.\\
2 & HypER & Papers citing MVFA are scored 0/1/2 (unrelated / inspired / dependent); six reasoning chains are built; one literature-grounded seed idea per problem and chain, with provenance.\\
3 & HypoEvolve & Population of six seeds per problem; three generations of LLM crossover (combination, inspiration) and mutation (substitution, out-of-box); fitness from pairwise comparisons fitted with Bradley--Terry; $(\mu+\lambda)$ replacement.\\
4 & Ranking & Twelve highest-fitness candidates per problem are reviewed on five criteria with a literature novelty search; final score combines reviews (70\%) and fitness (30\%); at most two ideas per method family in the top five.\\
\bottomrule\end{tabularx}
"""
    s += "\n" + tex("Shared evaluation protocol (limitation group C, attached to every problem and idea): checkpoints and hyper-parameters are selected on a held-out split of the source modalities and never on the target test set, and the MVFA baseline is re-run under the same rule; results are reported under a modality-disjoint leave-one-modality-out split (RESC and OCT17 held out together); patient-level separation where identifiers exist; at least 3 seeds and 3 few-shot support draws with mean and standard deviation.") + "\n"

    s += r"\section{Research problems}" + "\n"
    s += tex("ResearchAgent produced six candidate problems from limitation groups A (label-efficient adaptation), B (score calibration and aggregation) and D (compact normality modelling). Each candidate was scored on its five problem criteria (1–5, human-induced rubric) and refined twice; the two highest-scoring problems, from different groups, were selected.") + "\n\n"
    s += problem_scores_table()
    for pid in ORDER:
        p = D["problems"][pid]
        g = gaps_of(pid)
        s += f"\\subsection{{Problem {PNUM[pid]}: {tex(PSHORT[pid])}}}\n"
        s += f"\\lbl{{Problem statement}}{para(p['problem'])}\n\n\\lbl{{Rationale}}{para(p['rationale'])}\n\n"
        s += "\\lbl{Limitations addressed}\\begin{itemize}[leftmargin=1.2em,itemsep=1pt,topsep=2pt]\n" + "\n".join(
            f"\\item \\textbf{{{m}}} {tex(GAP_TITLES[m]['title'])}" for m in g["members"]) + "\n\\end{itemize}\n"

    for pid in ORDER:
        s += f"\\section{{Ideas for Problem {PNUM[pid]}}}\n"
        ev = D["evolution"][pid]
        n = len(ev["candidates"])
        s += tex(f"HypoEvolve produced {n} candidate ideas for this problem ({n - 18} HypER seeds and 18 offspring over three generations, {ev['comparisons']} pairwise judgments). The table ranks the twelve highest-fitness candidates on the five criteria (1–5); Mean is the review mean, Fit. the Bradley–Terry fitness, Final = 0.7 × normalised review mean + 0.3 × normalised fitness. Stars mark the five selected ideas.") + "\n"
        s += ranking_table(pid)
        if D["ranking"][pid]["skipped"]:
            s += "\\begin{note}\\small " + tex("Diversity rule: " + "; ".join(f"{x['id']} was not selected because its {x['reason']}." for x in D["ranking"][pid]["skipped"])) + "\\end{note}\n"
        for i, cid in enumerate(D["ranking"][pid]["top5"], 1):
            s += idea_block(pid, i, ev["candidates"][cid])

    s += r"\section{Limitations of this study}" + "\n\\begin{itemize}[leftmargin=1.2em]\n"
    for t in [
        "The same model generated and judged ideas (as in ResearchAgent and HypoEvolve). The judging steps can be repeated with an independent model (e.g. Llama-70B on Groq) by a provided script, and the explorer page collects human ratings to measure human–model agreement as ResearchAgent did.",
        "Pairwise comparisons show each pair once, as in HypoEvolve; position bias is not corrected. A both-orders variant is implemented behind a switch.",
        "HypER's distilled small model was not trained; its chain construction and relevance scoring were run with a large model, as the authors did to create their training data.",
        "Novelty was checked with a literature search per idea, not by experiments; ideas whose core statistical tools exist elsewhere received lower novelty scores accordingly.",
        "No idea has been run yet; every hypothesis states the result that would falsify it, to be tested in the next assignment.",
        "This is the second run of the pipeline, on the final Task 1 review; the first run, on an earlier version of the review, is archived. The same model produced both runs, so the second is not independent of the first (details in the documentation).",
    ]:
        s += f"\\item {tex(t)}\n"
    s += "\\end{itemize}\n"
    s += r"""\section*{References}
\begin{enumerate}[leftmargin=1.6em,itemsep=1pt]
\item C. Huang et al. Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images. CVPR 2024. \href{https://github.com/MediaBrain-SJTU/MVFA-AD}{Code}.
\item J. Baek et al. ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models. \href{https://arxiv.org/pdf/2404.07738}{arXiv:2404.07738}.
\item R. Vasu et al. HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance. EMNLP 2025. \href{https://arxiv.org/pdf/2506.12937}{arXiv:2506.12937}.
\item J. Liu et al. HypoEvolve: Genetic Algorithms Enable Multi-Agent LLMs to Discover Scientific Hypotheses. \href{https://arxiv.org/pdf/2609.15938}{arXiv:2609.15938}.
\item Prior work cited in the idea boxes is linked inline.
\end{enumerate}
"""
    compile_tex("Task3_Submission", s, "Task 3 · Research problem discovery and ideation · MVFA")


# =============================================================================================== DOCUMENTATION
DECISIONS = [
    ("Which Task 1 literature review to build on", "GitHub version (arXiv, 14 papers) or submitted PDF (SurveyGen/OpenAlex, 16 papers)", "Submitted PDF version (run 1)", "K", "Task 3 must continue the submitted work."),
    ("How LLM calls are answered", "(A) file backend answered by Claude now, (B) user runs everything with an API key", "Hybrid: (A) now, re-judge later with Groq (Llama-70B)", "K", "Results today; an independent judge later addresses self-preference."),
    ("Literature corpus", "SurveyGen output files or web-collected corpus", "SurveyGen files (final_ref_set.json, paper_after_reranking.csv); replaced by the final QUAL-SG files in run 2 (decision 15)", "K", "Reproducible and identical to Task 1 at the time."),
    ("HypER small-model distillation (Phi-3 on Colab)", "Include now, skip, decide later", "Deferred; documented as future work", "K", "To be added when the pipeline is improved."),
    ("Deliverables", "PDF only or PDF + explorer", "PDF + interactive explorer; later two documents (submission + documentation)", "K", ""),
    ("Limitation groups used for problems", "A, B, D (C as protocol) or A, B, D, E", "A, B, D with C as shared protocol", "K", "E is not codable on BMAD (no subtype labels)."),
    ("Forward citations for HypER", "Run fetch_citations.py locally or collect by web search", "Run locally (Semantic Scholar API)", "K", "Faithful to HypER's Step 2 and reproducible."),
    ("Use AutoSurvey material", "Upload or not", "Not used", "Claude (recommended), K agreed", "Task 3 continues the submitted SurveyGen review; years come from the citation script."),
    ("Paper/code discrepancies in MVFA", "Continue Stage 1 or verify code and re-run", "Add Stage 0b, update protocol C, re-run Stage 1; archive the first Stage 1 attempt", "K", "Problems must hold in the code the next assignment modifies."),
    ("Research problems", "B1 and A1, or B1 and B2", "B1 and A1", "K", "Different gap groups: evaluation validity and supervision cost."),
    ("Human validation of the judge", "Groq re-judge only or also human ratings", "Both; rating form on the explorer page", "K", "ResearchAgent validates its judge with human agreement."),
    ("Pairwise comparison order", "(a) one order as in the paper, (b) both orders with ties", "(a) now; (b) implemented behind BOTH_ORDERS, to be run later", "K", ""),
    ("Stage 4 pool", "(a) final population of 6, (b) all 24 candidates with a diversity rule", "(b)", "K", "Broader menu for the next assignment."),
    ("Documents", "One document or two", "Two: a submission document and a full documentation", "K", ""),
    ("Re-run on a new Task 1 version", "Keep run 1 or re-run the whole pipeline", "Re-run everything on the new corpus; archive run 1", "K", "The final Task 1 review changed (259-paper corpus, 60-paper review set)."),
    ("Review set for run 2", "30-paper pipeline_ref (older), Prompted_ref (59, no abstracts), the 60 'selected' papers", "The new 60-paper pipeline_ref.json (= the 60 'selected' papers)", "K", "Matches the final corpus; all 60 have abstracts."),
    ("Entities without final_ref_set.json", "Upload OpenAlex records or extract entities from abstracts", "Extract key phrases from abstracts (no preference stated; Claude's recommended option)", "K / Claude", "No OpenAlex keywords exist for the final corpus."),
    ("Problems in run 2", "Keep B1 and A1 or let Stage 1 choose again", "Let Stage 1 choose again", "K", "A clean re-run; the result was again one problem from group A and one from group B."),
]
DEFAULTS = [
    ("Embeddings", "TF-IDF instead of sentence-transformers for similarity", "Hugging Face downloads were blocked in the run environment."),
    ("Entity store (run 2)", "RAKE-style key phrases from titles and abstracts: multi-word phrases between stop words (extended with generic research verbs), plus acronyms and internally capitalised names; entities in at least two papers are retrievable", "ResearchAgent used the BLINK linker on abstracts; key-phrase extraction is a dependency-free approximation. Run 1 used OpenAlex concepts."),
    ("Related papers for ResearchAgent", "59 other papers of the review set + the 10 MVFA-citing papers most similar to MVFA's abstract that are not already in the review (69 in total)", "ResearchAgent narrows citing papers by abstract similarity; many review papers already cite MVFA, so duplicates are skipped."),
    ("Problem selection", "Best two candidates by mean score from different limitation groups", "Two problems from one group would address the same limitations."),
    ("ResearchAgent refinement prompt", "Written in the style of Tables 6-8", "The paper describes refinement but does not print the prompt."),
    ("HypER chains", "Branch the six best first-hop papers, then extend greedily; second hop pre-filtered to 20 most similar citing papers", "HypER builds one greedy chain per source paper; we have one source."),
    ("HypER few-shot examples", "The authors' two clinical-trial examples, verbatim", "Faithful to the released code; domain mismatch noted."),
    ("HypER seed prompt", "Listing 4 plus research problem, code map, protocol and Method/Experiment/Provenance fields", "Seeds become HypoEvolve's initial population and must be codable."),
    ("HypoEvolve operators", "Inspiration and component substitution written in the style of Appendix D; combination and out-of-box verbatim", "Only two of four operator prompts are printed."),
    ("Offspring sampling", "Each offspring's operator call has its own sample index", "Identical prompts must not share one cached answer."),
    ("Bradley-Terry", "MM algorithm with a 0.1 pseudo-tie per compared pair; initial scale mean 50, spread 60; offset anchored on survivors", "Finite strengths for undefeated candidates; paper's scaling."),
    ("Stage 4 reviews", "One call per idea scores all five criteria, each against its own ResearchAgent rubric", "Five times fewer calls; ResearchAgent uses one call per criterion."),
    ("Stage 4 weights", "Final = 0.7 x normalised review mean + 0.3 x normalised fitness; shortlist 12; at most 2 per family", "Reviews are the assignment's criteria; fitness keeps the evolutionary signal."),
    ("Novelty evidence", "Web search per idea (queries saved); novelty_search.py re-runs them on Semantic Scholar", "Semantic Scholar was not reachable from the run environment."),
    ("Later code findings", "E18-E20 recorded but not merged into gap texts", "Merging would have changed Stage 1 prompts after they ran."),
]
FUTURE = [
    ("Independent re-judging (planned)", "Set JUDGE_BACKEND=openai, JUDGE_BASE_URL=https://api.groq.com/openai/v1, JUDGE_API_KEY and JUDGE_MODEL (e.g. llama-3.3-70b-versatile) and run python3 rejudge.py (464 judge calls of this run; python3 rejudge.py stage1 stage4 for a 114-call subset). It re-asks the new judge exactly this run's judge prompts and reports Spearman and exact agreement for problem reviews, kappa for HypER relevance, winner agreement and a Bradley-Terry refit for the comparisons, and the top-5 recomputed from the new reviews. Simply re-running the stages with only JUDGE_* changed does not work, because refinements and parent selection depend on the judgments and create new generator prompts."),
    ("Position-bias correction (planned)", "Set BOTH_ORDERS = True in stage3_evolve.py. Each pair is judged in both orders and disagreement counts as a tie; about 1.8x the comparison calls."),
    ("HypER checker on Colab (deferred)", "Fine-tune Phi-3-mini-128k-instruct with LoRA (rank 8, lr 2e-5, attention layers) on the released HypER chains (github.com/rosnikv/HypER, dataset/training_data) and use it as a second, independent validator of our chains (valid/invalid, breakpoints). Report the domain mismatch: HypER's chains are clinical-trial literature."),
    ("Human agreement", "Collect ratings on the explorer page (you and teammates), export them and run agreement.py ratings.json for Spearman and quadratic-weighted kappa per criterion, as in ResearchAgent Table 1."),
    ("Per-criterion reviews", "Split the Stage 4 review into one call per criterion (as ResearchAgent does) to remove interaction between criteria in one answer."),
    ("Larger search", "HypoEvolve's sensitivity analysis also tests a population of ten and five generations; increase MU and G in stage3_evolve.py. Cost grows quadratically with the pool."),
    ("Final tournament", "Re-score all 24 candidates of a problem in one Bradley-Terry fit so that fitness is fully comparable across generations (currently each candidate keeps its most recent fitness)."),
    ("Better retrieval", "Replace TF-IDF with a sentence encoder (all-MiniLM-L6-v2, as in the Task 1 repo) and re-run stages 1-2; expand hop-2 citations beyond the 40 most-cited first-hop papers (fetch_citations.py --hop2-top)."),
    ("Novelty re-check", "Run novelty_search.py on your machine and compare the Semantic Scholar hits with the saved web-search evidence."),
    ("Implementation order for the next assignment", "Problem 2 (B1) ideas are inference-only on retrained MVFA checkpoints and are the fastest to test: implement the temperature-calibrated log-odds fusion first (B1-g0-C5), then the conformal few-shot p-value (B1-g1-o2) and its Cauchy level-wise extension (B1-g3-o5), which share the leave-one-out support statistics; online FDR control (B1-g3-o4) is an evaluation layer on top. For Problem 1 (A1), start with cross-level persistence pseudo-masks (training-free) and the detection-branch veto, which share the det-only stage-1 model needed by occlusion attribution and the gated mean teacher."),
    ("New Task 1 version", "Replace data/paper_after_reranking.csv and data/pipeline_ref.json, move outputs/ to runs/, clear logs/calls.jsonl, and run the stages; Stage 0, 0b and HypER relevance replay from the cache."),
]


def documentation():
    s = r"""
\begin{center}
{\sffamily\bfseries\huge Task 3 Pipeline: Full Documentation}\\[4pt]
{\sffamily\large How every result was produced, every decision taken, and what can be changed}\\[6pt]
{\color{muted}Base paper: MVFA (CVPR 2024) \quad·\quad Methods: ResearchAgent, HypER, HypoEvolve \quad·\quad Run 2 of 20 October 2026}
\end{center}
\tableofcontents
\newpage
"""
    s += r"\section{What this document covers}" + "\n" + tex(
        "The submission document presents the two research problems and the top-5 ideas for each. This document explains how they were produced: the inputs, the architecture of the code, each stage with what the source paper prescribes and what we implemented (including every adaptation and why), the full decision log, the evidence found in MVFA's code, the limitations of the study, and a prioritised list of changes that can be made later. Every number here is read from the pipeline's output files.") + "\n"

    s += r"\section{Inputs}" + "\n\\begin{itemize}[leftmargin=1.2em]\n"
    for t in [
        "Task 2 research gaps (Task_3_Research_Gap_Identification.pdf): 15 limitations of MVFA and one broad problem statement.",
        f"Task 1 corpus from the QUAL-SG pipeline (final version): paper_after_reranking.csv with {CS['papers']} papers (297 rows, 38 empty), all with abstracts, publication years, LLM relevance (1-5) and composite scores, 60 tagged 'selected'.",
        "Task 1 review set: pipeline_ref.json, the 60 references selected by the QUAL-SG pipeline (MMR selection, relevance gate 3), identical to the 60 'selected' rows of the CSV and including MVFA itself. An earlier 30-reference version built on a 367-paper candidate pool, and a 59-reference list from a deep-research prompt (Prompted_ref.json, no abstracts, some unverifiable identifiers), were superseded and are kept in data/unused/.",
        "Forward citations (data/s2_citations.json, fetched by fetch_citations.py on the user's machine from the Semantic Scholar API): 147 papers citing MVFA (121 with abstracts, 113 with citation contexts), 1,391 second-hop papers for the 40 most-cited first-hop papers and MVFA's 65 references. These do not depend on the Task 1 corpus and are unchanged from run 1.",
        "MVFA's public code, snapshot of commit a30bf7b (2024-04-06), for code verification.",
        "HypER's released code and data (github.com/rosnikv/HypER): the relevance prompt (v002) and the two few-shot examples.",
        "The three method papers: ResearchAgent (arXiv:2404.07738), HypER (arXiv:2506.12937), HypoEvolve (arXiv:2609.15938).",
    ]:
        s += f"\\item {tex(t)}\n"
    s += "\\end{itemize}\n"

    s += r"\section{Architecture}" + "\n" + tex(
        "All LLM calls go through llm_client.py. Each call has a role: generator (problems, seeds, offspring) or judge (reviews, relevance scores, pairwise comparisons). The backend is set by environment variables: 'openai' sends the call to any OpenAI-compatible endpoint (Groq, AgentRouter, OpenAI), 'file' writes the prompt to logs/pending and stops the stage. In this run every call used the file backend and was answered by the configured model (claude-opus-5-5) acting as generator and judge; answers were written to logs/pending/<key>.response.txt and ingested into logs/cache. Each cached record holds the role, model, tag, temperature, sample index, system prompt, user prompt and response. Because calls are cached by a hash of (model, system, user, sample) and all randomness is seeded, re-running any stage replays it exactly and continues where it stopped. Judge calls can be pointed at a different model with JUDGE_* variables. Because later generator prompts depend on earlier judgments, an independent re-judge is done by rejudge.py, which re-asks a new judge exactly the judge prompts of this run and measures agreement; a full re-run with an API model sets LLM_* for both roles.") + "\n\n"
    s += r"\begin{tabularx}{\linewidth}{@{}l Y@{}}\toprule File & Purpose\\\midrule" + "\n"
    for f, p in [("llm_client.py", "Backends, roles, cache, pending/ingest mechanism"), ("answer.py", "Helper to answer pending prompts by tag (file backend)"),
                 ("corpus.py", "Corpus loader (QUAL-SG CSV), review set, keyphrase entities, entity store (ResearchAgent Eq. 2)"), ("stage0_gaps.py", "Gap parsing, TF-IDF cross-check, grouping"),
                 ("stage0b_verify.py", "Model-free code verification with file/line evidence"), ("stage1_problems.py", "ResearchAgent problem discovery, review and refinement"),
                 ("stage2_hyper.py", "HypER chain construction and seed generation"), ("stage3_evolve.py", "HypoEvolve genetic algorithm"),
                 ("stage4_rank.py", "Shortlist, reviews, families, final score, diversity selection"), ("prompts/", "All prompt templates and the ResearchAgent rubrics (verbatim)"),
                 ("fetch_citations.py, novelty_search.py", "Semantic Scholar scripts to run locally"), ("agreement.py", "Human-model agreement from exported ratings"), ("rejudge.py", "Independent re-judging of this run's judge prompts with another model"),
                 ("export_explorer_data.py, make_reports.py", "Explorer data and the two PDFs"), ("run_all.py", "Runs every stage in order")]:
        s += f"\\texttt{{{tex(f)}}} & {tex(p)}\\\\\n"
    s += "\\bottomrule\\end{tabularx}\n"

    # ---------------------------------------------------------------- stages
    s += r"\section{Stage by stage}" + "\n"
    s += r"\subsection{Stage 0: grouping the limitations}" + "\n" + tex(
        "The 15 limitations were parsed from the Task 2 text (no LLM). A TF-IDF cosine clustering gave a model-free view of similar gaps, passed to one judge call as a weak hint. The call tagged each gap (component, kind, codability on BMAD with one GPU) and grouped them so that one fine-grained problem could address a whole group. Reason for grouping: ResearchAgent writes a problem from whatever it is given; all 15 at once reproduces the everything-at-once problem of Task 2.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y l@{}}\\toprule Group & Members & Use\\\\\\midrule\n"
    use = {"Label-efficient adaptation": "source of Problem 1 (A1)", "Score calibration and robust aggregation": "source of Problem 2 (B1)",
           "Leakage-free generalisation protocol": "shared evaluation protocol", "Compact normality modelling": "candidate source (D1, D2)", "Clinical granularity and scope": "not used"}
    for g in D["groups"]:
        s += f"{tex(g['name'])} & {', '.join(g['members'])} & {tex(use.get(g['name'], ''))}\\\\\n"
    s += "\\bottomrule\\end{tabularx}\n"

    s += r"\subsection{Stage 0b: verifying the limitations against MVFA's code}" + "\n" + tex(
        "While reviewing a Stage 1 candidate, we opened MVFA's public code and found several differences from the paper. Since the next assignment will modify this code, problems must rest on what the code does. Stage 0b locates every claim by a regular expression (the script fails if the code does not contain it) and writes corrected gap records; groups B and C received the new gaps. In run 1 the first Stage 1 attempt, built on the uncorrected gaps, was discarded; Stage 0 and 0b are unchanged in run 2.") + "\n"
    s += "{\\small\\begin{longtable}{@{}l P{3.6cm} P{10.8cm}@{}}\\toprule ID & File:line & Finding\\\\\\midrule\\endhead\n"
    for e in D["evidence"] + D["later_evidence"]:
        s += f"{e['id']} & \\href{{{e['url']}}}{{\\texttt{{{tex(e['file']).replace('/', '/\\allowbreak{}')}:{','.join(map(str, e['lines']))}}}}} & {tex(e['finding'])}\\\\\n"
    s += "\\bottomrule\\end{longtable}}\n"
    s += tex("Resulting changes: G3, G4, G8 and G9 corrected; G10 received evidence; G16 (checkpoint selection on the target test set) and G17 (paper says max pooling, code uses mean) added. E18-E20 (learning rate 1e-4 instead of 1e-3 for zero-shot training; batch size 1 instead of 16) were found during Stage 2 and are recorded without changing the gap texts that Stage 1 had already used. These discrepancies are also ready-made test cases for the group's reproducibility-gap auditor.") + "\n"

    s += r"\subsection{Stage 1: problem discovery with ResearchAgent}" + "\n" + tex(
        "Faithful elements: the problem-identification prompt (Table 6), the ReviewingAgent prompt (Table 9) with one call per criterion, the human-induced rubrics of Table 13 copied verbatim, entity retrieval by Eq. 2 over an entity co-occurrence store, citing papers narrowed by abstract similarity, and iterative refinement (initial version plus two refinements; the paper reports saturation after about three). Inputs: MVFA's title and abstract, 69 related papers (the 59 other papers of the 60-paper review set and the 10 MVFA-citing papers most similar to MVFA's abstract that are not already in the review), and 20 entities retrieved from an entity store over the whole corpus. Entities: the final corpus has no OpenAlex keywords, so entities are key phrases extracted from titles and abstracts (RAKE-style splitting at stop words; multi-word phrases plus acronyms and internally capitalised names such as CLIP or BMAD; phrases in at least two papers are retrievable), giving " + str(CS['entities']) + " entities, " + str(CS['retrievable']) + " retrievable. Adaptations: the gap group is an additional input; a feasibility statement describes the resources (public MVFA code, BMAD, one Colab GPU); protocol C is attached; the refinement prompt was written in the paper's style because it is not printed. Two candidates per group; the second was asked to differ from the first.") + "\n\n" + problem_scores_table() + "\n" + tex(
        "The two problems are selected as the best-scoring candidates from different limitation groups. A1 (4.4) refined a pseudo-mask question after reviewers asked for an operational rule and a label-cost curve; B1 (4.2) was stable across rounds. The second candidates (A2: normal-only few-shot adaptation; B2: pooling; D1, D2: compact normality and learned prompts) scored 3.8-4.2, mostly limited by originality because the new review set already contains the corresponding methods (MediCLIP, SpatioPrompt, CANOR, AnomalyCLIP and others).") + "\n"

    s += r"\subsection{Stage 2: literature chains and seed ideas with HypER}" + "\n" + tex(
        f"Implemented from the paper and its released code: citing papers of the current paper are taken from the two following years, grouped by year, scored in chunks of 10 with the authors' relevance prompt (0 unrelated, 1 inspired, 2 dependent) and two few-shot examples; the model also returns each chunk's top-3; candidates are ranked by 0.7 x relevance + 0.3 x normalised citations; the best becomes the next source. Of the {sum(D['hop1_relevance'].values())} MVFA-citing papers from 2025-2026 with abstracts, {D['hop1_relevance'].get('2', 0)} were scored 2, {D['hop1_relevance'].get('1', 0)} scored 1 and {D['hop1_relevance'].get('0', 0)} scored 0. Adaptations: with a single source paper, the six best first-hop papers each start a chain (HypER obtains many chains from many sources); second-hop candidates were pre-filtered to the 20 most similar abstracts; citation contexts were added to the relevance prompt. In run 2 the chain construction replays from the cache: the relevance prompts depend only on MVFA and its citing papers, which did not change, so the 14 relevance calls of run 1 are reused and only the 12 seed ideas are new. Seed ideas use HypER's generation prompt (Listing 4) with the research problem, a code-verified map of MVFA's code, the protocol, and output fields for method, experiment and provenance.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y@{}}\\toprule Chain & Papers (year, relevance) and status\\\\\\midrule\n"
    for c in D["chains"]:
        s += f"{c['id']} & MVFA $\\rightarrow$ " + " $\\rightarrow$ ".join(f"{tex(p['title'])} ({p['year']}, {p['relevance']})" for p in c["papers"]) + f". {{\\color{{muted}}{tex(c['note'])}}}\\\\\n"
    s += "\\bottomrule\\end{tabularx}\n"

    s += r"\subsection{Stage 3: evolution with HypoEvolve}" + "\n" + tex(
        "Implemented as Algorithm 1 with the paper's settings: population 6, 6 offspring per generation, 3 generations, crossover probability 0.6, mutation probability 0.15 (forced when crossover is skipped), size-2 tournaments, all unordered pairs of the pool compared by the judge, Bradley-Terry fitness (initial scale mean 50, spread 60; later fits keep the multiplier and anchor survivors), and (mu+lambda) replacement with lineage records. The pairwise prompt keeps the structure of Appendix D.2 (specificity, evidence, testability; explicit list of what not to reward; REASONING/CONFIDENCE/WINNER). Pairs already judged in an earlier generation are answered from the cache, equivalent to a deterministic judge. Each pair is shown in one order chosen by a hash of the two ids.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y Y@{}}\\toprule & Problem 1 (A1) & Problem 2 (B1)\\\\\\midrule\n"
    rows = {}
    for pid in ("B1", "A1"):
        ev = D["evolution"][pid]
        means = [round(sum(h["fitness"].values()) / len(h["fitness"]), 1) for h in ev["history"]]
        rows[pid] = dict(means=" / ".join(map(str, means)), best=ev["candidates"][ev["best"]]["title"], comps=ev["comparisons"])
    s += f"Population mean fitness, gen 0-3 & {rows['A1']['means']} & {rows['B1']['means']}\\\\\n"
    s += f"Pairwise judgments & {rows['A1']['comps']} & {rows['B1']['comps']}\\\\\n"
    s += f"Best idea (highest fitness) & {tex(rows['A1']['best'])} & {tex(rows['B1']['best'])}\\\\\n\\bottomrule\\end{{tabularx}}\n"
    s += tex("Fitness rises partly by construction, because survivors are re-ranked against new offspring by the same judge; it is not an independent quality measure. HypoEvolve uses external biological data for that purpose; here the Stage 4 reviews and novelty search play that role. The file-backend judge was fully transitive (no ties in 336 judgments): verdicts were written from an explicit assessment of every candidate and a considered ranking per generation (logs/answers/run2_qualsg/assess.py), unlike a sampled LLM judge.") + "\n"

    s += r"\subsection{Stage 4: multi-criteria ranking}" + "\n" + tex(
        "Pool: all 24 candidates per problem (user's choice). Shortlist: the 12 with the highest most-recent fitness. For each shortlisted idea a literature search looked for the closest prior work (one query per idea, 24 in total, saved with the hits in data/novelty_search.json), and one ResearchAgent-style review scored Novelty (method innovativeness rubric), Feasibility (experiment feasibility rubric), Clarity (method clarity), Impact (problem significance) and Validity (method validity). One call per problem grouped all candidates into method families. Final score = 0.7 x (review mean - 1)/4 + 0.3 x normalised fitness; ideas were selected greedily with at most two per family.") + "\n"
    for pid in ORDER:
        s += f"\\subsubsection*{{Problem {PNUM[pid]} ({pid})}}\n" + ranking_table(pid)
        fam = D["ranking"][pid]["families"]
        s += "{\\small Families: " + "; ".join(f"{tex(k)} ({len(v)})" for k, v in fam.items()) + ".}\n\n"
    s += tex("Novelty evidence lowered several scores. For Problem 1, anomaly-guided weakly supervised OCT lesion segmentation and pseudo-label refinement surveys cover the general idea of letting a weak signal steer segmentation, COIN derives masks from counterfactual changes in a classifier, size-constraint losses (Kervadec et al.) and patch-grid phase marginalisation exist, and RegAD compares registered query and normal images. For Problem 2, leave-one-out conformal anomaly detectors (arXiv 2402.16388), layer-wise multiple-testing fusion of p-values for OOD detection (arXiv 2403.10803), weighted conformal anomaly detection in low-data regimes (arXiv 2603.23205), CLIP zero-shot temperature calibration (arXiv 2303.12748), extreme-value score normalisation and online FDR control for anomaly detection streams all exist. No idea scored above 3 for novelty; the ideas are new mainly in how they use MVFA's own components (its detection branch, its four adapted levels, its two differently natured branches).") + "\n"

    # ---------------------------------------------------------------- run history
    s += r"\section{Run history: run 1 and run 2}" + "\n" + tex(
        "Run 1 (19-20 October 2026) used the earlier SurveyGen corpus (341 OpenAlex papers, 16-paper review set, OpenAlex concepts as entities). After the final Task 1 review was produced with the QUAL-SG pipeline, the whole pipeline was re-run on it (run 2, this document); run 1 is archived in runs/run1_surveygen341/ with its own PDFs, explorer data and call log. What stayed: Stage 0 and 0b (they do not use the corpus), the HypER citation chains (they depend only on MVFA's citing papers), all prompts, rubrics and settings. What changed: the related papers and entities given to ResearchAgent, hence the problems, seeds, offspring and rankings.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y Y@{}}\\toprule & Run 1 (SurveyGen) & Run 2 (QUAL-SG, this run)\\\\\\midrule\n"
    s += f"Corpus / review set & 341 / 16 papers & {CS['papers']} / {CS['review']} papers\\\\\n"
    s += f"Related papers in Stage 1 & 26 & 69\\\\\n"
    s += f"Selected problems (Stage 1 mean) & B1 {RUN1['problems']['B1']['trajectory'][-1]}, A1 {RUN1['problems']['A1']['trajectory'][-1]} & A1 {D['problems']['A1']['trajectory'][-1]}, B1 {D['problems']['B1']['trajectory'][-1]}\\\\\n"
    for pid, name in (("A1", "Mask-free localisation (A1), top-5"), ("B1", "Per-image scoring (B1), top-5")):
        t1 = "; ".join(tex(RUN1["evolution"][pid]["candidates"][i]["title"].split(":")[0]) for i in RUN1["ranking"][pid]["top5"])
        t2 = "; ".join(tex(D["evolution"][pid]["candidates"][i]["title"].split(":")[0]) for i in D["ranking"][pid]["top5"])
        s += f"{tex(name)} & {t1} & {t2}\\\\\n"
    s += f"LLM calls & {RUN1['total_calls']} & {D['total_calls']} new + {sum(D['reused_calls'].values())} reused\\\\\n\\bottomrule\\end{{tabularx}}\n"
    s += tex("Both runs chose one problem from group A and one from group B. The A1 problem moved from 'image-level labels in general' to a specific label-restricted pseudo-mask rule, and its best ideas moved from positive-unlabelled learning and MIL (run 1) to using MVFA's own detection branch as a teacher or filter (run 2). The B1 problem is nearly the same; its best ideas moved from p-value combination rules with region pooling to conformal p-values for the few-shot branch combined with a temperature-calibrated zero-shot probability. The new review set, which contains many 2025-2026 medical CLIP anomaly detectors, lowered the originality of problems whose solutions it already contains (A2, B2, D2).") + "\n"

    # ---------------------------------------------------------------- decisions
    s += r"\section{Decision log}" + "\n" + tex("Decisions taken by the user (K) during the run, in order:") + "\n"
    s += "\\begin{longtable}{@{}r P{3.3cm} P{3.9cm} P{3.6cm} P{3.4cm}@{}}\\toprule \\# & Decision & Options & Choice & Why\\\\\\midrule\\endhead\n"
    for i, (d, o, c, by, w) in enumerate(DECISIONS, 1):
        s += f"{i} & {tex(d)} & {tex(o)} & {tex(c)} {{\\color{{muted}}\\footnotesize({tex(by)})}} & {tex(w)}\\\\\\addlinespace[3pt]\n"
    s += "\\bottomrule\\end{longtable}\n" + tex("Implementation choices made by Claude, with the reason for each:") + "\n"
    s += "\\begin{longtable}{@{}P{3.2cm} P{6.6cm} P{5.8cm}@{}}\\toprule Topic & Choice & Reason\\\\\\midrule\\endhead\n"
    for t, c, r in DEFAULTS:
        s += f"{tex(t)} & {tex(c)} & {tex(r)}\\\\\\addlinespace[3pt]\n"
    s += "\\bottomrule\\end{longtable}\n"

    # ---------------------------------------------------------------- validity
    s += r"\section{Limitations and threats to validity}" + "\n\\begin{itemize}[leftmargin=1.2em]\n"
    for t in [
        "Single model as generator and judge. ResearchAgent and HypoEvolve do the same; HypER reports that LLM judges agree with experts least on originality and feasibility (Spearman -0.08 for feasibility), the two criteria that matter most here. Mitigations: code verification and novelty search as non-LLM evidence; planned Groq re-judging; human ratings on the explorer page.",
        "Carry-over between runs. The file-backend model produced run 1 (and, within run 1, a discarded first Stage 1 attempt) before run 2, so run 2 is not independent of run 1: the same limitation groups were chosen again and one idea family (p-value combination) recurs, although the run-2 seeds, offspring and top-5 ideas are different. An API model re-running the pipeline from scratch would not have this memory.",
        "Deterministic, transitive judge. A sampled judge produces intransitive and noisy verdicts; Bradley-Terry results here are cleaner than they would be with an API model.",
        "Position bias not corrected (option (a)); implemented as option (b) for a later run.",
        "Retrieval quality. TF-IDF similarity and abstract key phrases are weaker than sentence embeddings and a dedicated entity linker; some retrieved entities are generic (e.g. 'feature matching') or industrial ('defect type'), because the corpus contains many industrial zero-shot anomaly detection papers.",
        "Short chains. MVFA is from 2024, so its citation lineage has at most two further hops; four of six chains stop after one paper.",
        "Domain mismatch of HypER's few-shot examples (clinical trials).",
        "Fitness comparability across generations relies on the anchored offset; a final tournament would make it exact.",
        "Novelty search breadth. One query per idea with a general web search; absence of a hit is not proof of novelty.",
    ]:
        s += f"\\item {tex(t)}\n"
    s += "\\end{itemize}\n"

    s += r"\section{Changes that can be made}" + "\n" + tex("Ordered roughly by value for effort.") + "\n\\begin{enumerate}[leftmargin=1.6em]\n"
    for t, d in FUTURE:
        s += f"\\item \\textbf{{{tex(t)}.}} {tex(d)}\n"
    s += "\\end{enumerate}\n"

    s += r"\section{How to reproduce}" + "\n" + r"""\begin{enumerate}[leftmargin=1.6em]
\item Replay this run from the cache (no API key): \texttt{python3 run\_all.py}. Every stage reads its answers from \texttt{logs/cache}.
\item Re-run with a model of your choice: \texttt{export LLM\_BACKEND=openai LLM\_BASE\_URL=... LLM\_API\_KEY=... LLM\_MODEL=...}, then \texttt{python3 run\_all.py}. New prompts are sent to the API; cached ones are reused only for the same model name.
\item Independent judge on this run's outputs: set the \texttt{JUDGE\_*} variables and run \texttt{python3 rejudge.py}; results go to \texttt{outputs/rejudge\_<model>.json}.
\item Refresh literature: \texttt{python3 fetch\_citations.py} and \texttt{python3 novelty\_search.py} (Semantic Scholar, standard library only).
\item Rebuild documents: \texttt{python3 export\_explorer\_data.py \&\& python3 make\_reports.py}.
\end{enumerate}
"""
    s += "\\subsection*{LLM calls in this run}\n\\begin{tabular}{@{}l r@{}}\\toprule Stage & Calls\\\\\\midrule\n"
    names = {"stage0": "Stage 0 (grouping)", "stage1": "Stage 1 (ResearchAgent)", "stage2": "Stage 2 (HypER)", "stage3": "Stage 3 (HypoEvolve)", "stage4": "Stage 4 (ranking)"}
    s = s.replace("LLM calls in this run}\n\\begin{tabular}{@{}l r@{}}\\toprule Stage & Calls", "LLM calls in this run}\n\\begin{tabular}{@{}l r r@{}}\\toprule Stage & New calls & Reused from run 1")
    for k in ["stage0", "stage1", "stage2", "stage3", "stage4"]:
        s += f"{tex(names[k])} & {D['calls'].get(k, 0)} & {D['reused_calls'].get(k, 0)}\\\\\n"
    s += f"\\midrule Total & {D['total_calls']} & {sum(D['reused_calls'].values())}\\\\\\bottomrule\\end{{tabular}}\n\n"
    s += tex(f"Reused calls are prompts identical to run 1 (Stage 0 grouping and HypER relevance scoring), answered from the cache. Run 1 made {RUN1['total_calls']} calls; its outputs, documents and call log are archived in runs/run1_surveygen341/.") + "\n"
    compile_tex("Task3_Documentation", s, "Task 3 · Full documentation · MVFA")


if __name__ == "__main__":
    submission()
    documentation()
