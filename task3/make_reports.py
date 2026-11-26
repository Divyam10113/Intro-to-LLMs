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
ORDER = ["B1", "A1"]  # Problem 1 = B1 (highest Stage 1 score in run 3), Problem 2 = A1
PNUM = {"B1": 1, "A1": 2}
PSHORT = {"A1": "Training MVFA's segmentation adapters from image-level labels", "B1": "Inductive per-image scoring for few-shot MVFA"}
RUN1 = json.loads((ROOT / "runs" / "run1_surveygen341" / "outputs" / "explorer_data.json").read_text())
RUN2 = json.loads((ROOT / "runs" / "run2_qualsg_v4" / "outputs" / "explorer_data.json").read_text())
IN = D["inputs"]


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


URLS = ["https://api.groq.com/openai/v1", "github.com/rosnikv/HypER", "https://generativelanguage.googleapis.com/v1beta/openai/"]


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
    for p in sorted(D["problems"].values(), key=lambda p: -max(p["trajectory"])):
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
    s += tex(f"Starting from the {IN['task2_limitations']} limitations of MVFA identified by our Task 2 pipeline, we formulated two fine-grained research problems with ResearchAgent and discovered ideas for each with a pipeline that grounds seed ideas in citation chains (HypER), evolves them with a generational genetic algorithm (HypoEvolve), and ranks them on Novelty, Feasibility, Clarity, Impact and Validity. Before formulating problems, every implementation-level limitation was checked against MVFA's public code: of the ten limitations it could test, the code confirmed four, resolved three open questions, showed two to be worse than stated and corrected one, and it revealed {IN['code_gaps']} new ones, including model selection on the target test set and score normalisation with test-set statistics.") + "\n\n"
    for pid in ORDER:
        s += f"\\textbf{{Problem {PNUM[pid]} ({tex(PSHORT[pid])}).}} Top-5 ideas: " + "; ".join(
            f"({i}) {tex(c['title'])}" for i, c in enumerate(tops[pid], 1)) + ".\n\n"
    s += tex(f"The literature comes from our Task 1 output ({CS['papers']}-paper corpus with OpenAlex keywords and topics, {CS['review']}-paper review set) and {IN['citers']} papers citing MVFA (Task 1 and Semantic Scholar combined). In total the pipeline made {D['total_calls'] + sum(D['reused_calls'].values())} logged LLM calls; all prompts, answers, code and an interactive explorer of every intermediate result accompany this submission.") + "\n"

    s += r"\section{Approach}" + "\n"
    s += tex("The pipeline implements the three prescribed methods and connects them so that each one does the job it was designed for.") + "\n\n"
    s += r"""\begin{tabularx}{\linewidth}{@{}l l Y@{}}\toprule
Stage & Method & What it does here\\\midrule
0 & Pre-processing & The 39 Task 2 limitations are read from Task 2's output file and grouped into seven groups that a single problem could address.\\
0b & Code verification & Every implementation claim is checked against MVFA's public code (file and line evidence); gaps are corrected.\\
1 & ResearchAgent & Two candidate problems for each of the two groups that Task 4 builds on (labels and scoring), from the target paper, 68 related papers (our Task 1 review and papers citing MVFA) and 20 retrieved entities; ReviewingAgents score five criteria with human-induced rubrics; two refinement rounds; top-2 problems selected.\\
2 & HypER & Papers citing MVFA are scored 0/1/2 (unrelated / inspired / dependent); six reasoning chains are built; one literature-grounded seed idea per problem and chain, with provenance.\\
3 & HypoEvolve & Population of six seeds per problem; three generations of LLM crossover (combination, inspiration) and mutation (substitution, out-of-box); fitness from pairwise comparisons fitted with Bradley--Terry; $(\mu+\lambda)$ replacement.\\
4 & Ranking & Twelve highest-fitness candidates per problem are reviewed on five criteria with a literature novelty search; final score combines reviews (70\%) and fitness (30\%); at most two ideas per method family in the top five.\\
\bottomrule\end{tabularx}
"""
    s += "\n" + tex("Shared evaluation protocol (limitation group 'Leakage-free generalisation protocol', attached to every problem and idea): checkpoints and hyper-parameters are selected on a held-out split of the source modalities and never on the target test set, and the MVFA baseline is re-run under the same rule; results are reported under a modality-disjoint leave-one-modality-out split (RESC and OCT17 held out together); patient-level separation where identifiers exist; at least 3 seeds and 3 few-shot support draws with mean and standard deviation.") + "\n"

    s += r"\section{Research problems}" + "\n"
    s += tex("ResearchAgent produced four candidate problems, two from limitation group A (label-efficient adaptation) and two from group B (score calibration and robust aggregation), the two directions our Task 4 implementation builds on. Each candidate was scored on its five problem criteria (1–5, human-induced rubric) and refined twice; the two highest-scoring problems, from different groups, were selected.") + "\n\n"
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
        "The same model generated and judged ideas (as in ResearchAgent and HypoEvolve). An independent judge, Gemini 3.8 Flash, answered all 437 judging prompts of this run again. It agrees well on absolute scores (problem reviews: Spearman 0.70, 98% within one point; paper relevance: weighted kappa 0.68) but not on which of two ideas is stronger (same winner in 57% of A1 and 60% of B1 comparisons), although it is consistent with itself (91-92% when the two ideas are swapped). With Gemini's reviews and tournament the top 5 keeps 3 of 5 ideas for B1 (B1-g2-o1, B1-g2-o4, B1-g3-o5) and none for A1, where Gemini prefers the softer MIL variants ranked 7-12 here. The problems are therefore robust to the judge, and so is B1's core; the order of ideas within a problem, A1's above all, reflects the judge's preferences.",
        "Pairwise comparisons show each pair once, as in HypoEvolve. The independent check judged every pair in both orders: the idea shown first won 51% (A1) and 54% (B1) of the time, so position bias is small.",
        "HypER's chain construction and relevance scoring were run with a large model, as the authors did to create their training data. HypER's distilled checker (Phi-3 with LoRA) was trained afterwards and used as a second opinion on Stage 2: it finds all six chains valid and every seed supported by its chain, and it never scores 0 a paper our judge found relevant, but it is more lenient (score 2 for 88 of the 158 papers citing MVFA, against 7 from our judge). It was trained on clinical-trial chains, so it transfers to computer-vision papers only partly.",
        "Novelty was checked with a literature search per idea, not by experiments; ideas whose core statistical tools exist elsewhere received lower novelty scores accordingly.",
        "No idea has been run yet; every hypothesis states the result that would falsify it, to be tested in the next assignment.",
        "This is the third run of the pipeline, on Task 1's run-v6 output and Task 2's 39 limitations; the two earlier runs, on earlier versions of the review, are archived. The same model produced all runs, so this run is not independent of the earlier ones (details in the documentation).",
        "Task 2's limitations were produced from Task 1's run v4 and were not regenerated for Task 1's run v6; the limitations used here are mostly derived from the MVFA paper itself, so the effect is small, but the versions differ.",
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
    ("HypER small-model distillation (Phi-3 on Colab)", "Include now, skip, decide later", "Deferred at first; run later on Modal (see the HypER Phi-3 checker decision)", "K", "To be added when the pipeline is improved."),
    ("Deliverables", "PDF only or PDF + explorer", "PDF + interactive explorer; later two documents (submission + documentation)", "K", ""),
    ("Limitation groups used for problems", "A, B, D (C as protocol) or A, B, D, E", "A, B, D with C as shared protocol", "K", "E is not codable on BMAD (no subtype labels)."),
    ("Forward citations for HypER", "Run fetch_citations.py locally or collect by web search", "Run locally (Semantic Scholar API)", "K", "Faithful to HypER's Step 2 and reproducible."),
    ("Use AutoSurvey material", "Upload or not", "Not used", "Claude (recommended), K agreed", "Task 3 continues the submitted SurveyGen review; years come from the citation script."),
    ("Paper/code discrepancies in MVFA", "Continue Stage 1 or verify code and re-run", "Add Stage 0b, update protocol C, re-run Stage 1; archive the first Stage 1 attempt", "K", "Problems must hold in the code the next assignment modifies."),
    ("Research problems", "B1 and A1, or B1 and B2", "B1 and A1", "K", "Different gap groups: evaluation validity and supervision cost."),
    ("Human validation of the judge", "Groq re-judge only or also human ratings", "Both at first; human ratings were later dropped", "K", "ResearchAgent validates its judge with human agreement."),
    ("Pairwise comparison order", "(a) one order as in the paper, (b) both orders with ties", "(a) now; (b) implemented behind BOTH_ORDERS, to be run later", "K", ""),
    ("Stage 4 pool", "(a) final population of 6, (b) all 24 candidates with a diversity rule", "(b)", "K", "Broader menu for the next assignment."),
    ("Documents", "One document or two", "Two: a submission document and a full documentation", "K", ""),
    ("Re-run on a new Task 1 version", "Keep run 1 or re-run the whole pipeline", "Re-run everything on the new corpus; archive run 1", "K", "The final Task 1 review changed (259-paper corpus, 60-paper review set)."),
    ("Review set for run 2", "30-paper pipeline_ref (older), Prompted_ref (59, no abstracts), the 60 'selected' papers", "The new 60-paper pipeline_ref.json (= the 60 'selected' papers)", "K", "Matches the final corpus; all 60 have abstracts."),
    ("Entities without final_ref_set.json", "Upload OpenAlex records or extract entities from abstracts", "Extract key phrases from abstracts (no preference stated; Claude's recommended option)", "K / Claude", "No OpenAlex keywords exist for the final corpus."),
    ("Problems in run 2", "Keep B1 and A1 or let Stage 1 choose again", "Let Stage 1 choose again", "K", "A clean re-run; the result was again one problem from group A and one from group B."),
    ("Re-run on Task 1 run v6 (run 3)", "Keep run 2 or re-run on the new Task 1 output", "Re-run everything; archive run 2 in runs/run2_qualsg_v4/", "K", "Task 1 v6 adds OpenAlex keywords and topics, one output file, MVFA-citing papers and years; only 25 of run 2's 60 review papers are among the new 59."),
    ("Task 2 input for run 3", "Task 2 PDF (15 limitations) or the task2 branch's structured output", "The task2 branch's full_nim/final_limitations.json (39 limitations with aspect, importance, suggested fix, status, evidence)", "K", "A teammate had updated Task 2; the structured file is Task 2's main result."),
    ("Re-run Task 2 on Task 1 v6", "Re-run Task 2 first or use it as it stands", "Use it as it stands; document the version difference", "K (after explanation)", "Same Task 2 code, different input; the limitations of groups A and B are derived from the MVFA paper, so a re-run would change little."),
    ("Which groups get problems (option a/b)", "(a) only the two groups Task 4 builds on (labels A, scoring B) or (b) all candidate groups", "(a): two candidates each for groups A and B", "K", "The ideas are meant for the Task 4 implementation; fewer problems means more effort per problem."),
    ("Both-orders pairwise judging", "Now or later", "Later (option b noted for future work)", "K", ""),
    ("Task 1 code", "Use run-v6 output without its code or wait", "Use the output now; ask the Task 1 teammate to push the v6 code", "K / Claude", "The output is final; the code should be in the repository for reproducibility."),
    ("Independent judge", "Open model on Colab, a free API, or a paid API", "Gemini 3.8 Flash through Google's Gemini API (API calls)", "K", "A model from another company; Colab Pro had ended."),
    ("HypER Phi-3 checker", "Run on Colab, on another GPU, or skip", "Trained on a free GPU (a Modal A100 on its free monthly credit)", "K", "Colab Pro had ended."),
    ("Both orders for the independent judge", "Skip or ask the 336 comparisons again with the ideas swapped", "Run (rejudge.py stage3swap)", "K", "Separates position bias and noise from real disagreement between the judges."),
]
DEFAULTS = [
    ("Embeddings", "TF-IDF instead of sentence-transformers for similarity", "Hugging Face downloads were blocked in the run environment."),
    ("Entity store (run 3)", "OpenAlex keywords and topics of every Task 1 paper (as in run 1); entities in at least two papers are retrievable", "Task 1 v6 provides these fields again; they are closer to ResearchAgent's linked entities than run 2's key phrases."),
    ("Related papers for ResearchAgent", "58 other papers of the review set + the 10 MVFA-citing papers most similar to MVFA's abstract that are not already in the review (68 in total)", "ResearchAgent narrows citing papers by abstract similarity; many review papers already cite MVFA, so duplicates are skipped."),
    ("Papers citing MVFA", "Union of Task 1's 'cites_mvfa' papers and Semantic Scholar's citing papers (s2_citations.json); Semantic Scholar records first because they carry citation contexts", "Task 1 found 47 citing papers that Semantic Scholar did not list; Semantic Scholar adds 27 and the sentences in which papers cite MVFA."),
    ("Stage 0 input", "Task 2's limitation records (limitation, explanation, aspect, importance, suggested fix, status) instead of parsing a PDF", "Structured input; the TF-IDF hint uses 8 clusters for 39 limitations."),
    ("Code notes in limitations", "Each code-verified limitation carries its verdict and a one-sentence code note in its text; code-only findings become new gaps C1-C4 and join the group of the limitation they extend", "Problems must rest on what the code does; all 20 checks (E1-E20) are now merged."),
    ("HypER hop-1 relevance", "Papers already judged in run 2 keep their run-2 score and explanation (110 papers); the 48 papers new in run 3 were read and scored", "The prompt, source paper and judge are the same, so the judgment should be too; it keeps the judge consistent across runs."),
    ("Seed ideas", "Where a chain and problem match an earlier run, the earlier seed is reused in adapted form and rewritten to the run-3 problem", "Four of six chains are the same as in run 2; the problems changed in their details."),
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
    ("Novelty evidence", "14 thematic web searches covering the 24 shortlisted ideas (queries saved per idea); novelty_search.py re-runs them on Semantic Scholar", "Semantic Scholar was not reachable from the run environment; ideas of one theme share a query."),
]
FUTURE = [
    ("Deciding between the two judges", "The two judges disagree on the order of ideas within a problem (section Independent checks). A third model can be added with the same command (set JUDGE_MODEL and run python3 rejudge.py stage1 stage2 stage3 stage3swap stage4), for example Gemini 3.1 Pro at about three times the cost of Flash. For Task 4, ideas both judges place in the top 5 (B1-g2-o1, B1-g2-o4, B1-g3-o5) are the safest choices."),
    ("Position-bias correction (planned)", "Set BOTH_ORDERS = True in stage3_evolve.py. Each pair is judged in both orders and disagreement counts as a tie; about 1.8x the comparison calls."),
    ("HypER checker in our domain", "The checker was trained on HypER's clinical-trial chains for 1 epoch (HypER: 10). It agrees with our judge on which papers are irrelevant but scores many topically similar papers 2. Training it longer, or adding computer-vision chains labelled by our Stage 2 judge, would test whether the gap is domain transfer; colab/modal_hyper.py runs either with --epochs or a changed training file."),
    ("Per-criterion reviews", "Split the Stage 4 review into one call per criterion (as ResearchAgent does) to remove interaction between criteria in one answer."),
    ("Larger search", "HypoEvolve's sensitivity analysis also tests a population of ten and five generations; increase MU and G in stage3_evolve.py. Cost grows quadratically with the pool."),
    ("Final tournament", "Re-score all 24 candidates of a problem in one Bradley-Terry fit so that fitness is fully comparable across generations (currently each candidate keeps its most recent fitness)."),
    ("Better retrieval", "Replace TF-IDF with a sentence encoder (all-MiniLM-L6-v2, as in the Task 1 repo) and re-run stages 1-2; expand hop-2 citations beyond the 40 most-cited first-hop papers (fetch_citations.py --hop2-top)."),
    ("Novelty re-check", "Run novelty_search.py on your machine and compare the Semantic Scholar hits with the saved web-search evidence."),
    ("Implementation order for the next assignment", "Problem 1 (B1) ideas are inference-only on retrained MVFA checkpoints and are the fastest to test: build the leave-one-out support statistics once, then the alignment-adaptive residual CDF (B1-g2-o1) and its exact leave-one-out version (B1-g3-o5), the GPD tail extension (B1-g2-o4) and the conformal threshold (B1-g1-o5) as an evaluation layer; the content-conditioned regressor (B1-g3-o4) needs a small model trained on source normals. For Problem 2 (A1), start with normal-bank-gated MIL (A1-g1-o2), whose normal bank is reused by per-image k (A1-g3-o2) and gated mosaic MIL (A1-g3-o3); witness-rate MIL (A1-g1-o6) and mosaic MIL (A1-g2-o2) change only the loss and the data loader."),
    ("Task 2 on Task 1 v6", "Re-run Task 2 on Task 1's run-v6 corpus and compare its limitations with the 39 used here; if groups A and B change, re-run from Stage 0."),
    ("New Task 1 or Task 2 version", "Replace data/task1/final_ref_set.json or data/task2/final_limitations.json, update data/inputs_manifest.json, move outputs/ to runs/, clear logs/calls.jsonl, and run the stages; unchanged prompts replay from the cache."),
]


def independent_checks():
    """Section on the two independent checks run after the pipeline: re-judging with Gemini 3.8 Flash (rejudge.py) and
    HypER's distilled checker trained on Modal (colab/modal_hyper.py). Each part is written only if its result file exists."""
    rj = sorted(O.glob("rejudge_*.json"))
    hc = O / "stage2b_hyper_check.json"
    if not rj and not hc.exists():
        return ""
    items = lambda xs: "\\begin{itemize}[leftmargin=1.2em]\n" + "".join("\\item " + tex(x) + "\n" for x in xs) + "\\end{itemize}\n"
    P = lambda title, text: "\\paragraph{" + tex(title) + "} " + tex(text) + "\n\n"
    code = lambda text: "{\\small\\begin{verbatim}\n" + text.strip("\n") + "\n\\end{verbatim}}\n"

    def table(spec, head, rows):
        t = "\\begin{tabularx}{\\linewidth}{@{}" + spec + "@{}}\\toprule\n" + " & ".join(tex(h) for h in head) + "\\\\\\midrule\n"
        for r in rows:
            t += " & ".join(tex(c) for c in r) + "\\\\\n"
        return t + "\\bottomrule\\end{tabularx}\n\n"

    s = "\\section{Independent checks}\n" + tex(
        "The same model (Claude) generated and judged everything in this run, as in ResearchAgent and HypoEvolve, so the judgments could share "
        "the generator's preferences. Two checks were run afterwards with models from elsewhere. Neither changes the run's outputs; both only "
        "measure how far other judges agree with ours.") + "\n" + items([
        "An independent judge: Gemini 3.8 Flash (Google) answered every judging prompt of stages 1-4 again, word for word, and every pairwise "
        "comparison a second time with the two ideas swapped (rejudge.py, through API calls to Google's Gemini API).",
        "HypER's own distilled checker: Phi-3-mini fine-tuned with LoRA on HypER's released citation chains, the small-model half of HypER that the "
        "run had skipped, used to re-check Stage 2 (colab/modal_hyper.py, trained on a free GPU: a Modal A100 on its free monthly credit)."])

    for f in rj:
        R = json.loads(f.read_text())
        judge = R.get("new_judge", f.stem)
        pretty = {"gemini-3.8-flash": "Gemini 3.8 Flash"}.get(judge, judge)
        s += "\\subsection{Re-judging with " + tex(pretty) + "}\n"
        s += P("Why.", "A judge tends to agree with its own way of writing and reasoning. If a model from another company, given exactly the "
               "same prompts, reaches the same scores and the same choices, the run's judgments do not depend on the judge being Claude. Where it "
               "does not, the report says which conclusions depend on the judge.")
        s += P("How it was run.", "Through API calls to Google's Gemini API (Gemini 3.8 Flash, OpenAI-compatible endpoint, low reasoning "
               "effort), with no change to the pipeline's code: llm_client.py already speaks the OpenAI protocol. The answers are cached in "
               "logs/cache/, so the run can be replayed for free.")
        s += P("What was re-asked.", "rejudge.py reads logs/calls.jsonl, the list of every model call this run made, and selects the judge calls "
               "of stages 1-4 (plus the second-hop relevance chunks that this run reused from the cache of run 2). It sends the identical system "
               "prompt, user prompt and temperature to the new model, so any difference comes from the judge alone. Three judge calls of the run "
               "are not re-asked because they are not quality judgments: one grouped Task 2's limitations and two named the method families of "
               "the shortlisted ideas.")
        s += table("l Y r r", ["Stage", "What the judge does", "Prompts", "Answers compared"], [
            ["1", "Rates 4 candidate problems, each in 3 versions, on 5 criteria (one call per criterion, 1-5)", "60", "60 ratings"],
            ["2", "Scores the relevance (0/1/2) of the papers citing MVFA and of the second-hop papers, several per call", "17", "167 scores"],
            ["3", "Picks the stronger of two ideas: all 15 pairs of the 6 seeds, then 51 new pairs in each later generation", "336", "336 winners"],
            ["4", "Reviews each of the 12 shortlisted ideas per problem on 5 criteria", "24", "120 ratings"],
            ["3, swapped", "The same 336 comparisons with Idea A and Idea B exchanged", "336", "336 winners"],
            ["Total", "", "773", ""]])
        s += P("Both orders.", "For each comparison, the user prompt was split at its 'Idea A:' and 'Idea B:' blocks and rebuilt with the two "
               "ideas exchanged, everything else unchanged. If Gemini prefers the same idea in both orders, its verdict is consistent; if it "
               "prefers whichever idea it reads first (or second), that is position bias. For the combined tournament, each pair's two verdicts "
               "are averaged, so a pair on which Gemini contradicts itself counts as a tie, and Bradley-Terry fitness is fitted again.")
        s += P("Measures.", "Spearman correlation: whether the two judges order items the same way (1 = same order, 0 = unrelated). Exact "
               "agreement and within one point: the share of identical ratings, or ratings at most one point apart. Weighted kappa: agreement "
               "on the 0/1/2 relevance scores beyond chance, counting a 0-versus-2 disagreement more than 0-versus-1 (above 0.6 is usually "
               "called substantial). Winner agreement: the share of comparisons where both judges pick the same idea (0.5 is chance). Top-5 "
               "overlap, in three versions: Gemini's reviews with our tournament fitness; Gemini's reviews with Gemini's tournament; and Gemini's "
               "reviews with Gemini's both-orders tournament. All three keep our shortlist of 12 ideas, the 0.7/0.3 weights and the rule of at "
               "most two ideas per method family.")
        xs = []
        if "stage1" in R:
            r = R["stage1"]
            xs.append("Stage 1 problem reviews (%s of %s answers readable): Spearman %s, exact agreement %s, within one point %s; mean score %s (ours) vs %s (new judge)."
                      % (r["n"], r.get("asked", r["n"]), r["spearman"], r["exact"], r["within1"], r["mean_original"], r["mean_new"]))
        if "stage2" in R:
            r = R["stage2"]
            xs.append("Stage 2 relevance 0/1/2 (n=%s): exact agreement %s, weighted kappa %s." % (r["n"], r["exact"], r["qwk"]))
        for pid, r in (R.get("stage3") or {}).items():
            x = ("Stage 3 %s (%s comparisons): same winner %s, Spearman of the Bradley-Terry fitness %s; best idea %s (ours) vs %s (new judge)."
                 % (pid, r["n"], r["winner_agreement"], r["bt_spearman"], r["original_best"], r["new_best"]))
            if "share_A_new" in r:
                x += (" Idea shown first chosen: %s (ours) vs %s (new judge); ties: %s vs %s."
                      % (r["share_A_original"], r["share_A_new"], r["ties_original"], r["ties_new"]))
            xs.append(x)
        for pid, r in (R.get("stage3_swap") or {}).items():
            xs.append("Stage 3 %s judged again with the two ideas swapped (%s comparisons): the new judge picks the same idea in both orders %s of the time; "
                      "the idea shown first wins %s of the time (0.5 = no position bias). Where both orders agree (n=%s), it picks the same winner as our judge %s of the time. "
                      "Bradley-Terry on both orders averaged: Spearman with ours %s; best idea %s (ours) vs %s."
                      % (pid, r["n"], r["self_consistency"], r["first_shown_wins"], r["n_consistent"], r["agreement_when_consistent"],
                         r["bt_spearman_both_orders"], r["original_best"], r["new_best_both_orders"]))
        for pid, r in (R.get("stage4") or {}).items():
            x = ("Stage 4 %s: Spearman of the review means %s; top-5 overlap %s of 5 with the new reviews and our tournament fitness (new top 5: %s)."
                 % (pid, r["spearman_mean"], r["top5_overlap"], ", ".join(r["top5_new"])))
            fi = r.get("top5_fully_independent")
            if fi:
                x += " With the new judge's tournament fitness as well: overlap %s of 5 (%s)." % (fi["overlap"], ", ".join(fi["top5"]))
            fb = r.get("top5_fully_independent_both_orders")
            if fb:
                x += " With that tournament judged in both orders: overlap %s of 5 (%s)." % (fb["overlap"], ", ".join(fb["top5"]))
            xs.append(x)
        s += "\\paragraph{Results.}\n" + items(xs)
        if "stage3_swap" in R:
            s += P("What the numbers say.",
                "The two judges agree on absolute scores: which problems are strong, which papers are relevant. They disagree on which of two "
                "ideas for the same problem is stronger, and this is not noise: Gemini picks the same idea in about nine of ten pairs when the order is swapped, "
                "and neither judge favours the idea shown first. The ideas in a generation are close in quality and the judges weigh the "
                "criteria differently, so their tournaments rank the ideas differently. For B1, three of the five selected ideas are also in Gemini's top 5, "
                "including the first one (B1-g2-o1). For A1, Gemini's top 5 consists of the shortlisted ideas ranked 7-12 here (posterior-weighted, "
                "consensus, co-teaching and matched-pair MIL variants), and Gemini's tournament winner is a seed (A1-g0-C1), so by its judgement evolution did not improve on the seeds of A1. "
                "Conclusions: the choice of problems and B1's core ideas hold under a second judge; the order of ideas within a problem, A1's above all, "
                "should be read as one judge's preference.")
        s += P("Limits.", "One second judge, asked once, at low reasoning effort; a stronger model or several samples could agree more or less. "
               "Gemini rates more generously on average, which matters little for rank correlations but makes exact agreement look lower. Its "
               "reviews cover only the 12 ideas per problem that our pipeline shortlisted, so an idea outside the shortlist that Gemini's tournament "
               "favours (A1-g0-C1) was never reviewed, and the fully independent top 5 still uses our shortlist and our method families. Ties "
               "were allowed but never used by either judge.")
        s += P("How to reproduce.", "From task3/, with a Gemini API key (the cached answers make the same command free to replay with the same "
               "settings):")
        s += code("""export JUDGE_BACKEND=openai
export JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export JUDGE_API_KEY=...        # never written to a file
export JUDGE_MODEL=gemini-3.8-flash
export JUDGE_CONCURRENCY=2 JUDGE_MAX_TOKENS=2048
export JUDGE_EXTRA_BODY='{"reasoning_effort": "low"}'
python3 rejudge.py stage1 stage2 stage3 stage3swap stage4
python3 make_reports.py""")

    if hc.exists():
        S = json.loads(hc.read_text())
        ti = S.get("train_info") or {}
        s += "\\subsection{HypER's distilled checker (Phi-3 + LoRA) on Modal}\n"
        s += P("Why.", "HypER has two halves. A large model builds citation chains and scores how far each paper depends on the one it cites; "
               "the authors then fine-tune a small model, Phi-3-mini-128k-instruct, on those chains so that it can check new chains cheaply. "
               "Our Stage 2 ran only the first half, with Claude as the large model. Training the second half and running it on our chains gives a "
               "check of Stage 2 by a model that learned HypER's notion of relevance from HypER's own data, not from our judge. In HypER's "
               "relevance prompt, 2 means that the paper's key hypothesis at least partly depends on the source paper's findings, 1 means it "
               "is related, 0 means it is not.")
        s += P("Three tasks.", "The checker answers in JSON, in HypER's three formats, rebuilt by colab/hyper_checker.py from the released "
               "chains: 1-hop relevance (score one paper against the paper it cites, 0/1/2); chain validity, c1 (is a whole citation chain "
               "valid, and which papers break it); and hypothesis support, c2 (does a valid chain support a given hypothesis). The same formats "
               "are built for our Stage 2: 158 papers citing MVFA and 9 second-hop papers, our 6 chains, and the 12 seed ideas (6 per problem, "
               "one per chain).")
        s += P("How it was run.", "On a free GPU: an A100 40 GB on Modal, covered by its free monthly credit, with colab/modal_hyper.py, "
               "which runs the same scripts and settings as the Colab notebook. Training took about 1 hour 45 minutes. Settings compared with HypER's:")
        s += table("l Y Y", ["Setting", "HypER's released configuration", "This run"], [
            ["Base model", "Phi-3-mini-128k-instruct", "same"],
            ["LoRA", "r 8, alpha 16, dropout 0.05, all linear layers", "same"],
            ["Optimisation", "lr 2e-4, cosine schedule, 10 warm-up steps, micro-batch 1, gradient accumulation 4, loss on the answer only", "same"],
            ["Precision", "8-bit loading", "bf16 weights"],
            ["Maximum length", "16,384 tokens", "%s tokens (%s of %s examples longer, skipped)" % (format(ti.get("max_len", 8192), ","), ti.get("skipped", 464), format((ti.get("examples") or 0) + (ti.get("skipped") or 0), ","))],
            ["Training", "10 epochs on 8 GPUs", "%s epoch on one A100 40 GB: %s examples, %s optimizer steps" % (ti.get("epochs", 1), format(ti.get("examples") or 0, ","), format(ti.get("steps") or 0, ","))]])
        if "hyper_test_base" in S and "hyper_test_lora" in S:
            rows = []
            names = {"hyper-1-hop": "1-hop relevance (0/1/2)", "hyper-multi-hop2_c1": "Chain validity (c1)", "hyper-multi-hop2_c2": "Hypothesis support (c2)"}
            for t in S["hyper_test_base"]:
                b, l = S["hyper_test_base"][t], S["hyper_test_lora"][t]
                rows.append([names.get(t, t), str(b["n"]), "%s / %s / %s" % (b["macro_f1"], b["accuracy"], b["unparsed"]),
                             "%s / %s / %s" % (l["macro_f1"], l["accuracy"], l["unparsed"])])
            ab = round(sum(v["macro_f1"] for v in S["hyper_test_base"].values()) / 3, 3)
            al = round(sum(v["macro_f1"] for v in S["hyper_test_lora"].values()) / 3, 3)
            rows.append(["Average macro F1", "", str(ab), str(al)])
            s += P("Did the training work?", "Both models answered a stratified sample of HypER's test split, 100 questions per task. Cells "
                   "show macro F1 / accuracy / unreadable answers.")
            s += table("Y r l l", ["Task", "n", "Base Phi-3", "Fine-tuned checker"], rows)
        xs = []
        for k, lab in (("hop1", "papers citing MVFA"), ("hop2", "second-hop papers")):
            r = S["relevance"].get(k) or {}
            if r.get("parsed"):
                xs.append("Relevance of %s (n=%s): same score as our judge for %s, weighted kappa %s; our scores %s, checker %s."
                          % (lab, r["parsed"], r["exact"], r["qwk"], r["ours_distribution"], r["checker_distribution"]))
        st = S["relevance"].get("chain_starters_scored_0_by_checker") or []
        xs.append("Chain-starting papers that the checker scores 0: " + ("; ".join(st) if st else "none") + ".")
        xs.append("Chain validity: " + ", ".join("%s %s" % (c["chain"], c["validity"]) for c in S["chains"]) + ".")
        xs.append("Seed ideas whose chain the checker finds supports the seed hypothesis: %s of %s." % (sum(x["supported"] for x in S["seeds"]), len(S["seeds"])))
        s += "\\paragraph{Results on our Stage 2.}\n" + items(xs)
        cm = (S["relevance"].get("hop1") or {}).get("confusion_ours_rows_checker_cols")
        if cm:
            s += tex("Relevance of the papers citing MVFA, our judge (rows) against the checker (columns):") + "\n\n"
            s += "\\begin{center}\\begin{tabular}{lccc}\\toprule\n & checker 0 & checker 1 & checker 2\\\\\\midrule\n" + \
                 "".join("ours %d & %s\\\\\n" % (i, " & ".join(str(v) for v in row)) for i, row in enumerate(cm)) + "\\bottomrule\\end{tabular}\\end{center}\n\n"
        s += P("What the numbers say.",
            "The training worked: on HypER's own test questions the checker's average macro F1 rises from 0.49 (base Phi-3) to 0.75; "
            "HypER reports 0.47 to 0.62 on its full test set. It never predicts the middle relevance score 1, which only 5 of the 100 test papers have. "
            "On our Stage 2 it agrees less with our judge: the same 0/1/2 score for 49% of the 158 papers citing MVFA (weighted kappa 0.21). "
            "The disagreement goes one way. It scores 2 for 57 papers our judge scored 0, mostly anomaly-detection and CLIP-adaptation papers that share MVFA's topic, "
            "and it never scores 0 a paper our judge found relevant; every chain-starting paper is relevant to it as well. "
            "It finds all six chains valid and all twelve seeds supported by their chains, so it raises no objection to Stage 2; "
            "because it answered 'valid' every time on our data, this is weak evidence either way. HypER's training chains are clinical-trial literature, "
            "so the gap mostly measures how far the checker transfers to computer vision.")
        s += P("Limits.", "One epoch instead of ten, inputs cut at 8,192 tokens instead of 16,384, and bf16 instead of 8-bit loading; "
               "the test-split comparison uses 300 of HypER's test questions, not all of them; the checker learned from clinical-trial chains "
               "and is applied to computer-vision papers; on our six chains and twelve seeds it never answered 'invalid', so those two results "
               "cannot separate good chains from bad ones.")
        s += P("How to reproduce.", "From task3/, after pip install modal and modal setup (a Modal account with credit):")
        s += code("""modal run colab/modal_hyper.py --step smoke     # 10-minute test
modal deploy colab/modal_hyper.py               # once, and after editing the script
modal run colab/modal_hyper.py --step start     # training on Modal; the laptop can sleep
modal run colab/modal_hyper.py --step eval      # laptop open, about 20 minutes
modal run colab/modal_hyper.py --step mvfa      # laptop open, a few minutes
python3 stage2b_hyper_check.py && python3 make_reports.py""")
    return s


def documentation():
    s = r"""
\begin{center}
{\sffamily\bfseries\huge Task 3 Pipeline: Full Documentation}\\[4pt]
{\sffamily\large How every result was produced, every decision taken, and what can be changed}\\[6pt]
{\color{muted}Base paper: MVFA (CVPR 2024) \quad·\quad Methods: ResearchAgent, HypER, HypoEvolve \quad·\quad Run 3 of 19 November 2026}
\end{center}
\tableofcontents
\newpage
"""
    s += r"\section{What this document covers}" + "\n" + tex(
        "The submission document presents the two research problems and the top-5 ideas for each. This document explains how they were produced: how Tasks 1 to 3 connect, the inputs, the architecture of the code, each stage with what the source paper prescribes and what we implemented (including every adaptation and why), the history of the three runs, the full decision log, the evidence found in MVFA's code, the limitations of the study, and a prioritised list of changes that can be made later. Every number here is read from the pipeline's output files.") + "\n"

    s += r"\section{How Tasks 1, 2 and 3 connect}" + "\n" + tex(
        "Task 1 (literature review) supplies the papers: its corpus provides the entity store (OpenAlex keywords and topics), its selected papers are the related papers that ResearchAgent reads, and its MVFA-citing papers, together with Semantic Scholar's, are the first hop of the HypER chains. Task 2 (research gaps) supplies the limitations of MVFA, which Stage 0 groups and Stage 0b checks against MVFA's code. Task 3 turns one limitation group into one research problem and the problem into ranked ideas. Task 4 implements ideas from the two problems in MVFA's code. The versions used here:") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y@{}}\\toprule Input & Version\\\\\\midrule\n"
    s += f"Task 1 & {tex(IN['manifest']['task1']['version'])}; code: {tex(IN['manifest']['task1']['code_branch'])}\\\\\n"
    s += f"Task 2 & {tex(IN['manifest']['task2']['source'])}; built on {tex(IN['manifest']['task2']['built_on'])}\\\\\n"
    s += "\\bottomrule\\end{tabularx}\n\n" + tex(
        "Task 2 was produced from Task 1's run v4 and was not regenerated for run v6 (decision 21). Of Task 2's 39 limitations, 23 cite literature evidence and 6 rest on literature alone; the eight limitations of the two groups used for problems are derived mainly from the MVFA paper, so a Task 2 re-run on the new corpus would change little for this run.") + "\n"

    s += r"\section{Inputs}" + "\n\\begin{itemize}[leftmargin=1.2em]\n"
    org = IN["task1_origins"]
    for t in [
        f"Task 2 limitations (data/task2/final_limitations.json): {IN['task2_limitations']} limitations of MVFA, each with an explanation, aspect, provenance, importance (1-5), suggested fix, status against later literature and evidence. Task 2's research-gap summary is kept in data/task2/task2_research_gaps.md.",
        f"Task 1 corpus (data/task1/final_ref_set.json): {IN['task1_papers']} papers with DOI, title, abstract, year, venue, citation count, OpenAlex keywords and topics, relevance and composite scores, and origin (search {org.get('search', 0)}, snowball {org.get('snowball', 0)}, citing MVFA {org.get('cites_mvfa', 0)}, MVFA's references {org.get('mvfa_reference', 0)}, co-citation {org.get('co_citation', 0)}, anchor 1 = MVFA). {IN['task1_review']} papers are marked 'selected' (the review set, including MVFA). Papers excluded by Task 1 are kept in final_ref_set_excluded.json and not used.",
        f"Papers citing MVFA: {IN['citers']} in total, {IN['citers_both']} found by both Task 1 and Semantic Scholar, {IN['citers_task1_only']} only by Task 1 and {IN['citers_s2_only']} only by Semantic Scholar (data/s2_citations.json, fetched by fetch_citations.py, which also gives the sentences in which papers cite MVFA and the second-hop citations of the 40 most-cited first-hop papers). {IN['citers_eligible']} of them have an abstract and a 2025-2026 publication year, the window HypER uses.",
        "MVFA's public code, snapshot of commit a30bf7b (2024-04-06), for code verification.",
        "HypER's released code and data (github.com/rosnikv/HypER): the relevance prompt (v002) and the two few-shot examples.",
        "The three method papers: ResearchAgent (arXiv:2404.07738), HypER (arXiv:2506.12937), HypoEvolve (arXiv:2609.15938).",
    ]:
        s += f"\\item {tex(t)}\n"
    s += "\\end{itemize}\n"

    s += r"\section{Architecture}" + "\n" + tex(
        "All LLM calls go through llm_client.py. Each call has a role: generator (problems, seeds, offspring) or judge (reviews, relevance scores, pairwise comparisons). The backend is set by environment variables: 'openai' sends the call to any OpenAI-compatible endpoint (Groq, AgentRouter, OpenAI), 'file' writes the prompt to logs/pending and stops the stage. In this run every call used the file backend and was answered by the configured model (claude-opus-5-5) acting as generator and judge; answers were written by small scripts in logs/answers/run3/ and ingested into logs/cache. Each cached record holds the role, model, tag, temperature, sample index, system prompt, user prompt and response. Because calls are cached by a hash of (model, system, user, sample) and all randomness is seeded, re-running any stage replays it exactly and continues where it stopped. Judge calls can be pointed at a different model with JUDGE_* variables. Because later generator prompts depend on earlier judgments, an independent re-judge is done by rejudge.py, which re-asks a new judge exactly the judge prompts of this run and measures agreement; a full re-run with an API model sets LLM_* for both roles.") + "\n\n"
    s += r"\begin{tabularx}{\linewidth}{@{}l Y@{}}\toprule File & Purpose\\\midrule" + "\n"
    for f, p in [("llm_client.py", "Backends, roles, cache, pending/ingest mechanism"), ("answer.py", "Helper to answer pending prompts by tag (file backend)"),
                 ("corpus.py", "Task 1 loader, review set, MVFA-citing papers (Task 1 + Semantic Scholar), OpenAlex entity store (ResearchAgent Eq. 2)"), ("stage0_gaps.py", "Reads Task 2's limitations, TF-IDF cross-check, grouping"),
                 ("stage0b_verify.py", "Model-free code verification with file/line evidence, verdicts and code-only gaps"), ("stage1_problems.py", "ResearchAgent problem discovery, review and refinement"),
                 ("stage2_hyper.py", "HypER chain construction and seed generation"), ("stage3_evolve.py", "HypoEvolve genetic algorithm"),
                 ("stage4_rank.py", "Shortlist, reviews, families, final score, diversity selection"), ("prompts/", "All prompt templates and the ResearchAgent rubrics (verbatim)"),
                 ("fetch_citations.py, novelty_search.py", "Semantic Scholar scripts to run locally"), ("rejudge.py", "Independent re-judging of this run's judge prompts with another model"),
                 ("export_explorer_data.py, make_reports.py", "Explorer data and the two PDFs"), ("run_all.py", "Runs every stage in order")]:
        s += f"\\texttt{{{tex(f)}}} & {tex(p)}\\\\\n"
    s += "\\bottomrule\\end{tabularx}\n"

    # ---------------------------------------------------------------- stages
    s += r"\section{Stage by stage}" + "\n"
    s += r"\subsection{Stage 0: grouping the limitations}" + "\n" + tex(
        f"The {IN['task2_limitations']} limitations were read from Task 2's output file (no LLM). A TF-IDF cosine clustering (8 clusters) gave a model-free view of similar limitations, passed to one judge call as a weak hint. The call tagged each limitation (component, kind, codability on BMAD with one GPU) and grouped them so that one fine-grained problem could address a whole group; limitations that fit no group stay ungrouped. Reason for grouping: ResearchAgent writes a problem from whatever it is given; all {IN['task2_limitations']} at once reproduces the everything-at-once problem.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}P{4.6cm} Y P{3.6cm}@{}}\\toprule Group & Members & Use\\\\\\midrule\n"
    use = {"Label-efficient adaptation": "source of A1 (Problem 2)", "Score calibration and robust aggregation": "source of B1 (Problem 1)",
           "Leakage-free generalisation protocol": "shared evaluation protocol"}
    for g in D["groups"]:
        s += f"{tex(g['name'])} & {', '.join(g['members'])} & {tex(use.get(g['name'], 'not used in run 3 (option a)'))}\\\\\n"
    s += f"Ungrouped & {', '.join(D['ungrouped'])} & \\\\\n"
    s += "\\bottomrule\\end{tabularx}\n"

    s += r"\subsection{Stage 0b: verifying the limitations against MVFA's code}" + "\n" + tex(
        "Since the next assignment modifies MVFA's public code, problems must rest on what the code does. Stage 0b locates every claim by a regular expression (the script fails if the code does not contain it), records file-and-line evidence (E1-E20), gives each testable Task 2 limitation a verdict with a one-sentence code note that is appended to its text, and adds code-only findings as new limitations (C1-C4) in the group of the limitation they extend.") + "\n"
    s += "{\\small\\begin{longtable}{@{}l P{3.6cm} P{10.8cm}@{}}\\toprule ID & File:line & Finding\\\\\\midrule\\endhead\n"
    for e in D["evidence"] + D["later_evidence"]:
        s += f"{e['id']} & \\href{{{e['url']}}}{{\\texttt{{{tex(e['file']).replace('/', '/\\allowbreak{}')}:{','.join(map(str, e['lines']))}}}}} & {tex(e['finding'])}\\\\\n"
    s += "\\bottomrule\\end{longtable}}\n"
    s += "{\\small\\begin{longtable}{@{}l l P{12.4cm}@{}}\\toprule ID & Verdict & Code note (or new limitation)\\\\\\midrule\\endhead\n"
    for g in D["gaps"]:
        cc = g.get("code_check")
        if isinstance(cc, str):
            try:
                import ast
                cc = ast.literal_eval(cc)
            except Exception:
                cc = None
        if cc:
            note = cc.get("note", "") if g["id"].startswith("L") else f"{g['title']}: {g['text']}"
            s += f"{g['id']} & {tex(cc.get('verdict', ''))} & {tex(note)}\\\\\n"
    s += "\\bottomrule\\end{longtable}}\n"
    s += tex("These discrepancies are also ready-made test cases for the group's reproducibility-gap auditor.") + "\n"

    s += r"\subsection{Stage 1: problem discovery with ResearchAgent}" + "\n" + tex(
        "Faithful elements: the problem-identification prompt (Table 6), the ReviewingAgent prompt (Table 9) with one call per criterion, the human-induced rubrics of Table 13 copied verbatim, entity retrieval by Eq. 2 over an entity co-occurrence store, citing papers narrowed by abstract similarity, and iterative refinement (initial version plus two refinements; the paper reports saturation after about three). Inputs: MVFA's title and abstract, 68 related papers (the 58 other papers of Task 1's 59-paper review set and the 10 MVFA-citing papers most similar to MVFA's abstract that are not already in the review), and 20 entities retrieved from an entity store over the whole corpus, built from the papers' OpenAlex keywords and topics: " + str(CS['entities']) + " entities, " + str(CS['retrievable']) + " occurring in at least two papers. Adaptations: the limitation group is an additional input; a feasibility statement describes the resources (public MVFA code, BMAD, one Colab GPU); the shared protocol is attached; the refinement prompt was written in the paper's style because it is not printed. Following decision 22 (option a), problems were generated only for groups A and B, two candidates each; the second was asked to differ from the first.") + "\n\n" + problem_scores_table() + "\n" + tex(
        f"B1 ({max(D['problems']['B1']['trajectory'])}) asks whether MVFA's transductive test-set normalisation can be replaced by an inductive one built from the K normal supports; reviewers asked for a defined statistic and for separating primary from secondary outcomes, which the first refinement did. A1 ({max(D['problems']['A1']['trajectory'])}) asks whether MVFA's segmentation adapters can be trained from image-level labels by multiple-instance learning; the refinements added a testable hypothesis about the text anchors and a control that replaces them. The second candidates (A2: synthetic abnormal supports; B2: image-level pooling) scored 3.8, limited by originality because synthetic anomalies and pooling ablations are well covered in the review set.") + "\n"

    s += r"\subsection{Stage 2: literature chains and seed ideas with HypER}" + "\n" + tex(
        f"Implemented from the paper and its released code: citing papers of the current paper are taken from the two following years, grouped by year, scored in chunks of 10 with the authors' relevance prompt (0 unrelated, 1 inspired, 2 dependent) and two few-shot examples; the model also returns each chunk's top-3; candidates are ranked by 0.7 x relevance + 0.3 x normalised citations; the best becomes the next source. Of the {sum(D['hop1_relevance'].values())} MVFA-citing papers from 2025-2026 with abstracts, {D['hop1_relevance'].get('2', 0)} were scored 2, {D['hop1_relevance'].get('1', 0)} scored 1 and {D['hop1_relevance'].get('0', 0)} scored 0. 110 of these papers had been judged in run 2 with the same prompt and kept their judgment; the 48 new ones (mostly found only by Task 1) were read and scored. Adaptations: with a single source paper, the six best first-hop papers each start a chain (HypER obtains many chains from many sources); second-hop candidates were pre-filtered to the 20 most similar abstracts; citation contexts were added to the relevance prompt where Semantic Scholar provides them. Papers found only by Task 1 have no citation contexts and no second-hop citations, so their chains stop after one paper. Seven papers scored 2; the six with the highest HypER score start chains (ties at zero citations were broken by order of discovery, which left out 'Mitigating Source-Domain Dependency for Target-Free Zero-Shot Medical Anomaly Detection'). Seed ideas use HypER's generation prompt (Listing 4) with the research problem, a code-verified map of MVFA's code, the protocol, and output fields for method, experiment and provenance.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y@{}}\\toprule Chain & Papers (year, relevance) and status\\\\\\midrule\n"
    for c in D["chains"]:
        s += f"{c['id']} & MVFA $\\rightarrow$ " + " $\\rightarrow$ ".join(f"{tex(p['title'])} ({p['year']}, {p['relevance']})" for p in c["papers"]) + f". {{\\color{{muted}}{tex(c['note'])}}}\\\\\n"
    s += "\\bottomrule\\end{tabularx}\n"
    s += tex("Two chains are new in run 3 and come from papers that only Task 1 found: ADAN (anatomy-decoupling adapters on BMAD) and 'Do Words Matter?', which proves that a trained adapter reads a pair of text anchors only through their difference and its length and tests this in MVFA. The second paper bears directly on Problem 2's hypothesis that the text anchors make weakly supervised localisation transfer; its seed turns that hypothesis into a control with scale-matched random anchors.") + "\n"

    s += r"\subsection{Stage 3: evolution with HypoEvolve}" + "\n" + tex(
        "Implemented as Algorithm 1 with the paper's settings: population 6, 6 offspring per generation, 3 generations, crossover probability 0.6, mutation probability 0.15 (forced when crossover is skipped), size-2 tournaments, all unordered pairs of the pool compared by the judge, Bradley-Terry fitness (initial scale mean 50, spread 60; later fits keep the multiplier and anchor survivors), and (mu+lambda) replacement with lineage records. The pairwise prompt keeps the structure of Appendix D.2 (specificity, evidence, testability; explicit list of what not to reward; REASONING/CONFIDENCE/WINNER). Pairs already judged in an earlier generation are answered from the cache, equivalent to a deterministic judge. Each pair is shown in one order chosen by a hash of the two ids.") + "\n\n"
    s += "\\begin{tabularx}{\\linewidth}{@{}l Y Y@{}}\\toprule & Problem 1 (B1) & Problem 2 (A1)\\\\\\midrule\n"
    rows = {}
    for pid in ("B1", "A1"):
        ev = D["evolution"][pid]
        means = [round(sum(h["fitness"].values()) / len(h["fitness"]), 1) for h in ev["history"]]
        rows[pid] = dict(means=" / ".join(map(str, means)), best=ev["candidates"][ev["best"]]["title"], comps=ev["comparisons"])
    s += f"Population mean fitness, gen 0-3 & {rows['B1']['means']} & {rows['A1']['means']}\\\\\n"
    s += f"Pairwise judgments & {rows['B1']['comps']} & {rows['A1']['comps']}\\\\\n"
    s += f"Best idea (highest fitness) & {tex(rows['B1']['best'])} & {tex(rows['A1']['best'])}\\\\\n\\bottomrule\\end{{tabularx}}\n"
    ncomp = rows['A1']['comps'] + rows['B1']['comps']
    s += tex(f"Fitness rises partly by construction, because survivors are re-ranked against new offspring by the same judge; it is not an independent quality measure. HypoEvolve uses external biological data for that purpose; here the Stage 4 reviews and novelty search play that role. The file-backend judge was fully transitive (no ties in {ncomp} judgments): verdicts were written from an explicit assessment of every candidate (strength and weakness, logs/answers/run3/s3_assess.py) and a considered ranking per generation that kept the order of earlier generations, unlike a sampled LLM judge.") + "\n"

    s += r"\subsection{Stage 4: multi-criteria ranking}" + "\n" + tex(
        "Pool: all 24 candidates per problem. Shortlist: the 12 with the highest most-recent fitness. For each shortlisted idea the closest prior work was looked up with a web search (14 thematic queries covering the 24 ideas, saved with the hits in data/novelty_search.json), and one ResearchAgent-style review scored Novelty (method innovativeness rubric), Feasibility (experiment feasibility rubric), Clarity (method clarity), Impact (problem significance) and Validity (method validity). One call per problem grouped all candidates into method families. Final score = 0.7 x (review mean - 1)/4 + 0.3 x normalised fitness; ideas were selected greedily with at most two per family.") + "\n"
    for pid in ORDER:
        s += f"\\subsubsection*{{Problem {PNUM[pid]} ({pid})}}\n" + ranking_table(pid)
        fam = D["ranking"][pid]["families"]
        s += "{\\small Families: " + "; ".join(f"{tex(k)} ({len(v)})" for k, v in fam.items()) + ".}\n\n"
    s += tex("Novelty evidence lowered several scores. For Problem 1, a 2026 study already calibrates few-shot anomaly thresholds by leave-one-image-out scoring of the target's normal images and analyses distribution-free certification (industrial data), weighted conformal anomaly detection in low-data regimes has dedicated work, CDF normalisation from held-out normal scores is implemented in anomaly-detection libraries, extreme-value score normalisation exists, and registration-based few-shot detection augments its support images. For Problem 2, MIL lesion segmentation from image labels (FMaMIL and earlier histopathology work), medical anomaly segmentation against a normal feature memory (MIAS-SAM), class-prior estimation for positive-unlabelled learning and the MIL ranking loss of video anomaly detection all exist; no prior use of mosaics with known quadrant labels for weakly supervised anomaly localisation was found. Ideas are new mainly in how they use MVFA's own components (its K normal supports, its two branches, its segmentation adapters and the MVFA-specific leave-one-modality-out setting).") + "\n"

    # ---------------------------------------------------------------- run history
    s += r"\section{Run history: runs 1, 2 and 3}" + "\n" + tex(
        "Run 1 (19-20 October 2026) used the SurveyGen corpus (341 OpenAlex papers, 16-paper review set, OpenAlex concepts as entities). Run 2 (20 October) used the QUAL-SG review (259-paper corpus, 60-paper review set, key-phrase entities). Run 3 (19 November, this document) uses Task 1's run-v6 output and Task 2's 39 structured limitations; runs 1 and 2 are archived in runs/ with their own PDFs, explorer data and call logs. What stayed: all prompts, rubrics and settings, the code verification checks and the Semantic Scholar citation data. What changed in run 3: the limitations (39 instead of 15, hence new groups and verdict notes), the corpus, review set, entities and MVFA-citing papers, the restriction to two groups (option a), and therefore the problems, seeds, offspring and rankings.") + "\n\n"
    s += "{\\small\\begin{tabularx}{\\linewidth}{@{}P{2.9cm} Y Y Y@{}}\\toprule & Run 1 (SurveyGen) & Run 2 (QUAL-SG) & Run 3 (Task 1 v6, this run)\\\\\\midrule\n"
    s += f"Task 2 limitations & 15 (+2 from code) & 15 (+2 from code) & {IN['task2_limitations']} (+{IN['code_gaps']} from code)\\\\\n"
    s += f"Corpus / review set & 341 / 16 & 259 / 60 & {CS['papers']} / {CS['review']}\\\\\n"
    s += f"Related papers in Stage 1 & 26 & 69 & 68\\\\\n"
    s += f"MVFA-citing papers scored & {sum(RUN1['hop1_relevance'].values())} & {sum(RUN2['hop1_relevance'].values())} & {sum(D['hop1_relevance'].values())}\\\\\n"
    s += f"Selected problems (Stage 1 mean) & B1 {max(RUN1['problems']['B1']['trajectory'])}, A1 {max(RUN1['problems']['A1']['trajectory'])} & A1 {max(RUN2['problems']['A1']['trajectory'])}, B1 {max(RUN2['problems']['B1']['trajectory'])} & B1 {max(D['problems']['B1']['trajectory'])}, A1 {max(D['problems']['A1']['trajectory'])}\\\\\n"
    for pid, name in (("A1", "Label-supervision problem (A1), top-5"), ("B1", "Scoring problem (B1), top-5")):
        t = ["{\\footnotesize " + " ".join(f"({j}) " + tex(X["evolution"][pid]["candidates"][i]["title"].split(":")[0]) for j, i in enumerate(X["ranking"][pid]["top5"], 1)) + "}" for X in (RUN1, RUN2, D)]
        s += f"{tex(name)} & {t[0]} & {t[1]} & {t[2]}\\\\\n"
    s += f"LLM calls (new) & {RUN1['total_calls']} & {RUN2['total_calls']} & {D['total_calls']}\\\\\n\\bottomrule\\end{{tabularx}}}}\n"
    s += tex("All three runs chose one problem about label supervision and one about scoring. In run 3 the label problem returned to multiple-instance learning from image labels (run 1's direction) because the code check now states plainly that half of BMAD never trains the segmentation branch; its best ideas change which patches MIL supervises (a normal-patch bank, mosaics with known quadrant labels, a per-dataset lesion fraction). The scoring problem now keeps MVFA's fusion weight and max pooling fixed and changes only the normaliser, which moved the best ideas from probability calibration and p-value combination (runs 1-2) to anatomy-aware, support-based normalisers and their small-K corrections.") + "\n"

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

    # ---------------------------------------------------------------- optional independent checks
    s += independent_checks()

    # ---------------------------------------------------------------- validity
    s += r"\section{Limitations and threats to validity}" + "\n\\begin{itemize}[leftmargin=1.2em]\n"
    for t in [
        "Single model as generator and judge. ResearchAgent and HypoEvolve do the same; HypER reports that LLM judges agree with experts least on originality and feasibility (Spearman -0.08 for feasibility), the two criteria that matter most here. Mitigations: code verification and novelty search as non-LLM evidence; an independent re-judge with Gemini 3.8 Flash (section Independent checks).",
        "Carry-over between runs. The file-backend model produced runs 1 and 2 before run 3, so run 3 is not independent of them: hop-1 relevance judgments of papers seen before were carried over, and seeds on chains shared with earlier runs were adapted from earlier seeds. The run-3 problems, offspring, reviews and top-5 ideas are new. An API model re-running the pipeline from scratch would not have this memory.",
        "Input versions. Task 2's limitations were produced from Task 1's run v4, while the literature used here is run v6; Task 1's v6 code is not yet in the repository.",
        "Papers found only by Task 1 have no citation contexts or second-hop citations, so two of the six chains (C5, C6) stop after one paper by construction.",
        "Deterministic, transitive judge. A sampled judge produces intransitive and noisy verdicts; Bradley-Terry results here are cleaner than they would be with an API model. The API judge used for the independent check was nevertheless consistent with itself in 91-92% of swapped pairs.",
        "Position bias not corrected in the run (option (a)); implemented as option (b). The independent judge, asked in both orders, chose the first-shown idea 51% (A1) and 54% (B1) of the time.",
        "Retrieval quality. TF-IDF similarity is weaker than sentence embeddings; OpenAlex keywords are closer to linked entities than key phrases, but some retrieved entities are generic ('transfer learning') or off-domain ('video anomaly detection', 'manufacturing quality control'), because the corpus contains many non-medical anomaly detection papers.",
        "Short chains. MVFA is from 2024, so its citation lineage has at most two further hops; five of six chains stop after one paper.",
        "Domain mismatch of HypER's few-shot examples (clinical trials).",
        "Fitness comparability across generations relies on the anchored offset; a final tournament would make it exact.",
        "Novelty search breadth. 14 thematic web queries for 24 ideas; absence of a hit is not proof of novelty.",
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
    s = s.replace("LLM calls in this run}\n\\begin{tabular}{@{}l r@{}}\\toprule Stage & Calls", "LLM calls in this run}\n\\begin{tabular}{@{}l r r@{}}\\toprule Stage & New calls & Reused from earlier runs")
    for k in ["stage0", "stage1", "stage2", "stage3", "stage4"]:
        s += f"{tex(names[k])} & {D['calls'].get(k, 0)} & {D['reused_calls'].get(k, 0)}\\\\\n"
    s += f"\\midrule Total & {D['total_calls']} & {sum(D['reused_calls'].values())}\\\\\\bottomrule\\end{{tabular}}\n\n"
    s += tex(f"Reused calls are prompts identical to an earlier run (the second-hop relevance chunk of chain C1), answered from the cache. In addition, 110 hop-1 relevance judgments were carried over from run 2 inside new chunk prompts (see Stage 2). Runs 1 and 2 made {RUN1['total_calls']} and {RUN2['total_calls']} calls; their outputs, documents and call logs are archived in runs/.") + "\n"
    compile_tex("Task3_Documentation", s, "Task 3 · Full documentation · MVFA")


if __name__ == "__main__":
    submission()
    documentation()
