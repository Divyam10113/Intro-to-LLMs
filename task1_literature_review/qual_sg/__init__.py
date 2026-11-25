"""QUAL-SG: a from-scratch, documented implementation of

    SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models
    (Bao, Nayeem, Rafiei, Zhang; EMNLP 2025; arXiv:2508.17647)

Stages (paper Figure 2 / Section 2.4):
    step1  literature retrieval            (topic -> candidate pool D)
    step2  quality-signal augmentation     (D -> D_ex, co-citation expansion + metadata)
    step3  quality-aware re-ranking        (D_ex -> top-K by avg rank over S_t, S_a, S_d)
    step4  outline generation              (LLM)
    step5  section-wise survey generation  (LLM)
    step6  evaluation                      (citation / content / structure)

Run everything with:  python3 -m qual_sg.run_all
"""

__version__ = "1.0.0"
