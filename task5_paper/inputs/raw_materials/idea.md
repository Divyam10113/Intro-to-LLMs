# Paper idea and constraints (final: Task 4 rounds 1-3 + follow-up)

## Framing
Mixed paper (PAPER_PLAN §5.2, partial win). Headline 1: image labels alone (plain top-k MIL) nearly match mask
supervision for MVFA's localisation on unseen modalities (H1, confirmed in three rounds). Headline 2 (Brain only):
masks for 10 % of the abnormal source images plus MIL beat full-mask MVFA (pre-registered follow-up, 5/5 seeds).
Secondary: MIL added to full masks matches MVFA (H4). Everything else (H2, H3, H5, round 2's ideas #1 and #3, round 1's
PU) goes into one compact "what did not hold" table. Never claim to beat MVFA overall or the published MVFA numbers.

## Method
MVFA (frozen CLIP ViT-L/14 + residual adapters at 4 levels) trains segmentation adapters with Dice + Focal on the
masked sources only. MIL replaces that with, per level, BCE(mean of the top-k = 3 patch probabilities, image label)
on every source image. A mask budget b keeps masks for a seeded share b of the masked sources' abnormal images.
MIL+100 % keeps all masks and adds MIL on the mask-less sources. Tested refinements (from the LLM pipeline):
normal-bank-gated top-k selection, and nnPU with an estimated or fixed class prior.

## Structure and 4+1 budget
1 Introduction; 2 Related Work (2.1 vision-language medical anomaly detection; 2.2 weak supervision / MIL;
2.3 PU learning); 3 Problem Formulation; 4 Methodology (4.1 data and protocol; 4.2 study design); 5 Experiments
(Table 1 round 3, Figure 1 label budget, Table 2 hypothesis record, diagnostics, limitations); 6 Conclusion.
Content ends on page 4; page 5 = references only. Numbers only from results.json; citations only from refs.bib.
