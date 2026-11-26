# Paper idea and constraints (Round 1)

## Working title and framing
How Far Do Image-Level Labels Go for Localisation in Medical Anomaly Detection?

Frame this as an honest **analysis paper** under PAPER_PLAN.md §5.1 (F-C). The study found no consistent PU gain over plain fine-tuning, failure of the MVFA-score prior estimator, and a strong but unregistered top-k MIL observation. Never claim improvement over MVFA.

## Problem
MVFA's segmentation adapters learn from pixel masks on only the masked BMAD sources; the three mask-less sources provide image-level labels but do not train localization. Study patch-level image-label supervision in leave-one-modality-out evaluation on Brain MRI, Liver CT and RESC.

## Round 1 method as run
- M-h warm-starts the segmentation adapters from R0, retaining MVFA's Dice + Focal loss on masked sources and adding λ times non-negative PU risk on mask-less sources. M-h's π uses the mean true anomalous-patch fraction from masked sources (decision D1); this is not mask-free prior estimation.
- M-h-ctrl performs the matched mask-loss fine-tune without the PU iterations.
- Fully mask-free M retrains both adapter sets from scratch using image labels only, with fixed π = 0.30 (decision D2).
- B0 retrains both adapter sets with top-k MIL and no pixel masks.
- The prior estimator was corrected to a lower-tail form, but it still fails on MVFA's real patch scores; use its failure evidence and name the irreducibility assumption.
- Last-step models were evaluated; target tests were not used for these fine-tunes. R0's released checkpoints were selected using target test performance.

## Objective
For normal-image patches N sampled from p₀ and abnormal-image patches U sampled from πp₁+(1−π)p₀, with z=s₁−s₀ and sigmoid(z) the patch anomaly probability, use

`R_nnPU = (1−π) mean_N ℓ(z,0) + max(0, mean_U ℓ(z,1) − (1−π) mean_N ℓ(z,1))`.

On masked sources M-h also uses MVFA's Dice + Focal loss; the hybrid control omits the PU branch. M uses the PU risk without masks. B0 uses top-k MIL with k=max(3,1% of patches).

## Contributions supported by Round 1
1. A controlled PU-vs-fine-tuning comparison: no consistent gain under the registered rule, with a target-specific Brain signal.
2. A failure analysis showing the prior estimator is unreliable on MVFA patch scores relative to known masked-source fractions.
3. A mask-free top-k MIL observation that retains much of R0 localization in these runs; label it unregistered and single-seed on two targets.
4. An implementation audit of the released MVFA training path, especially target-test checkpoint selection and which branches receive masks.

## Forced structure and 4+1 page budget
1 Introduction (background/problem + contributions); 2 Related Work (2.1 medical anomaly detection and vision-language adaptation; 2.2 weakly supervised and mask-free anomaly localization; 2.3 positive-unlabelled learning and mixture-proportion estimation); 3 Problem Formulation; 4 Methodology (4.1 Dataset; 4.2 Approach); 5 Experiments (setup, results, discussion, limitations); 6 Conclusion. Abstract is written last. No appendix.

Budgets: title/authors/abstract 0.30 page; Introduction 0.55; Related Work 0.50; Problem Formulation 0.40; Methodology including Figure 1 0.60; Experiments with two tables and Figure 2 1.45; Conclusion 0.20. All content through Conclusion ends by page 4; page 5 is references only. Cut prose rather than shrinking fonts or margins.

Only numeric source: `results.json`. Only citations: keys in `refs.bib` / `citation_pool.json`. Never pool Round 1 and Round 2.
