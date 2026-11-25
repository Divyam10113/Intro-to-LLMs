# Task 2 — Research gaps and research problem for MVFA

**Base paper:** *Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images* (MVFA), CVPR 2024, arXiv:2403.12570.

**Main result:** the new multi-agent pipeline run on MiniMax M3.1 Flash, through NVIDIA NIM.
**Second run for comparison:** the same pipeline on gpt-6-astra, through AgentRouter. It appears in Section 4.

Each bullet ends with a tag: `[limitation ids | aspect | where the issue came from | status in later work]`.
- **Where it came from:** *Author-stated* = the MVFA authors say it themselves; *Inferred* = reasoned from the paper; *Peer-review* = a reviewer-style critique; *Literature* = found in later papers.
- **Status:** *open* = not addressed by later papers; *partially_addressed* = some later work covers part of it; *addressed* = later work has fixed it.

Full evidence for every limitation, including verbatim quotes from the paper, is in
[`minimax_full_with_evidence.md`](minimax_full_with_evidence.md).

---

## 1. Research gaps (main result: MiniMax)

- **Few-shot estimates and ablations lack statistical support.** All few-shot results derive from a single random K-shot draw, with no repeated runs, error bars or significance tests, and accuracy is non-monotonic in K (HIS AC 85.10 at K=8 versus 82.62 at K=16). The reported ablation margins (0.13 AC for multi-level ensembling, +1.29 AC for the dual adapter) therefore cannot be separated from support-sampling noise, so the abstract's "significantly surpasses" claim is not statistically supported. [L1, L27 | Experimental Design; Results & Analysis | Inferred; Peer-review; Literature | partially_addressed]
- **Split integrity and leakage are unverified for several benchmarks.** MVFA itself reports no patient-, volume- or slide-level split check for any dataset, so correlated slices or patches from the same case may straddle the train/test boundary; even where slide separation holds (HIS), patch-level AUCs from 115 WSIs are treated as independent observations. BMAD, the benchmark MVFA takes its data from, documents patient-ID partitioning for BrainMRI and case/slide separation for RESC and HIS; we found no such documentation for LiverCT, OCT17 or ChestXray, and no later study re-runs MVFA under verified leakage-free splits. [L2 | Experimental Design | Inferred; Peer-review; Literature | partially_addressed]
- **"Zero-shot" transfer still depends on dense medical supervision.** Adapters are trained on labeled images and pixel masks from five source datasets with unreported per-fold budgets, and the resulting supervised model is compared against the training-free WinCLIP. Later "strictly zero-shot" work (TAFT) and target-free frameworks still train on source domains and report no label/mask accounting, so the gains cannot be read as transfer without medical labels. [L3 | Experimental Design | Inferred; Peer-review | partially_addressed]
- **The few-shot protocol is under-specified.** The paper says only that K annotated images are drawn at random; it does not state their normal/anomalous composition, how subsets are seeded, or whether target samples are used alone or jointly with labeled data from the other five datasets. Later work (MadCLIP, UniVAD, TAFT) specifies its own support composition but leaves MVFA's protocol unrecovered, so cross-paper comparisons inherit the ambiguity. [L4 | Experimental Design | Inferred; Peer-review; Literature | partially_addressed]
- **The multi-level alignment loss is undefined without pixel masks.** Dice and Focal terms are applied at every level to all datasets, yet OCT17, ChestXray and HIS provide only image-level labels and the treatment of missing masks is never specified, so the effective objective differs across folds and cannot be reproduced from the text for half the benchmark. MadCLIP and TAFT acknowledge the granularity split but still define mask-based losses without a documented fallback. [L5 | Methodology | Inferred | partially_addressed]
- **The few-shot branch remains a one-class normal-only detector.** The memory bank contains only labeled normal images, so abnormal support samples influence the training loss but never serve as test-time references, even though the paper stresses the value of a few abnormal samples (it uses them only as training supervision). MadCLIP adds a learned abnormality branch yet keeps abnormal exemplars out of the reference set, and TAFT, a text-adapted follow-up to MVFA, retains the normal-only bank. [L6 | Methodology | Inferred; Peer-review; Literature | partially_addressed]
- **Few-shot adaptation presupposes expert pixel masks.** Every training tuple includes a pixel-level lesion mask (the labeled pools for BrainMRI, LiverCT and RESC hold 83, 166 and 115 mask-annotated images), yet annotation cost, mask count and mask-free performance are never quantified. UniVAD auto-generates component masks in a normal-only regime and TAFT still requires region masks, so the scarce-data motivation remains only partially substantiated. [L18 | Clinical & Deployment | Inferred | partially_addressed]
- **Test-time fusion is uncalibrated and under-specified.** The final score mixes a softmax probability with a cosine distance at fixed 0.5/0.5 weights without scale calibration, and the paper does not state whether the memory-bank branch is active in zero-shot folds or report branch-wise scores. TAFT separates the two paths but still fixes the weights and omits calibration. [L10 | Methodology | Inferred; Peer-review | partially_addressed]
- **Headline averages and unequal supervision budgets obscure regressions.** AC averages span six datasets while AS averages span only the three mask datasets, the +7.33 AC gain over APRIL-GAN is dominated by LiverCT, and MVFA loses on BrainMRI zero-shot AS and on OCT17 at K=2; baselines are also trained under unequal budgets (thousands of normal images versus K labeled images). Later work improves the partitioning and qualification of comparisons but does not re-audit the original numbers. [L7, L12 | Results & Analysis; Experimental Design | Author-stated; Inferred; Peer-review; Literature | partially_addressed]
- **Evaluation is restricted to ROC AUC.** No Dice/IoU, PRO/AUPRO, boundary or lesion-level metric, PR-AUC, threshold-based sensitivity/specificity or calibration is reported, so background-dominated pixel AUC cannot be read as localization quality and classification operating points remain unassessed. BMAD includes PRO, but much of the later medical AD work we found still reports AUROC only. [L8 | Results & Analysis | Inferred; Peer-review; Literature | partially_addressed]
- **Prompt construction is ambiguous and unaudited.** The main text describes generic normal/anomalous states while the appendix substitutes target organ names, the 35 templates are averaged into a single 2×d feature without a stated selection criterion, and no prompt ablation is reported. Learnable or filtered prompts in later work do not clarify whether MVFA relies on privileged target-anatomy metadata at inference, leaving the "unseen anatomical region" claim unresolved. [L9 | Methodology | Inferred; Peer-review; Literature | partially_addressed]
- **A single leave-one-out fold with confounded source pools defines the zero-shot claim.** One source combination per target makes the averages depend on which five datasets, and their order-of-magnitude size differences, form the training pool, and the OCT17 target retains RESC, another retinal OCT dataset, so "unseen modality and anatomical region" does not hold. The later work we found keeps the single fold and does not re-run OCT17 without RESC. [L11, L22 | Experimental Design | Inferred; Peer-review; Literature | partially_addressed]
- **The training recipe is undocumented and model selection is unvalidated.** Only input size, optimizer, learning rate, batch size and epoch count are given, with no per-modality intensity or stain normalization, augmentation, scheduler or seeds, and no validation split or early-stopping rule for a fixed 50-epoch schedule at K=2. BMAD, the benchmark MVFA builds on, already uses validation-based checkpointing and five-seed repetition, but MVFA does not adopt either. [L19, L20 | Methodology; Experimental Design | Inferred; Peer-review | partially_addressed]
- **The multi-level design is not separated from added capacity or compute.** The adapter-versus-projector ablation varies structure and capacity without parameter counts, and no matched-budget LoRA, prefix/VPT or full fine-tuning baselines are given; the "lightweight" claim is unquantified (no trainable parameters, FLOPs, latency, peak memory or memory-bank size). Later PEFT precedents and cost tables do not supply a capacity-matched comparison for MVFA. [L14, L15 | Methodology | Inferred; Peer-review; Literature | partially_addressed]
- **Binary labels conflate heterogeneous anomaly semantics.** ChestXray's positive class mixes "Lung Opacity" with "No Lung Opacity/Not Normal" cases and OCT17's three anomaly types are collapsed into one positive class, with only aggregate AUCs reported. No later work re-labels these datasets or reports per-anomaly-type results, so it remains unknown whether the model detects diverse pathologies or exploits dataset-level artifacts. [L13 | Experimental Design | Inferred; Peer-review; Literature | open]

## 2. Research problem (main result: MiniMax)

Current CLIP-based few-shot medical anomaly detectors assume supervision that clinical sites rarely have: MadCLIP and TAFT (a text-adapted follow-up to MVFA) train on expert pixel masks, score test images against a normal-only memory bank, and fuse uncalibrated softmax and cosine-distance scores. The problem is to build a mask-free, support-composition-robust detector that uses normal and abnormal labeled support images symmetrically as test-time references and returns one calibrated score. The setting is multi-modality medical anomaly detection (MRI, CT, OCT, X-ray, histology) with a frozen CLIP backbone and MVFA-style multi-level residual adapters. At training time it assumes source-domain labeled images (masks optional) plus K target-domain image-level labels; at test time only unlabeled target images and the K support images, with no target masks or per-target retraining. It must (i) replace mask-dependent Dice/Focal alignment with an objective well defined under image-level supervision alone, (ii) learn a bidirectional reference model in which abnormal exemplars constrain the boundary at test time, (iii) remain stable across balanced and unbalanced support compositions, and (iv) fuse text and exemplar evidence by calibrated rather than fixed-weight combination. None provides these jointly: MadCLIP adds a learned abnormality branch but still requires dense masks and keeps abnormal exemplars out of the reference set; UniVAD is training-free and normal-only; TAFT retains region masks and single-draw K=4 evaluation. Success will be measured on the six BMAD benchmarks with patient/slide-level splits and repeated K ∈ {2, 4, 8, 16} draws (balanced and unbalanced), reporting image-level AUROC/AUPRC, ECE/Brier calibration, sensitivity at fixed specificity, pixel-level PRO and Dice, per-anomaly-type results for OCT17/ChestXray, and matched-budget comparisons against WinCLIP, APRIL-GAN, AnomalyCLIP, MadCLIP, UniVAD and TAFT.

Addresses: L5, L6, L10, L18

---

## 3. How the gaps were found

The pipeline is in `limitations/`. It extends the methods of the three assigned papers.

1. **Corpus.** 125 papers that cite MVFA and 62 papers it cites, collected through OpenAlex, plus the papers selected in Task 1. Full text was downloaded where available and split into sections.
2. **Four writer agents** each list limitations:
   - **Extractor:** what the authors admit.
   - **Analyzer:** method and experiment flaws.
   - **Reviewer:** a peer-review style critique.
   - **Citation:** what later papers say about MVFA. It finds them through keyword search plus meaning-based search, then an LLM re-ranks the results.

   A **Judge** scores each agent's list and sends it back for one rewrite if the score is below 8/10. This step follows arXiv:2601.11578 and BAGELS (arXiv:2505.18207).
3. **Verifier (ours).** Every limitation must quote its proof word-for-word. The quote is checked by exact string match, and then an LLM checks that the quote supports the claim.
4. **Master.** Merges duplicates, rates importance from 1 to 5, and tags where each point came from. Aspects follow the LimitGen categories (arXiv:2507.02694), plus our own Clinical & Deployment category.
5. **Gap status (ours).** Each limitation is checked against papers from 2024 onward and marked open, partially addressed or addressed.
6. **Problem statement.** Built from 3–5 important gaps that are still open or only partially addressed.

## 4. Second run for comparison: gpt-6-astra

Same pipeline, different model. A few calls fell back to gpt-6.1-sol when astra was unavailable. The full evidence is in
[`astra_full_with_evidence.md`](astra_full_with_evidence.md).

### 4.1 Research gaps (astra)

- **Unverified modality and anatomy holdouts.** RESC and OCT17 both contain retinal OCT, but no supplied account explicitly verifies their joint exclusion during zero-shot adaptation. Consequently, genuinely unseen-modality/anatomy transfer remains insufficiently distinguished from transfer within a previously seen domain. [L1 | Experimental Design | Inferred, Peer-review, Literature | open]

- **Unestablished patient and support independence.** Later dataset descriptions partially clarify patient separation, but complete patient-/volume-disjoint MRI/CT partitions and labeled-support provenance remain undocumented. This constitutes an unresolved leakage risk, not demonstrated overlap. [L2 | Experimental Design | Inferred, Peer-review | partially_addressed]

- **Underspecified mixed-granularity supervision.** The treatment of image-label-only OCT17, ChestXray and HIS under mask-dependent Dice/focal objectives remains unspecified, leaving supervision and gradient flow unclear. MadCLIP distinguishes classification and segmentation losses, but does not resolve MVFA’s handling of missing masks during mixed-source or classification-only few-shot adaptation. [L3 | Methodology | Inferred, Peer-review | partially_addressed]

- **Unaccounted few-shot reference budgets.** The K-shot protocol does not establish normal/anomalous support composition, guaranteed normal references, or whether every memory-bank image counts toward K. Effective anomaly supervision and normal-reference coverage therefore remain unclear, particularly at small support budgets. [L4 | Experimental Design | Inferred, Peer-review | open]

- **Unverified baseline implementation parity.** Per-baseline checkpoints, optimization schedules, adaptation configurations and validation-search budgets remain insufficiently documented, especially for APRIL-GAN. Implementation and tuning differences consequently remain possible contributors to the reported gains. [L5 | Experimental Design | Peer-review | open]

- **Missing support- and seed-level uncertainty.** Reported point estimates do not quantify variability from training seeds and support selection, although supports influence both adapter fitting and memory-bank construction. RetExpert provides seed-based confidence intervals in fundus detection, but equivalent support-draw uncertainty remains unestablished for MVFA-style few-shot detection. [L6 | Results & Analysis | Inferred, Peer-review | partially_addressed]

- **Insufficient lesion-sensitive and operational validation.** ROC-AUC does not establish lesion overlap, boundary accuracy, lesion-wise misses, calibrated thresholds, or predictive values at deployment prevalence. BMAD’s PRO and narrower task-specific evaluations provide partial precedents, but comprehensive validation remains unshown for the supplied MVFA-style successors. [L7 | Results & Analysis | Inferred, Peer-review, Literature | partially_addressed]

- **Incomplete architecture and input specifications.** Written descriptions omit adapter widths, the exact CLIP checkpoint, dual-stream propagation and sufficiently explicit intensity, resizing and positional-embedding handling. LiverCT’s preprocessing variant is also unidentified; annotation-derived organ crops, if required, would necessitate an annotation-free localization step for raw-image deployment. [L8 | Methodology | Peer-review, Literature | open]

- **Unestablished complete-examination capability.** Selected MRI/CT slices and histology patches do not establish complete-volume or whole-slide detection, spatial consistency, or examination-level sensitivity and accumulated false alarms. Later peripheral-brain analyses partially probe coverage, but complete-examination aggregation remains unestablished. [L9 | Clinical & Deployment | Author-stated, Inferred, Peer-review, Literature | partially_addressed]

- **Unvalidated small-lesion spatial fidelity.** Resizing to 240 pixels and interpolating patch-grid predictions can discard subtle findings and boundary detail. Spatial-FAD mitigates these architectural constraints through dense embeddings and boundary refinement, but controlled resolution/upsampling ablations and lesion-size-stratified, boundary-specific validation remain missing from the supplied evidence. [L10 | Methodology | Inferred, Literature | partially_addressed]

- **Uncalibrated reference-dependent fusion.** Equal-weight fusion combines softmax scores and cosine distances on different scales, with distributions changing as normal-reference coverage changes. Calibration and text-only, memory-only and fused comparisons from identical few-shot checkpoints remain absent; ROC-AUC does not validate a likelihood interpretation. [L11 | Methodology | Inferred, Peer-review | open]

- **Unresolved acquisition and cohort dependence.** Later source-dependence mitigation does not establish scanner/site robustness or controlled sequence, CT-window and intensity robustness, particularly after few-shot adaptation. LiverCT additionally lacks same-source LiTS normal controls and documented support provenance, preventing separation of disease-detection gains from possible differences in target-cohort access. [L15 | Methodology | Peer-review, Literature | partially_addressed] [L16 | Experimental Design | Inferred | partially_addressed]

- **Unverified medical-language contribution.** Region-level descriptions and adaptive text modules reduce reliance on fixed generic prototypes, but matched text-free, nonsemantic-prototype and binary-head controls remain absent. Meaningful linguistic benefit beyond learned binary separation, and semantic adequacy for zero-shot regional localization without regional annotations, remain unverified. [L19 | Methodology | Inferred, Literature | partially_addressed]

- **Unestablished end-to-end efficiency.** Freezing CLIP does not eliminate backbone execution, dual-task multilevel processing or reference-bank storage and search costs. CLIPDxNet supplies deployment measurements and Spatial-FAD compresses references, but MVFA’s total/trainable parameters, training costs, latency, peak memory and support-/resolution-dependent scaling still lack matched measurement. [L25 | Clinical & Deployment | Inferred, Peer-review, Literature | partially_addressed]

### 4.2 Research problem (astra)

The methodological problem is to build a streaming, resolution-adaptive extension of MVFA for few-normal-shot detection and localization in complete FLAIR brain MRI and axial liver CT volumes. Assume access during adaptation to patient-indexed source volumes, image labels and lesion masks; inference receives a complete volume from a domain excluded from adaptation and K verified normal reference slices, but no target lesion annotations or parameter updates. The method must jointly learn budgeted native-resolution token refinement, cross-slice feature exchange and compressed reference comparison through multilevel adapters on frozen CLIP, together with volume-level anomaly aggregation. It must preserve small-lesion evidence and spatial consistency while controlling examination-level false positives and bounding peak accelerator memory as volumes and reference banks grow. Spatial-FAD addresses local resolution and storage through dense embeddings, boundary refinement and compact prototypes, while UniVAD establishes normal-reference-only deployment; neither establishes, in the supplied evidence, a method coupling these capabilities to cross-slice reasoning and budgeted examination-level aggregation. Success would require improved localization and examination-level detection against MVFA and Spatial-FAD under matched source-annotation, reference and memory budgets, using the original complete volumes underlying BrainMRI and LiverCT in leave-one-modality-out, patient-disjoint source/support/test protocols. Evaluation would report lesion-size-stratified Dice and PRO, 95th-percentile Hausdorff distance, lesion sensitivity at fixed false positives per volume, volume ROC-AUC, trainable/total parameters, training/inference time, and memory/storage scaling with volume length, resolution and reference count, with confidence intervals accounting for patients, support draws and seeds.

Addresses: L9, L10, L25

---

## 5. Evaluation

**Answer key:** MVFA-discussing papers that cite it were split in half. One half was used as evidence. The other 7 papers were never shown to the pipeline, and the critiques of MVFA in them form the answer key. That gave 3 critiques:
1. fixed hand-written text prompts;
2. noisy, imprecise segmentation masks;
3. an expensive memory bank.

The judge was Nemotron 3 Ultra, a different model from the ones that generated the lists.

| System | Model | # limitations | Known critiques caught (of 3) | Importance (1–5) |
|---|---|---|---|---|
| **New pipeline (main)** | MiniMax M3.1 Flash | 39 | **3** | **4.62** |
| New pipeline | gpt-6-astra | 27 | 2 | 3.74 |
| New pipeline without the Citation agent | gpt-6-astra | 25 | 1 | 3.72 |
| Old pipeline (re-run on the same model) | gpt-6-astra | 11 | 2 | 4.00 |
| Single prompt (zero-shot) | gpt-6-astra | 15 | 1 | 4.60 |

**Proof check (deterministic, no LLM):**
- MiniMax run: 259 of 259 quotes from the MVFA paper were found verbatim, and 32 of 32 quotes from other papers were verified.
- astra run: 118 of 118 quotes were verified.

**What the results show:**
- The new pipeline with the Citation agent caught at least as many known critiques as the old pipeline and the single prompt. The MiniMax run was the only system to catch all three.
- Removing the Citation agent lost one critique, so reading later papers helps.
- Unlike the old pipeline, every limitation from the new pipeline comes with checkable proof and a status in later work.

**Limits of this evaluation, stated honestly:**
- The answer key has only 3 items, so one critique more or less is a 33-point jump. The differences above are indicative, not proven.
- The judge gave every system 5/5 for faithfulness and soundness, so those scores are not informative and are left out of the table.
- Compared side by side, the old and new pipelines cover mostly the same topics. The new pipeline's main advantage is that its output is grounded, checkable and organised, not that it finds many new issues.
- The gap-status labels rarely discriminate: 35 of the 39 MiniMax limitations are marked "partially addressed".
- **Manual fact-check.** Every claim in the MiniMax gaps and problem was checked against the MVFA paper and the corpus. All quoted numbers are correct. A few statements were edited by hand:
  - **One factual error fixed.** The output said patient-level splits were documented "for HIS and RESC only". BMAD also documents patient-ID partitioning for BrainMRI.
  - **One chronology error fixed.** BMAD's validation-based checkpointing and five-seed repetition were described as later work. BMAD is the benchmark MVFA itself builds on.
  - **One duplicate removed.** TAFT was listed twice, once by name and once as "the text-adapted dual-path follow-up".
  - **Overstatements softened:**
    - "contradicts" the paper's use of abnormal samples (the paper uses them only for training);
    - "unsupported" became "not statistically supported";
    - "non-reproducible" and "unreproducible" were reworded or removed;
    - "never" and "still only" became "the later work we found".
