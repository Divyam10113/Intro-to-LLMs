# Related Work outline (curated Task 1 pool + approved extra refs only)

## 2.1 Medical anomaly detection and vision-language adaptation

- `\cite{huang2024mvfa}` Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images: MVFA is the study’s base model and benchmark protocol.
- `\cite{bmad2024}` BMAD: Benchmarks for Medical Anomaly Detection: BMAD defines the six-dataset medical anomaly benchmark.
- `\cite{clip2021}` Learning Transferable Visual Models From Natural Language Supervision: CLIP is the frozen vision-language backbone.
- `\cite{winclip2023}` WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation: Representative zero/few-shot CLIP anomaly detection and localization.
- `\cite{aprilgan2023}` APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD: CLIP-based anomaly detection/localization baseline family.
- `\cite{anomalyclip2023}` AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection: Representative object-agnostic CLIP adaptation for anomaly maps.
- `\cite{adaclip2024}` AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection: Representative prompt adaptation for zero-shot anomaly detection.
- `\cite{vcpclip2024}` VCP-CLIP: A Visual Context Prompting Model for Zero-Shot Anomaly Segmentation: Representative CLIP-based zero-shot anomaly segmentation.
- `\cite{iqeclip2026}` IQE-CLIP: Instance-Aware Query Embedding for Zero-/Few-Shot Anomaly Detection in Medical Domain: Recent medical-domain zero/few-shot anomaly detection.
- `\cite{madpot2026}` MADPOT: Medical Anomaly Detection with CLIP Adaptation and Partial Optimal Transport: Recent medical-domain CLIP adaptation.
- `\cite{mediclip2024}` MediCLIP: Adapting CLIP for Few-shot Medical Image Anomaly Detection: Recent few-shot medical anomaly detection with CLIP.
- `\cite{medicalad_survey2021}` Deep Learning for Medical Anomaly Detection – A Survey: Provides the medical anomaly-detection background taxonomy.

## 2.2 Weakly supervised and mask-free anomaly localisation

- `\cite{mvtec2019}` MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection: Introduces the anomaly-segmentation benchmark underlying the per-region metric lineage.
- `\cite{adfa2023}` ADFA: Attention-augmented Differentiable top-k Feature Adaptation for Unsupervised Medical Anomaly Detection: Related patch-selection/top-k anomaly detection.
- `\cite{imagelevel2026}` Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision: Directly relevant image-level supervision for anomaly localization.
- `\cite{cycle_seg2021}` Unsupervised Anomaly Segmentation using Image-Semantic Cycle Translation: Context for localization without dense anomaly labels.
- `\cite{ilse2018attentionmil}` Attention-based Deep Multiple Instance Learning: Multiple-instance learning background for image-level labels.
- `\cite{patchcore2021}` Towards Total Recall in Industrial Anomaly Detection: Memory bank of normal patch features; the basis of the normal-bank gate in Task 3's idea #1 (added 2026-11-25, Task 1 relevance 3).

## 2.3 Positive-unlabelled learning and mixture-proportion estimation

- `\cite{elkan2008pu}` Learning Classifiers from Only Positive and Unlabeled Data: Foundational positive-unlabelled learning background.
- `\cite{kiryo2017nnpu}` Positive-Unlabeled Learning with Non-Negative Risk Estimator: Source of the non-negative PU risk used in M-h and M.
- `\cite{ramaswamy2016mpe}` Mixture Proportion Estimation via Kernel Embedding of Distributions: Mixture-proportion estimation assumptions and identifiability.

