import csv,json,math,shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path('paper')
DATA=ROOT/'inputs'
RAW=DATA/'raw_materials'
FIG=RAW/'figures'
FIG.mkdir(parents=True,exist_ok=True)
results=json.loads((DATA/'results.json').read_text(encoding='utf-8'))
round1=results['rounds']['r1']; protocol=results['protocol']
# Keep every number used by the generation model sourced from results.json.
def stat(s):
 return f"{s['mean']:.2f}"+(f" ± {s['std']:.2f}" if s['std'] is not None else " (single seed)")
def val(x): return f"{x:.2f}"

# Build a concise, numeric-source-controlled experiment log.
lines=["# Experimental log — Round 1 (all result values from `results.json`)","",
"## Framing and protocol","",
"Pre-registered rule §5.1 selects an **analysis paper (F-C)**. Neither the hybrid PU fine-tune nor the fully mask-free PU model meets the registered rule. B0 is an informative, unregistered observation. Do not describe these results as an improvement over MVFA.","",
f"The study uses the {protocol['benchmark']} leave-one-modality-out benchmark with {protocol['dataset_count']} datasets. Targets are {', '.join(protocol['targets'])}; the source pool includes {protocol['masked_source_count']} datasets with pixel masks and {protocol['maskless_source_count']} without them. Each source is subsampled to at most {protocol['max_train_images_per_class_per_source']} normal and {protocol['max_train_images_per_class_per_source']} abnormal training images, with a held-out {protocol['pi_holdout_fraction']:.0%} subset for prior estimation. The backbone is frozen {protocol['frozen_backbone']} with {protocol['adapter_levels']} adapter levels. Main runs use {round1['configs']['mh']['Liver']['steps'][0]} updates and seeds {', '.join(map(str,round1['configs']['mh']['Liver']['seeds']))}. Pixel metrics use AUPRO integrated through FPR {protocol['aUpro_false_positive_limit']:.2f}.",
"Model selection in the released R0 checkpoints used the target test set; these fine-tunes use their last training step and do not evaluate the target during training.","",
"## Table 1 source — M-h versus the matched control","",
"Values are mean ± sample standard deviation across seeds; R0 is shown separately as published and reproduced.","",
"| Target | Model | Image AUROC | Pixel AUROC | AUPRO |","|---|---|---:|---:|---:|"]
for target in ['Liver','Brain','Retina_RESC']:
 pub=round1['r0_published'][target]; r0=round1['r0_reproduced'][target]
 lines.append(f"| {target} | R0 (published) | {val(pub['img_auroc'])} | {val(pub['pix_auroc'])} | — |")
 lines.append(f"| {target} | R0 (reproduced) | {val(r0['img_auroc'])} | {val(r0['pix_auroc'])} | {val(r0['aupro'])} |")
 for cfg,label in [('mh','M-h'),('mh_ctrl','M-h-ctrl')]:
  row=round1['configs'][cfg][target]
  lines.append(f"| {target} | {label} | {stat(row['img_auroc'])} | {stat(row['pix_auroc'])} | {stat(row['aupro'])} |")
pair=round1['paired']['mh-mh_ctrl']
lines += ["",f"Paired M-h minus M-h-ctrl AUPRO was positive in {pair['positive']['aupro']} of {pair['n_pairs']} matched runs (one-sided sign-test p = {pair['sign_test_p_aupro']:.4f}). Mean paired AUPRO differences were {pair['per_target_mean']['Brain']['aupro']:+.2f} on Brain, {pair['per_target_mean']['Liver']['aupro']:+.2f} on Liver, and {pair['per_target_mean']['Retina_RESC']['aupro']:+.2f} on RESC. The pre-registered criterion is not met. The Brain result is target-specific: AUPRO improved in all {sum(1 for x in pair['per_pair'] if x['target']=='Brain' and x['aupro']>0)} paired Brain seeds; RESC is mixed and Liver is near zero.","",
"## Table 2 source — fully mask-free models","",
"M and B0 reinitialize both adapter sets; M uses fixed π = 0.30 and B0 uses top-k MIL. Values outside Liver are single-seed observations.","",
"| Target | Model | Image AUROC | Pixel AUROC | AUPRO | Pixel retention vs R0 | AUPRO/R0 |","|---|---|---:|---:|---:|---:|---:|"]
for target in ['Liver','Brain','Retina_RESC']:
 for cfg,label in [('m','M'),('b0','B0')]:
  row=round1['configs'][cfg][target]; ret=round1['retention'][cfg][target]
  n=row['img_auroc']['n']; seednote='' if n>1 else ' (single seed)'
  lines.append(f"| {target} | {label}{seednote} | {stat(row['img_auroc'])} | {stat(row['pix_auroc'])} | {stat(row['aupro'])} | {ret['seed0']:.3f} | {ret['aupro_ratio_seed0']:.3f} |")
lines += ["",f"M's mean seed-0 pixel-retention across targets is {round1['retention']['m']['_mean_seed0_over_targets']:.3f}; B0's is {round1['retention']['b0']['_mean_seed0_over_targets']:.3f}. M exceeds B0 in pixel AUROC and AUPRO on {sum(1 for t in ['Liver','Brain','Retina_RESC'] if round1['configs']['m'][t]['pix_auroc']['mean']>round1['configs']['b0'][t]['pix_auroc']['mean'] and round1['configs']['m'][t]['aupro']['mean']>round1['configs']['b0'][t]['aupro']['mean'])} of {protocol['masked_source_count']} targets. This evidence does not meet the label-efficiency framing rule. B0's 0.93–0.99 retention range is an unregistered single-seed observation.","",
"## Why π estimation failed","",
"The corrected lower-tail estimator was evaluated against known patch fractions on masked source data. True fractions in `results.json` range from 0.008 to 0.119. At the deepest segmentation level, the corresponding estimates range from -0.286 to 1.000 (reported raw, before clipping), far from the true rates in several cases. This violates the irreducibility assumption needed for mixture-proportion estimation on these MVFA patch-score distributions. M-h therefore used the mean true patch fraction from masked sources (D1), while mask-free M used the pre-registered fixed π = 0.30 (D2); these are different information regimes.","",
"## Secondary sensitivity runs","",
"Sensitivity runs are kept separate and are not included in means or pre-registered decisions. Their π overrides were selected after inspecting mask statistics. For example, M on Liver with π = 0.05 produced pixel AUROC {next(x['pix_auroc'] for x in round1['secondary'] if x['run_id']=='m_Liver_s0_pi0.05'):.2f} and AUPRO {next(x['aupro'] for x in round1['secondary'] if x['run_id']=='m_Liver_s0_pi0.05'):.2f}; this is post-hoc evidence only.","",
"## Limitations and disclosure","",
"The study is zero-shot and uses a subsampled training pool. M-h and its control have three seeds; M and B0 have a single seed outside Liver, and the Liver pixel metric for M has high seed variability. R0 inherits target-test checkpoint selection bias. RESC is evaluated while OCT2017 is in the source pool, so they share a retinal OCT modality. Pixel AUROC on Liver is dominated by the large normal background. The π overrides in secondary runs are post-hoc. Round 2 is absent and must remain a separate experiment if added later.",
"The work used an AI-augmented pipeline across literature review, limitation extraction, ideation, code implementation, and manuscript drafting; human verification is recorded in the supplied Task 4 transcript and the paper verification log."]
(RAW/'experiment_log.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Idea specification: method and exact section/page constraints.
idea=f'''# Paper idea and constraints (Round 1)\n\n## Working title and framing\nHow Far Do Image-Level Labels Go for Localisation in Medical Anomaly Detection?\n\nFrame this as an honest **analysis paper** under PAPER_PLAN.md §5.1 (F-C). The study found no consistent PU gain over plain fine-tuning, failure of the MVFA-score prior estimator, and a strong but unregistered top-k MIL observation. Never claim improvement over MVFA.\n\n## Problem\nMVFA's segmentation adapters learn from pixel masks on only the masked BMAD sources; the three mask-less sources provide image-level labels but do not train localization. Study patch-level image-label supervision in leave-one-modality-out evaluation on Brain MRI, Liver CT and RESC.\n\n## Round 1 method as run\n- M-h warm-starts the segmentation adapters from R0, retaining MVFA's Dice + Focal loss on masked sources and adding λ times non-negative PU risk on mask-less sources. M-h's π uses the mean true anomalous-patch fraction from masked sources (decision D1); this is not mask-free prior estimation.
- M-h-ctrl performs the matched mask-loss fine-tune without the PU iterations.
- Fully mask-free M retrains both adapter sets from scratch using image labels only, with fixed π = 0.30 (decision D2).
- B0 retrains both adapter sets with top-k MIL and no pixel masks.
- The prior estimator was corrected to a lower-tail form, but it still fails on MVFA's real patch scores; use its failure evidence and name the irreducibility assumption.
- Last-step models were evaluated; target tests were not used for these fine-tunes. R0's released checkpoints were selected using target test performance.
\n## Objective\nFor normal-image patches N sampled from p₀ and abnormal-image patches U sampled from πp₁+(1−π)p₀, with z=s₁−s₀ and sigmoid(z) the patch anomaly probability, use\n\n`R_nnPU = (1−π) mean_N ℓ(z,0) + max(0, mean_U ℓ(z,1) − (1−π) mean_N ℓ(z,1))`.\n\nOn masked sources M-h also uses MVFA's Dice + Focal loss; the hybrid control omits the PU branch. M uses the PU risk without masks. B0 uses top-k MIL with k=max(3,1% of patches).\n\n## Contributions supported by Round 1\n1. A controlled PU-vs-fine-tuning comparison: no consistent gain under the registered rule, with a target-specific Brain signal.
2. A failure analysis showing the prior estimator is unreliable on MVFA patch scores relative to known masked-source fractions.
3. A mask-free top-k MIL observation that retains much of R0 localization in these runs; label it unregistered and single-seed on two targets.
4. An implementation audit of the released MVFA training path, especially target-test checkpoint selection and which branches receive masks.
\n## Forced structure and 4+1 page budget\n1 Introduction (background/problem + contributions); 2 Related Work (2.1 medical anomaly detection and vision-language adaptation; 2.2 weakly supervised and mask-free anomaly localization; 2.3 positive-unlabelled learning and mixture-proportion estimation); 3 Problem Formulation; 4 Methodology (4.1 Dataset; 4.2 Approach); 5 Experiments (setup, results, discussion, limitations); 6 Conclusion. Abstract is written last. No appendix.\n\nBudgets: title/authors/abstract 0.30 page; Introduction 0.55; Related Work 0.50; Problem Formulation 0.40; Methodology including Figure 1 0.60; Experiments with two tables and Figure 2 1.45; Conclusion 0.20. All content through Conclusion ends by page 4; page 5 is references only. Cut prose rather than shrinking fonts or margins.\n\nOnly numeric source: `results.json`. Only citations: keys in `refs.bib` / `citation_pool.json`. Never pool Round 1 and Round 2.\n'''
(RAW/'idea.md').write_text(idea,encoding='utf-8')
# Related-work outline limited to our citations.
pool=json.loads((DATA/'citation_pool.json').read_text(encoding='utf-8'))
sections={'2.1':'Medical anomaly detection and vision-language adaptation','2.2':'Weakly supervised and mask-free anomaly localisation','2.3':'Positive-unlabelled learning and mixture-proportion estimation'}
lines=['# Related Work outline — curated Task 1 pool and approved extra refs','']
for sid,title in sections.items():
 lines += [f'## {sid} {title}','']
 for p in pool:
  if p.get('section')==sid:
   lines.append(f"- `\\cite{{{p['citation_key']}}}` — {p['title']}: {p.get('inclusion_reason','')}")
 lines.append('')
lines.append('Use only these items. State that metrics and numeric results from other papers are not directly compared because the protocols differ.')
(RAW/'related_work_outline.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copy2(DATA/'citation_pool.json',RAW/'citation_pool.json')
shutil.copy2(DATA/'refs.bib',RAW/'refs.bib')
# Figure 2 is the preselected, shared Liver heatmap comparison from Task 4.
shutil.copy2(DATA/'task4_r1/report/paper_inputs/liver_heatmaps.pdf',FIG/'fig2_liver_heatmaps.pdf')
# Figure 1: faithful diagram of the actual hybrid and mask-free training paths.
fig,ax=plt.subplots(figsize=(9.2,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis('off')
def box(x,y,w,h,text,fc='#f4f6f8',ec='#34495e',fs=9.5):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',facecolor=fc,edgecolor=ec,linewidth=1.2)
 ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True)
def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color='#44546a',linewidth=1.25))
box(.15,3.55,1.45,.85,'BMAD sources\nimage labels; some\npixel masks',fc='#eaf2f8')
box(2.0,3.55,1.45,.85,'Frozen CLIP\nViT-L/14',fc='#e8f6f3')
arrow(1.6,3.98,2.0,3.98)
box(4.0,3.55,2.15,.85,'R0 warm start\nsegmentation adapters\n(M-h / control)',fc='#fcf3cf')
arrow(3.45,3.98,4.0,3.98)
box(6.8,4.02,2.65,.75,'Masked sources\nDice + Focal',fc='#fdebd0')
box(6.8,2.95,2.65,.75,'Mask-less sources\nλ · nnPU, π from masks (D1)',fc='#f5eef8')
arrow(6.15,4.0,6.8,4.4); arrow(6.15,3.92,6.8,3.33)
box(.15,1.15,1.45,.85,'All source images\nimage labels only',fc='#eaf2f8')
box(2.0,1.15,1.45,.85,'Frozen CLIP\nViT-L/14',fc='#e8f6f3')
arrow(1.6,1.58,2.0,1.58)
box(4.0,1.15,2.15,.85,'Fresh detection +\nsegmentation adapters\n(M / B0)',fc='#fcf3cf')
arrow(3.45,1.58,4.0,1.58)
box(6.8,1.75,2.65,.7,'M: nnPU, fixed π = 0.30',fc='#f5eef8')
box(6.8,.75,2.65,.7,'B0: top-k MIL',fc='#e8daef')
arrow(6.15,1.65,6.8,2.05); arrow(6.15,1.52,6.8,1.10)
ax.set_title('Training paths in Task 4 Round 1',fontsize=11,pad=8)
fig.tight_layout(); fig.savefig(FIG/'fig1_method.png',dpi=220,bbox_inches='tight'); fig.savefig(FIG/'fig1_method.pdf',bbox_inches='tight'); plt.close(fig)
print('Wrote raw materials and figures to',RAW)
