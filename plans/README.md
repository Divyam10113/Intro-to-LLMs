# Plans: Task 4 (codification) and the final paper

**Status (2026-11-25): both are done.**
- **Task 4:** closed after round 3 and a pre-registered Brain follow-up. See `../task4_share/SHARE_README.md`, and the verdict in `../task4_share/task4r3/report/VERDICT.md`.
- **Final paper:** `../paper/runs/final/latex_writeup/final_paper.pdf` (4+1, all checks pass). See `../paper/README.md` and `PAPER_PLAN.md` §12. The open human items are in `../paper/OPEN_ITEMS.md`.

| File | What it does | Writes to |
|---|---|---|
| `TASK4_PLAN.md` | **Tier 0, base (≈ 3.5 GPU h).** Start from MVFA's released checkpoints and briefly fine-tune only the segmentation adapters, adding a PU loss (witness-rate MIL → nnPU) on the three mask-less datasets. Compare with MVFA (R0) and with the same fine-tune without the PU term (control). One seed: suggestive only | `task4/` |
| `TASK4_PLAN_CREDIBLE.md` | **Tier 1, credible (≈ 10 GPU h, ≈ 12–13 with buffer).** The base plan plus 3 seeds for M-h and the control on all 3 targets, AUPRO, and pre-registered sign-test rules. The minimum for a workshop submission | `task4/` |
| `TASK4_PLAN_RECOMMENDED.md` | **Tier 2, recommended (≈ 14 GPU h, ≈ 17–18 with buffer).** The credible plan plus mask-free M (PU) and B0 (top-k MIL) on all targets, giving retention without any masks. **Use this one** for a reputable AI-generated-papers workshop | `task4/` |
| `TASK4_ROUND2_PLAN.md` | **Round 2, after the recommended tier's results.** Traces Task 3's top-5 → round 1 (idea #3, PU) → its feedback → round 2: codify Task 3's idea #1 (normal-bank-gated MIL) in Task 3's own protocol, plus Task 3's mask-budget points, against MVFA re-run under the same rule. Refines round 1's idea (#3, PU with a fixed π instead of the failed estimator) and codifies Task 3's #1 (gated MIL) side by side; 65 runs, 5 epochs, ≈ 2.5–3 h. Run by Claude Code with one simple loop and three teammate gates; ends with `VERDICT.md` (how good the results are, whether they can go to a workshop). Links round 1's decisions (D1–D6) at the end; kickoff prompt in the appendix | `task4r2/` |
| `TASK4_ROUND3_PLAN.md` | **Round 3, final and simple (≈ 3.4× round 2's compute).** Tier 1 confirms rounds 1–2's positive results (plain MIL keeps most of MVFA's localisation without masks; MIL beats MVFA on Brain; PU+100 % ≥ MVFA) by rerunning MVFA, MIL and PU+100 % at 20 epochs on fresh seeds. Tier 2 extends them with two new arms that need ~3 lines of code: MIL+100 % and MIL with 10 % / 25 % masks. 58 runs, ≈ 5½ h on 2 GPUs; frozen rules H1–H5; ends with `VERDICT.md` and `PAPER_CLAIMS.md` (paper built around what holds). The earlier full draft (mosaic MIL, 138 runs) is in `archive/TASK4_ROUND3_PLAN_v1_full.md` | `task4r3/` |
| `TASK5_STAGE_A_PLAN.md` | **Detailed steps A0–A7 for PAPER_PLAN v2's Phases 0–2** (status banner at the top says what changed): everything for the paper that needs no results and no GPU (decisions, hand-off agreement, citation pool, drafts of background, related work, formulation and method, PaperOrchestra fork and dummy pilot, check scripts). About 3–4 working days | `paper/` |
| `PAPER_PLAN.md` | **v2 (2026-11-22).** The final 4+1 paper (4 content pages + 1 references page) from the latest Task 1–4 outputs, with PaperOrchestra; every section verified. Task 4 round 1 is in hand, so the framing is an **analysis paper** (PU gave no consistent gain; π not estimable; plain MIL keeps 93–99 % without masks). §11 says how to swap in round 2 | `paper/` |
| `archive/TASK4_PLAN_v1_agentic.md` | The earlier full agentic plan (ladder R0–M-h, improvement loop), kept for reference only. **Too compute-heavy; not used** | — |

## Order of execution
```
TASK4 round 1   TASK4_PLAN_RECOMMENDED.md  ──►  results (done, 2026-11-22)  ──►  TASK4_ROUND2_PLAN.md  ──►  VERDICT.md
                                                      │                                                       │
PAPER_PLAN v2   Phases 0–2 (setup, pool, drafts, pilot) → 3–6 (round 1 results, framing, generation, checks)    │
                                                                         └── §11: swap in round 2 when it arrives ◄┘
```

- The paper can be written **now** from round 1 (an analysis paper); round 2 updates the Experiments section later.
- `Task5_inputs_v2.zip` (project root) has everything the paper agent needs, including round 1's outputs; `MANIFEST.md` inside lists it.
- The `TASK4_PLAN*.md` tiers share section and step numbers; round 1 ran the recommended tier.

## Sources (latest only)
The team repository `Divyam10113/Intro-to-LLMs`:
- `Vismay`: Task 1 v6;
- `task2`: Task 2;
- `task3_kevalina`: Task 3 run 3.

Do not use older local iterations or Drive copies.
