# Independent review (PAPER_PLAN §7 check 7), 2026-11-25

**Reviewer:** a different model (Claude Sonnet) from the writer (Claude Opus), run as a separate agent. It read the manuscript against `results.json`, the round 1–3 and follow-up reports, the frozen rules and the training code. It edited nothing.

**Reviewer's overall verdict:**
- **Numbers:** they match `results.json` and the reports, including Table 1, Table 2 and the p-values.
- **Risks:** they were interpretive. The main ones were consistency across rounds, the wording "top-ranked", and how far the PU "failure" extends.
- **Venue:** acceptable for a student or workshop venue after fixing findings 1–4.

## Findings and what was done

| # | Severity | Finding | Fix in the manuscript |
|---|---|---|---|
| 1 | major | "Held three times" / "consistently across rounds" mixes baselines (round 1 = one seed against R0; round 2 RESC 0.716; round 3 against R0 = 0.916, RESC 0.826) | Abstract: "two earlier, weaker tests point the same way". Contribution 1 and §5.1 now say that only round 3 used a frozen rule, and report 0.716 and the R0-relative 0.916 (RESC 0.826) |
| 2 | major | "Top-ranked" contradicts "ranks 1 and 3" | Changed to "the two refinements we implemented (ranks 1 and 3 of five)" in the abstract, contributions and conclusion |
| 3 | major | The PU "failed" claim is broader than the evidence (π fixed at 0.01, single-seed π = 0.05 run) | §5.3 now states π = 0.01, the RESC −13.83, the single-seed π = 0.05 run, and "we did not tune π, so this tests our implementation, not PU learning in general". π is also given in §3 and Table 2 |
| 4 | major/minor | "Falsified" is strong for a 3-seed test | Kept the frozen rule's own label (the caption now says labels follow the rules). Table 2's caption and §5.3 state 3 seeds and 5 epochs |
| 5 | minor | The in-mask bullet mixes rounds | Round 3 values are marked "(round 3, source validation images)"; the brain-fold values are marked "the follow-up's brain fold" |
| 6 | minor | "Long enough for model selection" is unsupported | Replaced: 18/58 runs chose one of the last two epochs, so longer training could still change the results |
| 7 | minor | Seeds 6–7 are never mentioned | §4.2 and Table 2's caption name seeds 3–7 (round 3 brain test) and 8–12 (follow-up) |
| 8 | minor | Eq. 2 vs the code at b = 1; ℓ, π and p undefined | Text states how normal images are handled at b < 1 and at b = 1 (matches `build_epoch`); p = σ(s₁ − s₀); ℓ = logistic loss; π = 0.01 |
| 9 | minor | Retention definition is ambiguous; no uncertainty given | Text says ρ is computed per target from seed means and then averaged, has no confidence interval, and ρ > 1 means the arm exceeds MVFA |
| 10 | minor | Holm-adjusted p is not shown; pooled pairs | Table 2's caption gives the Holm-adjusted values (1.0, 1.0, 0.5078); §4.2 adds that pooling targets makes the sign test coarse |
| 11 | minor | "RESC instability is gone" | Now "smaller than in round 2 (pixel SD 3.71 vs 9.90) but still the largest in Table 1" |
| 12 | minor | The heat-map statement reads as a finding | Now "in the four fixed examples … its maps look more diffuse" |
| 13 | minor | "Three of its six source datasets" | Now "pixel masks, which BMAD provides for only three of its six datasets" |
| 14 | minor | Image-AUROC numbers have no table; causal wording | New Table 3 (round 3 image AUROC, generated from `results.json`); "cannot separate source-only selection from the schedule" |
| 15 | minor | Brain 10 % result: post-hoc origin and its scope | Contribution 2 says "on brain MRI" and adds "trails on liver CT; no image-AUROC gain". §5.2 says the result was "one of three budgets on three targets", that it is one target and one test, and that MVFA's image AUROC is noisy (63.29 ± 9.80) |
| 16 | nit | R0 is shown beside re-trained arms | Table 1's caption says R0 enters no retention or test |
| 17 | nit | The published-number comparison sits next to the headline; "every arm" is ambiguous | The published number is moved to the end of the paragraph, with an explicit no-claim; "every re-trained arm" |
| 18 | nit | Audit claims are weakly sourced; the duplicates' effect is not stated | "MVFA's released training script selects checkpoints on the target test set"; the duplicates were removed from source validation, so target scores are unaffected |

**After the fixes:**
- **Checks:** every check passes (`check_all.sh`): 554 reconciled values; numbers; citations (21); claim audit 55/55, with new checks for the review's numbers and Table 3; 4+1 pages. The heat-map figure was narrowed to 85 % width to keep the content on 4 pages.
- **Not re-reviewed:** the revised text has not been sent back to the reviewer for a second pass.
