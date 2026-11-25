# Pipeline handoffs

Each stage reads a copy of the previous stage's output and keeps it inside its own folder. That way each task folder still runs on its own, as it did on its own branch. `pipeline/sync_handoffs.py` copies the files forward and records their sha256 in `pipeline/handoffs_manifest.json`:

```bash
make sync-dry-run   # which copies would change (nothing is written)
make sync           # copy every changed source forward byte-for-byte, rewrite the manifest
make check          # fail if any copy differs from its source or from the manifest
```

As of the restructure (2026-11-26), all nine copies were already byte-identical to their sources. `make sync` copied nothing and only wrote the manifest.

## The handoffs

| Name | Source | Copy (what the next stage reads) | Why |
|---|---|---|---|
| `task1.v6_to_handoff` | `task1_literature_review/runs/v6_2026-11-16/final_ref_set.json` | `task1_literature_review/task3_handoff/final_ref_set.json` | Task 1's export. The hand-off folder is the published interface |
| | `…/runs/v6_2026-11-16/intermediates/final_ref_set_excluded.json` | `…/task3_handoff/final_ref_set_excluded.json` | papers left out, with reasons |
| `task1_to_task3` | `task1_literature_review/task3_handoff/final_ref_set.json` | `task3_ideation/data/task1/final_ref_set.json` | Task 3 stage 1 (literature for ResearchAgent) and stage 2 (papers citing MVFA) |
| | `…/task3_handoff/final_ref_set_excluded.json` | `task3_ideation/data/task1/final_ref_set_excluded.json` | |
| | `…/task3_handoff/README.md` | `task3_ideation/data/task1/README_task1_output.md` | field descriptions |
| `task2_to_task3` | `task2_limitations/outputs/full_nim/final_limitations.json` | `task3_ideation/data/task2/final_limitations.json` | Task 3 stage 0 groups these 39 limitations. Also recorded in `task3_ideation/data/inputs_manifest.json` (left unchanged) |
| | `task2_limitations/deliverables/task2/task2_research_gaps.md` | `task3_ideation/data/task2/task2_research_gaps.md` | the human-readable deliverable |
| `task3_to_task4` | `task3_ideation/outputs/stage3_evolution_A1.json` | `task4_weak_supervision/plans/stage3_evolution_A1.json` | Problem A1 (training MVFA's segmentation adapters from image-level labels with a MIL loss) and its evolved ideas, which Task 4 tests |
| | `task3_ideation/outputs/stage4_ranking_A1.json` | `task4_weak_supervision/plans/stage4_ranking_A1.json` | A1's top 5 (normal-bank-gated MIL first). Task 4 round 1 tested a PU loss, round 2 gated MIL and refined PU, round 3 confirmed at 20 epochs and added mask budgets; the follow-up confirmed MIL+10 % masks on Brain |

From Task 3 onward, the handoff is an idea rather than a data file. Task 4 does not read these JSON files programmatically. Its plans (`plans/TASK4_PLAN_RECOMMENDED.md`, `plans/TASK4_ROUND2_PLAN.md`, `plans/TASK4_ROUND3_PLAN.md`) cite them, and the copies record exactly which version of A1 was tested.

## Task 5 (to be added)

Task 5, the paper, will be added later in `task5_paper/`. It is expected to read Task 4's:
- `round3/report/PAPER_CLAIMS.md`;
- `round3/report/paper_inputs/task4_r3/` (and round 1's `report/paper_inputs/`);
- the follow-up's `report/FU_REPORT.md` and `results/analysis_fu.md`.

If Task 5 keeps copies of these files, add them as a `task4_to_task5` entry in `HANDOFFS` in `pipeline/sync_handoffs.py` and run `make sync`; see "Adding a task" in [README.md](README.md#adding-a-task-eg-task-5). Until then `make task5` and `make task5-dry-run` only report that Task 5 is not added yet.

## Known mismatch: Task 2 used Task 1 run v4

Task 2 was built before Task 1's run v6 existed:
- **Task 2 (39 limitations)** read Task 1's **run v4** review and its reference list. Task 2's corpus `task2_limitations/data/papers.json` has 223 papers:
  - 125 citing MVFA;
  - 62 cited by MVFA;
  - 59 from Task 1.
- **Task 3** uses Task 1's **run v6** (594 candidate papers), and Task 2's limitations as they were.

**About 176 of Task 2's 223 papers are also among v6's 594.** The exact figure depends on how papers are matched:
- 176 is the figure from the parent session's analysis;
- re-checking during the restructure gave 178 when matching on DOI or normalised title, and 164 on DOI alone.

So roughly a fifth of the papers behind Task 2's limitations are not in the literature that Task 3 later used.

**This was left as it is on purpose.** Re-running Task 2 on v6 would need paid NIM/AgentRouter calls. It would also produce a different set of limitations from the one Task 3 actually used: the 39 in `data/task2/`, recorded in `task3_ideation/data/inputs_manifest.json` (`"built_on": "Task 1 run v4"`). The results stay consistent with each other as long as Task 3 is read as "built on Task 2 as delivered".

## How the stages are run without keys or GPUs

`pipeline/smoke.py` (`make smoke`, `make taskN-dry-run`) copies each stage to a scratch folder and runs it there under these conditions:
- every API-key variable is removed;
- `LLM_BACKEND=file` and `QUAL_SG_BACKEND=file` are set: when an answer is not cached, this backend writes the prompt to `logs/pending/` and stops, and it never calls a model;
- the HTTP(S) proxy points to the dead address `127.0.0.1:9`;
- `CUDA_VISIBLE_DEVICES` is empty.

The run fails if `git status` changes. What each stage covers, and what it cannot, is in [README.md](README.md#running-each-stage).
