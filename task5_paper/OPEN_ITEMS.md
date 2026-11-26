# Open items for the final human review (Gate 1), updated 2026-11-25

- [ ] **Authors and affiliation**: replace "Anonymous authors / Anonymous affiliation" in `runs/final/latex_writeup/final_paper.tex` before any submission.
- [x] **Independent review by a different model**: done (Claude Sonnet); 18 findings, all addressed (`logs/INDEPENDENT_REVIEW.md`). A second pass on the revised text was not run.
- [ ] **AI-use statement**: removed from the paper at the human's request (DECISIONS F-8). Before any submission beyond the course, check the venue's LLM-disclosure policy; the record is in `logs/`.
- [ ] **PaperOrchestra re-run (optional)**: final raw materials are in `inputs/raw_materials/`; it needs a new Gemini key (revoke the one shared earlier). Not required for the course (DECISIONS F-1).
- [ ] **Reference formatting**: Task 1's pool has no author names, so most BibTeX entries show title, venue, year and DOI only. Add authors if the venue requires them.
- [ ] **Workshop target**: none chosen. Once a call is chosen, check page limit, template and anonymity rules against it.
- [ ] **Trained adapters**: on the GPU server only (`/mnt/HDD1/divyam/task4/{r2,r3,fu}/runs/`); archive them if the paper is submitted.
- [x] Task 4 rounds 2-3 and the follow-up are in the paper (replaces the round-1-only item).
- [x] 4+1 format, numbers, citations and claim audit pass (`check_all.sh`).
