# Issue 10: Final eval + README results + demo rehearsal

- **Spec:** `docs/implementation-spec.md` §10–§11 Phase 7 (ship part)
- **Depends on:** #8, #9

## Goal

Complete evidence set, honest numbers in the README, and a rehearsed demo.

## Tasks

- [ ] Complete the real set to ~20 cars in the §10.1 layout; run full eval (`--split all`, `--cold`) and archive `eval_report.json`
- [ ] README: results table (all seven metrics vs targets, real/synthetic split), what was cut (if anything), how to run, one annotated sample report
- [ ] Demo rehearsal: fresh car upload→report live, plus cache-replay fallback; note wall time and cost
- [ ] Final `pytest` full-suite green; remove scratch/debug artifacts

## Acceptance criteria

- README reports real measured numbers (targets met or missed, either way) with the eval date and split details
- Demo path rehearsed twice (live + replay) without errors
- Full test suite green on the final tree

## Out of scope

- Nothing further — this is the last issue
