# Issue 5: Agent loop + report assembly + golden test

- **Spec:** `docs/implementation-spec.md` §8.2, §9 (`report.py` part), §11 Phase 4 (agent part), §12 golden test
- **Depends on:** #2, #3, #4
- **Blocks:** #7, #9

## Goal

The Responses-API reconcile loop working part-by-part with parallel tool calls, report assembly, and the golden end-to-end test.

## Tasks

- [ ] `agent.py`: loop per §8.2 pseudocode (normalize all → per-part match → ordered flag decisions 1–5 → `record_flag`; `MAX_AGENT_ROUNDS=3`; stop on no-new-proposals)
- [ ] Hidden-part lines → "needs explanation" notes, never flags; feature-flag gating for labor (step 5) and repair-vs-replace (step 4)
- [ ] Parallel batching for independent tool calls; high reasoning effort; transcript + tool log persisted per run
- [ ] `report.py`: `build_report()` (flags by confidence desc, per-type shop-question templates, notes, counts, run stats) + `Report` JSON shape per §9
- [ ] `tests/test_pipeline_golden.py`: one synthetic car with all five injected error types → asserts the five expected flags and no extras (recorded/cached API)

## Acceptance criteria

- Golden test green: exactly the five expected flags on the fixture car
- Re-run of a logged run shows identical flags (determinism check)
- `tool_calls` fully audits the golden run; round cap respected (assert max rounds in test)

## Out of scope

- Eval harness (#6, #7), serving/UI (#9), tuning (#8)
