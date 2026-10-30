# Issue 8: Harden — failure fix, effort tuning, caching, thresholds

- **Spec:** `docs/implementation-spec.md` §11 Phase 6, §13
- **Depends on:** #7

## Goal

Close the worst eval gap and land time/cost within budget (or invoke the cut order deliberately).

## Tasks

- [ ] Failure analysis on the eval report: pick the single worst failure mode, fix it, re-run eval to show movement
- [ ] Per-step reasoning-effort tuning (low for extraction/listing, high for reconcile); prompt caching on for the taxonomy block
- [ ] Threshold review from eval data: `OVERLAP_MIN_INTERSECT_PX`, `MATCH_MIN_SCORE`, `MIN_PART_CONFIDENCE`, `MIN_ASSESS_CONFIDENCE` — change only with measured justification
- [ ] Verify p50 time <2 min and mean cost <$0.50/car on `--cold` runs; if missed, invoke the cut order (#14: labor flags, then repair-vs-replace) via feature flags and re-run

## Acceptance criteria

- Before/after eval numbers recorded (in the issue or `eval_report` history) showing the worst mode improved without regressing other metrics
- Budget verdict documented: within budget, or cut order invoked with disabled flags noted
- All previously green tests still green

## Out of scope

- New features or flag types, UI changes (#9), README (#10)
