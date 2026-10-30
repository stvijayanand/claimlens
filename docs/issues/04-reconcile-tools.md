# Issue 4: Reconcile tools — normalize, match, labor check, record_flag

- **Spec:** `docs/implementation-spec.md` §8.1, §11 Phase 4 (tools part)
- **Depends on:** #1
- **Parallel with:** #2, #3, #6

## Goal

The four deterministic agent tools, fully unit-tested, with the validating `record_flag` sink enforcing invariant I2.

## Tasks

- [ ] `tools.py`: `normalize_part_name()` (exact → longest-substring alias lookup, side mapping, `unmatched` status, side-missing confidence penalty)
- [ ] `tools.py`: `match_damage()` (0.5 part + 0.3 side + 0.2 confidence scoring, `basis` string, `MATCH_MIN_SCORE` threshold, writes `matches`)
- [ ] `tools.py`: `check_labor_range()` (`labor_ranges` lookup, `in_range|high|low|unknown_part`, never guesses; gated by `FEATURE_LABOR_CHECK`)
- [ ] `tools.py`: `record_flag()` (Pydantic validate + evidence refs exist in DB; writes `flags` only on accept; structured rejection otherwise)
- [ ] JSON tool definitions (name, description, parameters) for all four, importable by the agent loop
- [ ] Every invocation logged to `tool_calls` (I3)
- [ ] Seed `hidden=1` taxonomy rows for behind-bumper internals
- [ ] `tests/test_tools.py` + `tests/test_record_flag.py` (accept/reject matrix: off-schema, no evidence, dangling refs)

## Acceptance criteria

- `pytest tests/test_tools.py tests/test_record_flag.py` green with no API calls
- `record_flag` rejects: schema violation, zero evidence refs, unknown crop/line ids — and accepts a well-formed flag
- Idempotency: repeating any tool call with the same input yields the same output

## Out of scope

- The agent loop itself (#5), vision/extraction internals, UI
