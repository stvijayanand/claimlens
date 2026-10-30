# Issue 7: Eval runner + metrics

- **Spec:** `docs/implementation-spec.md` §10.3, §11 Phase 5 (runner part)
- **Depends on:** #5 (pipeline), #6 (data)

## Goal

One command scores the full pipeline against seeded errors, real and synthetic separately.

## Tasks

- [ ] `eval/run_eval.py --split real|synthetic|all [--cold]`: per-car full pipeline run, flag-vs-`injected_errors.json` comparison, `eval_report.json` output + printed metrics table
- [ ] Metric implementations per §10.3: recall, precision, extraction accuracy, synthetic part detection (IoU≥0.5), schema pass rate, p50/p95 time, mean cost
- [ ] Real/synthetic splits always scored and printed separately
- [ ] Eval smoke: 2-car synthetic run completes and emits the table

## Acceptance criteria

- `run_eval.py --split synthetic` on the staged cars finishes and prints all seven metrics with the §10.3 targets alongside
- `--cold` demonstrably bypasses `model_cache` (cache-miss count in output)
- A deliberately broken pipeline change moves the metrics (sanity: the harness can detect regression — verify once, then revert)

## Out of scope

- Tuning fixes (#8), UI (#9), README writeup (#10)
