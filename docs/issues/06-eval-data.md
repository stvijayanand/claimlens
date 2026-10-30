# Issue 6: Eval data — datasets, synthetic generation, error seeding

- **Spec:** `docs/implementation-spec.md` §10.1–§10.2, §11 Phase 5 (data part)
- **Depends on:** #1 (models, `meta_client`)
- **Parallel with:** #2, #3, #4

## Goal

Eval-ready data layout with real cars, labeled synthetic cars, and deterministic seeded-error estimates.

## Tasks

- [ ] `eval/datasets.py`: loader for `data/eval/real/<car>/` and `data/eval/synthetic/<car>/`; hard assertion that `synthetic:true` files can never load through the real-evidence path (I4)
- [ ] `eval/generate_synthetic.py`: Muse Image edit CLI (`<clean_photo> --damage --at --out`); writes photo + `ground_truth.json` + matching `estimate_correct.json`; add a small `edit_image()` helper to `meta_client.py` if no suitable call exists (owned by this issue)
- [ ] `eval/seed_errors.py`: `add_phantom`, `drop_line`, `swap_repair_replace`, `duplicate_line`, `inflate_hours` (×2.5); each writes `injected_errors.json` entries
- [ ] `tests/test_seed_errors.py`: all five transforms on fixture estimates, incl. record correctness
- [ ] Choose the public real-photo dataset (labeled, redistributable crops); stage ≥5 real cars + ≥5 synthetic cars in the §10.1 layout (full 20-car real set completes in #10)

## Acceptance criteria

- Loader test proves a synthetic file is rejected from the real path
- Every staged car has `estimate_correct.json` + `estimate_seeded.json` + `injected_errors.json`; all five error types covered across cars
- `pytest tests/test_seed_errors.py` green

## Out of scope

- The eval runner/metrics (#7), pipeline changes, UI
