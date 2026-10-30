# Issue 2: Vision pipeline — listing, SAM segmentation, overlap, crops

- **Spec:** `docs/implementation-spec.md` §6, §11 Phase 2
- **Depends on:** #1
- **Parallel with:** #3, #4, #6

## Goal

Photos in, damaged-part crops + masks in DuckDB.

## Tasks

- [ ] `vision.py`: `list_visible_parts()` (Spark, low effort, taxonomy prompt attached, `MIN_PART_CONFIDENCE` filter)
- [ ] `vision.py`: `segment_photo()` (one part-phrase batch + one `DAMAGE_PHRASES` batch per photo via `segment()`)
- [ ] `vision.py`: `compute_overlaps()` + pure `is_damaged()` per `OVERLAP_MIN_INTERSECT_PX`
- [ ] `vision.py`: `make_crops()` (`CROP_PADDING_FRAC`, clamped); parts without damage get no crop/assessment
- [ ] `tests/test_overlap.py`: touching / identical / disjoint / empty mask cases

## Acceptance criteria

- One real car's photos produce inspectable `masks`, `visible_parts`, and `crops` rows in DuckDB
- SAM call count per car ≤ photos × 2 batches (assert in test with a stubbed `segment()`)
- `pytest tests/test_overlap.py` green

## Out of scope

- Damage assessment (§7a is #3), estimate work, agent, UI
