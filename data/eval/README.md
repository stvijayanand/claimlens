# ClaimLens eval data (Issue #6)

Layout per `docs/implementation-spec.md` §10.1:

- `real/<car_id>/{photo_1.*, estimate_correct.json, estimate_seeded.json,
  injected_errors.json, source.json}` — 5 cars.
- `synthetic/<car_id>/{photo.jpg, ground_truth.json, estimate_correct.json,
  estimate_seeded.json, injected_errors.json, source.json}` — 5 cars.

Every staged car has `estimate_correct.json` + `estimate_seeded.json` +
`injected_errors.json`, each with ≥1 injected error. All five error types
(`phantom_line`, `missing_line`, `repair_vs_replace`, `duplicate`,
`labor_out_of_range`) are covered in each split:

| car | damage | seed |
|---|---|---|
| real001 | front + rear door crushed (L side) | drop_line → missing_line |
| real002 | hood / bumper / headlight (FL) | duplicate_line → duplicate |
| real003 | front door torn + fender (FL) | swap_repair_replace → repair_vs_replace |
| real004 | rear door + quarter panel (RR) | inflate_hours → labor_out_of_range |
| real005 | front bumper + grille | add_phantom → phantom_line |
| syn001 | trunk lid dent | drop_line → missing_line |
| syn002 | fender scratch (FL) | duplicate_line → duplicate |
| syn003 | rear door dent (RL) | swap_repair_replace → repair_vs_replace |
| syn004 | hood dent | inflate_hours → labor_out_of_range |
| syn005 | front bumper crack | add_phantom → phantom_line |

## Real photos

Source: `DrBimmer/car-parts-and-damage-dataset` (Hugging Face, MIT
license), photos + Supervisely part-polygon annotations. Original bytes are
kept unmodified; per-car `source.json` records the original filename and the
annotated part labels. `estimate_correct.json` was authored by visual
inspection of each photo against its part polygons, then seeded with
`claimlens.eval.seed_errors` transforms.

## Synthetic photos

Clean inputs: Wikimedia Commons car photos (CC BY-SA; per-car `source.json`
has the file page + license). Damage added with
`python -m claimlens.eval.generate_synthetic` via `edit_image` on model
`muse-image-1.0` (~$0.01/image).

Observed API behavior (2026-10-30): `images.edit` returns a fresh
composition rather than a true edit of the input photo, and rejects some
requests transiently (generic 400 / content-filter 400 — retries with the
same prompt succeeded). Because outputs do not preserve input geometry, all
`ground_truth.json` boxes were authored by inspecting the output photo
(`box_source: "author"`) instead of trusting the `--at` point; the loader
still records the nominal box when generating. `photo.jpg` holds the raw
API response bytes (WebP-encoded); readers must sniff content, not suffix.

`config.IMAGE_MODEL` still defaults to the provisional `muse-image` id,
which the API rejects (404) — generation passes `muse-image-1.0`
explicitly until the config default is corrected (not owned by this issue).

## Separation invariant (I4)

All synthetic JSON files carry `"synthetic": true`. The loader
(`claimlens.eval.datasets.load_real_car`) scans every JSON file in a real
car directory and raises `SyntheticInRealPathError` if any marker is found,
so synthetic fixtures can never load through the real-evidence path.
