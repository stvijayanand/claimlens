# claimlens

After an accident, car owners get repair estimates they can't read: dozens of part codes and labor hours. ClaimLens brings check to the owner, built on Meta's models. SAM 3.1 segments car parts and damage in 4–10 photos. Muse Spark judges each damaged part and reads the estimate PDF, both as structured output against my schemas, then runs as a tool-calling agent that matches lines to damage and records flags with evidence and confidence. Muse Image creates labeled test photos with damage at known spots. I'll evaluate on real and synthetic photo sets with estimates seeded with known errors. Targets: 80%+ of errors caught, 75%+ precision, under 2 minutes and $0.50 per car.

## Demo video

[![ClaimLens 3-minute demo](https://img.youtube.com/vi/ZA-hRAe1bfM/hqdefault.jpg)](https://www.youtube.com/watch?v=ZA-hRAe1bfM)

3-minute hackathon demo: live upload → report, code tour, eval results.
Watch: https://www.youtube.com/watch?v=ZA-hRAe1bfM

## Hackathon originality

Everything demoed here was built during the hackathon (Oct 27–31, 2026). The repo
started as a one-line README (commit `0743db6`); the design, spec, all source code,
eval fixtures, tests, and UI were written in that window — see `git log`.

Original work (built during the hackathon, and what the demo highlights):

- `src/claimlens/`: vision pipeline (part listing → SAM segmentation → overlap →
  crops), damage assessment, estimate extraction, reconcile agent + tools
  (`normalize_part_name`, `match_damage`, `check_labor_range`, `record_flag`),
  report assembly, FastAPI app, Streamlit UI, config, DuckDB storage, and the
  taxonomy/labor-range seed tables.
- `eval/`: eval runner + metrics, deterministic error-seeding transforms, and the
  synthetic-generation script.
- Eval fixtures authored during the hackathon: every `estimate_correct.json`
  (written by visual inspection of its photo), the seeded estimates,
  `injected_errors.json` records, synthetic ground-truth boxes, and the rendered
  demo estimate image.
- `tests/`: the full test suite. `docs/`: design doc, implementation spec, issue
  plans. This README's results, rehearsal log, and sample report.

Not original work — third-party inputs, used but not claimed:

- Meta's models (SAM 3.1, Muse Spark, Muse Image), accessed via the Meta Model
  API. The demo shows my code calling them, not the models themselves.
- Real eval/demo photos: `DrBimmer/car-parts-and-damage-dataset` (Hugging Face,
  MIT license), used unmodified; per-car provenance is recorded in `source.json`.
- Synthetic base photos: Wikimedia Commons (CC BY-SA); damage was added via the
  Muse Image API during the hackathon.
- Open-source libraries (FastAPI, Streamlit, DuckDB, Pydantic, and the rest — see
  `pyproject.toml`).

## Results (final eval, 2026-11-01)

Cold replay eval (`python eval/run_eval.py --split all --cold`): 20 real + 5 synthetic
cars, one injected estimate error per car, all five error types covered in each split
(real: 4 per type; synthetic: 1 per type). Per-car detail is archived in
[eval_report.json](eval_report.json) (before/after hardening history:
[eval_report_before.json](eval_report_before.json),
[eval_report_after.json](eval_report_after.json)).

| metric | target | real (n=20) | synthetic (n=5) | status |
|---|---|---|---|---|
| recall (injected errors caught) | ≥80% | 100% (20/20) | 100% (5/5) | PASS |
| precision (flags that match an injected error) | ≥75% | 100% (20/20) | 100% (5/5) | PASS |
| extraction accuracy (seeded lines transcribed) | ≥90% | 100% (40/40) | 100% (6/6) | PASS |
| synthetic part detection (crop IoU ≥0.5) | ≥85% | n/a (synthetic-only metric) | 100% (5/5) | PASS |
| schema pass rate | 100% | 100% (76/76) | 100% (15/15) | PASS |
| time per car | p50 <2 min | p50 0.4s / p95 0.5s | p50 0.4s / p95 0.4s | PASS |
| mean cost per car | <$0.50 | $0.019 (estimated) | $0.015 (estimated) | PASS |

Notes:

- The runner replays model calls from staged fixtures (deterministic, no network);
  the vision overlap/crop code, extraction validation, `model_cache` behavior, agent
  loop, and report assembly all run for real. See `eval/run_eval.py` for the split
  between replayed and real stages.
- Cost is fallback-priced per call (replayed calls carry no usage payload), hence
  `estimated`. Live 4-photo runs measured ~$0.18–0.19/car with exact token pricing
  (see Demo rehearsal) — inside the $0.50 budget either way.
- Eval wall time is replay speed (sub-second). Live wall time is ~2 minutes/car,
  dominated by SAM segmentation calls.

## What was cut

Nothing. The cut order (`docs/implementation-spec.md` §14) was not invoked: both
feature flags stayed on (`FEATURE_LABOR_CHECK`, `FEATURE_REPAIR_REPLACE`), and all
five flag types ship.

Known limitations (honest, not cuts):

- Live damage recall is the weakest link. In rehearsal, SAM with the v1 damage
  phrases (`dent`, `scratch`, `crack`) returned only 3 stray `crack` masks across
  8 photo-runs, none overlapping a part mask — so live runs found 0 crops and read
  every estimate line as phantom. Torn bumpers and smashed headlights do not match
  those phrases well. Widening the damage vocabulary (or a whole-photo fallback) is
  the top follow-up; the replay eval does not cover live SAM recall, so this needs
  live measurement.
- Eval photos are 1 per car; the live API requires 4–10 photos per run.
- One estimate layout is supported in v1 (single-column numbered printout); nested
  CCC/Mitchell-style layouts are out of scope.

## Demo rehearsal (2026-11-01)

Live path (`uvicorn claimlens.api:app` + Meta Model API): 4 front-damage photos
composited as one demo car (third-party eval photos real006/009/016/017 — see
Hackathon originality) + a 5-line rendered estimate (line 5, mirror, phantom by
design). Three end-to-end upload→report runs, all `done` without errors:

| run | estimate input | lines extracted | flags | wall time | cost |
|---|---|---|---|---|---|
| live 1 | tiny bitmap-font PNG | 0 (model correctly refused to guess) | 0 | 128s | $0.189 |
| live 2 | crisp Arial PNG | 5/5 | 5 phantom | 122s | $0.185 |
| live 3 | same as live 2 (repeatability) | 5/5 | 5 phantom | 127s | $0.179 |

Replay path (fallback): server restarted with a bogus API key — a fresh run fails
loudly (`error`, 401 cause, never a silent partial report), while the stored live
report renders byte-identical in ~0.1s with zero model calls.

## How to run

Prerequisites: Python 3.11+, a Meta Model API key.

```bash
python -m venv .venv && .venv/bin/activate     # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .

# Create .env with your key (see src/claimlens/config.py for all overrides):
# META_MODEL_API_KEY=...
# META_MODEL_BASE_URL=https://api.meta.ai/v1

# Terminal 1: API
python -m uvicorn claimlens.api:app --port 8000

# Terminal 2: UI (Windows PowerShell: $env:CLAIMLENS_API_BASE="http://127.0.0.1:8000")
CLAIMLENS_API_BASE=http://127.0.0.1:8000 python -m streamlit run src/claimlens/ui/app.py

# Full eval (cold, both splits) and test suite
python eval/run_eval.py --split all --cold --out eval_report.json
python -m pytest -q
```

## Sample report (annotated)

Live run 2 above, trimmed to one flag (`GET /cars/{id}/report`):

```json
{
  "car_id": "car-7db4c39f544f",   // case id from POST /cars
  "run_id": "run-a2ca12fec855",   // pipeline run; the UI polls it, then reads this
  "duration_ms": 121939,          // upload→report wall time, ~at the 2-min budget
  "api_cost_usd": 0.1845,         // exact token pricing + $0.0025/SAM image
  "flags": [
    {
      "flag_type": "phantom_line",                    // billed but not visible
      "canonical_part": "front bumper cover",         // taxonomy-normalized part
      "reason": "Estimate line 1 ('frt bpr cvr') has no matching visible damage.",
      "confidence": 1.0,                              // 1.0: no crop matched at all
      "evidence_crop_ids": [],                        // empty: the point of a phantom
      "evidence_line_nos": [1],                       // the cited estimate line
      "shop_questions": [                             // per-type question template
        "Can you show me the damage behind line 1 (front bumper cover) in the photos?"
      ]
    }
    // ... +4 more phantom_line flags (hood, fender, headlight, mirror)
  ],
  "notes": [],            // hidden-part "needs explanation" items would land here
  "counts": {"total": 5, "phantom_line": 5, "missing_line": 0,
             "repair_vs_replace": 0, "duplicate": 0, "labor_out_of_range": 0},
  "unparsed_lines": 0     // all 5 estimate lines transcribed cleanly
}
```

Reading it honestly: the estimate transcription and flag machinery worked (5/5 lines,
evidence-linked flags with shop questions), but the five phantoms over-fire — lines
1–4 cover genuinely damaged parts that live vision failed to segment (see Known
limitations). Line 5 (mirror) is a true phantom by design.
