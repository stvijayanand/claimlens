# ClaimLens — Implementation Spec

- **Status:** Draft (spec-only; no code yet)
- **Date:** 2026-10-30
- **Sources:** `docs/design.md` (binding for scope, pipeline, and decisions), `ClaimLens — Meta Hackathon Submission.docx`
- **Purpose:** Turn the design into buildable units: exact files, contracts, interfaces, and phase gates. Anything here that conflicts with the design doc loses to the design doc.

## 0. Global decisions and requirements

- **Runtime:** Python 3.11+, Pydantic v2, FastAPI, DuckDB (single file, e.g. `data/claimlens.duckdb`), Streamlit, OpenAI-compatible SDK pointed at the Meta Model API base URL. (Assumption; confirm availability in Phase 1.)
- **Config:** All model IDs, base URL, key (`META_MODEL_API_KEY` env var), effort levels, thresholds, and budgets live in `config.py` + env overrides. No hardcoded model names or prices in pipeline code.
- **Invariants:**
  - I1: Every model output is validated against its Pydantic model before use. Target: 100% pass.
  - I2: A flag exists only if `record_flag` accepts it (schema-valid + ≥1 evidence ref).
  - I3: Every tool call is logged to DuckDB with input, output, and latency.
  - I4: Synthetic test images are labeled and never rendered as real evidence.
  - I5: Per-car wall time and API cost are logged on every run.
- **Feature flags (for the cut order):** `FEATURE_LABOR_CHECK`, `FEATURE_REPAIR_REPLACE` (both default on; cut in that order if behind). Missing/phantom-line detection is unconditional.

## 1. Repository layout

```
pyproject.toml
README.md                        # results added in final phase
docs/design.md
docs/implementation-spec.md      # this file
data/
  claimlens.duckdb               # created at runtime, not committed
  taxonomy.csv                   # canonical part, side handling, aliases
  labor_ranges.csv               # canonical part, operation, low_hrs, high_hrs
  eval/real/<car_id>/            # photos + estimate_correct.json + injected variants
  eval/synthetic/<car_id>/       # generated: photo + ground_truth.json + estimates
src/claimlens/
  __init__.py
  config.py                      # settings, effort levels, thresholds, budgets
  meta_client.py                 # SDK client, structured() helper, response cache
  models.py                      # §2: all Pydantic contracts
  db.py                          # connect, DDL (§3), row helpers
  taxonomy.py                    # load taxonomy, alias lookup
  api.py                         # §5: FastAPI app + endpoints
  vision.py                      # §6: listing, SAM calls, overlap, crops
  assess.py                      # §7a: damage assessment
  estimate.py                    # §7b: estimate extraction
  agent.py                       # §8: reconcile loop
  tools.py                       # §8: the four tool implementations
  report.py                      # §9: report assembly
  ui/app.py                      # §9: Streamlit app
  eval/datasets.py               # eval set loading
  eval/seed_errors.py            # error injection transforms
  eval/generate_synthetic.py     # Muse Image synthesis script
  eval/run_eval.py               # eval runner + metrics
tests/
  test_models.py                 # schema round-trip + flag invariant
  test_taxonomy.py               # alias/side normalization cases
  test_overlap.py                # mask-overlap rule cases
  test_tools.py                  # each tool: happy path + rejection cases
  test_record_flag.py            # I2 enforcement (accept/reject matrix)
  test_seed_errors.py            # injection transforms on fixture estimates
  test_pipeline_golden.py        # one synthetic car end-to-end, expected flags
```

## 2. Data contracts (`models.py`)

Pydantic v2 models are the single source: `model_json_schema()` output is passed as the structured-output schema for the matching model call, and every model response is parsed with `model_validate`.

```python
from pydantic import BaseModel, Field, model_validator

class VisiblePart(BaseModel):
    canonical_part: str
    side: str | None = None            # FL/FR/RL/RR/center/unknown
    confidence: float = Field(ge=0, le=1)

class VisibleParts(BaseModel):
    photo_id: str
    parts: list[VisiblePart]

class DamageAssessment(BaseModel):
    photo_id: str
    crop_id: str                       # evidence pointer (FK -> crops)
    canonical_part: str
    side: str | None = None
    damage_type: str                   # dent | scratch | crack | shattered | ...
    severity: str                      # minor | moderate | severe
    repair_replace: str                # repair | replace
    likelihood: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    visibility: str                    # visible | not_visible

class EstimateLine(BaseModel):
    line_no: int
    raw_part_text: str
    canonical_part: str | None = None  # filled by normalize_part_name
    side: str | None = None
    operation: str                     # repair | replace | RI | refinish | ...
    labor_hours: float | None = None
    price: float | None = None

class EstimateDoc(BaseModel):
    lines: list[EstimateLine]
    unparsed_count: int = 0            # rejected lines, never guessed

class Flag(BaseModel):
    flag_type: str                     # phantom_line | missing_line |
                                       # repair_vs_replace | duplicate |
                                       # labor_out_of_range
    canonical_part: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    evidence_crop_ids: list[str] = []
    evidence_line_nos: list[int] = []
    shop_questions: list[str] = []

    @model_validator(mode="after")
    def _evidence_required(self):
        if not self.evidence_crop_ids and not self.evidence_line_nos:
            raise ValueError("flag needs >=1 evidence ref")
        return self

class Match(BaseModel):
    line_no: int
    crop_id: str
    score: float = Field(ge=0, le=1)
    basis: str                         # e.g. "part+side+location"
```

String enums are deliberately open (plain `str`) in v1: the taxonomy table owns the canonical vocabulary, and unknown values flow to `unmatched`/`unparsed` paths instead of failing validation. Tighten to `Literal` only if eval shows drift.

## 3. Storage (`db.py`, DuckDB)

`db.py` owns `connect(path)`, `init_schema(conn)` (idempotent `CREATE TABLE IF NOT EXISTS`), and thin row helpers. Blobs (photo bytes, mask bytes, crop bytes) are stored inline as `BLOB` — no sidecar files, so a run is fully auditable from one DB file.

```sql
CREATE TABLE IF NOT EXISTS cars (
  car_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT now());

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, car_id TEXT NOT NULL REFERENCES cars(car_id),
  started_at TIMESTAMP DEFAULT now(), duration_ms INTEGER,
  api_cost_usd DOUBLE, note TEXT);

CREATE TABLE IF NOT EXISTS photos (
  photo_id TEXT PRIMARY KEY, car_id TEXT NOT NULL REFERENCES cars(car_id),
  photo_idx INTEGER NOT NULL, width INTEGER, height INTEGER,
  img BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS estimate_files (
  car_id TEXT PRIMARY KEY REFERENCES cars(car_id),
  source_type TEXT NOT NULL,            -- pdf | image
  blob BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS visible_parts (
  photo_id TEXT NOT NULL REFERENCES photos(photo_id),
  canonical_part TEXT NOT NULL, side TEXT, confidence DOUBLE NOT NULL);

CREATE TABLE IF NOT EXISTS masks (
  mask_id TEXT PRIMARY KEY, photo_id TEXT NOT NULL REFERENCES photos(photo_id),
  kind TEXT NOT NULL,                   -- part | damage
  label TEXT NOT NULL,                  -- phrase used ("front bumper", "dent")
  x0 INTEGER NOT NULL, y0 INTEGER NOT NULL,
  x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
  mask BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS crops (
  crop_id TEXT PRIMARY KEY, photo_id TEXT NOT NULL REFERENCES photos(photo_id),
  canonical_part TEXT NOT NULL, x0 INTEGER NOT NULL, y0 INTEGER NOT NULL,
  x1 INTEGER NOT NULL, y1 INTEGER NOT NULL, img BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS damage_assessments (
  crop_id TEXT PRIMARY KEY REFERENCES crops(crop_id),
  assessment JSON NOT NULL);            -- DamageAssessment JSON

CREATE TABLE IF NOT EXISTS estimate_lines (
  car_id TEXT NOT NULL REFERENCES cars(car_id),
  line_no INTEGER NOT NULL, line JSON NOT NULL,   -- EstimateLine JSON
  PRIMARY KEY (car_id, line_no));

CREATE TABLE IF NOT EXISTS matches (
  car_id TEXT NOT NULL, line_no INTEGER NOT NULL,
  crop_id TEXT NOT NULL, score DOUBLE NOT NULL, basis TEXT NOT NULL,
  PRIMARY KEY (car_id, line_no, crop_id));

CREATE TABLE IF NOT EXISTS flags (
  flag_id TEXT PRIMARY KEY, car_id TEXT NOT NULL REFERENCES cars(car_id),
  run_id TEXT NOT NULL REFERENCES runs(run_id), flag JSON NOT NULL);

CREATE TABLE IF NOT EXISTS tool_calls (
  seq INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  tool TEXT NOT NULL, input JSON NOT NULL,
  output JSON, latency_ms INTEGER, accepted BOOLEAN);

CREATE TABLE IF NOT EXISTS model_cache (
  input_hash TEXT PRIMARY KEY, response JSON NOT NULL,
  created_at TIMESTAMP DEFAULT now());

CREATE TABLE IF NOT EXISTS part_taxonomy (
  canonical_part TEXT NOT NULL, side_scope TEXT NOT NULL,  -- sided | unsided
  alias TEXT NOT NULL,                          -- lowercase match key
  hidden INTEGER NOT NULL DEFAULT 0,            -- 1 = cannot be photographed
  PRIMARY KEY (alias));

CREATE TABLE IF NOT EXISTS labor_ranges (
  canonical_part TEXT NOT NULL, operation TEXT NOT NULL,
  low_hrs DOUBLE NOT NULL, high_hrs DOUBLE NOT NULL,
  PRIMARY KEY (canonical_part, operation));
```

Seed data: `data/taxonomy.csv` and `data/labor_ranges.csv` load into the last two tables at init. Start with ~30 common exterior parts, sided where applicable (doors, fenders, lights, mirrors).

## 4. Meta API client (`meta_client.py`)

- `get_client() -> SDK client`: OpenAI-compatible client pointed at the configured base URL, auth from `META_MODEL_API_KEY`. Single construction site.
- `structured(*, kind, model, effort, schema, messages, files=None) -> dict`:
  1. Builds cache key `sha256(kind | model | effort | schema | messages | file_bytes)`.
  2. On hit, returns cached JSON (marks `tool_calls`/logs as cached; excluded from fresh-cost accounting but included in run log).
  3. On miss, calls the API with the JSON schema as structured-output constraint + `effort` (low: extraction/listing; high: reconcile-adjacent judgments), parses with the matching Pydantic model (I1), stores to `model_cache`, returns it.
  4. On schema failure: one retry with a "repair to schema" follow-up; then raise `SchemaViolation` (caller records unparsed/unmatched — never a guessed value).
- `segment(image_bytes, phrases) -> list[Mask]`: SAM 3.1 call wrapper; one call per phrase batch; returns boxes + masks; persists nothing itself (caller writes `masks` rows).
- Taxonomy prompt block: built once in `taxonomy.py`, passed with listing calls, flagged cacheable per API prompt-caching support.
- Cost accounting: `meta_client` records per-call usage (whatever the API returns: images/tokens) into the current run; `report.py` converts to USD via `config.py` price table (seeded from doc figures, corrected in Phase 1).

## 5. Intake API (`api.py`, FastAPI)

All endpoints are synchronous in v1 except the run trigger, which returns immediately with a `run_id` the UI polls.

| Method + path | Request | Response | Notes |
|---|---|---|---|
| `POST /cars` | `{}` | `201 {car_id}` | Creates case row. |
| `POST /cars/{id}/photos` | multipart `files[]` (jpeg/png, ≤10MB each) | `200 {photo_ids[]}` | Enforces 4–10 total; stores dims. `422` outside range. |
| `POST /cars/{id}/estimate` | multipart `file` (pdf/jpeg/png) | `200 {source_type}` | One estimate per car; re-upload replaces. |
| `POST /cars/{id}/run` | `{}` | `202 {run_id}` | Requires ≥4 photos + estimate; starts pipeline (§6–§9). `422` if incomplete. |
| `GET /cars/{id}/runs/{run_id}` | — | `200 {status, progress, report?}` | `status`: queued/running/done/error. |
| `GET /cars/{id}/report` | — | `200 Report` | Latest successful run's report JSON. |
| `GET /crops/{crop_id}` | — | image bytes | Serves evidence crops to the UI. |

Errors: JSON `{detail}` with 4xx for caller errors (bad count/type, unknown id), `500` only for unexpected failures; model/schema failures inside a run mark the run `error` with a logged cause, never a partial silent report.

## 6. Vision pipeline (`vision.py`)

Runs per photo, in this order. Tunables live in `config.py`: `DAMAGE_PHRASES = ["dent", "scratch", "crack"]`, `OVERLAP_MIN_INTERSECT_PX` (default 1, i.e. any overlap; raise only if eval shows noise), `CROP_PADDING_FRAC = 0.15`.

1. `list_visible_parts(photo_id) -> VisibleParts` — Spark, low effort, taxonomy prompt block attached. Writes `visible_parts` rows. Parts below `MIN_PART_CONFIDENCE` (0.3) are dropped before SAM.
2. `segment_photo(photo_id, parts)` — for the photo, one `segment()` batch over the listed part phrases + one batch over `DAMAGE_PHRASES`. Writes `masks` rows (box + mask bytes).
3. `compute_overlaps(photo_id)` — for each part mask × damage mask pair on the photo, damaged iff intersection area ≥ `OVERLAP_MIN_INTERSECT_PX`. Pure function `is_damaged(part_mask, damage_mask) -> bool` (unit-tested in `test_overlap.py`).
4. `make_crops(photo_id) -> list[crop_id]` — one crop per damaged part: part box expanded by `CROP_PADDING_FRAC`, clamped to image bounds. Writes `crops` rows.
5. Parts segmented with no overlapping damage get **no** crop and **no** assessment row — downstream code treats "no assessment" as "no visible damage", except low-confidence-visible parts, which `assess.py` may still assess and mark `not_visible` (see §7a).

SAM call budget per car: ≤ (photos × 2 batches). No blind whole-taxonomy prompting, ever.

## 7a. Damage assessment (`assess.py`)

- `assess_crop(crop_id) -> DamageAssessment` — Spark image understanding, low effort, structured output. Input: crop bytes + parent photo bytes (context) + canonical part hint. Output validated (I1); on `SchemaViolation`, retry once, then record `visibility: "not_visible", confidence: 0` with a logged cause.
- `assess_car(car_id)` — maps over all crops; writes `damage_assessments` rows.
- Low-confidence rule: if the model returns `confidence < MIN_ASSESS_CONFIDENCE` (0.4), force `visibility = "not_visible"`. The report renders these as "not visible in photos", never "undamaged".

## 7b. Estimate extraction (`estimate.py`)

- `extract_estimate(car_id) -> EstimateDoc` — Spark file handling, low effort, structured output. Input: estimate bytes + `source_type` hint. No OCR step.
- Each returned line is validated individually: valid → `EstimateLine` row; invalid → counted in `unparsed_count` with the raw text logged. `canonical_part` stays `None` until the agent normalizes it.
- Scope gate: support one common printed layout first (chosen in Phase 3 against real samples). Layout detection beyond that is out of scope.

## 8. Reconcile agent (`agent.py`, `tools.py`)

### 8.1 Tool implementations (`tools.py`)

Pure/deterministic, idempotent, no model calls. Each has a JSON tool definition (name, description, parameters schema) registered for the Responses-API loop, and each invocation is logged to `tool_calls` (I3).

- `normalize_part_name(raw, line_no) -> {canonical_part|None, side|None, confidence, status}` — lowercase/trim → `part_taxonomy` alias lookup (exact, then longest-substring fallback) → side tokens mapped to FL/FR/RL/RR/center (`LF→FL`, `frt→front`, …). Unknown → `{status: "unmatched"}`. Side required for sided parts; missing side → `confidence` penalty, not rejection.
- `match_damage(line_no, crop_id) -> {score, basis}` — score = weighted part-equality (0.5) + side-equality (0.3) + assessment-confidence factor (0.2); `basis` records which fired. Writes `matches` row. Threshold `MATCH_MIN_SCORE` (0.6): below it the pair is recorded but not matchable.
- `check_labor_range(canonical_part, operation, hours) -> {low, high, verdict}` — lookup in `labor_ranges`; `verdict ∈ {in_range, high, low, unknown_part}`. No row → `unknown_part`, never a guessed range. Gated by `FEATURE_LABOR_CHECK`.
- `record_flag(flag: Flag) -> {accepted, errors[]}` — the validating sink (I2): `Flag.model_validate` + evidence-nonempty (also enforced by the model validator) + evidence refs must exist in DB (crop id in `crops`, line no in `estimate_lines` for this car). Writes `flags` row only on accept. Returns structured rejection otherwise.

### 8.2 Agent loop (`agent.py`)

Responses-API loop, high effort, tools callable in parallel. Pseudocode per run:

```
normalize all lines (parallel normalize_part_name)
for each canonical part (grouped across lines and assessments):
    candidate pairs = match_damage(line, crop) for all line×crop of that part (parallel)
    decide, in this order:
      1. damage with no matchable line            -> propose missing_line
      2. line with no matchable damage:
           hidden-part (taxonomy hidden flag*)    -> "needs explanation" note, NOT a flag
           else                                   -> propose phantom_line
      3. two lines match same damage (score>=min)-> propose duplicate
      4. line op=replace vs assessment=repair(lik>=0.6) or vice versa
                                                 -> propose repair_vs_replace (if FEATURE_REPAIR_REPLACE)
      5. check_labor_range verdict high|low      -> propose labor_out_of_range (if FEATURE_LABOR_CHECK)
    record_flag each proposal; rejections are logged, not retried blindly
stop after MAX_AGENT_ROUNDS=3 passes or no new proposals
```

\* `hidden` on `part_taxonomy` (§3) marks parts that cannot be photographed (behind-bumper internals): seed a short list in Phase 4; hidden-part lines render as "needs explanation".

Agent guardrails: round cap; parallel batches only for independent calls; temperature/effort from `config.py`; full transcript + tool log persisted per run for audit.

## 9. Report assembly + UI (`report.py`, `ui/app.py`)

- `build_report(car_id, run_id) -> Report`: loads accepted flags + their evidence + run stats, sorts flags by confidence desc, attaches per-flag `shop_questions` (question templates per flag type in `report.py`, e.g. phantom_line → "Can you show me the damage behind line N in the photos?"). Hidden-part notes ride along as non-flag informational items.
- `Report` JSON shape: `{car_id, run_id, duration_ms, api_cost_usd, flags[], notes[], counts{by_type}, unparsed_lines}`.
- Streamlit app (`ui/app.py`), single-column mobile-friendly layout:
  1. Upload view: photos (4–10) + estimate file → calls `POST /cars`, `/photos`, `/estimate`, `/run`.
  2. Progress view: polls `GET /runs/{run_id}` until done/error.
  3. Report view: flags with crop thumbnails (via `GET /crops/{id}`), cited line text, reason, confidence bar, shop questions; notes section; footer with time + cost.
- UI reads only through `api.py` endpoints or `report.py` — never direct model calls.

## 10. Evaluation harness (`eval/`)

### 10.1 Data layout

- Real: `data/eval/real/<car_id>/{photo_*.jpg, estimate_correct.json, estimate_seeded.json, injected_errors.json}`. ~20 cars from a public car-damage dataset (chosen in Phase 5; must permit redistribution of eval crops in the README).
- Synthetic: `generate_synthetic.py <clean_photo> --damage dent --at x,y --out data/eval/synthetic/<car_id>/` calls Muse Image to add damage at a known spot; writes `photo.jpg`, `ground_truth.json` (part, damage type, box), plus `estimate_correct.json` authored to match. All synthetic files carry `"synthetic": true` and are excluded from any real-evidence path by a loader assertion.

### 10.2 Error seeding (`seed_errors.py`)

Deterministic transforms on `estimate_correct.json`, each recording `{error_type, line_no, detail}` into `injected_errors.json`:

| Transform | Effect |
|---|---|
| `add_phantom` | Appends a plausible line for an undamaged part |
| `drop_line` | Removes a line covering real damage |
| `swap_repair_replace` | Flips operation repair↔replace on one line |
| `duplicate_line` | Copies a line with a new line_no |
| `inflate_hours` | Multiplies one line's hours by 2.5 |

Each seeded estimate gets ≥1 injected error; the set covers all five types across cars.

### 10.3 Runner + metrics (`run_eval.py`)

- `run_eval.py --split real|synthetic|all [--cold]`: runs the full pipeline per car (`--cold` bypasses `model_cache` for honest time/cost), compares flags vs `injected_errors.json`, writes `eval_report.json` + prints the metrics table. Real and synthetic are always scored and reported separately.
- Metric definitions:
  - recall = injected errors with a matching flag (same type + same part/line) / total injected. Target ≥80%.
  - precision = flags matching an injected error or author-verified real issue / total flags. Target ≥75%.
  - extraction accuracy = lines exactly matching `estimate_correct.json` (part+op+hours) / total lines. Target ≥90%.
  - synthetic part detection = ground-truth damaged parts with IoU≥0.5 crop / total. Target ≥85%.
  - schema pass rate = validated outputs / total model outputs. Target 100%.
  - time = upload→report wall ms per car (p50 + p95). Target p50 <2 min.
  - cost = USD per car (mean). Target <$0.50.

## 11. Build phases and acceptance gates

| Phase | Work | Done when |
|---|---|---|
| 1. Foundations | `config`, `meta_client.structured()` + cache, `db` DDL, `models`, taxonomy/labor seeds | First structured-output call validates; unit: `test_models`, `test_taxonomy` green; live API prices confirmed in `config` |
| 2. Vision | `vision.py` listing→segment→overlap→crops on one car | One car's crops + masks inspectable in DuckDB; `test_overlap` green |
| 3. Extraction | `assess.py`, `estimate.py`, first static report (no agent: naive line↔damage join) | `test_seed_errors` green; one static end-to-end report renders |
| 4. Agent | `tools.py`, `agent.py` loop, `record_flag`, logging, `hidden` taxonomy markers | `test_tools`, `test_record_flag` green; golden car produces expected flags |
| 5. Eval | datasets, `generate_synthetic.py`, `seed_errors.py`, `run_eval.py` | Full eval runs on ≥5 cars; real/synthetic split reported |
| 6. Harden | Worst-failure fix, effort tuning, prompt caching on, thresholds from eval | Metrics move toward §10.3 targets; p50 time/cost within budget or cut order invoked |
| 7. Ship | Streamlit UI, demo cache replay, final eval, README results | Upload→report demo works; README reports real numbers |

Rough end-to-end (Phase 3) by day 5; remainder is accuracy + evaluation, per the design.

## 12. Test plan

- Unit (`tests/test_*.py`, no API calls): schema round-trips, flag-evidence rejection matrix, taxonomy alias/side cases, overlap edge cases (touching/identical/empty masks), each tool's happy path + unknown-input behavior, seed transforms on fixture estimates.
- Golden (`test_pipeline_golden.py`, cached or recorded API): one synthetic car with all five injected error types → asserts the five expected flags and no extras. This is the highest-value test; write it in Phase 4.
- Eval smoke: `run_eval.py --split synthetic` on 2 cars must complete and emit the metrics table before Phase 6 tuning begins.
- Manual: upload view on a phone-width viewport; demo replay with API key removed (cache-only run succeeds).

## 13. Instrumentation (I5)

Every run records: per-stage latencies, per-call model usage (images/tokens as returned), computed USD cost, cache hit/miss counts, flag counts by type, schema retry count. Stored on `runs` + `tool_calls`; surfaced in the report footer and eval output. If the API returns no usage payload, fall back to configured per-call price estimates and mark figures `estimated: true`.

## 14. Cut order (binding)

If behind: (1) set `FEATURE_LABOR_CHECK=off` and skip `labor_out_of_range` flags; (2) set `FEATURE_REPAIR_REPLACE=off` and skip those flags. Missing-line and phantom-line detection ship regardless. The UI hides disabled flag types; eval reports which flags were active.

## 15. Assumptions and open items

- A1: Meta Model API key + SDK/endpoint docs available at Phase 1; doc-quoted prices verified then.
- A2: Python 3.11+, Pydantic v2, DuckDB single-file store (revisit only if Phase 1 shows otherwise).
- A3: Real eval dataset chosen in Phase 5 (labeled, redistributable crops).
- A4: One estimate layout supported in v1; layout chosen from real samples in Phase 3.
- No open product questions: scope, flag types, metrics, and cut order are decided by the design doc.
