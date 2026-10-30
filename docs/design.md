# ClaimLens — Design Doc

- **Status:** Draft (design-only; no implementation yet)
- **Date:** 2026-10-30
- **Source:** `ClaimLens — Meta Hackathon Submission.docx` (§1–§6) and `README.md`
- **Decision:** Stack and pipeline follow the submission doc exactly; this doc turns it into buildable components, contracts, and interfaces.

## 1. Context and goals

After an accident, car owners receive repair estimates (dozens of lines like "R&I frt bumper cover", "refinish 2.4 hrs") they cannot evaluate. ClaimLens reconciles the estimate against photos of the visible damage and produces a plain-language, evidence-backed report for the owner.

**Goals**

- G1: Owner uploads 4–10 damage photos plus the estimate (PDF or photo of the printout); system returns a report in under 2 minutes at under $0.50/car in Meta Model API cost.
- G2: Every finding cites evidence (photo crop and/or estimate line) with a reason and confidence, plus questions to ask the shop.
- G3: All model I/O is schema-checked JSON; every reconcile decision is auditable via a logged tool-call trail.
- G4: All model calls go through the Meta Model API with one key and its OpenAI-compatible SDK.

**Non-goals**

- No repair pricing, no insurance advice — the report shows what to ask about, with evidence.
- No voice input (Muse Voice Transcribe deliberately unused).
- One common printed estimate layout first; other layouts are future work.
- Private on-device mode (Muse Glimmer) is a noted future direction, not this build.

## 2. Architecture overview

Eight stages; photos and the estimate travel separate paths until the reconcile agent joins them. All persistent state lives in DuckDB.

```
Photos (4-10) ──▶ [1. Visible-parts listing] ──▶ parts per photo
      │              (Muse Spark, low effort)
      ▼
                [2. SAM 3.1 segmentation] ──▶ part masks + damage masks
      (prompted ONLY for listed parts +   overlap ──▶ damaged parts
       "dent"/"scratch"/"crack")
      ▼
                [3. Crop + assess] ──▶ DamageAssessment JSON per part
                     (Muse Spark image understanding, structured output)

Estimate PDF/photo ──▶ [4. Line extraction] ──▶ EstimateLine JSON per line
                        (Muse Spark file handling, structured output, no OCR step)

DamageAssessments + EstimateLines ──▶ [5. Reconcile agent] ──▶ Flags
   (Responses API loop, high effort;   tools: normalize_part_name,
    parallel tool calls)               match_damage, check_labor_range,
                                       record_flag (validating sink)
      ▼
                [6. Report] ──▶ Streamlit UI: flags + evidence + questions
```

Supporting stores: DuckDB (photos, masks, assessments, lines, tool-call log, model-response cache by input hash), part-name taxonomy table, labor-range reference table.

## 3. Component design

### 3.1 Upload and intake (FastAPI)

- `POST /cars` → creates car case, returns `car_id`.
- `POST /cars/{id}/photos` → stores 4–10 photos (blob + metadata: index, width/height).
- `POST /cars/{id}/estimate` → stores estimate file (PDF or image), records `source_type`.
- Validation: photo count, file types/sizes; estimate accepted as PDF or image file.

### 3.2 Visible-parts listing (Spark, low effort, cached taxonomy prompt)

- Input: one photo. Output: `VisibleParts { photo_id, parts: [{canonical_part, side, confidence}] }`.
- Purpose: cost control — SAM is only prompted for parts actually visible.
- The part-taxonomy prompt block is shared across all cars and prompt-cached.

### 3.3 Segmentation (SAM 3.1 via Meta Model API)

- Calls per photo: one batch for listed part phrases ("front bumper", "left front door", …) + one batch for damage phrases ("dent", "scratch", "crack").
- Output per match: bounding box + mask; persist masks in DuckDB.
- Overlap rule: a part is *damaged* iff a damage mask overlaps its part mask above threshold T (tune on eval set; start T = any-overlap, refine by failure analysis).
- Pricing assumption (per doc, verify at build): $2.50/1k images.

### 3.4 Damage assessment (Spark image understanding, structured output)

- Input: each damaged-part crop (+ full photo for context).
- Output: `DamageAssessment` (see §5). Includes repair-vs-replace likelihood and confidence.
- Fallback for subtle damage: parts segmented but damage-uncertain are marked `visibility: "not_visible"` — never "undamaged" on low confidence.

### 3.5 Estimate extraction (Spark file handling, structured output)

- Input: estimate PDF (or photo) passed directly through API file handling — no separate OCR step.
- Output: `EstimateLine[]` (see §5). Lines failing schema validation are rejected (recorded as unparsed), never guessed.
- Scope: one common printed layout first.

### 3.6 Reconcile agent (Responses API + tool calling)

The agent works car part-by-part, calling tools, in parallel where independent. **Agent proposes, code decides**: a flag exists only if `record_flag` accepts it.

Tools (deterministic, idempotent, all calls logged):

| Tool | Signature (sketch) | Behavior |
|---|---|---|
| `normalize_part_name` | `(raw: str) -> {canonical_part, side, confidence}` | Maps "frt bpr cvr" → front bumper cover via taxonomy table + alias rules; unknown → `unmatched` (never invented). |
| `match_damage` | `(estimate_line_id, damage_assessment_id) -> {score, basis}` | Entity resolution on normalized part + side + location; returns scored pair. |
| `check_labor_range` | `(canonical_part, operation) -> {hours_ref_low, hours_ref_high, verdict}` | Compares line hours against DuckDB labor-range table. |
| `record_flag` | `(flag: Flag) -> {accepted, errors[]}` | Validating sink: rejects off-schema flags and flags without at least one evidence ref (crop id or line id). |

Agent guardrails: cap tool-call rounds per car (start: 3 passes); low reasoning effort for extraction steps, high for reconcile decisions; full tool-call log persisted for audit.

### 3.7 Report (Streamlit)

- Flag types: (a) line with no visible damage, (b) damage with no line, (c) repair-vs-replace mismatch, (d) duplicate lines, (e) unusual labor hours.
- Each flag renders: evidence crops + cited lines, reason, confidence, and suggested questions for the shop.
- Hidden-part lines (e.g., behind-bumper parts that can't be photographed) render as "needs explanation", never as errors.
- Demo resilience: model-response cache in DuckDB keyed by input hash; demo replays cached runs if the API is slow.

## 4. Data flow (per car)

1. Intake stores photos + estimate → `car_id`.
2. Per photo: visible parts → SAM masks → overlap → crops.
3. Per crop: damage assessment JSON (schema-validated).
4. Estimate → line JSON array (schema-validated; rejects logged).
5. Agent: normalize → match → labor-check → record flags (validated).
6. Report renders flags + evidence + questions.

## 5. Data contracts (Pydantic → JSON schemas)

Pydantic models are the single source: they generate the structured-output schemas for the API **and** validate everything the models return (target: 100% of outputs pass validation).

```python
class DamageAssessment(BaseModel):
    photo_id: str
    canonical_part: str        # from taxonomy
    side: str | None           # FL/FR/RL/RR/center/unknown
    damage_type: str           # dent | scratch | crack | ...
    severity: str              # minor | moderate | severe
    repair_replace: str        # repair | replace + likelihood: float
    likelihood: float
    confidence: float
    visibility: str            # visible | not_visible
    crop_id: str               # evidence pointer

class EstimateLine(BaseModel):
    line_no: int
    raw_part_text: str
    canonical_part: str | None # filled by normalize_part_name
    operation: str             # repair | replace | R&I | refinish | ...
    labor_hours: float | None
    price: float | None

class Flag(BaseModel):
    flag_type: str             # phantom_line | missing_line | repair_vs_replace
                               # duplicate | labor_out_of_range
    reason: str
    confidence: float
    evidence_crop_ids: list[str] = []
    evidence_line_nos: list[int] = []
    shop_questions: list[str] = []
    # invariant: at least one evidence ref required (enforced by record_flag)
```

## 6. Storage (DuckDB)

Tables: `cars`, `photos`, `estimate_files`, `visible_parts`, `masks` (part + damage, with boxes), `crops`, `damage_assessments`, `estimate_lines`, `matches` (line↔damage scores), `flags`, `tool_calls` (full agent audit log), `model_cache` (input-hash → response), `part_taxonomy` (canonical names + aliases + sides), `labor_ranges` (part + operation → low/high hours).

## 7. Evaluation design

Two photo sources, **reported separately** so synthetic can't flatter the numbers:

- **Real:** ~20 damaged-car photo sets from a public car-damage dataset.
- **Synthetic:** Muse Image edits of clean car photos adding a dent/scratch/crack at a chosen spot (~$0.01/image); labeled ground truth; test-set only, never shown as real evidence.

Per car: author a correct estimate, copy it, inject known errors — phantom line, missing line, replace↔repair swap, duplicate, inflated hours. Then measure:

| Measure | Target |
|---|---|
| Injected errors caught (recall) | ≥ 80% |
| Flags that are real problems (precision) | ≥ 75% |
| Estimate lines extracted correctly | ≥ 90% |
| Damaged parts found (synthetic) | ≥ 85% |
| Outputs passing schema validation | 100% |
| Upload → report time | < 2 min |
| API cost per car | < $0.50 |

The README reports real numbers either way. Eval harness runs from the first static report onward; worst failure mode drives tuning (overlap threshold, effort levels, caching).

## 8. Cost and latency budget

- SAM calls bounded by visible-parts pre-listing (no blind whole-taxonomy prompting).
- One Spark call per photo (listing) + one per crop (assessment) + one per estimate (extraction) + bounded agent rounds.
- Cache taxonomy prompts; cache all model responses by input hash.
- Verify live SAM/Spark pricing in step 1; per-car cost logged every run.

## 9. Risks and fallbacks (from doc §6)

| Risk | Fallback (decided) |
|---|---|
| SAM misses subtle damage | Judge damage on part crops too; "not visible" wording, never false "undamaged" |
| Estimate formats vary | One layout first; reject unparsed lines, don't guess |
| Agent loops/wasteful calls | Round caps; deterministic idempotent tools |
| Synthetic damage looks unrealistic | Real-photo numbers stay the headline; separate reporting |
| API limits/latency at demo | DuckDB response cache + replay |
| Hidden damage unphotographable | "Needs explanation", not errors |
| Scope creep | Cut order: labor ranges, then repair-vs-replace; missing + phantom lines ship regardless |

## 10. Build order (summary)

Foundations (client, schemas, DuckDB, taxonomy) → vision path on one car → extraction path + first static report → agent loop → test set + eval harness → harden (worst failure mode, effort tuning, caching) → Streamlit UI + final eval + README. Rough end-to-end by day 5; remainder is accuracy and evaluation.

## 11. Assumptions

- A Meta Model API key and SDK/endpoint docs are available at build time; doc-quoted prices verified against the live API in the first build step.
- No open product questions: scope, stack, flag types, metrics, and cut order are taken as decided by the submission doc.
