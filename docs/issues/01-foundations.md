# Issue 1: Foundations — config, client, contracts, DB, seeds

- **Spec:** `docs/implementation-spec.md` §0–§4 (scaffolding parts), §11 Phase 1
- **Depends on:** nothing (first)
- **Blocks:** all other issues

## Goal

Runnable skeleton: settings, Meta API client with cache, all Pydantic contracts, DuckDB schema, taxonomy/labor seeds, and the first unit tests.

## Tasks

- [ ] `pyproject.toml` (Python 3.11+, deps: fastapi, duckdb, pydantic v2, streamlit, openai-compatible SDK, pillow, pytest, httpx) + `src/claimlens/` package layout per spec §1
- [ ] `config.py`: settings, model IDs, effort levels, thresholds, budgets, feature flags, price table
- [ ] `meta_client.py`: `get_client()`, `structured()` with input-hash cache + one schema-repair retry + `SchemaViolation`, `segment()` wrapper, per-call usage hooks
- [ ] `models.py`: `VisiblePart(s)`, `DamageAssessment`, `EstimateLine`, `EstimateDoc`, `Flag` (evidence validator), `Match`
- [ ] `db.py`: `connect()`, idempotent `init_schema()` (§3 DDL), row helpers
- [ ] `taxonomy.py`: CSV load + alias lookup + cached prompt-block builder
- [ ] `data/taxonomy.csv` (~30 exterior parts, sided/unsided, aliases, `hidden` seeds) + `data/labor_ranges.csv`
- [ ] `tests/test_models.py` (schema round-trip, flag-evidence rejection) + `tests/test_taxonomy.py` (alias/side cases)

## Acceptance criteria

- `pytest tests/` green
- `structured()` validates a sample call against its schema and returns the cached object on repeat (cache-HIT observable in logs)
- `init_schema()` runs twice without error; blobs round-trip
- Live Meta API model IDs and prices confirmed and recorded in `config.py` (verify doc-quoted figures)

## Out of scope

- Any pipeline, agent, UI, or eval logic
