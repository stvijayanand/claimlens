# Issue 9: Serve — FastAPI endpoints + Streamlit UI + demo replay

- **Spec:** `docs/implementation-spec.md` §5, §9 (`ui` part), §11 Phase 7 (serve part)
- **Depends on:** #5 (report JSON)
- **Parallel with:** #7 (once #5 is done)

## Goal

A mobile-friendly app: upload → progress → evidence-backed report, with cache-only demo replay.

## Tasks

- [ ] `api.py`: all seven endpoints per the §5 table (`POST /cars`, `/photos`, `/estimate`, `/run`; `GET /runs/{id}`, `/report`, `/crops/{id}`) with stated status codes and error shapes
- [ ] `ui/app.py`: single-column Streamlit app — upload view → polling progress view → report view (crop thumbnails, cited lines, reasons, confidence, shop questions, notes, time+cost footer); hidden flag types stay hidden per feature flags
- [ ] Demo replay: with the API key removed, a previously run car renders its full report from `model_cache` alone
- [ ] UI reads only via `api.py`/`report.py` — no direct model calls

## Acceptance criteria

- Manual upload→report flow works end to end through the UI on a test car
- Phone-width viewport check: single column, readable, images visible (screenshot or notes in the issue)
- Cache-only replay succeeds with no API key set
- Run failures surface as run `error` with cause — never a partial silent report

## Out of scope

- Pipeline/agent changes, eval, README (#10)
