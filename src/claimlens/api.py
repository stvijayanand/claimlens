"""Intake API: FastAPI app + run orchestration (spec section 5).

Endpoints (spec section 5 table)::

    POST /cars                      -> 201 {car_id}
    POST /cars/{id}/photos          -> 200 {photo_ids[]}   (multipart files[])
    POST /cars/{id}/estimate        -> 200 {source_type}   (multipart file)
    POST /cars/{id}/run             -> 202 {run_id}
    GET  /cars/{id}/runs/{run_id}   -> 200 {status, progress, report?}
    GET  /cars/{id}/report          -> 200 Report
    GET  /crops/{crop_id}           -> image bytes

Errors are JSON ``{detail}``: 4xx for caller errors (unknown id, bad
count/type, incomplete run prerequisites), 500 only for unexpected
failures. Model/schema failures inside a run mark the run ``error`` with
a logged cause -- never a partial silent report.

Run state: the v1 ``runs`` table has no status column, so live status
lives in the module-level ``_RUNS`` map (status/progress/error), while
the durable outcome persists in the ``runs`` row: a JSON transcript in
``note`` on success (written by :mod:`claimlens.agent`), or
``{"error": ..., "cause": ...}`` on failure. Status and report reads
fall back to the database, so a previously run car renders its full
report after a restart -- and with no API key set (``build_report`` is
reads-only; demo replay needs no model calls).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image

from . import agent, assess, config, db, estimate, meta_client, report, vision

log = logging.getLogger(__name__)

#: Photo upload limits (spec section 5: jpeg/png, <=10MB each, 4-10 total).
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MIN_PHOTOS = 4
MAX_PHOTOS = 10
MAX_ESTIMATE_BYTES = 10 * 1024 * 1024

_PHOTO_CONTENT_TYPES = {"image/jpeg", "image/png"}
_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}
_ESTIMATE_CONTENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}
_ESTIMATE_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}

#: Database file for the API. Tests monkeypatch this to a temp path.
DB_PATH = config.DB_PATH

#: Serializes all DuckDB access (single process, one file DB).
_DB_LOCK = threading.Lock()

#: Live run state: run_id -> {car_id, status, progress, error?, cause?}.
#: ``status`` is queued | running | done | error.
_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()

#: Pipeline entry point. Overridable by tests (monkeypatch) to stage
#: deterministic run outcomes over HTTP without model calls.
PIPELINE = None  # set to run_pipeline below (after its definition)


def _db_path() -> Path:
    """Resolve the database file (env override wins for server processes)."""
    override = os.environ.get("CLAIMLENS_DB_PATH")
    if override:
        return Path(override)
    return Path(DB_PATH)


def _ensure_seeds(conn: Any) -> None:
    """Load taxonomy/labor CSVs into empty tables (idempotent)."""
    count = conn.execute("SELECT COUNT(*) FROM part_taxonomy").fetchone()[0]
    if count == 0 and config.TAXONOMY_CSV.exists():
        with config.TAXONOMY_CSV.open(newline="", encoding="utf-8") as fh:
            rows = [
                (
                    r["canonical_part"].strip(),
                    r["side_scope"].strip(),
                    r["alias"].strip(),
                    int(r["hidden"]),
                )
                for r in csv.DictReader(fh)
            ]
        if rows:  # DuckDB rejects executemany with an empty list
            conn.executemany(
                "INSERT INTO part_taxonomy (canonical_part, side_scope, alias,"
                " hidden) VALUES (?, ?, ?, ?)",
                rows,
            )
    count = conn.execute("SELECT COUNT(*) FROM labor_ranges").fetchone()[0]
    if count == 0 and config.LABOR_CSV.exists():
        with config.LABOR_CSV.open(newline="", encoding="utf-8") as fh:
            rows = [
                (
                    r["canonical_part"].strip(),
                    r["operation"].strip(),
                    float(r["low_hrs"]),
                    float(r["high_hrs"]),
                )
                for r in csv.DictReader(fh)
            ]
        if rows:  # DuckDB rejects executemany with an empty list
            conn.executemany(
                "INSERT INTO labor_ranges (canonical_part, operation, low_hrs,"
                " high_hrs) VALUES (?, ?, ?, ?)",
                rows,
            )


def _connect() -> Any:
    """Open the API database with schema + seeds ready."""
    conn = db.connect(_db_path())
    db.init_schema(conn)
    _ensure_seeds(conn)
    return conn


def _feature_flags() -> dict[str, bool]:
    """Current server-side feature flags (spec section 14)."""
    return {
        "labor_check": bool(config.FEATURE_LABOR_CHECK),
        "repair_replace": bool(config.FEATURE_REPAIR_REPLACE),
    }


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _set_run(run_id: str, **fields: Any) -> None:
    with _RUNS_LOCK:
        state = _RUNS.setdefault(run_id, {})
        state.update(fields)


def _get_run(run_id: str) -> dict[str, Any] | None:
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)
        return dict(state) if state is not None else None


def _memory_status_payload(
    run_id: str, car_id: str, state: dict[str, Any]
) -> dict[str, Any]:
    """Build a status payload from in-memory run state (no database)."""
    status = state.get("status", "queued")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "car_id": car_id,
        "status": status,
        "progress": float(state.get("progress", 0.0)),
        "feature_flags": _feature_flags(),
    }
    if status == "error":
        payload["error"] = state.get("error", "run failed")
        payload["cause"] = state.get("cause", "unknown")
    return payload


def _photo_count(conn: Any, car_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE car_id = ?", [car_id]
    ).fetchone()
    return int(row[0])


def _has_estimate(conn: Any, car_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM estimate_files WHERE car_id = ?", [car_id]
    ).fetchone()
    return row is not None


def _require_car(conn: Any, car_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM cars WHERE car_id = ?", [car_id]
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown car_id: {car_id}")


def _run_note(conn: Any, run_id: str) -> tuple[str | None, Any]:
    """Return (car_id, parsed note) for a runs row, or (None, None)."""
    row = conn.execute(
        "SELECT car_id, note FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    if row is None:
        return None, None
    note: Any = None
    if row[1]:
        try:
            note = json.loads(row[1])
        except (TypeError, ValueError):
            note = None
    return str(row[0]), note


def _successful_run_id(conn: Any, car_id: str) -> str | None:
    """Newest run of ``car_id`` that completed (transcript, no error).

    Skips queued/running rows (no transcript yet) and errored rows, so a
    report is always from a fully successful run -- never partial.
    """
    rows = conn.execute(
        "SELECT run_id, note FROM runs WHERE car_id = ? ORDER BY started_at DESC",
        [car_id],
    ).fetchall()
    for run_id, raw_note in rows:
        if not raw_note:
            continue
        try:
            note = json.loads(raw_note)
        except (TypeError, ValueError):
            continue
        if not isinstance(note, dict) or "error" in note:
            continue
        if "agent" in note:
            return str(run_id)
        flagged = conn.execute(
            "SELECT 1 FROM flags WHERE run_id = ? LIMIT 1", [run_id]
        ).fetchone()
        if flagged is not None:
            return str(run_id)
    return None


def _status_from_db(
    conn: Any, car_id: str, run_id: str
) -> dict[str, Any] | None:
    """Rebuild run status from the database (restart/replay fallback)."""
    db_car_id, note = _run_note(conn, run_id)
    if db_car_id is None or db_car_id != car_id:
        return None
    if isinstance(note, dict) and "error" in note:
        return {
            "run_id": run_id,
            "car_id": car_id,
            "status": "error",
            "progress": 1.0,
            "error": note.get("error", "run failed"),
            "cause": note.get("cause", note.get("error", "unknown")),
            "feature_flags": _feature_flags(),
        }
    if isinstance(note, dict) and "agent" in note:
        built = report.build_report(conn, car_id, run_id)
        return {
            "run_id": run_id,
            "car_id": car_id,
            "status": "done",
            "progress": 1.0,
            "report": built,
            "feature_flags": _feature_flags(),
        }
    flagged = conn.execute(
        "SELECT 1 FROM flags WHERE run_id = ? LIMIT 1", [run_id]
    ).fetchone()
    if flagged is not None:
        built = report.build_report(conn, car_id, run_id)
        return {
            "run_id": run_id,
            "car_id": car_id,
            "status": "done",
            "progress": 1.0,
            "report": built,
            "feature_flags": _feature_flags(),
        }
    # A row with no transcript and no flags never completed (e.g. the
    # server restarted mid-run): surface it as an error, never as a
    # silent partial report.
    return {
        "run_id": run_id,
        "car_id": car_id,
        "status": "error",
        "progress": 1.0,
        "error": "run did not complete",
        "cause": "no run transcript found (the server may have restarted"
        " while the run was queued or running)",
        "feature_flags": _feature_flags(),
    }


def run_pipeline(
    car_id: str,
    run_id: str,
    *,
    listing_transport: Any = None,
    segment_transport: Any = None,
    assess_transport: Any = None,
    extract_transport: Any = None,
) -> dict[str, Any]:
    """Run vision -> assess -> extract -> agent -> report for one run.

    Raises on any failure (the caller marks the run ``error`` with the
    cause). Transports override network calls (tests / replay harnesses);
    ``None`` uses the live Meta Model API.
    """
    started = time.perf_counter()
    usage_events: list[dict[str, Any]] = []

    def on_usage(event: dict[str, Any]) -> None:
        usage_events.append(event)

    with _DB_LOCK:
        conn = _connect()
        try:
            photos = [
                row[0]
                for row in conn.execute(
                    "SELECT photo_id FROM photos WHERE car_id = ?"
                    " ORDER BY photo_idx",
                    [car_id],
                ).fetchall()
            ]
            if len(photos) < MIN_PHOTOS:
                raise ValueError(
                    f"incomplete car: {len(photos)} photos, need >={MIN_PHOTOS}"
                )
            if not _has_estimate(conn, car_id):
                raise ValueError("incomplete car: no estimate file uploaded")

            for idx, photo_id in enumerate(photos):
                vision.process_photo(
                    conn,
                    photo_id,
                    list_transport=listing_transport,
                    segment_transport=segment_transport,
                    on_usage=on_usage,
                )
                _set_run(
                    run_id, progress=0.05 + 0.50 * (idx + 1) / len(photos)
                )

            assess.assess_car(
                conn, car_id, transport=assess_transport, on_usage=on_usage
            )
            _set_run(run_id, progress=0.65)

            estimate.extract_estimate(
                conn, car_id, transport=extract_transport, on_usage=on_usage
            )
            _set_run(run_id, progress=0.75)

            # Cost accounting (spec section 13): token usage prices
            # exactly via event_cost_usd, falling back per call when the
            # API returns no usage payload. Live SAM calls emit one
            # usage event per request (each billed as one segmented
            # image); with stub transports (no live SAM) each photo
            # budgets the two batches (parts + damage) as before.
            live_sam_images = sum(
                1
                for event in usage_events
                if event.get("kind") == "segment"
                and not event.get("cached")
            )
            if live_sam_images == 0:
                live_sam_images = len(photos) * 2
            spark_cost = sum(
                meta_client.event_cost_usd(event)[0]
                for event in usage_events
                if event.get("kind") != "segment"
            )
            api_cost = (
                spark_cost + live_sam_images * config.SAM_USD_PER_IMAGE
            )
            conn.execute(
                "UPDATE runs SET api_cost_usd = ? WHERE run_id = ?",
                [api_cost, run_id],
            )

            agent.run_agent(conn, car_id, run_id)
            _set_run(run_id, progress=0.9)

            duration_ms = int((time.perf_counter() - started) * 1000)
            conn.execute(
                "UPDATE runs SET duration_ms = ? WHERE run_id = ?",
                [duration_ms, run_id],
            )
            built = report.build_report(conn, car_id, run_id)
            _set_run(run_id, progress=1.0)
            return built
        finally:
            conn.close()


PIPELINE = run_pipeline


def _execute_run(car_id: str, run_id: str) -> None:
    """Background worker: run the pipeline, record done/error + cause."""
    started = time.perf_counter()
    _set_run(run_id, status="running", progress=0.01)
    try:
        PIPELINE(car_id, run_id)
    except Exception as exc:  # noqa: BLE001 - every failure becomes run error
        cause = f"{type(exc).__name__}: {exc}"
        log.exception("run %s for car %s failed", run_id, car_id)
        duration_ms = int((time.perf_counter() - started) * 1000)
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "UPDATE runs SET duration_ms = ?, note = ?"
                    " WHERE run_id = ?",
                    [
                        duration_ms,
                        json.dumps({"error": "run failed", "cause": cause}),
                        run_id,
                    ],
                )
            finally:
                conn.close()
        _set_run(
            run_id,
            status="error",
            progress=1.0,
            error="run failed",
            cause=cause,
        )
    else:
        _set_run(run_id, status="done", progress=1.0)


def _read_upload(
    upload: UploadFile,
    *,
    allowed_types: set[str],
    allowed_suffixes: set[str],
    max_bytes: int,
    label: str,
) -> bytes:
    suffix = Path(upload.filename or "").suffix.lower()
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type not in allowed_types and suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=422,
            detail=f"{label} must be one of"
            f" {sorted(allowed_suffixes)}; got {upload.filename!r}",
        )
    data = upload.file.read()
    if not data:
        raise HTTPException(
            status_code=422, detail=f"{label} {upload.filename!r} is empty"
        )
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"{label} {upload.filename!r} exceeds"
            f" {max_bytes // (1024 * 1024)}MB",
        )
    return data


app = FastAPI(title="ClaimLens API")


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Any, exc: HTTPException) -> Any:
    """Render caller errors as exactly ``{"detail": ...}`` (spec section 5).

    Registered explicitly so the error shape never depends on the
    framework's default serialization.
    """
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Any, exc: Any) -> Any:
    """Render request-validation failures under the same ``detail`` key."""
    return JSONResponse(
        status_code=422, content={"detail": jsonable_encoder(exc.errors())}
    )


@app.post("/cars", status_code=201)
def create_car() -> dict[str, str]:
    """Create a case row; return its car id."""
    car_id = f"car-{_new_id()}"
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute("INSERT INTO cars (car_id) VALUES (?)", [car_id])
        finally:
            conn.close()
    return {"car_id": car_id}


@app.post("/cars/{car_id}/photos")
def upload_photos(
    car_id: str, files: list[UploadFile] = File(...)
) -> dict[str, list[str]]:
    """Store photo bytes + dims; enforce the 4-10 total (spec section 5).

    The per-request minimum is 1 file; the 4-photo minimum is enforced at
    ``POST /run`` so cars can be built up incrementally (and so the
    incomplete-run 422 is reachable). Totals above 10 are rejected here.
    """
    if not files:
        raise HTTPException(
            status_code=422, detail="at least one photo file is required"
        )
    blobs: list[tuple[str, bytes]] = []
    for upload in files:
        blobs.append(
            (
                upload.filename or "photo",
                _read_upload(
                    upload,
                    allowed_types=_PHOTO_CONTENT_TYPES,
                    allowed_suffixes=_PHOTO_SUFFIXES,
                    max_bytes=MAX_PHOTO_BYTES,
                    label="photo",
                ),
            )
        )
    dims: list[tuple[int, int]] = []
    for filename, blob in blobs:
        try:
            with Image.open(io.BytesIO(blob)) as img:
                dims.append((img.width, img.height))
        except Exception:
            raise HTTPException(
                status_code=422,
                detail=f"photo {filename!r} is not a readable jpeg/png image",
            ) from None
    with _DB_LOCK:
        conn = _connect()
        try:
            _require_car(conn, car_id)
            existing = _photo_count(conn, car_id)
            if existing + len(blobs) > MAX_PHOTOS:
                raise HTTPException(
                    status_code=422,
                    detail=f"too many photos: {existing} stored +"
                    f" {len(blobs)} new exceeds the {MAX_PHOTOS} maximum",
                )
            start_idx_row = conn.execute(
                "SELECT COALESCE(MAX(photo_idx), -1) FROM photos"
                " WHERE car_id = ?",
                [car_id],
            ).fetchone()
            next_idx = int(start_idx_row[0]) + 1
            photo_ids: list[str] = []
            for offset, ((_, blob), (width, height)) in enumerate(
                zip(blobs, dims)
            ):
                photo_id = f"{car_id}-p{next_idx + offset:02d}-{_new_id()}"
                conn.execute(
                    "INSERT INTO photos (photo_id, car_id, photo_idx, width,"
                    " height, img) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        photo_id,
                        car_id,
                        next_idx + offset,
                        width,
                        height,
                        blob,
                    ],
                )
                photo_ids.append(photo_id)
        finally:
            conn.close()
    return {"photo_ids": photo_ids}


@app.post("/cars/{car_id}/estimate")
def upload_estimate(
    car_id: str, file: UploadFile = File(...)
) -> dict[str, str]:
    """Store (or replace) the car's estimate file."""
    blob = _read_upload(
        file,
        allowed_types=_ESTIMATE_CONTENT_TYPES,
        allowed_suffixes=_ESTIMATE_SUFFIXES,
        max_bytes=MAX_ESTIMATE_BYTES,
        label="estimate",
    )
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        source_type = "pdf"
    else:
        source_type = "image"
    with _DB_LOCK:
        conn = _connect()
        try:
            _require_car(conn, car_id)
            conn.execute(
                "INSERT OR REPLACE INTO estimate_files (car_id, source_type,"
                " blob) VALUES (?, ?, ?)",
                [car_id, source_type, blob],
            )
        finally:
            conn.close()
    return {"source_type": source_type}


@app.post("/cars/{car_id}/run", status_code=202)
def start_run(car_id: str) -> dict[str, str]:
    """Trigger the pipeline; return immediately with a run id for polling."""
    with _DB_LOCK:
        conn = _connect()
        try:
            _require_car(conn, car_id)
            count = _photo_count(conn, car_id)
            if count < MIN_PHOTOS:
                raise HTTPException(
                    status_code=422,
                    detail=f"incomplete car: {count} photos, need"
                    f" >={MIN_PHOTOS} to run",
                )
            if not _has_estimate(conn, car_id):
                raise HTTPException(
                    status_code=422,
                    detail="incomplete car: no estimate file uploaded",
                )
            run_id = f"run-{_new_id()}"
            conn.execute(
                "INSERT INTO runs (run_id, car_id) VALUES (?, ?)",
                [run_id, car_id],
            )
        finally:
            conn.close()
    _set_run(
        run_id,
        car_id=car_id,
        status="queued",
        progress=0.0,
    )
    worker = threading.Thread(
        target=_execute_run, args=(car_id, run_id), daemon=True
    )
    worker.start()
    return {"run_id": run_id}


@app.get("/cars/{car_id}/runs/{run_id}")
def get_run(car_id: str, run_id: str) -> dict[str, Any]:
    """Poll run status; the full report rides along once done.

    Live state is served from memory without touching the database lock,
    so progress polls answer while the pipeline holds the lock. The
    database is read only to build a done report and for the restart
    fallback, when no run holds the lock.
    """
    state = _get_run(run_id)
    if state is not None:
        if state.get("car_id") != car_id:
            raise HTTPException(
                status_code=404,
                detail=f"run {run_id} does not belong to car {car_id}",
            )
        if state.get("status", "queued") != "done":
            return _memory_status_payload(run_id, car_id, state)
        # Done: fall through to build the report from the database. The
        # worker releases the lock before marking done, so this never
        # waits on a live run.
    with _DB_LOCK:
        conn = _connect()
        try:
            _require_car(conn, car_id)
            state = _get_run(run_id)
            if state is not None:
                if state.get("car_id") != car_id:
                    raise HTTPException(
                        status_code=404,
                        detail=f"run {run_id} does not belong to car {car_id}",
                    )
                if state.get("status", "queued") != "done":
                    return _memory_status_payload(run_id, car_id, state)
                payload = _memory_status_payload(run_id, car_id, state)
                payload["report"] = report.build_report(conn, car_id, run_id)
                return payload
            fallback = _status_from_db(conn, car_id, run_id)
            if fallback is None:
                raise HTTPException(
                    status_code=404, detail=f"unknown run_id: {run_id}"
                )
            return fallback
        finally:
            conn.close()


@app.get("/cars/{car_id}/report")
def get_report(car_id: str) -> dict[str, Any]:
    """Latest successful run's report JSON (reads-only: no model calls)."""
    with _DB_LOCK:
        conn = _connect()
        try:
            _require_car(conn, car_id)
            run_id = _successful_run_id(conn, car_id)
            if run_id is None:
                raise HTTPException(
                    status_code=404,
                    detail="no successful run yet for this car",
                )
            return report.build_report(conn, car_id, run_id)
        finally:
            conn.close()


@app.get("/crops/{crop_id}")
def get_crop(crop_id: str) -> Any:
    """Serve one evidence crop's image bytes to the UI."""
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT img FROM crops WHERE crop_id = ?", [crop_id]
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"unknown crop_id: {crop_id}"
        )
    blob = bytes(row[0])
    media_type = (
        "image/png" if blob.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
    )
    return Response(content=blob, media_type=media_type)
