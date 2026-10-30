"""DuckDB storage for ClaimLens.

``connect()`` opens the database file (``:memory:`` allowed for tests) and
``init_schema()`` creates all tables idempotently. Blobs (photo/mask/crop
bytes) are stored inline so a run is fully auditable from one DB file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from . import config

_DDL = (
    """CREATE TABLE IF NOT EXISTS cars (
         car_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS runs (
         run_id TEXT PRIMARY KEY, car_id TEXT NOT NULL REFERENCES cars(car_id),
         started_at TIMESTAMP DEFAULT now(), duration_ms INTEGER,
         api_cost_usd DOUBLE, note TEXT)""",
    """CREATE TABLE IF NOT EXISTS photos (
         photo_id TEXT PRIMARY KEY, car_id TEXT NOT NULL REFERENCES cars(car_id),
         photo_idx INTEGER NOT NULL, width INTEGER, height INTEGER,
         img BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS estimate_files (
         car_id TEXT PRIMARY KEY REFERENCES cars(car_id),
         source_type TEXT NOT NULL, blob BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS visible_parts (
         photo_id TEXT NOT NULL REFERENCES photos(photo_id),
         canonical_part TEXT NOT NULL, side TEXT, confidence DOUBLE NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS masks (
         mask_id TEXT PRIMARY KEY, photo_id TEXT NOT NULL REFERENCES photos(photo_id),
         kind TEXT NOT NULL, label TEXT NOT NULL,
         x0 INTEGER NOT NULL, y0 INTEGER NOT NULL,
         x1 INTEGER NOT NULL, y1 INTEGER NOT NULL, mask BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS crops (
         crop_id TEXT PRIMARY KEY, photo_id TEXT NOT NULL REFERENCES photos(photo_id),
         canonical_part TEXT NOT NULL, x0 INTEGER NOT NULL, y0 INTEGER NOT NULL,
         x1 INTEGER NOT NULL, y1 INTEGER NOT NULL, img BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS damage_assessments (
         crop_id TEXT PRIMARY KEY REFERENCES crops(crop_id),
         assessment JSON NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS estimate_lines (
         car_id TEXT NOT NULL REFERENCES cars(car_id),
         line_no INTEGER NOT NULL, line JSON NOT NULL,
         PRIMARY KEY (car_id, line_no))""",
    """CREATE TABLE IF NOT EXISTS matches (
         car_id TEXT NOT NULL, line_no INTEGER NOT NULL,
         crop_id TEXT NOT NULL, score DOUBLE NOT NULL, basis TEXT NOT NULL,
         PRIMARY KEY (car_id, line_no, crop_id))""",
    """CREATE TABLE IF NOT EXISTS flags (
         flag_id TEXT PRIMARY KEY, car_id TEXT NOT NULL REFERENCES cars(car_id),
         run_id TEXT NOT NULL REFERENCES runs(run_id), flag JSON NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS tool_calls (
         seq INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
         tool TEXT NOT NULL, input JSON NOT NULL,
         output JSON, latency_ms INTEGER, accepted BOOLEAN)""",
    """CREATE TABLE IF NOT EXISTS model_cache (
         input_hash TEXT PRIMARY KEY, response JSON NOT NULL,
         created_at TIMESTAMP DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS part_taxonomy (
         canonical_part TEXT NOT NULL, side_scope TEXT NOT NULL,
         alias TEXT NOT NULL, hidden INTEGER NOT NULL DEFAULT 0,
         PRIMARY KEY (alias))""",
    """CREATE TABLE IF NOT EXISTS labor_ranges (
         canonical_part TEXT NOT NULL, operation TEXT NOT NULL,
         low_hrs DOUBLE NOT NULL, high_hrs DOUBLE NOT NULL,
         PRIMARY KEY (canonical_part, operation))""",
)


def connect(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection. ``:memory:`` gives an ephemeral test DB."""
    target = str(path or config.DB_PATH)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(target)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables. Safe to call repeatedly."""
    for statement in _DDL:
        conn.execute(statement)


def cache_get(conn: duckdb.DuckDBPyConnection, key: str) -> dict[str, Any] | None:
    """Return the cached model response for ``key``, or None on miss."""
    row = conn.execute(
        "SELECT response FROM model_cache WHERE input_hash = ?", [key]
    ).fetchone()
    if row is None:
        return None
    value = row[0]
    return json.loads(value) if isinstance(value, str) else dict(value)


def cache_put(
    conn: duckdb.DuckDBPyConnection, key: str, response: dict[str, Any]
) -> None:
    """Store a model response under ``key`` (insert or replace)."""
    conn.execute(
        "INSERT OR REPLACE INTO model_cache (input_hash, response) VALUES (?, ?)",
        [key, json.dumps(response)],
    )
