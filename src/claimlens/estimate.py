"""Estimate extraction: estimate file bytes -> EstimateLine rows.

Spec section 7b. ``extract_estimate`` passes the stored estimate file (PDF or
image) directly through Spark file handling (low effort, structured output) --
no OCR step. Each returned line is validated individually against
``EstimateLine``: valid lines become ``estimate_lines`` rows; invalid lines
are counted in ``unparsed_count`` with their raw text logged, never guessed.
``canonical_part`` stays ``None`` until the reconcile agent normalizes it.

Supported v1 layout (spec scope gate / A4): the common single-column
numbered-line shop printout, as PDF or a photo of the printout. Each row
carries a line number, a part-description text, an operation code
(repair / replace / R&I / RI / refinish / ...), optional labor hours, and an
optional price. Multi-column CCC/Mitchell-style layouts with nested sub-rows,
section headers that look like lines, and handwritten annotations are out of
scope for v1.

Layout-choice note: no ``data/eval`` samples existed when this module was
written (eval data is owned by sibling Issues #6/#7), so the choice follows
the most common printed-shop-estimate shape from the design doc. Revisit it
against real samples in Phase 5.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import duckdb
from pydantic import BaseModel, ValidationError

from . import config, meta_client
from .meta_client import FileAttachment, Transport, UsageHook
from .models import EstimateDoc, EstimateLine

log = logging.getLogger(__name__)

_EXTRACT_SYSTEM = (
    "You transcribe a printed auto body-shop repair estimate into numbered "
    "line items. For each line report: the printed line number, the raw part "
    "description exactly as printed, the operation code (repair, replace, "
    "R&I, RI, refinish, or as printed), labor hours if shown, and price if "
    "shown. Transcribe only lines that are actually printed; never invent, "
    "merge, or renumber lines. Skip section headers and totals."
)


class _RawEstimateLine(BaseModel):
    """One line exactly as the model emitted it; validated field-by-field.

    All fields are ``Any`` so that a malformed line fails the per-line
    ``EstimateLine`` check (and is counted unparsed) instead of failing the
    whole document.
    """

    line_no: Any = None
    raw_part_text: Any = None
    canonical_part: Any = None
    side: Any = None
    operation: Any = None
    labor_hours: Any = None
    price: Any = None


class _RawEstimateDoc(BaseModel):
    """Loose extraction envelope; per-line validation happens after."""

    lines: list[_RawEstimateLine]


def _guess_image_mime(blob: bytes) -> str:
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return "image/jpeg"


def extract_estimate(
    conn: duckdb.DuckDBPyConnection,
    car_id: str,
    *,
    transport: Transport | None = None,
    on_usage: UsageHook | None = None,
) -> EstimateDoc:
    """Extract ``car_id``'s estimate; write ``estimate_lines`` rows.

    Returns the parsed doc (valid lines + ``unparsed_count``). Raises
    ``ValueError`` when the car has no stored estimate file, and propagates
    ``SchemaViolation`` when the model cannot return a line list at all --
    an unparseable whole is reported, never guessed line by line.
    ``transport`` overrides the network call (tests).
    """
    row = conn.execute(
        "SELECT source_type, blob FROM estimate_files WHERE car_id = ?",
        [car_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"car has no stored estimate file: {car_id!r}")
    source_type, blob = str(row[0]), bytes(row[1])
    mime_type = (
        _guess_image_mime(blob)
        if source_type == "image"
        else "application/pdf"
    )
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Transcribe this {source_type} repair estimate into its "
                "numbered line items."
            ),
        },
    ]
    raw = meta_client.structured(
        kind="estimate_extraction",
        response_model=_RawEstimateDoc,
        model=config.SPARK_MODEL,
        effort=config.EFFORT_LOW,
        messages=messages,
        files=[
            FileAttachment(
                content=blob, mime_type=mime_type, name=f"estimate:{car_id}"
            )
        ],
        conn=conn,
        transport=transport,
        on_usage=on_usage,
    )
    assert isinstance(raw, _RawEstimateDoc)

    valid: list[EstimateLine] = []
    seen_line_nos: set[int] = set()
    unparsed_count = 0
    for raw_line in raw.lines:
        payload = raw_line.model_dump()
        try:
            line = EstimateLine.model_validate(payload)
        except ValidationError as exc:
            unparsed_count += 1
            log.warning(
                "extract_estimate %s: unparsed line %s: %s",
                car_id,
                json.dumps(payload, default=str),
                exc,
            )
            continue
        if line.line_no in seen_line_nos:
            unparsed_count += 1
            log.warning(
                "extract_estimate %s: duplicate line_no %d counted unparsed: %s",
                car_id,
                line.line_no,
                json.dumps(payload, default=str),
            )
            continue
        seen_line_nos.add(line.line_no)
        # Canonical part is the reconcile agent's job (normalize_part_name).
        valid.append(line.model_copy(update={"canonical_part": None}))

    conn.execute("DELETE FROM estimate_lines WHERE car_id = ?", [car_id])
    conn.executemany(
        "INSERT INTO estimate_lines (car_id, line_no, line) VALUES (?, ?, ?)",
        [(car_id, line.line_no, line.model_dump_json()) for line in valid],
    )
    return EstimateDoc(lines=valid, unparsed_count=unparsed_count)
