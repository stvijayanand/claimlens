"""Report assembly: accepted flags + notes + run stats -> Report JSON.

Spec section 9. :func:`build_report` loads one run's accepted flags, sorts
them by confidence (descending, with deterministic tie-breaks), attaches
per-flag shop questions from the templates below, and bundles the run's
"needs explanation" notes, per-type counts, and run stats.

Shapes
------
``Report`` JSON: ``{car_id, run_id, duration_ms, api_cost_usd, flags[],
notes[], counts, unparsed_lines}``. ``counts`` always carries ``total``
plus all five flag types (zero-filled) so consumers can rely on the keys.

``notes`` are read back from the agent transcript stored in ``runs.note``
(:mod:`claimlens.agent` persists them there; there is no notes table in
v1). ``unparsed_lines`` counts stored estimate lines whose
``canonical_part`` is still ``None`` — the closest persisted signal to
extraction rejects, since ``unparsed_count`` has no v1 storage column.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
from pydantic import BaseModel, Field

from .models import DamageAssessment, EstimateLine, Flag


class ReportNote(BaseModel):
    """One informational "needs explanation" item (never a flag)."""

    note_type: str = "needs_explanation"
    line_no: int | None = None
    canonical_part: str | None = None
    reason: str


class ReportCounts(BaseModel):
    """Flag counts: total plus every flag type, zero-filled."""

    total: int = 0
    phantom_line: int = 0
    missing_line: int = 0
    repair_vs_replace: int = 0
    duplicate: int = 0
    labor_out_of_range: int = 0


class Report(BaseModel):
    """Assembled owner-facing report for one car run (spec section 9)."""

    car_id: str
    run_id: str | None = None
    duration_ms: int | None = None
    api_cost_usd: float | None = None
    flags: list[Flag] = Field(default_factory=list)
    notes: list[ReportNote] = Field(default_factory=list)
    counts: ReportCounts = Field(default_factory=ReportCounts)
    unparsed_lines: int = 0


def _row_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _load_lines(
    conn: duckdb.DuckDBPyConnection, car_id: str
) -> dict[int, EstimateLine]:
    rows = conn.execute(
        "SELECT line FROM estimate_lines WHERE car_id = ? ORDER BY line_no",
        [car_id],
    ).fetchall()
    lines = [EstimateLine.model_validate(_row_json(row[0])) for row in rows]
    return {line.line_no: line for line in lines}


def _load_assessments(
    conn: duckdb.DuckDBPyConnection, car_id: str
) -> dict[str, DamageAssessment]:
    rows = conn.execute(
        "SELECT d.assessment FROM damage_assessments d"
        " JOIN crops c ON d.crop_id = c.crop_id"
        " JOIN photos p ON c.photo_id = p.photo_id"
        " WHERE p.car_id = ?",
        [car_id],
    ).fetchall()
    assessments = [
        DamageAssessment.model_validate(_row_json(row[0])) for row in rows
    ]
    return {a.crop_id: a for a in assessments}


def _latest_run_id(
    conn: duckdb.DuckDBPyConnection, car_id: str
) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM runs WHERE car_id = ? ORDER BY started_at DESC",
        [car_id],
    ).fetchone()
    return row[0] if row is not None else None


def _format_lines(nos: list[int]) -> str:
    ordered = sorted(nos)
    if len(ordered) == 1:
        return f"line {ordered[0]}"
    return "lines " + " and ".join(str(no) for no in ordered)


def shop_questions(
    flag: Flag,
    lines: dict[int, EstimateLine],
    assessments: dict[str, DamageAssessment],
) -> list[str]:
    """Render the per-type shop-question template for ``flag``.

    Evidence refs are guaranteed to exist by ``record_flag``, but an
    assessment row may be absent for a cited crop; every lookup degrades to
    a generic wording instead of raising.
    """
    nos = sorted(flag.evidence_line_nos)
    parts = [lines[no].canonical_part or flag.canonical_part for no in nos]
    part = parts[0] if parts else flag.canonical_part

    if flag.flag_type == "phantom_line":
        return [
            f"Can you show me the damage behind {_format_lines(nos)}"
            f" ({part}) in the photos?"
        ]
    if flag.flag_type == "missing_line":
        details = []
        for crop_id in flag.evidence_crop_ids:
            assessment = assessments.get(crop_id)
            if assessment is not None:
                details.append(
                    f"{assessment.severity} {assessment.damage_type}"
                    f" on the {assessment.canonical_part}"
                )
        what = "; ".join(details) if details else f"{part} damage"
        return [
            f"The photos show {what} that is not listed on the estimate —"
            " why is there no line for it?"
        ]
    if flag.flag_type == "repair_vs_replace":
        ops = sorted({lines[no].operation for no in nos if no in lines})
        billed = ops[0] if ops else "billed"
        verdicts = sorted(
            {
                assessments[crop_id].repair_replace
                for crop_id in flag.evidence_crop_ids
                if crop_id in assessments
            }
        )
        suggested = verdicts[0] if verdicts else "assessed"
        return [
            f"{_format_lines(nos).capitalize()} says '{billed}' for the"
            f" {part}, but the photos suggest '{suggested}' — can you"
            " explain the difference?"
        ]
    if flag.flag_type == "duplicate":
        return [
            f"{_format_lines(nos).capitalize()} both seem to cover the same"
            f" {part} damage — is one of them a duplicate?"
        ]
    if flag.flag_type == "labor_out_of_range":
        first = lines[nos[0]] if nos and nos[0] in lines else None
        if first is not None and first.labor_hours is not None:
            billed_hours: Any = first.labor_hours
        else:
            billed_hours = "billed"
        return [
            f"{_format_lines(nos).capitalize()} bills {billed_hours} hours"
            f" for {first.operation if first else 'the operation'} on the"
            f" {part} — why is it outside the typical range?"
        ]
    return [
        f"Can you walk me through {_format_lines(nos) if nos else 'this item'}"
        f" ({flag.flag_type})?"
    ]


def _sort_key(flag: Flag) -> tuple[Any, ...]:
    return (
        -flag.confidence,
        flag.flag_type,
        flag.canonical_part,
        sorted(flag.evidence_line_nos),
        sorted(flag.evidence_crop_ids),
    )


def build_report(
    conn: duckdb.DuckDBPyConnection,
    car_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the report dict for ``car_id`` (and ``run_id`` or latest).

    Reads only; makes no model calls. With no runs row, stats are ``None``
    and flags/notes are empty.
    """
    resolved = run_id or _latest_run_id(conn, car_id)
    duration_ms: int | None = None
    api_cost_usd: float | None = None
    notes: list[ReportNote] = []
    if resolved is not None:
        row = conn.execute(
            "SELECT duration_ms, api_cost_usd, note FROM runs WHERE run_id = ?",
            [resolved],
        ).fetchone()
        if row is not None:
            duration_ms, api_cost_usd, raw_note = row
            if raw_note:
                try:
                    transcript = json.loads(raw_note)
                except (TypeError, ValueError):
                    transcript = {}
                if isinstance(transcript, dict):
                    for item in transcript.get("notes", []):
                        try:
                            notes.append(ReportNote.model_validate(item))
                        except ValueError:
                            continue

    flags: list[Flag] = []
    if resolved is not None:
        rows = conn.execute(
            "SELECT flag FROM flags WHERE car_id = ? AND run_id = ?",
            [car_id, resolved],
        ).fetchall()
        flags = [Flag.model_validate(_row_json(row[0])) for row in rows]

    lines = _load_lines(conn, car_id)
    assessments = _load_assessments(conn, car_id)
    flags = [
        flag.model_copy(update={"shop_questions": shop_questions(flag, lines, assessments)})
        for flag in sorted(flags, key=_sort_key)
    ]

    counts = ReportCounts(total=len(flags))
    for flag in flags:
        if flag.flag_type in ReportCounts.model_fields:
            setattr(counts, flag.flag_type, getattr(counts, flag.flag_type) + 1)

    report = Report(
        car_id=car_id,
        run_id=resolved,
        duration_ms=duration_ms,
        api_cost_usd=api_cost_usd,
        flags=flags,
        notes=notes,
        counts=counts,
        unparsed_lines=sum(
            1 for line in lines.values() if line.canonical_part is None
        ),
    )
    return report.model_dump(mode="json")
