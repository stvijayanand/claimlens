"""Reconcile-agent tools: normalize, match, labor check, record_flag.

The four deterministic agent tools (spec section 8.1). Pure: no model calls,
no network. Every invocation is logged to ``tool_calls`` (spec invariant I3)
and every tool is idempotent: repeating a call with the same input yields the
same output (DB writes use deterministic ids with ``INSERT OR REPLACE``).

``TOOL_DEFINITIONS`` holds the JSON tool definitions (name, description,
parameters) for the Responses-API agent loop; the loop injects the
``ToolContext`` (connection + car/run ids) and dispatches by name, e.g. via
``call_tool``.
"""

from __future__ import annotations

import hashlib
import json
import string
import time
from dataclasses import dataclass
from typing import Any

import duckdb
from pydantic import ValidationError

from . import config, taxonomy
from .models import DamageAssessment, EstimateLine, Flag

#: Confidence by alias-match path: exact alias hit vs substring fallback.
_EXACT_CONFIDENCE = 1.0
_FALLBACK_CONFIDENCE = 0.9
#: Penalty when a sided part resolves with no side (missing side is never
#: a rejection, per spec).
SIDED_MISSING_SIDE_PENALTY = 0.2

#: match_damage score weights: part equality + side equality + confidence.
_PART_WEIGHT = 0.5
_SIDE_WEIGHT = 0.3
_CONFIDENCE_WEIGHT = 0.2


@dataclass(frozen=True)
class ToolContext:
    """Execution context the agent loop injects into every tool call."""

    conn: duckdb.DuckDBPyConnection
    car_id: str
    run_id: str


def _row_json(value: Any) -> Any:
    """Decode a DuckDB JSON column value (str or already-decoded)."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _log_tool_call(
    ctx: ToolContext,
    tool: str,
    tool_input: dict[str, Any],
    output: dict[str, Any],
    latency_ms: int,
    accepted: bool,
) -> None:
    """Append one audit row to ``tool_calls`` (I3)."""
    seq = ctx.conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM tool_calls"
    ).fetchone()[0]
    ctx.conn.execute(
        "INSERT INTO tool_calls (seq, run_id, tool, input, output,"
        " latency_ms, accepted) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            seq,
            ctx.run_id,
            tool,
            json.dumps(tool_input, sort_keys=True, default=str),
            json.dumps(output, sort_keys=True, default=str),
            latency_ms,
            accepted,
        ],
    )


def _tokens(raw: str) -> list[str]:
    """Split raw part text into lowercase side-candidate tokens.

    Whitespace split with surrounding punctuation stripped; interior
    characters (e.g. the ``&`` in ``R&I``) are kept so operation prefixes
    never parse as side tokens (``R&I`` must not read as right).
    """
    return [
        tok.strip(string.punctuation).lower()
        for tok in raw.split()
        if tok.strip(string.punctuation)
    ]


def _normalize_core(raw: str) -> dict[str, Any]:
    """Pure part-name resolution: alias lookup + side folding."""
    key = " ".join(raw.strip().lower().split())
    aliases = taxonomy.get_taxonomy()
    entry = taxonomy.find(raw)
    if entry is not None:
        base = _EXACT_CONFIDENCE if key in aliases else _FALLBACK_CONFIDENCE
        side = (
            taxonomy.combine_side(_tokens(raw))
            if entry.side_scope == "sided"
            else None
        )
        confidence = base
        if entry.side_scope == "sided" and side is None:
            confidence -= SIDED_MISSING_SIDE_PENALTY
        return {
            "canonical_part": entry.canonical_part,
            "side": side,
            "confidence": round(max(confidence, 0.0), 4),
            "status": "matched",
            "hidden": entry.hidden,
        }
    # Bare door synonyms ("door", "door shell", ...) resolve only with a
    # front/rear side; without one the part is genuinely ambiguous.
    generic_hit = any(alias in key for alias in taxonomy.GENERIC_DOOR_ALIASES)
    if generic_hit:
        words = _tokens(raw)
        lowered = {w.lower() for w in words}
        front = bool(lowered & {"f", "frt", "front"})
        rear = bool(lowered & {"rear", "back", "bk"})
        if front != rear:
            canonical = "front door" if front else "rear door"
            side = taxonomy.combine_side(words)
            confidence = _FALLBACK_CONFIDENCE
            if side is None:
                confidence -= SIDED_MISSING_SIDE_PENALTY
            return {
                "canonical_part": canonical,
                "side": side,
                "confidence": round(max(confidence, 0.0), 4),
                "status": "matched",
                "hidden": False,
            }
    return {
        "canonical_part": None,
        "side": None,
        "confidence": 0.0,
        "status": "unmatched",
        "hidden": False,
    }


def normalize_part_name(
    ctx: ToolContext, raw: str, line_no: int
) -> dict[str, Any]:
    """Resolve estimate text to a canonical part + side.

    Exact alias match first, then longest-substring fallback, then the
    generic-door rule; side tokens fold via ``taxonomy.combine_side``.
    Unknown text yields ``status: "unmatched"`` (never invented). A sided
    part with no resolvable side takes a confidence penalty, not rejection.
    The ``estimate_lines`` row for this car/line is updated when present.
    """
    start = time.perf_counter()
    result = _normalize_core(raw)
    if result["status"] == "matched":
        row = ctx.conn.execute(
            "SELECT line FROM estimate_lines WHERE car_id = ? AND line_no = ?",
            [ctx.car_id, line_no],
        ).fetchone()
        if row is not None:
            line = EstimateLine.model_validate(_row_json(row[0]))
            line.canonical_part = result["canonical_part"]
            line.side = result["side"]
            ctx.conn.execute(
                "INSERT OR REPLACE INTO estimate_lines (car_id, line_no, line)"
                " VALUES (?, ?, ?)",
                [ctx.car_id, line_no, json.dumps(line.model_dump(mode="json"))],
            )
    latency_ms = int((time.perf_counter() - start) * 1000)
    _log_tool_call(
        ctx, "normalize_part_name", {"raw": raw, "line_no": line_no},
        result, latency_ms, True,
    )
    return result


def _match_core(
    line: EstimateLine, assessment: DamageAssessment
) -> dict[str, Any]:
    """Pure line-vs-damage scoring."""
    line_part = (line.canonical_part or "").strip().lower()
    assess_part = (assessment.canonical_part or "").strip().lower()
    part_hit = bool(line_part) and line_part == assess_part
    side_hit = (
        line.side is not None
        and assessment.side is not None
        and line.side.strip().upper() == assessment.side.strip().upper()
    )
    score = (
        (_PART_WEIGHT if part_hit else 0.0)
        + (_SIDE_WEIGHT if side_hit else 0.0)
        + _CONFIDENCE_WEIGHT * assessment.confidence
    )
    basis_bits = (
        (["part"] if part_hit else [])
        + (["side"] if side_hit else [])
        + [f"confidence={assessment.confidence:.2f}"]
    )
    score = round(score, 4)
    return {
        "score": score,
        "basis": "+".join(basis_bits),
        "matchable": score >= config.MATCH_MIN_SCORE,
    }


def match_damage(
    ctx: ToolContext, line_no: int, crop_id: str
) -> dict[str, Any]:
    """Score one estimate-line/damage pair; record it in ``matches``.

    Score = 0.5 part-equality + 0.3 side-equality + 0.2 x assessment
    confidence; ``basis`` records which components fired. Pairs below
    ``MATCH_MIN_SCORE`` are recorded but not matchable. Missing line or
    assessment rows yield score 0.0 with no ``matches`` write.
    """
    start = time.perf_counter()
    line_row = ctx.conn.execute(
        "SELECT line FROM estimate_lines WHERE car_id = ? AND line_no = ?",
        [ctx.car_id, line_no],
    ).fetchone()
    assess_row = ctx.conn.execute(
        "SELECT assessment FROM damage_assessments WHERE crop_id = ?",
        [crop_id],
    ).fetchone()
    accepted = True
    if line_row is None or assess_row is None:
        missing = "line" if line_row is None else "assessment"
        result: dict[str, Any] = {
            "score": 0.0,
            "basis": f"missing-{missing}",
            "matchable": False,
        }
        accepted = False
    else:
        try:
            line = EstimateLine.model_validate(_row_json(line_row[0]))
            assessment = DamageAssessment.model_validate(
                _row_json(assess_row[0])
            )
        except ValidationError:
            result = {
                "score": 0.0,
                "basis": "invalid-row",
                "matchable": False,
            }
            accepted = False
        else:
            result = _match_core(line, assessment)
            ctx.conn.execute(
                "INSERT OR REPLACE INTO matches"
                " (car_id, line_no, crop_id, score, basis)"
                " VALUES (?, ?, ?, ?, ?)",
                [ctx.car_id, line_no, crop_id, result["score"], result["basis"]],
            )
    latency_ms = int((time.perf_counter() - start) * 1000)
    _log_tool_call(
        ctx, "match_damage", {"line_no": line_no, "crop_id": crop_id},
        result, latency_ms, accepted,
    )
    return result


def _normalize_operation(operation: str) -> str:
    """Fold estimate operation spellings to labor-table keys (``R&I``->``ri``)."""
    return "".join(ch for ch in operation.strip().lower() if ch.isalnum())


def _labor_row(
    conn: duckdb.DuckDBPyConnection, canonical_part: str, operation: str
) -> tuple[float, float] | None:
    """Fetch (low, high) hours, retrying via taxonomy alias resolution."""
    row = conn.execute(
        "SELECT low_hrs, high_hrs FROM labor_ranges"
        " WHERE canonical_part = ? AND operation = ?",
        [canonical_part, operation],
    ).fetchone()
    if row is None:
        entry = taxonomy.find(canonical_part)
        if entry is not None and entry.canonical_part != canonical_part:
            row = conn.execute(
                "SELECT low_hrs, high_hrs FROM labor_ranges"
                " WHERE canonical_part = ? AND operation = ?",
                [entry.canonical_part, operation],
            ).fetchone()
    if row is None:
        return None
    return float(row[0]), float(row[1])


def check_labor_range(
    ctx: ToolContext,
    canonical_part: str,
    operation: str,
    hours: float | None,
) -> dict[str, Any]:
    """Compare billed hours against the ``labor_ranges`` reference table.

    Verdict is ``in_range`` (bounds inclusive), ``high``, ``low``, or
    ``unknown_part`` when no reference row exists -- ranges are never
    guessed. ``hours=None`` yields verdict ``unknown`` with the reference
    bounds still reported. Returns verdict ``disabled`` with no lookup when
    ``FEATURE_LABOR_CHECK`` is off.
    """
    start = time.perf_counter()
    tool_input: dict[str, Any] = {
        "canonical_part": canonical_part,
        "operation": operation,
        "hours": hours,
    }
    if not config.FEATURE_LABOR_CHECK:
        result: dict[str, Any] = {
            "low": None,
            "high": None,
            "verdict": "disabled",
        }
        accepted = False
    else:
        part = " ".join(canonical_part.strip().lower().split())
        op = _normalize_operation(operation)
        bounds = _labor_row(ctx.conn, part, op)
        if bounds is None:
            result = {"low": None, "high": None, "verdict": "unknown_part"}
        elif hours is None:
            low, high = bounds
            result = {"low": low, "high": high, "verdict": "unknown"}
        else:
            low, high = bounds
            if hours < low:
                verdict = "low"
            elif hours > high:
                verdict = "high"
            else:
                verdict = "in_range"
            result = {"low": low, "high": high, "verdict": verdict}
        accepted = True
    latency_ms = int((time.perf_counter() - start) * 1000)
    _log_tool_call(
        ctx, "check_labor_range", tool_input, result, latency_ms, accepted
    )
    return result


def _flag_id(car_id: str, run_id: str, flag_json: str) -> str:
    """Deterministic flag id so repeat submissions are idempotent."""
    return hashlib.sha256(
        f"{car_id}\0{run_id}\0{flag_json}".encode("utf-8")
    ).hexdigest()


def record_flag(
    ctx: ToolContext, flag: Flag | dict[str, Any]
) -> dict[str, Any]:
    """Validating sink (I2): accept a flag only if schema-valid with live refs.

    ``flag`` may be a ``Flag`` or its dict form. Acceptance requires Pydantic
    validation (including >=1 evidence ref), every ``evidence_crop_ids``
    entry present in ``crops``, and every ``evidence_line_nos`` entry present
    in ``estimate_lines`` for this car. The ``flags`` row is written only on
    accept; rejections return ``{accepted: False, errors: [...]}``.
    """
    start = time.perf_counter()
    raw: dict[str, Any] = (
        flag.model_dump(mode="json") if isinstance(flag, Flag) else dict(flag)
    )
    errors: list[str] = []
    try:
        valid = Flag.model_validate(raw)
    except ValidationError as exc:
        for err in exc.errors(include_url=False):
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"{loc}: {err['msg']}" if loc else str(err["msg"]))
        valid = None
    if valid is not None:
        unknown_crops = [
            cid
            for cid in valid.evidence_crop_ids
            if ctx.conn.execute(
                "SELECT 1 FROM crops WHERE crop_id = ?", [cid]
            ).fetchone()
            is None
        ]
        unknown_lines = [
            no
            for no in valid.evidence_line_nos
            if ctx.conn.execute(
                "SELECT 1 FROM estimate_lines WHERE car_id = ? AND line_no = ?",
                [ctx.car_id, no],
            ).fetchone()
            is None
        ]
        if unknown_crops:
            errors.append(f"unknown evidence_crop_ids: {unknown_crops}")
        if unknown_lines:
            errors.append(f"unknown evidence_line_nos: {unknown_lines}")
    if errors:
        result: dict[str, Any] = {
            "accepted": False,
            "errors": errors,
            "flag_id": None,
        }
        accepted = False
    else:
        assert valid is not None
        flag_json = json.dumps(valid.model_dump(mode="json"), sort_keys=True)
        flag_id = _flag_id(ctx.car_id, ctx.run_id, flag_json)
        ctx.conn.execute(
            "INSERT OR REPLACE INTO flags (flag_id, car_id, run_id, flag)"
            " VALUES (?, ?, ?, ?)",
            [flag_id, ctx.car_id, ctx.run_id, flag_json],
        )
        result = {"accepted": True, "errors": [], "flag_id": flag_id}
        accepted = True
    latency_ms = int((time.perf_counter() - start) * 1000)
    _log_tool_call(ctx, "record_flag", {"flag": raw}, result, latency_ms, accepted)
    return result


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "normalize_part_name",
        "description": (
            "Resolve raw estimate part text to a canonical part name and side"
            " (FL/FR/RL/RR/center). Returns canonical_part, side, confidence,"
            " status (matched|unmatched), and hidden (true when the part"
            " cannot be photographed)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raw": {
                    "type": "string",
                    "description": "Raw part text from the estimate line.",
                },
                "line_no": {
                    "type": "integer",
                    "description": "Estimate line number to normalize.",
                },
            },
            "required": ["raw", "line_no"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "match_damage",
        "description": (
            "Score an estimate-line/damage pair (0.5 part + 0.3 side + 0.2"
            " confidence). Returns score, basis, and matchable (score above"
            " the threshold). Records the pair either way."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "line_no": {
                    "type": "integer",
                    "description": "Estimate line number.",
                },
                "crop_id": {
                    "type": "string",
                    "description": "Damage crop id to pair with the line.",
                },
            },
            "required": ["line_no", "crop_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_labor_range",
        "description": (
            "Compare billed labor hours against the reference range for a"
            " part+operation. Returns low, high, and verdict"
            " (in_range|high|low|unknown_part)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "canonical_part": {
                    "type": "string",
                    "description": "Canonical part name.",
                },
                "operation": {
                    "type": "string",
                    "description": "Operation (repair, replace, ri, refinish).",
                },
                "hours": {
                    "type": ["number", "null"],
                    "description": "Billed labor hours from the estimate line.",
                },
            },
            "required": ["canonical_part", "operation", "hours"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "record_flag",
        "description": (
            "Record a reconcile finding. Accepts only schema-valid flags with"
            " at least one evidence ref pointing at a real crop or estimate"
            " line; otherwise returns structured errors."
        ),
        "parameters": {
            "type": "object",
            "properties": {"flag": Flag.model_json_schema()},
            "required": ["flag"],
            "additionalProperties": False,
        },
    },
]


def call_tool(
    ctx: ToolContext, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch one agent tool call by name (loops inject ``ctx``)."""
    if name == "normalize_part_name":
        return normalize_part_name(ctx, args["raw"], args["line_no"])
    if name == "match_damage":
        return match_damage(ctx, args["line_no"], args["crop_id"])
    if name == "check_labor_range":
        return check_labor_range(
            ctx, args["canonical_part"], args["operation"], args["hours"]
        )
    if name == "record_flag":
        return record_flag(ctx, args["flag"])
    raise ValueError(f"unknown tool: {name!r}")
