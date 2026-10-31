"""Reconcile agent: deterministic section 8.2 loop over the four tools.

The agent joins ``damage_assessments`` with ``estimate_lines`` for one car by
driving the deterministic tools in :mod:`claimlens.tools` (normalize, match,
labor check, record_flag). There are no model calls here: the "high-effort
reconcile policy" is the fixed ordered decision list from spec section 8.2,
and ``config.EFFORT_HIGH`` is recorded in the run transcript for audit.

Loop shape per run::

    normalize every line (one parallel batch)
    match every same-part line x assessment pair (one parallel batch)
    labor-check every matched, non-hidden line (one parallel batch)
    decision rounds (up to MAX_AGENT_ROUNDS, stop on no new proposals):
        1. visible damage with no matchable line      -> missing_line
        2. line with no matchable damage:
             hidden part                             -> note, never a flag
             same-part damage exists but not_visible  -> note, never a flag
             otherwise                               -> phantom_line
        3. two lines match the same damage           -> duplicate
        4. replace-vs-repair mismatch (lik >= 0.6)   -> repair_vs_replace
           (skipped unless FEATURE_REPAIR_REPLACE)
        5. labor verdict high|low                    -> labor_out_of_range
           (skipped unless FEATURE_LABOR_CHECK)
        record_flag each new proposal (rejections logged, never retried)

"Matchable" means score >= ``MATCH_MIN_SCORE`` against a ``visible``
assessment. ``not_visible`` assessments never trigger ``missing_line``
(uncertain damage is "not visible", never "undamaged"), and hidden-part
lines are excluded from every flag step: they render as "needs explanation"
notes. Parallel batches dispatch through a thread pool while a lock
serializes the shared DuckDB connection (in-memory test DBs cannot use
per-thread connections).

Persistence per run: the ``tool_calls`` audit log grows via the tools
(spec invariant I3); the ``runs`` row gains ``duration_ms`` and a JSON
transcript in ``note`` (rounds, batches, proposals, notes, feature flags).
``api_cost_usd`` is preserved when earlier pipeline stages already set it,
else 0.0 — the reconcile loop itself makes no model calls.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from typing import Any, Callable, TypeVar

import duckdb

from . import config, tools
from .models import DamageAssessment, EstimateLine, Flag
from .tools import ToolContext

#: Likelihood floor for a repair-vs-replace disagreement (spec 8.2 step 4).
REPAIR_LIKELIHOOD_MIN = 0.6

#: Fixed confidence for labor-range flags (the reference table has no spread).
LABOR_FLAG_CONFIDENCE = 0.9

#: Confidence for a phantom flag on an unrecognized (unmatched) part name.
UNMATCHED_PHANTOM_CONFIDENCE = 0.5

#: Agent identifier stamped into every run transcript.
AGENT_VERSION = "claimlens-reconcile-v1"

T = TypeVar("T")
R = TypeVar("R")


def _row_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _load_lines(conn: duckdb.DuckDBPyConnection, car_id: str) -> list[EstimateLine]:
    rows = conn.execute(
        "SELECT line FROM estimate_lines WHERE car_id = ? ORDER BY line_no",
        [car_id],
    ).fetchall()
    return [EstimateLine.model_validate(_row_json(row[0])) for row in rows]


def _load_assessments(
    conn: duckdb.DuckDBPyConnection, car_id: str
) -> list[DamageAssessment]:
    rows = conn.execute(
        "SELECT d.assessment FROM damage_assessments d"
        " JOIN crops c ON d.crop_id = c.crop_id"
        " JOIN photos p ON c.photo_id = p.photo_id"
        " WHERE p.car_id = ? ORDER BY d.crop_id",
        [car_id],
    ).fetchall()
    return [DamageAssessment.model_validate(_row_json(row[0])) for row in rows]


def _part_key(canonical: str | None) -> str | None:
    if canonical is None:
        return None
    key = " ".join(canonical.strip().lower().split())
    return key or None


def _run_batch(
    func: Callable[[T], R], items: list[T], max_workers: int
) -> list[R]:
    """Run ``func`` over ``items`` as one parallel batch, order-preserving."""
    if not items:
        return []
    if max_workers < 2 or len(items) == 1:
        return [func(item) for item in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(func, items))


def _flag_fingerprint(flag: dict[str, Any]) -> tuple[Any, ...]:
    return (
        flag.get("flag_type"),
        flag.get("canonical_part"),
        tuple(sorted(flag.get("evidence_line_nos", []))),
        tuple(sorted(flag.get("evidence_crop_ids", []))),
        flag.get("reason"),
    )


def _propose_flags(
    lines: list[EstimateLine],
    assessments: list[DamageAssessment],
    hidden_line_nos: set[int],
    normalize_conf: dict[int, float],
    match_results: dict[tuple[int, str], dict[str, Any]],
    labor_results: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply ordered decisions 1-5; return (proposals, notes)."""
    proposals: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []

    visible = [a for a in assessments if a.visibility == "visible"]
    not_visible_parts = {
        _part_key(a.canonical_part) for a in assessments if a.visibility != "visible"
    } - {None}

    def matchable(line_no: int, crop_id: str) -> bool:
        result = match_results.get((line_no, crop_id))
        if result is None or not result.get("matchable"):
            return False
        return any(
            a.crop_id == crop_id and a.visibility == "visible" for a in visible
        )

    def matchable_crops(line_no: int) -> list[str]:
        return sorted(
            a.crop_id for a in visible if matchable(line_no, a.crop_id)
        )

    def matchable_lines(crop_id: str) -> list[int]:
        return sorted(
            line.line_no
            for line in lines
            if line.line_no not in hidden_line_nos
            and matchable(line.line_no, crop_id)
        )

    assess_by_crop = {a.crop_id: a for a in visible}
    line_by_no = {line.line_no: line for line in lines}

    # Deterministic part order; hidden lines only ever produce notes.
    for line in sorted(lines, key=lambda ln: ln.line_no):
        if line.line_no in hidden_line_nos:
            notes.append(
                {
                    "note_type": "needs_explanation",
                    "line_no": line.line_no,
                    "canonical_part": line.canonical_part,
                    "reason": (
                        f"Line {line.line_no} covers"
                        f" {line.canonical_part or line.raw_part_text}, which"
                        " cannot be photographed — ask the shop to explain it."
                    ),
                }
            )

    # 1. Visible damage with no matchable line -> missing_line.
    for assessment in sorted(visible, key=lambda a: a.crop_id):
        if not matchable_lines(assessment.crop_id):
            proposals.append(
                {
                    "flag_type": "missing_line",
                    "canonical_part": assessment.canonical_part,
                    "reason": (
                        f"{assessment.canonical_part} shows"
                        f" {assessment.severity} {assessment.damage_type} in"
                        f" crop {assessment.crop_id} with no matching"
                        " estimate line."
                    ),
                    "confidence": assessment.confidence,
                    "evidence_crop_ids": [assessment.crop_id],
                    "evidence_line_nos": [],
                    "shop_questions": [],
                }
            )

    # 2. Line with no matchable damage -> note or phantom_line.
    for line in sorted(lines, key=lambda ln: ln.line_no):
        if line.line_no in hidden_line_nos:
            continue
        if matchable_crops(line.line_no):
            continue
        key = _part_key(line.canonical_part)
        if key is not None and key in not_visible_parts:
            notes.append(
                {
                    "note_type": "needs_explanation",
                    "line_no": line.line_no,
                    "canonical_part": line.canonical_part,
                    "reason": (
                        f"Line {line.line_no} covers {line.canonical_part},"
                        " but the damage was not visible in the photos — ask"
                        " the shop to point it out."
                    ),
                }
            )
        elif key is None:
            proposals.append(
                {
                    "flag_type": "phantom_line",
                    "canonical_part": line.raw_part_text.strip() or "unknown",
                    "reason": (
                        f"Estimate line {line.line_no}"
                        f" ({line.raw_part_text!r}) names a part the"
                        " taxonomy does not recognize and matches no visible"
                        " damage."
                    ),
                    "confidence": UNMATCHED_PHANTOM_CONFIDENCE,
                    "evidence_crop_ids": [],
                    "evidence_line_nos": [line.line_no],
                    "shop_questions": [],
                }
            )
        else:
            proposals.append(
                {
                    "flag_type": "phantom_line",
                    "canonical_part": line.canonical_part or key,
                    "reason": (
                        f"Estimate line {line.line_no}"
                        f" ({line.raw_part_text!r}) has no matching visible"
                        " damage."
                    ),
                    "confidence": normalize_conf.get(line.line_no, 0.9),
                    "evidence_crop_ids": [],
                    "evidence_line_nos": [line.line_no],
                    "shop_questions": [],
                }
            )

    # 3. Two lines match the same damage -> duplicate (one per extra line).
    for assessment in sorted(visible, key=lambda a: a.crop_id):
        hits = matchable_lines(assessment.crop_id)
        if len(hits) < 2:
            continue
        primary, *extras = hits
        for extra in extras:
            scores = [
                float(match_results[(no, assessment.crop_id)]["score"])
                for no in (primary, extra)
            ]
            proposals.append(
                {
                    "flag_type": "duplicate",
                    "canonical_part": assessment.canonical_part,
                    "reason": (
                        f"Estimate lines {primary} and {extra} both match the"
                        f" same {assessment.canonical_part} damage"
                        f" ({assessment.crop_id})."
                    ),
                    "confidence": round(min(scores), 4),
                    "evidence_crop_ids": [assessment.crop_id],
                    "evidence_line_nos": [primary, extra],
                    "shop_questions": [],
                }
            )

    # 4. Repair-vs-replace mismatch on matchable pairs (feature-gated).
    if config.FEATURE_REPAIR_REPLACE:
        pairs = sorted(
            (line_no, crop_id)
            for (line_no, crop_id) in match_results
            if line_no not in hidden_line_nos and matchable(line_no, crop_id)
        )
        for line_no, crop_id in pairs:
            line = line_by_no[line_no]
            assessment = assess_by_crop[crop_id]
            op = line.operation.strip().lower()
            verdict = assessment.repair_replace.strip().lower()
            if (
                op in {"repair", "replace"}
                and verdict in {"repair", "replace"}
                and op != verdict
                and assessment.likelihood >= REPAIR_LIKELIHOOD_MIN
            ):
                proposals.append(
                    {
                        "flag_type": "repair_vs_replace",
                        "canonical_part": assessment.canonical_part,
                        "reason": (
                            f"Line {line_no} says '{op}' but the"
                            f" {assessment.canonical_part} damage reads"
                            f" '{verdict}' (likelihood"
                            f" {assessment.likelihood:.2f})."
                        ),
                        "confidence": assessment.likelihood,
                        "evidence_crop_ids": [crop_id],
                        "evidence_line_nos": [line_no],
                        "shop_questions": [],
                    }
                )

    # 5. Labor verdict high|low (feature-gated).
    if config.FEATURE_LABOR_CHECK:
        for line_no in sorted(labor_results):
            if line_no in hidden_line_nos:
                continue
            result = labor_results[line_no]
            if result.get("verdict") in {"high", "low"}:
                line = line_by_no[line_no]
                proposals.append(
                    {
                        "flag_type": "labor_out_of_range",
                        "canonical_part": line.canonical_part or "unknown",
                        "reason": (
                            f"Line {line_no} bills {line.labor_hours}h for"
                            f" {line.operation} on"
                            f" {line.canonical_part} (typical"
                            f" {result['low']}–{result['high']}h)."
                        ),
                        "confidence": LABOR_FLAG_CONFIDENCE,
                        "evidence_crop_ids": [],
                        "evidence_line_nos": [line_no],
                        "shop_questions": [],
                    }
                )

    return proposals, notes


def run_agent(
    conn: duckdb.DuckDBPyConnection,
    car_id: str,
    run_id: str,
    *,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Reconcile ``car_id``'s estimate against its damage; persist the run.

    Returns ``{"car_id", "run_id", "rounds_used", "flags", "notes",
    "transcript"}``. ``flags`` are the accepted flag dicts for this run;
    ``transcript`` is also stored as JSON in ``runs.note``.
    """
    started = time.perf_counter()
    row = conn.execute(
        "SELECT api_cost_usd FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO runs (run_id, car_id) VALUES (?, ?)", [run_id, car_id]
        )
        prior_cost: float | None = None
    else:
        prior_cost = row[0]

    ctx = ToolContext(conn=conn, car_id=car_id, run_id=run_id)
    gate = threading.Lock()

    def guarded(func: Callable[..., R], *args: Any) -> R:
        with gate:
            return func(*args)

    # Normalize every line (parallel batch), then reload resolved rows.
    lines = _load_lines(conn, car_id)
    normalize_out = _run_batch(
        lambda line: guarded(
            tools.normalize_part_name, ctx, line.raw_part_text, line.line_no
        ),
        lines,
        max_workers,
    )
    norm_by_line = {line.line_no: out for line, out in zip(lines, normalize_out)}
    lines = _load_lines(conn, car_id)
    hidden_line_nos = {
        no for no, out in norm_by_line.items() if out.get("hidden")
    }
    normalize_conf = {
        no: float(out.get("confidence", 0.0)) for no, out in norm_by_line.items()
    }

    assessments = _load_assessments(conn, car_id)
    assess_parts = {_part_key(a.canonical_part) for a in assessments} - {None}

    # Match every same-part line x assessment pair (parallel batch).
    pairs = [
        (line.line_no, assessment.crop_id)
        for line in lines
        if line.line_no not in hidden_line_nos
        for assessment in assessments
        if _part_key(line.canonical_part) is not None
        and _part_key(line.canonical_part)
        == _part_key(assessment.canonical_part)
    ]
    match_out = _run_batch(
        lambda pair: guarded(tools.match_damage, ctx, pair[0], pair[1]),
        pairs,
        max_workers,
    )
    match_results = dict(zip(pairs, match_out))

    # Labor-check every matched, non-hidden line (parallel batch).
    labor_lines = [
        line
        for line in lines
        if line.line_no not in hidden_line_nos
        and _part_key(line.canonical_part) is not None
    ]
    labor_out = (
        _run_batch(
            lambda line: guarded(
                tools.check_labor_range,
                ctx,
                line.canonical_part or "",
                line.operation,
                line.labor_hours,
            ),
            labor_lines,
            max_workers,
        )
        if config.FEATURE_LABOR_CHECK
        else []
    )
    labor_results = {
        line.line_no: out for line, out in zip(labor_lines, labor_out)
    }

    # Decision rounds: record each new proposal once; stop when a full pass
    # proposes nothing new, or at MAX_AGENT_ROUNDS.
    proposed: set[tuple[Any, ...]] = set()
    accepted_flags: list[dict[str, Any]] = []
    round_log: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    run_notes: list[dict[str, Any]] = []
    record_calls = 0
    max_rounds = max(config.MAX_AGENT_ROUNDS, 1)
    for round_no in range(1, max_rounds + 1):
        proposals, notes = _propose_flags(
            lines,
            assessments,
            hidden_line_nos,
            normalize_conf,
            match_results,
            labor_results,
        )
        run_notes = notes
        fresh = [p for p in proposals if _flag_fingerprint(p) not in proposed]
        for proposal in fresh:
            proposed.add(_flag_fingerprint(proposal))
        outcomes = _run_batch(
            lambda proposal: guarded(tools.record_flag, ctx, proposal),
            fresh,
            max_workers,
        )
        record_calls += len(fresh)
        accepted = 0
        for proposal, outcome in zip(fresh, outcomes):
            if outcome.get("accepted"):
                accepted += 1
                accepted_flags.append(
                    Flag.model_validate(proposal).model_dump(mode="json")
                )
            else:
                rejections.append(
                    {
                        "round": round_no,
                        "flag": proposal,
                        "errors": outcome.get("errors", []),
                    }
                )
        round_log.append(
            {
                "round": round_no,
                "computed": len(proposals),
                "new": len(fresh),
                "accepted": accepted,
                "rejected": len(fresh) - accepted,
            }
        )
        if not fresh:
            break

    duration_ms = int((time.perf_counter() - started) * 1000)
    transcript: dict[str, Any] = {
        "agent": AGENT_VERSION,
        "effort": config.EFFORT_HIGH,
        "max_rounds": max_rounds,
        "rounds_used": len(round_log),
        "rounds": round_log,
        "batches": {
            "normalize": len(lines),
            "match": len(pairs),
            "labor": len(labor_results),
            "record_flag": record_calls,
            "max_workers": max_workers,
        },
        "assessed_parts": sorted(assess_parts),
        "hidden_line_nos": sorted(hidden_line_nos),
        "notes": run_notes,
        "rejections": rejections,
        "feature_flags": {
            "labor_check": config.FEATURE_LABOR_CHECK,
            "repair_replace": config.FEATURE_REPAIR_REPLACE,
        },
        "duration_ms": duration_ms,
    }
    conn.execute(
        "UPDATE runs SET duration_ms = ?, api_cost_usd = ?, note = ?"
        " WHERE run_id = ?",
        [
            duration_ms,
            prior_cost if prior_cost is not None else 0.0,
            json.dumps(transcript),
            run_id,
        ],
    )
    return {
        "car_id": car_id,
        "run_id": run_id,
        "rounds_used": len(round_log),
        "flags": accepted_flags,
        "notes": run_notes,
        "transcript": transcript,
    }
