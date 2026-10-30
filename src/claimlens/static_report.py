"""Static JSON report: Phase 3 read-path proof, pre-agent.

``build_static_report`` reads ``damage_assessments`` + ``estimate_lines`` for
one car and adds a naive part-name join for display only. It makes no model
calls and writes nothing: its only job is to prove the extraction read path
works before the reconcile agent exists.

The naive join compares normalized names (lowercase equality or substring in
either direction). It is display-only and must never feed reconcile
decisions; real entity resolution is the agent's ``match_damage`` tool.

Note: ``unparsed_count`` is returned by ``extract_estimate`` but has no
storage column in v1, so the static report cannot read it back; it reports
the stored line/assessment counts instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _naive_match(part: str, raw_text: str) -> bool:
    """Naive display-only name join: equality or substring, either way."""
    left, right = _norm(part), _norm(raw_text)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def build_static_report(
    conn: duckdb.DuckDBPyConnection, car_id: str
) -> dict[str, Any]:
    """Assemble the static report dict for ``car_id`` (reads only)."""
    assessment_rows = conn.execute(
        "SELECT d.assessment FROM damage_assessments d"
        " JOIN crops c ON d.crop_id = c.crop_id"
        " JOIN photos p ON c.photo_id = p.photo_id"
        " WHERE p.car_id = ? ORDER BY d.crop_id",
        [car_id],
    ).fetchall()
    assessments = [
        json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
        for row in assessment_rows
    ]
    line_rows = conn.execute(
        "SELECT line FROM estimate_lines WHERE car_id = ? ORDER BY line_no",
        [car_id],
    ).fetchall()
    lines = [
        json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
        for row in line_rows
    ]
    naive_matches = [
        {
            "crop_id": assessment["crop_id"],
            "line_no": line["line_no"],
            "basis": "naive-name",
        }
        for assessment in assessments
        for line in lines
        if _naive_match(
            str(assessment.get("canonical_part", "")),
            str(line.get("raw_part_text", "")),
        )
    ]
    visible = sum(1 for a in assessments if a.get("visibility") == "visible")
    return {
        "car_id": car_id,
        "assessments": assessments,
        "estimate_lines": lines,
        "naive_matches": naive_matches,
        "counts": {
            "assessments": len(assessments),
            "visible": visible,
            "not_visible": len(assessments) - visible,
            "estimate_lines": len(lines),
            "naive_matches": len(naive_matches),
        },
    }


def dump_static_report(
    conn: duckdb.DuckDBPyConnection,
    car_id: str,
    path: str | Path | None = None,
) -> str:
    """Render the static report as indented JSON; optionally write ``path``."""
    text = json.dumps(build_static_report(conn, car_id), indent=2)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
