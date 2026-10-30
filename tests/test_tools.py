"""Tool tests: normalize/match/labor happy paths, rejections, idempotency.

No API calls: tools are deterministic over an in-memory DuckDB plus the
checked-in taxonomy CSV.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from claimlens import config, db, taxonomy, tools
from claimlens.tools import ToolContext


def make_ctx() -> ToolContext:
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO cars (car_id) VALUES ('c1'), ('c2')")
    conn.execute("INSERT INTO runs (run_id, car_id) VALUES ('r1', 'c1')")
    conn.execute(
        "INSERT INTO photos (photo_id, car_id, photo_idx, width, height, img)"
        " VALUES ('p1', 'c1', 0, 10, 10, ?)",
        [b"img"],
    )
    return ToolContext(conn=conn, car_id="c1", run_id="r1")


def add_crop(ctx: ToolContext, crop_id: str = "crop1") -> None:
    ctx.conn.execute(
        "INSERT INTO crops (crop_id, photo_id, canonical_part,"
        " x0, y0, x1, y1, img) VALUES (?, 'p1', 'front door', 0, 0, 5, 5, ?)",
        [crop_id, b"crop"],
    )


def add_assessment(
    ctx: ToolContext,
    crop_id: str = "crop1",
    part: str = "front door",
    side: str | None = "FL",
    confidence: float = 0.8,
) -> None:
    ctx.conn.execute(
        "INSERT INTO damage_assessments (crop_id, assessment) VALUES (?, ?)",
        [
            crop_id,
            json.dumps(
                {
                    "photo_id": "p1",
                    "crop_id": crop_id,
                    "canonical_part": part,
                    "side": side,
                    "damage_type": "dent",
                    "severity": "moderate",
                    "repair_replace": "repair",
                    "likelihood": 0.7,
                    "confidence": confidence,
                    "visibility": "visible",
                }
            ),
        ],
    )


def add_line(
    ctx: ToolContext,
    line_no: int,
    raw: str = "LF front door",
    part: str | None = "front door",
    side: str | None = "FL",
    operation: str = "repair",
    hours: float | None = 3.0,
    car_id: str = "c1",
) -> None:
    ctx.conn.execute(
        "INSERT INTO estimate_lines (car_id, line_no, line) VALUES (?, ?, ?)",
        [
            car_id,
            line_no,
            json.dumps(
                {
                    "line_no": line_no,
                    "raw_part_text": raw,
                    "canonical_part": part,
                    "side": side,
                    "operation": operation,
                    "labor_hours": hours,
                    "price": None,
                }
            ),
        ],
    )


def add_labor(
    ctx: ToolContext, part: str, operation: str, low: float, high: float
) -> None:
    ctx.conn.execute(
        "INSERT INTO labor_ranges (canonical_part, operation, low_hrs, high_hrs)"
        " VALUES (?, ?, ?, ?)",
        [part, operation, low, high],
    )


def tool_rows(ctx: ToolContext, tool: str) -> list[tuple[Any, ...]]:
    return ctx.conn.execute(
        "SELECT input, output, latency_ms, accepted FROM tool_calls"
        " WHERE run_id = 'r1' AND tool = ? ORDER BY seq",
        [tool],
    ).fetchall()


# --- normalize_part_name ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "part", "side", "confidence", "hidden"),
    [
        ("frt bpr cvr", "front bumper cover", None, 1.0, False),
        ("front bumper cover outer", "front bumper cover", None, 0.9, False),
        ("LF fender", "fender", "FL", 0.9, False),  # substring alias + side
        ("left front door", "front door", "FL", 0.9, False),
        ("fender", "fender", None, 0.8, False),  # sided, side missing: penalty
        ("crash bar", "bumper reinforcement", None, 1.0, True),
        ("bumper foam", "bumper absorber", None, 1.0, True),
        ("front left door shell", "front door", "FL", 0.9, False),
    ],
)
def test_normalize_matched(raw, part, side, confidence, hidden):
    ctx = make_ctx()
    add_line(ctx, 1, raw=raw, part=None, side=None)
    out = tools.normalize_part_name(ctx, raw, 1)
    assert out == {
        "canonical_part": part,
        "side": side,
        "confidence": confidence,
        "status": "matched",
        "hidden": hidden,
    }
    # Repeat call: identical output (idempotent).
    assert tools.normalize_part_name(ctx, raw, 1) == out


@pytest.mark.parametrize(
    "raw",
    [
        "rocket engine",
        "door",  # generic door with no front/rear side: ambiguous
        "left door shell",  # lateral only: still ambiguous
    ],
)
def test_normalize_unmatched(raw):
    ctx = make_ctx()
    add_line(ctx, 1, raw=raw, part=None, side=None)
    out = tools.normalize_part_name(ctx, raw, 1)
    assert out == {
        "canonical_part": None,
        "side": None,
        "confidence": 0.0,
        "status": "unmatched",
        "hidden": False,
    }


def test_normalize_conflicting_tokens_longest_substring_wins():
    """Garbage-in determinism: longest alias substring wins; side unresolvable."""
    ctx = make_ctx()
    add_line(ctx, 1, raw="x", part=None, side=None)
    out = tools.normalize_part_name(ctx, "front rear door shell", 1)
    assert out["canonical_part"] == "rear door"
    assert out["side"] is None
    assert out["confidence"] == 0.7  # substring base 0.9 minus side penalty
    assert out["status"] == "matched"


def test_normalize_operation_prefix_not_a_side():
    """'R&I' must not parse as right side; unsided parts skip side parsing."""
    ctx = make_ctx()
    add_line(ctx, 1, raw="x", part=None, side=None)
    out = tools.normalize_part_name(ctx, "R&I frt bumper cover", 1)
    assert out["canonical_part"] == "front bumper cover"
    assert out["side"] is None


def test_normalize_persists_line_row():
    ctx = make_ctx()
    add_line(ctx, 4, raw="LF fender", part=None, side=None)
    tools.normalize_part_name(ctx, "LF fender", 4)
    (line_json,) = ctx.conn.execute(
        "SELECT line FROM estimate_lines WHERE car_id = 'c1' AND line_no = 4"
    ).fetchone()
    line = json.loads(line_json)
    assert line["canonical_part"] == "fender"
    assert line["side"] == "FL"
    assert line["raw_part_text"] == "LF fender"  # untouched fields preserved


def test_normalize_missing_line_row_still_resolves():
    ctx = make_ctx()
    out = tools.normalize_part_name(ctx, "hood", 99)
    assert out["status"] == "matched"
    assert out["canonical_part"] == "hood"


def test_normalize_logs_tool_call():
    ctx = make_ctx()
    add_line(ctx, 1, raw="hood", part=None, side=None)
    tools.normalize_part_name(ctx, "hood", 1)
    rows = tool_rows(ctx, "normalize_part_name")
    assert len(rows) == 1
    tool_in, tool_out, latency_ms, accepted = rows[0]
    assert json.loads(tool_in) == {"raw": "hood", "line_no": 1}
    assert json.loads(tool_out)["canonical_part"] == "hood"
    assert latency_ms is not None and latency_ms >= 0
    assert accepted is True


# --- match_damage -----------------------------------------------------------


def test_match_full_hit():
    ctx = make_ctx()
    add_line(ctx, 1)
    add_crop(ctx)
    add_assessment(ctx)
    out = tools.match_damage(ctx, 1, "crop1")
    assert out == {
        "score": 0.96,  # 0.5 part + 0.3 side + 0.2 * 0.8
        "basis": "part+side+confidence=0.80",
        "matchable": True,
    }
    row = ctx.conn.execute(
        "SELECT score, basis FROM matches"
        " WHERE car_id = 'c1' AND line_no = 1 AND crop_id = 'crop1'"
    ).fetchone()
    assert (row[0], row[1]) == (0.96, "part+side+confidence=0.80")
    assert tools.match_damage(ctx, 1, "crop1") == out  # idempotent
    count = ctx.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert count == 1


def test_match_side_mismatch_below_and_above_threshold():
    ctx = make_ctx()
    add_line(ctx, 1, side="FL")
    add_crop(ctx)
    add_assessment(ctx, side="FR", confidence=0.8)
    out = tools.match_damage(ctx, 1, "crop1")
    assert out["score"] == pytest.approx(0.5 + 0.2 * 0.8)
    assert out["basis"] == "part+confidence=0.80"
    assert out["matchable"] is True  # 0.66 >= MATCH_MIN_SCORE

    add_crop(ctx, "crop2")
    add_assessment(ctx, "crop2", side="FR", confidence=0.1)
    low = tools.match_damage(ctx, 1, "crop2")
    assert low["score"] == pytest.approx(0.5 + 0.2 * 0.1)
    assert low["matchable"] is False
    # Below-threshold pairs are still recorded.
    assert (
        ctx.conn.execute(
            "SELECT COUNT(*) FROM matches WHERE crop_id = 'crop2'"
        ).fetchone()[0]
        == 1
    )


def test_match_part_mismatch():
    ctx = make_ctx()
    add_line(ctx, 1, part="hood", side="FL")
    add_crop(ctx)
    add_assessment(ctx, part="front door", side="FL", confidence=0.5)
    out = tools.match_damage(ctx, 1, "crop1")
    assert out["score"] == pytest.approx(0.3 + 0.2 * 0.5)
    assert out["basis"] == "side+confidence=0.50"
    assert out["matchable"] is False


def test_match_missing_refs_score_zero_without_write():
    ctx = make_ctx()
    add_line(ctx, 1)
    out = tools.match_damage(ctx, 1, "nope")
    assert out == {"score": 0.0, "basis": "missing-assessment", "matchable": False}
    add_crop(ctx)
    add_assessment(ctx)
    out = tools.match_damage(ctx, 77, "crop1")
    assert out == {"score": 0.0, "basis": "missing-line", "matchable": False}
    assert ctx.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
    rows = tool_rows(ctx, "match_damage")
    assert len(rows) == 2
    assert all(r[3] is False for r in rows)


def test_match_threshold_reads_config(monkeypatch):
    ctx = make_ctx()
    add_line(ctx, 1)
    add_crop(ctx)
    add_assessment(ctx, confidence=0.8)  # score 0.96
    monkeypatch.setattr(config, "MATCH_MIN_SCORE", 0.99)
    assert tools.match_damage(ctx, 1, "crop1")["matchable"] is False


# --- check_labor_range ------------------------------------------------------


@pytest.fixture()
def labor_ctx() -> ToolContext:
    ctx = make_ctx()
    add_labor(ctx, "fender", "repair", 2.0, 6.0)
    add_labor(ctx, "front bumper cover", "replace", 1.0, 2.0)
    add_labor(ctx, "front bumper cover", "ri", 0.5, 1.0)
    return ctx


@pytest.mark.parametrize(
    ("part", "op", "hours", "verdict"),
    [
        ("fender", "repair", 3.5, "in_range"),
        ("fender", "repair", 2.0, "in_range"),  # bounds inclusive
        ("fender", "repair", 6.0, "in_range"),
        ("fender", "repair", 9.0, "high"),
        ("fender", "repair", 1.0, "low"),
        ("front bumper cover", "R&I", 0.7, "in_range"),  # op normalized
        ("frt bpr cvr", "replace", 1.5, "in_range"),  # alias resolved
    ],
)
def test_labor_verdicts(labor_ctx, part, op, hours, verdict):
    out = tools.check_labor_range(labor_ctx, part, op, hours)
    assert out["verdict"] == verdict
    assert out["low"] is not None and out["high"] is not None
    assert tools.check_labor_range(labor_ctx, part, op, hours) == out


@pytest.mark.parametrize(
    ("part", "op"),
    [
        ("rocket engine", "repair"),  # unknown part: never guessed
        ("fender", "teleport"),  # unknown operation
    ],
)
def test_labor_unknown_part(labor_ctx, part, op):
    assert tools.check_labor_range(labor_ctx, part, op, 1.0) == {
        "low": None,
        "high": None,
        "verdict": "unknown_part",
    }


def test_labor_none_hours_reports_bounds_without_verdict(labor_ctx):
    out = tools.check_labor_range(labor_ctx, "fender", "repair", None)
    assert out == {"low": 2.0, "high": 6.0, "verdict": "unknown"}


def test_labor_gated_by_feature_flag(labor_ctx, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_LABOR_CHECK", False)
    out = tools.check_labor_range(labor_ctx, "fender", "repair", 99.0)
    assert out == {"low": None, "high": None, "verdict": "disabled"}
    rows = tool_rows(labor_ctx, "check_labor_range")
    assert rows[-1][3] is False


def test_labor_logs_tool_call(labor_ctx):
    tools.check_labor_range(labor_ctx, "fender", "repair", 3.0)
    (tool_in, tool_out, latency_ms, accepted) = tool_rows(
        labor_ctx, "check_labor_range"
    )[0]
    assert json.loads(tool_in)["hours"] == 3.0
    assert json.loads(tool_out)["verdict"] == "in_range"
    assert accepted is True


# --- tool definitions + dispatch --------------------------------------------


def test_tool_definitions_importable_and_complete():
    defs = {d["name"]: d for d in tools.TOOL_DEFINITIONS}
    assert set(defs) == {
        "normalize_part_name",
        "match_damage",
        "check_labor_range",
        "record_flag",
    }
    for definition in tools.TOOL_DEFINITIONS:
        assert definition["type"] == "function"
        assert definition["description"]
        params = definition["parameters"]
        assert params["type"] == "object"
        assert params["required"]
    assert set(defs["normalize_part_name"]["parameters"]["properties"]) == {
        "raw",
        "line_no",
    }
    assert set(defs["match_damage"]["parameters"]["properties"]) == {
        "line_no",
        "crop_id",
    }
    assert set(defs["check_labor_range"]["parameters"]["properties"]) == {
        "canonical_part",
        "operation",
        "hours",
    }
    flag_schema = defs["record_flag"]["parameters"]["properties"]["flag"]
    assert {"flag_type", "evidence_crop_ids", "evidence_line_nos"} <= set(
        flag_schema["properties"]
    )


def test_call_tool_dispatches_all_four(labor_ctx):
    add_line(labor_ctx, 1, raw="hood", part=None, side=None)
    out = tools.call_tool(
        labor_ctx, "normalize_part_name", {"raw": "hood", "line_no": 1}
    )
    assert out["canonical_part"] == "hood"
    add_crop(labor_ctx)
    add_assessment(labor_ctx, part="hood", side=None, confidence=0.5)
    out = tools.call_tool(
        labor_ctx, "match_damage", {"line_no": 1, "crop_id": "crop1"}
    )
    assert out["score"] == pytest.approx(0.5 + 0.2 * 0.5)
    out = tools.call_tool(
        labor_ctx,
        "check_labor_range",
        {"canonical_part": "fender", "operation": "repair", "hours": 3.0},
    )
    assert out["verdict"] == "in_range"
    out = tools.call_tool(
        labor_ctx,
        "record_flag",
        {
            "flag": {
                "flag_type": "phantom_line",
                "canonical_part": "hood",
                "reason": "no damage visible",
                "confidence": 0.9,
                "evidence_crop_ids": ["crop1"],
            }
        },
    )
    assert out["accepted"] is True
    with pytest.raises(ValueError, match="unknown tool"):
        tools.call_tool(labor_ctx, "nope", {})


def test_taxonomy_module_contract_used_by_tools():
    """Pin the taxonomy surface tools.py relies on."""
    assert taxonomy.find("headlamp") is not None
    assert taxonomy.combine_side(["lf"]) == "FL"
    assert "door" in taxonomy.GENERIC_DOOR_ALIASES
