"""record_flag tests: I2 accept/reject matrix and idempotency.

No API calls: the validating sink is deterministic over in-memory DuckDB.
"""

from __future__ import annotations

import json
from typing import Any

from claimlens import db, tools
from claimlens.models import Flag
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
    conn.execute(
        "INSERT INTO crops (crop_id, photo_id, canonical_part,"
        " x0, y0, x1, y1, img) VALUES ('crop1', 'p1', 'hood', 0, 0, 5, 5, ?)",
        [b"crop"],
    )
    for car_id, line_no in (("c1", 1), ("c1", 2), ("c2", 1)):
        conn.execute(
            "INSERT INTO estimate_lines (car_id, line_no, line)"
            " VALUES (?, ?, ?)",
            [
                car_id,
                line_no,
                json.dumps(
                    {
                        "line_no": line_no,
                        "raw_part_text": "hood",
                        "canonical_part": "hood",
                        "side": None,
                        "operation": "repair",
                        "labor_hours": 3.0,
                        "price": None,
                    }
                ),
            ],
        )
    return ToolContext(conn=conn, car_id="c1", run_id="r1")


def flag_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "flag_type": "phantom_line",
        "canonical_part": "hood",
        "reason": "no damage visible in any crop",
        "confidence": 0.9,
        "evidence_crop_ids": ["crop1"],
        "evidence_line_nos": [1],
    }
    base.update(overrides)
    return base


def flag_count(ctx: ToolContext) -> int:
    return ctx.conn.execute("SELECT COUNT(*) FROM flags").fetchone()[0]


def test_accept_with_both_evidence_kinds():
    ctx = make_ctx()
    out = tools.record_flag(ctx, flag_dict())
    assert out["accepted"] is True
    assert out["errors"] == []
    assert out["flag_id"]
    assert flag_count(ctx) == 1
    (stored,) = ctx.conn.execute(
        "SELECT flag FROM flags WHERE flag_id = ?", [out["flag_id"]]
    ).fetchone()
    assert json.loads(stored)["reason"] == "no damage visible in any crop"


def test_accept_crop_only_and_line_only():
    ctx = make_ctx()
    crop_only = tools.record_flag(
        ctx, flag_dict(evidence_crop_ids=["crop1"], evidence_line_nos=[])
    )
    assert crop_only["accepted"] is True
    line_only = tools.record_flag(
        ctx, flag_dict(evidence_crop_ids=[], evidence_line_nos=[2])
    )
    assert line_only["accepted"] is True
    assert flag_count(ctx) == 2


def test_accept_flag_instance():
    ctx = make_ctx()
    flag = Flag.model_validate(flag_dict())
    out = tools.record_flag(ctx, flag)
    assert out["accepted"] is True
    assert out["errors"] == []


def test_reject_schema_violation():
    ctx = make_ctx()
    out = tools.record_flag(ctx, flag_dict(confidence=1.5))
    assert out["accepted"] is False
    assert out["flag_id"] is None
    assert any("confidence" in err for err in out["errors"])
    assert flag_count(ctx) == 0


def test_reject_missing_required_field():
    ctx = make_ctx()
    bad = flag_dict()
    del bad["reason"]
    out = tools.record_flag(ctx, bad)
    assert out["accepted"] is False
    assert any("reason" in err for err in out["errors"])
    assert flag_count(ctx) == 0


def test_reject_zero_evidence():
    ctx = make_ctx()
    out = tools.record_flag(
        ctx, flag_dict(evidence_crop_ids=[], evidence_line_nos=[])
    )
    assert out["accepted"] is False
    assert out["errors"]  # Flag model validator: >=1 evidence ref
    assert flag_count(ctx) == 0


def test_reject_unknown_crop_id():
    ctx = make_ctx()
    out = tools.record_flag(ctx, flag_dict(evidence_crop_ids=["ghost"]))
    assert out["accepted"] is False
    assert any("ghost" in err for err in out["errors"])
    assert flag_count(ctx) == 0


def test_reject_unknown_line_no():
    ctx = make_ctx()
    out = tools.record_flag(ctx, flag_dict(evidence_line_nos=[999]))
    assert out["accepted"] is False
    assert any("999" in err for err in out["errors"])
    assert flag_count(ctx) == 0


def test_reject_line_from_other_car():
    """Line-number refs are scoped to this car, not just any row."""
    ctx = make_ctx()
    ctx.conn.execute(
        "DELETE FROM estimate_lines WHERE car_id = 'c1' AND line_no = 2"
    )
    out = tools.record_flag(
        ctx, flag_dict(evidence_crop_ids=[], evidence_line_nos=[2])
    )
    assert out["accepted"] is False  # line 2 exists only for c2 now
    assert flag_count(ctx) == 0


def test_reject_mixed_valid_and_dangling_refs():
    ctx = make_ctx()
    out = tools.record_flag(
        ctx, flag_dict(evidence_crop_ids=["crop1"], evidence_line_nos=[404])
    )
    assert out["accepted"] is False
    assert any("404" in err for err in out["errors"])
    assert flag_count(ctx) == 0


def test_idempotent_repeat_same_output_single_row():
    ctx = make_ctx()
    first = tools.record_flag(ctx, flag_dict())
    second = tools.record_flag(ctx, flag_dict())
    assert first == second
    assert first["accepted"] is True
    assert flag_count(ctx) == 1  # deterministic flag_id + replace, no dupes


def test_every_invocation_logged():
    ctx = make_ctx()
    tools.record_flag(ctx, flag_dict())
    tools.record_flag(ctx, flag_dict(evidence_line_nos=[404]))
    rows = ctx.conn.execute(
        "SELECT tool, input, output, latency_ms, accepted FROM tool_calls"
        " WHERE run_id = 'r1' ORDER BY seq"
    ).fetchall()
    assert len(rows) == 2
    assert [[r[0], r[4]] for r in rows] == [
        ["record_flag", True],
        ["record_flag", False],
    ]
    assert all(r[3] is not None and r[3] >= 0 for r in rows)
    assert json.loads(rows[0][1])["flag"]["flag_type"] == "phantom_line"
    assert json.loads(rows[0][2])["accepted"] is True
    assert json.loads(rows[1][2])["accepted"] is False
