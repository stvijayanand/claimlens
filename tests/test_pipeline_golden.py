"""Golden pipeline test: one car with all five injected error types.

Stages the reconcile inputs directly (crops, damage assessments, estimate
lines with ``canonical_part=None`` as extraction leaves them) and runs the
deterministic agent loop + report assembly. No model calls, no network: the
agent and report are pure over DuckDB plus the checked-in taxonomy CSV.

Fixture design (each error in its own part, so same-part matching cannot
cross-talk)::

    line 1  "RH front side mirror"  replace 0.6h  (no mirror damage)
      -> phantom_line
    lines 2,3  front-door variants   repair  3.0h  (one FL door crop)
      -> duplicate
    line 4  "left front fender"     replace 2.0h  (crop reads repair, lik 0.8)
      -> repair_vs_replace
    line 5  "LF headlamp"           replace 5.0h  (ref range 0.5-1.5h)
      -> labor_out_of_range
    (hood crop with no line)                          -> missing_line
    line 6  "crash bar"             replace 1.0h  (hidden part)
      -> "needs explanation" note, never a flag
    line 7  "roof"                  repair  4.0h  (roof crop not_visible)
      -> "needs explanation" note, never a flag
"""

from __future__ import annotations

import json

import duckdb

from claimlens import agent, config, db, report

CAR_ID = "golden1"
RUN_ID = "golden-run-1"
PHOTO_ID = "golden-photo-1"


def _line(
    line_no: int, raw: str, operation: str, hours: float | None
) -> tuple[str, int, str]:
    payload = {
        "line_no": line_no,
        "raw_part_text": raw,
        "canonical_part": None,
        "side": None,
        "operation": operation,
        "labor_hours": hours,
        "price": None,
    }
    return (CAR_ID, line_no, json.dumps(payload))


def _assessment(
    crop_id: str,
    part: str,
    side: str | None,
    damage_type: str,
    severity: str,
    repair_replace: str,
    likelihood: float,
    confidence: float,
    visibility: str,
) -> tuple[str, str]:
    payload = {
        "photo_id": PHOTO_ID,
        "crop_id": crop_id,
        "canonical_part": part,
        "side": side,
        "damage_type": damage_type,
        "severity": severity,
        "repair_replace": repair_replace,
        "likelihood": likelihood,
        "confidence": confidence,
        "visibility": visibility,
    }
    return (crop_id, json.dumps(payload))


def make_golden_db() -> duckdb.DuckDBPyConnection:
    """Build the staged in-memory DB: photo, crops, assessments, lines."""
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO cars (car_id) VALUES (?)", [CAR_ID])
    conn.execute(
        "INSERT INTO photos (photo_id, car_id, photo_idx, width, height, img)"
        " VALUES (?, ?, 0, 10, 10, ?)",
        [PHOTO_ID, CAR_ID, b"photo"],
    )
    for crop_id, part in [
        ("crop-door", "front door"),
        ("crop-fender", "fender"),
        ("crop-headlight", "headlight"),
        ("crop-hood", "hood"),
        ("crop-roof", "roof"),
    ]:
        conn.execute(
            "INSERT INTO crops (crop_id, photo_id, canonical_part,"
            " x0, y0, x1, y1, img) VALUES (?, ?, ?, 0, 0, 5, 5, ?)",
            [crop_id, PHOTO_ID, part, b"crop"],
        )
    conn.executemany(
        "INSERT INTO damage_assessments (crop_id, assessment) VALUES (?, ?)",
        [
            _assessment(
                "crop-door", "front door", "FL", "dent", "moderate",
                "repair", 0.75, 0.9, "visible",
            ),
            _assessment(
                "crop-fender", "fender", "FL", "scratch", "moderate",
                "repair", 0.8, 0.9, "visible",
            ),
            _assessment(
                "crop-headlight", "headlight", "FL", "shattered", "severe",
                "replace", 0.9, 0.85, "visible",
            ),
            _assessment(
                "crop-hood", "hood", None, "dent", "minor",
                "repair", 0.7, 0.85, "visible",
            ),
            _assessment(
                "crop-roof", "roof", None, "unknown", "unknown",
                "unknown", 0.0, 0.0, "not_visible",
            ),
        ],
    )
    conn.executemany(
        "INSERT INTO estimate_lines (car_id, line_no, line) VALUES (?, ?, ?)",
        [
            _line(1, "RH front side mirror", "replace", 0.6),
            _line(2, "LF front door", "repair", 3.0),
            _line(3, "left front door shell", "repair", 3.0),
            _line(4, "left front fender", "replace", 2.0),
            _line(5, "LF headlamp", "replace", 5.0),
            _line(6, "crash bar", "replace", 1.0),
            _line(7, "roof", "repair", 4.0),
        ],
    )
    conn.executemany(
        "INSERT INTO labor_ranges (canonical_part, operation, low_hrs, high_hrs)"
        " VALUES (?, ?, ?, ?)",
        [
            ("front door", "repair", 2.0, 6.0),
            ("fender", "replace", 1.5, 3.0),
            ("headlight", "replace", 0.5, 1.5),
        ],
    )
    return conn


def _flag_key(flag: dict) -> tuple:
    return (
        flag["flag_type"],
        flag["canonical_part"],
        tuple(sorted(flag["evidence_line_nos"])),
        tuple(sorted(flag["evidence_crop_ids"])),
    )


def test_golden_exactly_five_expected_flags():
    conn = make_golden_db()
    result = agent.run_agent(conn, CAR_ID, RUN_ID)

    assert result["car_id"] == CAR_ID
    assert result["run_id"] == RUN_ID
    assert sorted(_flag_key(f) for f in result["flags"]) == sorted(
        [
            ("missing_line", "hood", (), ("crop-hood",)),
            ("phantom_line", "mirror", (1,), ()),
            ("duplicate", "front door", (2, 3), ("crop-door",)),
            ("repair_vs_replace", "fender", (4,), ("crop-fender",)),
            ("labor_out_of_range", "headlight", (5,), ()),
        ]
    )
    by_type = {f["flag_type"]: f for f in result["flags"]}
    assert by_type["missing_line"]["confidence"] == 0.85
    assert by_type["phantom_line"]["confidence"] == 0.9
    assert by_type["duplicate"]["confidence"] == 0.98
    assert by_type["repair_vs_replace"]["confidence"] == 0.8
    assert by_type["labor_out_of_range"]["confidence"] == 0.9


def test_golden_hidden_and_unseen_lines_are_notes_never_flags():
    conn = make_golden_db()
    result = agent.run_agent(conn, CAR_ID, RUN_ID)

    assert len(result["flags"]) == 5
    for flag in result["flags"]:
        assert 6 not in flag["evidence_line_nos"]
        assert 7 not in flag["evidence_line_nos"]
        assert "crash bar" not in flag["reason"]
        assert "crop-roof" not in flag["evidence_crop_ids"]
        assert flag["canonical_part"] != "roof"

    assert sorted(n["line_no"] for n in result["notes"]) == [6, 7]
    assert all(n["note_type"] == "needs_explanation" for n in result["notes"])
    by_line = {n["line_no"]: n for n in result["notes"]}
    assert "cannot be photographed" in by_line[6]["reason"]
    assert "not visible" in by_line[7]["reason"]


def test_golden_rerun_is_deterministic():
    conn = make_golden_db()
    first = agent.run_agent(conn, CAR_ID, RUN_ID)
    second = agent.run_agent(conn, CAR_ID, "golden-run-2", max_workers=2)

    pack = lambda flags: sorted(
        json.dumps(f, sort_keys=True) for f in flags  # noqa: E731
    )
    assert pack(second["flags"]) == pack(first["flags"])
    assert second["notes"] == first["notes"]
    assert len(second["flags"]) == 5


def test_golden_round_cap_and_tool_audit():
    conn = make_golden_db()
    result = agent.run_agent(conn, CAR_ID, RUN_ID)

    assert result["rounds_used"] <= config.MAX_AGENT_ROUNDS
    assert result["rounds_used"] == 2  # propose, then confirm no-new-proposals
    assert result["transcript"]["effort"] == config.EFFORT_HIGH

    rows = conn.execute(
        "SELECT tool, input, output, accepted FROM tool_calls"
        " WHERE run_id = ? ORDER BY seq",
        [RUN_ID],
    ).fetchall()
    by_tool: dict[str, list] = {}
    for tool, tool_in, tool_out, accepted in rows:
        by_tool.setdefault(tool, []).append(
            (json.loads(tool_in), json.loads(tool_out), accepted)
        )

    # Every line normalized, every same-part pair matched, every matched
    # non-hidden line labor-checked, every proposal recorded.
    assert sorted(
        call[0]["line_no"] for call in by_tool["normalize_part_name"]
    ) == [1, 2, 3, 4, 5, 6, 7]
    assert sorted(
        (call[0]["line_no"], call[0]["crop_id"])
        for call in by_tool["match_damage"]
    ) == [
        (2, "crop-door"),
        (3, "crop-door"),
        (4, "crop-fender"),
        (5, "crop-headlight"),
        (7, "crop-roof"),
    ]
    assert sorted(
        call[0]["canonical_part"] for call in by_tool["check_labor_range"]
    ) == ["fender", "front door", "front door", "headlight", "mirror", "roof"]
    assert len(by_tool["record_flag"]) == 5
    assert all(call[2] for call in by_tool["record_flag"])

    # Each accepted flag traces to its accepting record_flag call.
    recorded = sorted(
        _flag_key(call[0]["flag"]) for call in by_tool["record_flag"]
    )
    assert recorded == sorted(_flag_key(f) for f in result["flags"])


def test_golden_feature_flags_gate_labor_and_repair(monkeypatch):
    conn = make_golden_db()
    monkeypatch.setattr(config, "FEATURE_LABOR_CHECK", False)
    monkeypatch.setattr(config, "FEATURE_REPAIR_REPLACE", False)
    result = agent.run_agent(conn, CAR_ID, RUN_ID)

    # Binding cut order (spec 14): labor off first, then repair-vs-replace;
    # missing/phantom/duplicate ship regardless.
    assert sorted(f["flag_type"] for f in result["flags"]) == [
        "duplicate",
        "missing_line",
        "phantom_line",
    ]
    labor_calls = conn.execute(
        "SELECT COUNT(*) FROM tool_calls"
        " WHERE run_id = ? AND tool = 'check_labor_range'",
        [RUN_ID],
    ).fetchone()[0]
    assert labor_calls == 0
    assert result["transcript"]["feature_flags"] == {
        "labor_check": False,
        "repair_replace": False,
    }


def test_golden_report_shape_and_order():
    conn = make_golden_db()
    agent.run_agent(conn, CAR_ID, RUN_ID)
    built = report.build_report(conn, CAR_ID, RUN_ID)

    assert built["car_id"] == CAR_ID
    assert built["run_id"] == RUN_ID
    assert built["duration_ms"] is not None and built["duration_ms"] >= 0
    assert built["api_cost_usd"] == 0.0
    assert built["unparsed_lines"] == 0
    assert built["counts"] == {
        "total": 5,
        "phantom_line": 1,
        "missing_line": 1,
        "repair_vs_replace": 1,
        "duplicate": 1,
        "labor_out_of_range": 1,
    }

    # Confidence desc with deterministic tie-breaks.
    assert [f["flag_type"] for f in built["flags"]] == [
        "duplicate",  # 0.98
        "labor_out_of_range",  # 0.90, tie-break by type
        "phantom_line",  # 0.90
        "missing_line",  # 0.85
        "repair_vs_replace",  # 0.80
    ]
    for flag in built["flags"]:
        assert flag["shop_questions"], flag["flag_type"]
    by_type = {f["flag_type"]: f for f in built["flags"]}
    assert "line 1" in by_type["phantom_line"]["shop_questions"][0]
    assert "no line for it" in by_type["missing_line"]["shop_questions"][0]
    assert "duplicate" in by_type["duplicate"]["shop_questions"][0]
    assert "Line 4" in by_type["repair_vs_replace"]["shop_questions"][0]
    assert "5.0 hours" in by_type["labor_out_of_range"]["shop_questions"][0]

    assert sorted(n["line_no"] for n in built["notes"]) == [6, 7]
