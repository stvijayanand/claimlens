"""estimate.py tests: per-line validation, unparsed counting (stubbed)."""

from __future__ import annotations

import json
import logging

import pytest

from claimlens import db, estimate, meta_client


def _memdb():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def _seed(conn, car_id="car1", source_type="pdf", blob=b"%PDF-fake"):
    conn.execute("INSERT INTO cars (car_id) VALUES (?)", [car_id])
    conn.execute(
        "INSERT INTO estimate_files (car_id, source_type, blob)"
        " VALUES (?, ?, ?)",
        [car_id, source_type, blob],
    )


def _line(line_no, raw, operation, **overrides):
    data = {
        "line_no": line_no,
        "raw_part_text": raw,
        "canonical_part": None,
        "side": None,
        "operation": operation,
        "labor_hours": None,
        "price": None,
    }
    data.update(overrides)
    return data


def test_extract_valid_lines_writes_rows():
    conn = _memdb()
    _seed(conn)
    seen = []

    def stub(payload):
        seen.append(payload)
        assert payload["effort"] == "low"
        assert len(payload["file_objs"]) == 1  # file handling, no OCR step
        return {
            "data": {
                "lines": [
                    _line(1, "R&I frt bumper cover", "RI", labor_hours=1.5),
                    # A model-supplied canonical part must not stick: the
                    # agent fills it via normalize_part_name.
                    _line(
                        2,
                        "refinish hood",
                        "refinish",
                        canonical_part="hood",
                        labor_hours=2.4,
                    ),
                ]
            }
        }

    doc = estimate.extract_estimate(conn, "car1", transport=stub)
    assert len(doc.lines) == 2
    assert doc.unparsed_count == 0
    assert all(line.canonical_part is None for line in doc.lines)
    rows = conn.execute(
        "SELECT line_no, line FROM estimate_lines WHERE car_id = ?"
        " ORDER BY line_no",
        ["car1"],
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2]
    first = json.loads(rows[0][1]) if isinstance(rows[0][1], str) else rows[0][1]
    assert first["operation"] == "RI"
    assert first["labor_hours"] == 1.5
    conn.close()


def test_malformed_lines_counted_unparsed_never_guessed(caplog):
    conn = _memdb()
    _seed(conn)

    def stub(payload):
        return {
            "data": {
                "lines": [
                    _line(1, "replace headlamp assy", "replace"),
                    _line(2, "mystery line with no operation", None),
                    _line("abc", "line with bad number", "repair"),
                    {"unexpected": "shape entirely"},
                ]
            }
        }

    with caplog.at_level(logging.WARNING, logger="claimlens.estimate"):
        doc = estimate.extract_estimate(conn, "car1", transport=stub)
    assert [line.line_no for line in doc.lines] == [1]
    assert doc.unparsed_count == 3
    assert "mystery line with no operation" in caplog.text  # raw text logged
    rows = conn.execute(
        "SELECT COUNT(*) FROM estimate_lines WHERE car_id = ?", ["car1"]
    ).fetchone()
    assert rows[0] == 1
    conn.close()


def test_duplicate_line_no_second_counted_unparsed(caplog):
    conn = _memdb()
    _seed(conn)

    def stub(payload):
        return {
            "data": {
                "lines": [
                    _line(1, "repair fender", "repair"),
                    _line(1, "repair fender again", "repair"),
                ]
            }
        }

    with caplog.at_level(logging.WARNING, logger="claimlens.estimate"):
        doc = estimate.extract_estimate(conn, "car1", transport=stub)
    assert [line.raw_part_text for line in doc.lines] == ["repair fender"]
    assert doc.unparsed_count == 1
    assert "duplicate" in caplog.text
    conn.close()


def test_missing_estimate_file_raises():
    conn = _memdb()
    conn.execute("INSERT INTO cars (car_id) VALUES (?)", ["car1"])
    with pytest.raises(ValueError, match="no stored estimate"):
        estimate.extract_estimate(conn, "car1", transport=lambda p: {})
    conn.close()


def test_unparseable_whole_doc_propagates_schema_violation():
    conn = _memdb()
    _seed(conn)

    def bad(payload):
        return {"data": {"nope": True}}

    with pytest.raises(meta_client.SchemaViolation):
        estimate.extract_estimate(conn, "car1", transport=bad)
    conn.close()
