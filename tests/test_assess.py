"""assess.py tests: assessment flow, forcing rules, fallback (stubbed)."""

from __future__ import annotations

import json
import logging

from claimlens import assess, db


def _memdb():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def _seed(conn, car_id="car1", photo_id="p1", crop_id="c1", part="front bumper"):
    conn.execute("INSERT OR IGNORE INTO cars (car_id) VALUES (?)", [car_id])
    conn.execute(
        "INSERT OR IGNORE INTO photos"
        " (photo_id, car_id, photo_idx, width, height, img)"
        " VALUES (?, ?, 0, 10, 10, ?)",
        [photo_id, car_id, b"photo-bytes"],
    )
    conn.execute(
        "INSERT INTO crops (crop_id, photo_id, canonical_part,"
        " x0, y0, x1, y1, img) VALUES (?, ?, ?, 0, 0, 5, 5, ?)",
        [crop_id, photo_id, part, b"crop-bytes"],
    )


def _judgment(**overrides):
    data = {
        "photo_id": "p1",
        "crop_id": "c1",
        "canonical_part": "front bumper",
        "side": None,
        "damage_type": "dent",
        "severity": "moderate",
        "repair_replace": "repair",
        "likelihood": 0.8,
        "confidence": 0.9,
        "visibility": "visible",
    }
    data.update(overrides)
    return data


def test_assess_crop_happy_path_writes_row():
    conn = _memdb()
    _seed(conn)
    seen = []

    def stub(payload):
        seen.append(payload)
        assert payload["model"]  # model id comes from config, not hardcoded
        assert payload["effort"] == "low"
        assert len(payload["file_objs"]) == 2  # crop + parent photo context
        return {"data": _judgment()}

    out = assess.assess_crop(conn, "c1", transport=stub)
    assert out.visibility == "visible"
    assert out.damage_type == "dent"
    row = conn.execute(
        "SELECT assessment FROM damage_assessments WHERE crop_id = ?", ["c1"]
    ).fetchone()
    assert row is not None
    stored = json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
    assert stored["crop_id"] == "c1"
    assert stored["visibility"] == "visible"
    conn.close()


def test_low_confidence_forced_not_visible():
    conn = _memdb()
    _seed(conn)

    def stub(payload):
        return {"data": _judgment(confidence=0.1, visibility="visible")}

    out = assess.assess_crop(conn, "c1", transport=stub)
    assert out.visibility == "not_visible"
    assert out.confidence == 0.1  # only visibility is forced, never undamaged
    row = conn.execute(
        "SELECT assessment FROM damage_assessments WHERE crop_id = ?", ["c1"]
    ).fetchone()
    stored = json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
    assert stored["visibility"] == "not_visible"
    conn.close()


def test_schema_failure_records_not_visible_never_guesses(caplog):
    conn = _memdb()
    _seed(conn)
    calls = []

    def bad(payload):
        calls.append(payload)
        return {"data": {"nope": True}}

    with caplog.at_level(logging.WARNING, logger="claimlens.assess"):
        out = assess.assess_crop(conn, "c1", transport=bad)
    assert len(calls) == 2, "structured() retries once before giving up"
    assert out.visibility == "not_visible"
    assert out.confidence == 0.0
    assert out.damage_type == "unknown"
    assert out.crop_id == "c1"  # evidence pointer intact
    assert "not_visible" in caplog.text
    row = conn.execute(
        "SELECT assessment FROM damage_assessments WHERE crop_id = ?", ["c1"]
    ).fetchone()
    assert row is not None
    conn.close()


def test_assess_car_covers_every_crop():
    conn = _memdb()
    _seed(conn, crop_id="c1")
    _seed(conn, crop_id="c2")

    def stub(payload):
        return {"data": _judgment()}

    outs = assess.assess_car(conn, "car1", transport=stub)
    assert [o.crop_id for o in outs] == ["c1", "c2"]
    count = conn.execute("SELECT COUNT(*) FROM damage_assessments").fetchone()
    assert count[0] == 2
    conn.close()


def test_assess_crop_unknown_crop_raises():
    conn = _memdb()
    try:
        assess.assess_crop(conn, "missing", transport=lambda p: {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown crop_id")
    conn.close()
