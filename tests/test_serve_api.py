"""Serve-layer tests: the seven API endpoints + run orchestration.

All tests run against a temp file database (``api.DB_PATH`` monkeypatched)
through ``fastapi.testclient.TestClient``. Model calls are stubbed via
transports or by monkeypatching ``api.PIPELINE`` -- no network, no API key.
"""

from __future__ import annotations

import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from claimlens import api, config, db, meta_client

MIN_PHOTOS = 4


@pytest.fixture()
def db_file(tmp_path, monkeypatch):
    """Point the API at an empty temp database with clean run state."""
    target = tmp_path / "serve-test.duckdb"
    monkeypatch.setattr(api, "DB_PATH", target)
    monkeypatch.delenv("CLAIMLENS_DB_PATH", raising=False)
    with api._RUNS_LOCK:
        api._RUNS.clear()
    yield target
    with api._RUNS_LOCK:
        api._RUNS.clear()


@pytest.fixture()
def client(db_file):
    return TestClient(api.app)


def _png(color=(200, 30, 30), size=(24, 24)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _photo_files(n, prefix="p"):
    return [
        ("files", (f"{prefix}{i}.png", _png(), "image/png")) for i in range(n)
    ]


def _make_car(client, n_photos=MIN_PHOTOS, with_estimate=True):
    car_id = client.post("/cars").json()["car_id"]
    if n_photos:
        resp = client.post(
            f"/cars/{car_id}/photos", files=_photo_files(n_photos)
        )
        assert resp.status_code == 200, resp.text
    if with_estimate:
        resp = client.post(
            f"/cars/{car_id}/estimate",
            files={"file": ("est.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 200, resp.text
    return car_id


def _poll_done(client, car_id, run_id, timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/cars/{car_id}/runs/{run_id}")
        assert last.status_code == 200, last.text
        if last.json()["status"] in {"done", "error"}:
            return last.json()
        time.sleep(0.05)
    raise AssertionError(f"run never finished: {last.json() if last else None}")


# --- POST /cars ------------------------------------------------------------


def test_create_car_201(client):
    resp = client.post("/cars")
    assert resp.status_code == 201
    assert resp.json()["car_id"].startswith("car-")


# --- POST /cars/{id}/photos ------------------------------------------------


def test_upload_photos_happy_path_stores_dims(client, db_file):
    car_id = client.post("/cars").json()["car_id"]
    resp = client.post(f"/cars/{car_id}/photos", files=_photo_files(4))
    assert resp.status_code == 200
    photo_ids = resp.json()["photo_ids"]
    assert len(photo_ids) == 4

    conn = db.connect(db_file)
    try:
        rows = conn.execute(
            "SELECT width, height FROM photos WHERE car_id = ? ORDER BY photo_idx",
            [car_id],
        ).fetchall()
    finally:
        conn.close()
    assert [(w, h) for w, h in rows] == [(24, 24)] * 4


def test_upload_photos_unknown_car_404_with_detail_shape(client):
    resp = client.post("/cars/nope/photos", files=_photo_files(1))
    assert resp.status_code == 404
    assert "unknown car_id" in resp.json()["detail"]


def test_upload_photos_rejects_bad_type(client):
    car_id = client.post("/cars").json()["car_id"]
    resp = client.post(
        f"/cars/{car_id}/photos",
        files=[("files", ("note.txt", b"hello", "text/plain"))],
    )
    assert resp.status_code == 422


def test_upload_photos_rejects_unreadable_image(client):
    car_id = client.post("/cars").json()["car_id"]
    resp = client.post(
        f"/cars/{car_id}/photos",
        files=[("files", ("fake.png", b"not-an-image", "image/png"))],
    )
    assert resp.status_code == 422


def test_upload_photos_rejects_oversize(client):
    car_id = client.post("/cars").json()["car_id"]
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024)
    resp = client.post(
        f"/cars/{car_id}/photos",
        files=[("files", ("big.png", big, "image/png"))],
    )
    assert resp.status_code == 422


def test_upload_photos_rejects_total_above_ten(client):
    car_id = _make_car(client, n_photos=0, with_estimate=False)
    first = client.post(f"/cars/{car_id}/photos", files=_photo_files(10))
    assert first.status_code == 200
    resp = client.post(f"/cars/{car_id}/photos", files=_photo_files(1))
    assert resp.status_code == 422
    assert "exceeds" in resp.json()["detail"]


def test_upload_photos_allows_incremental_build(client):
    car_id = client.post("/cars").json()["car_id"]
    assert (
        client.post(f"/cars/{car_id}/photos", files=_photo_files(2)).status_code
        == 200
    )
    resp = client.post(f"/cars/{car_id}/photos", files=_photo_files(2, prefix="q"))
    assert resp.status_code == 200
    assert len(resp.json()["photo_ids"]) == 2


# --- POST /cars/{id}/estimate ----------------------------------------------


def test_upload_estimate_pdf_and_image(client, db_file):
    car_id = client.post("/cars").json()["car_id"]
    resp = client.post(
        f"/cars/{car_id}/estimate",
        files={"file": ("est.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"source_type": "pdf"}

    # Re-upload replaces the stored file.
    resp = client.post(
        f"/cars/{car_id}/estimate",
        files={"file": ("est.png", _png(), "image/png")},
    )
    assert resp.json() == {"source_type": "image"}
    conn = db.connect(db_file)
    try:
        rows = conn.execute(
            "SELECT source_type FROM estimate_files WHERE car_id = ?", [car_id]
        ).fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == ["image"]


def test_upload_estimate_rejects_bad_type(client):
    car_id = client.post("/cars").json()["car_id"]
    resp = client.post(
        f"/cars/{car_id}/estimate",
        files={"file": ("est.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 422


def test_upload_estimate_unknown_car_404(client):
    resp = client.post(
        "/cars/nope/estimate",
        files={"file": ("est.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 404


# --- POST /cars/{id}/run ----------------------------------------------------


def test_start_run_unknown_car_404(client):
    resp = client.post("/cars/nope/run")
    assert resp.status_code == 404


def test_start_run_rejects_too_few_photos(client):
    car_id = _make_car(client, n_photos=3, with_estimate=True)
    resp = client.post(f"/cars/{car_id}/run")
    assert resp.status_code == 422
    assert "photos" in resp.json()["detail"]


def test_start_run_rejects_missing_estimate(client):
    car_id = _make_car(client, n_photos=4, with_estimate=False)
    resp = client.post(f"/cars/{car_id}/run")
    assert resp.status_code == 422
    assert "estimate" in resp.json()["detail"]


def _fake_pipeline_factory(db_path, flag=None):
    def fake_pipeline(car_id, run_id):
        from claimlens import report as report_mod

        conn = db.connect(db_path)
        try:
            db.init_schema(conn)
            conn.execute(
                "UPDATE runs SET api_cost_usd = ?, note = ? WHERE run_id = ?",
                [
                    0.01,
                    json.dumps(
                        {
                            "agent": "test-fake",
                            "effort": "high",
                            "notes": [],
                            "feature_flags": {},
                        }
                    ),
                    run_id,
                ],
            )
            if flag is not None:
                conn.execute(
                    "INSERT INTO flags (flag_id, car_id, run_id, flag)"
                    " VALUES (?, ?, ?, ?)",
                    [f"flag-{run_id}", car_id, run_id, json.dumps(flag)],
                )
            return report_mod.build_report(conn, car_id, run_id)
        finally:
            conn.close()

    return fake_pipeline


def test_run_success_reports_done_with_report(client, db_file, monkeypatch):
    flag = {
        "flag_type": "phantom_line",
        "canonical_part": "mirror",
        "reason": "Estimate line 1 ('mirror') has no matching visible damage.",
        "confidence": 0.9,
        "evidence_crop_ids": [],
        "evidence_line_nos": [1],
        "shop_questions": [],
    }
    monkeypatch.setattr(api, "PIPELINE", _fake_pipeline_factory(db_file, flag))
    car_id = _make_car(client)
    conn = db.connect(db_file)
    try:
        conn.execute(
            "INSERT INTO estimate_lines (car_id, line_no, line) VALUES (?, ?, ?)",
            [
                car_id,
                1,
                json.dumps(
                    {
                        "line_no": 1,
                        "raw_part_text": "mirror",
                        "canonical_part": "mirror",
                        "side": None,
                        "operation": "replace",
                        "labor_hours": None,
                        "price": None,
                    }
                ),
            ],
        )
    finally:
        conn.close()

    resp = client.post(f"/cars/{car_id}/run")
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    payload = _poll_done(client, car_id, run_id)
    assert payload["status"] == "done"
    assert payload["progress"] == 1.0
    assert payload["report"]["car_id"] == car_id
    assert payload["report"]["run_id"] == run_id
    assert payload["report"]["counts"]["phantom_line"] == 1
    assert payload["feature_flags"] == {
        "labor_check": bool(config.FEATURE_LABOR_CHECK),
        "repair_replace": bool(config.FEATURE_REPAIR_REPLACE),
    }

    # Latest-successful-report endpoint serves the same run.
    got = client.get(f"/cars/{car_id}/report")
    assert got.status_code == 200
    assert got.json()["run_id"] == run_id


def test_run_failure_surfaces_error_with_cause_never_partial(
    client, db_file, monkeypatch
):
    def boom(car_id, run_id):
        raise RuntimeError("sam exploded")

    monkeypatch.setattr(api, "PIPELINE", boom)
    car_id = _make_car(client)
    run_id = client.post(f"/cars/{car_id}/run").json()["run_id"]

    payload = _poll_done(client, car_id, run_id)
    assert payload["status"] == "error"
    assert "sam exploded" in payload["cause"]

    # No successful run exists, so no partial silent report is served.
    got = client.get(f"/cars/{car_id}/report")
    assert got.status_code == 404

    # The error cause persists in the runs row (restart-safe).
    conn = db.connect(db_file)
    try:
        row = conn.execute(
            "SELECT note FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
    finally:
        conn.close()
    assert "sam exploded" in row[0]


def test_run_status_unknown_run_404(client):
    car_id = _make_car(client)
    resp = client.get(f"/cars/{car_id}/runs/run-missing")
    assert resp.status_code == 404


def test_run_status_wrong_car_404(client):
    car_id = _make_car(client)
    other = client.post("/cars").json()["car_id"]
    resp = client.get(f"/cars/{other}/runs/run-missing")
    assert resp.status_code == 404
    assert car_id  # silence unused-var; car_id anchors the fixture car


def test_status_and_report_survive_restart_and_need_no_key(
    client, db_file, monkeypatch
):
    """Simulated restart: memory cleared + API key removed, report renders."""
    monkeypatch.setattr(api, "PIPELINE", _fake_pipeline_factory(db_file))
    car_id = _make_car(client)
    run_id = client.post(f"/cars/{car_id}/run").json()["run_id"]
    assert _poll_done(client, car_id, run_id)["status"] == "done"

    # Simulate a fresh server process with no API key configured.
    with api._RUNS_LOCK:
        api._RUNS.clear()
    monkeypatch.setattr(config, "META_MODEL_API_KEY", "")

    def no_network():
        raise AssertionError("report path must not call the model API")

    monkeypatch.setattr(meta_client, "get_client", no_network)

    status = client.get(f"/cars/{car_id}/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "done"
    assert status.json()["report"]["run_id"] == run_id

    got = client.get(f"/cars/{car_id}/report")
    assert got.status_code == 200
    assert got.json()["run_id"] == run_id


def test_incomplete_run_row_reads_as_error_never_partial(client, db_file):
    """A run row with no transcript (e.g. restart mid-run) is an error."""
    car_id = _make_car(client)
    conn = db.connect(db_file)
    try:
        conn.execute(
            "INSERT INTO runs (run_id, car_id) VALUES (?, ?)",
            ["run-abandoned", car_id],
        )
    finally:
        conn.close()
    payload = client.get(f"/cars/{car_id}/runs/run-abandoned").json()
    assert payload["status"] == "error"
    assert payload["cause"]
    assert client.get(f"/cars/{car_id}/report").status_code == 404


def test_run_status_polls_while_pipeline_holds_lock(
    client, db_file, monkeypatch
):
    """Progress polls answer from memory while the run holds the DB lock."""
    entered = threading.Event()
    release = threading.Event()

    def slow_pipeline(car_id, run_id):
        with api._DB_LOCK:
            entered.set()
            assert release.wait(timeout=30)

    monkeypatch.setattr(api, "PIPELINE", slow_pipeline)
    car_id = _make_car(client)
    run_id = client.post(f"/cars/{car_id}/run").json()["run_id"]
    assert entered.wait(timeout=30)
    got: dict = {}
    errors: list = []

    def poll():
        try:
            got.update(api.get_run(car_id, run_id))
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    worker = threading.Thread(target=poll, daemon=True)
    worker.start()
    worker.join(timeout=10)
    release.set()
    assert not worker.is_alive(), "status poll blocked behind the lock"
    assert not errors
    assert got["status"] in ("queued", "running")
    assert 0.0 <= got["progress"] <= 1.0


# --- GET /cars/{id}/report --------------------------------------------------


def test_report_404_without_successful_run(client):
    car_id = _make_car(client)
    resp = client.get(f"/cars/{car_id}/report")
    assert resp.status_code == 404


def test_report_unknown_car_404(client):
    assert client.get("/cars/nope/report").status_code == 404


# --- GET /crops/{crop_id} ----------------------------------------------------


def test_get_crop_bytes_and_unknown_404(client, db_file):
    car_id = _make_car(client, n_photos=1, with_estimate=False)
    blob = _png()
    conn = db.connect(db_file)
    try:
        photo_id = conn.execute(
            "SELECT photo_id FROM photos WHERE car_id = ?", [car_id]
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO crops (crop_id, photo_id, canonical_part,"
            " x0, y0, x1, y1, img) VALUES (?, ?, ?, 0, 0, 5, 5, ?)",
            ["crop-1", photo_id, "hood", blob],
        )
    finally:
        conn.close()

    resp = client.get("/crops/crop-1")
    assert resp.status_code == 200
    assert resp.content == blob
    assert resp.headers["content-type"] == "image/png"
    assert client.get("/crops/crop-missing").status_code == 404


def test_post_run_crop_serves_pipeline_bytes(client, db_file, monkeypatch):
    """Crops written by a real pipeline run serve back their bytes.

    Regression guard: real crop ids (spaces/colons from part labels and
    boxes) and real ``make_crops`` PNG bytes must serve exactly like the
    synthetic row above.
    """

    def pipeline(car_id, run_id):
        return api.run_pipeline(
            car_id,
            run_id,
            listing_transport=_listing_stub,
            segment_transport=_segment_stub,
            assess_transport=_assess_stub,
            extract_transport=_extract_stub,
        )

    monkeypatch.setattr(api, "PIPELINE", pipeline)
    car_id = _make_car(client)
    run_id = client.post(f"/cars/{car_id}/run").json()["run_id"]
    status = None
    for _ in range(240):
        status = client.get(f"/cars/{car_id}/runs/{run_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert status["status"] == "done"
    conn = db.connect(db_file)
    try:
        rows = conn.execute("SELECT crop_id, img FROM crops").fetchall()
    finally:
        conn.close()
    assert rows, "pipeline produced no crops to serve"
    from urllib.parse import quote

    for crop_id, blob in rows:
        resp = client.get("/crops/" + quote(crop_id, safe=""))
        assert resp.status_code == 200
        assert resp.content == bytes(blob)
        assert resp.headers["content-type"] == "image/png"


# --- run_pipeline with stub transports (real stages, no network) -------------


def _listing_stub(payload):
    assert payload["effort"] == "low"
    photo_id = payload["messages"][1]["content"].split("photo ")[1].split(".")[0]
    return {
        "data": {
            "photo_id": photo_id,
            "parts": [
                {"canonical_part": "hood", "side": None, "confidence": 0.9}
            ],
        }
    }


def _segment_stub(payload):
    from claimlens.meta_client import SegmentMask  # noqa: F401 (re-export check)

    phrases = payload["phrases"]
    if set(phrases) == set(config.DAMAGE_PHRASES):
        return {
            "masks": [
                {
                    "label": "dent",
                    "x0": 5,
                    "y0": 5,
                    "x1": 15,
                    "y1": 15,
                    "mask": b"\x01" * 100,
                }
            ]
        }
    return {
        "masks": [
            {
                "label": phrase,
                "x0": 2,
                "y0": 2,
                "x1": 12,
                "y1": 12,
                "mask": b"\x01" * 100,
            }
            for phrase in phrases
        ]
    }


def _assess_stub(payload):
    return {
        "data": {
            "photo_id": "p",
            "crop_id": "c",
            "canonical_part": "hood",
            "side": None,
            "damage_type": "dent",
            "severity": "moderate",
            "repair_replace": "repair",
            "likelihood": 0.8,
            "confidence": 0.9,
            "visibility": "visible",
        }
    }


def _extract_stub(payload):
    return {
        "data": {
            "lines": [
                {
                    "line_no": 1,
                    "raw_part_text": "hood",
                    "canonical_part": None,
                    "side": None,
                    "operation": "repair",
                    "labor_hours": None,
                    "price": None,
                },
                {
                    "line_no": 2,
                    "raw_part_text": "front bumper",
                    "canonical_part": None,
                    "side": None,
                    "operation": "replace",
                    "labor_hours": 1.5,
                    "price": None,
                },
            ]
        }
    }


def test_run_pipeline_stubbed_transports_yields_phantom_flag(
    client, db_file
):
    """Full stage chain (vision->report) with stubbed model calls."""
    car_id = _make_car(client)
    conn = db.connect(db_file)
    try:
        conn.execute(
            "INSERT INTO runs (run_id, car_id) VALUES (?, ?)",
            ["run-stubbed", car_id],
        )
    finally:
        conn.close()

    built = api.run_pipeline(
        car_id,
        "run-stubbed",
        listing_transport=_listing_stub,
        segment_transport=_segment_stub,
        assess_transport=_assess_stub,
        extract_transport=_extract_stub,
    )
    assert built["car_id"] == car_id
    assert built["run_id"] == "run-stubbed"
    assert built["duration_ms"] is not None and built["duration_ms"] >= 0
    assert built["api_cost_usd"] is not None and built["api_cost_usd"] >= 0
    # Line 1 (hood) matches the hood crops; line 2 (bumper) is phantom.
    assert built["counts"]["total"] == 1
    assert built["flags"][0]["flag_type"] == "phantom_line"
    assert built["flags"][0]["shop_questions"]

    # The served report matches the pipeline output.
    served = client.get(f"/cars/{car_id}/report")
    assert served.status_code == 200
    assert served.json()["counts"]["total"] == 1


def test_run_pipeline_without_segment_transport_raises(client, db_file, monkeypatch):
    """Live SAM without credentials fails fast: the failure must propagate."""
    monkeypatch.setattr(meta_client, "_client", None)
    monkeypatch.setattr(config, "META_MODEL_BASE_URL", "")
    monkeypatch.setattr(config, "META_MODEL_API_KEY", "")
    car_id = _make_car(client)
    conn = db.connect(db_file)
    try:
        conn.execute(
            "INSERT INTO runs (run_id, car_id) VALUES (?, ?)",
            ["run-nosam", car_id],
        )
    finally:
        conn.close()
    with pytest.raises(meta_client.MetaClientError):
        api.run_pipeline(
            car_id,
            "run-nosam",
            listing_transport=_listing_stub,
            segment_transport=None,
            assess_transport=_assess_stub,
            extract_transport=_extract_stub,
        )
