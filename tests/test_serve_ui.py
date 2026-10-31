"""Serve-layer UI tests: Streamlit views via AppTest (headless, no browser).

The app under test talks HTTP only; here ``httpx.Client`` is faked so no
server is needed. Replay rendering, feature-flag hiding, and upload gating
are covered; run polling is covered at the API level in test_serve_api.py.
"""

from __future__ import annotations

import io

import httpx
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

# Relative to THIS file: AppTest resolves from_file() against its caller.
APP = "../src/claimlens/ui/app.py"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 60)).save(buf, format="PNG")
    return buf.getvalue()


def _report() -> dict:
    return {
        "car_id": "car-ui-1",
        "run_id": "run-ui-1",
        "duration_ms": 937,
        "api_cost_usd": 0.02,
        "flags": [
            {
                "flag_type": "missing_line",
                "canonical_part": "hood",
                "reason": "hood shows moderate dent with no matching line.",
                "confidence": 0.85,
                "evidence_crop_ids": ["crop-hood-1"],
                "evidence_line_nos": [],
                "shop_questions": [
                    "The photos show moderate dent on the hood — why is"
                    " there no line for it?"
                ],
            },
            {
                "flag_type": "labor_out_of_range",
                "canonical_part": "headlight",
                "reason": "Line 5 bills 5.0h (typical 0.5-1.5h).",
                "confidence": 0.9,
                "evidence_crop_ids": [],
                "evidence_line_nos": [5],
                "shop_questions": ["Line 5 bills 5.0 hours — why?"],
            },
            {
                "flag_type": "repair_vs_replace",
                "canonical_part": "fender",
                "reason": "Line 4 says replace but reads repair.",
                "confidence": 0.8,
                "evidence_crop_ids": ["crop-fender-1"],
                "evidence_line_nos": [4],
                "shop_questions": ["Line 4 says replace — why?"],
            },
        ],
        "notes": [
            {
                "note_type": "needs_explanation",
                "line_no": 6,
                "canonical_part": "crash bar",
                "reason": "Line 6 covers crash bar, which cannot be"
                " photographed — ask the shop to explain it.",
            }
        ],
        "counts": {
            "total": 3,
            "phantom_line": 0,
            "missing_line": 1,
            "repair_vs_replace": 1,
            "duplicate": 0,
            "labor_out_of_range": 1,
        },
        "unparsed_lines": 0,
    }


class _FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "fake error", request=None, response=self  # type: ignore[arg-type]
            )
        return self


class _FakeClient:
    """httpx.Client stand-in serving one canned report + crop bytes."""

    #: Next run-status payload (None = serve the report as done).
    next_status: dict | None = None

    def __init__(self, *args, **kwargs):
        self.report = kwargs.pop("report", _report())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        if "/runs/" in url:
            if _FakeClient.next_status is not None:
                return _FakeResponse(payload=_FakeClient.next_status)
            return _FakeResponse(
                payload={
                    "run_id": "run-ui-1",
                    "car_id": "car-ui-1",
                    "status": "done",
                    "progress": 1.0,
                    "report": self.report,
                    "feature_flags": {
                        "labor_check": True,
                        "repair_replace": True,
                    },
                }
            )
        if url.endswith("/report"):
            return _FakeResponse(payload=self.report)
        if "/crops/" in url:
            return _FakeResponse(content=_png())
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, **kwargs):
        raise AssertionError(f"unexpected POST {url}")


@pytest.fixture()
def fake_http(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _FakeClient)


def _replay(at: AppTest, car_id: str = "car-ui-1") -> AppTest:
    at.text_input(key="replay_car_id").set_value(car_id)
    labels = [b.label for b in at.button]
    assert labels == ["Start analysis", "Load report"], labels
    at.button[1].click().run()
    at.run()  # process the st.rerun() into the report view
    assert not at.exception, [e.value for e in at.exception]
    return at


def _texts(elements) -> list[str]:
    return [str(e.value) for e in elements]


def test_replay_renders_full_report(fake_http):
    at = AppTest.from_file(APP)
    at.run()
    assert not at.exception
    _replay(at)

    assert at.session_state["page"] == "report"
    subs = _texts(at.subheader)
    assert any("Visible damage not on the estimate" in s for s in subs), subs
    assert any("hood" in s for s in subs), subs

    bodies = _texts(at.markdown)
    assert any("moderate dent" in b for b in bodies), bodies
    assert any("Line 5" in b for b in bodies), bodies  # cited line
    assert any("why is there no line for it" in b for b in bodies), bodies
    assert any("cannot be photographed" in b for b in bodies), bodies  # notes

    assert len(at.image) == 2  # one thumbnail per evidence crop
    captions = _texts(at.caption)
    assert any("Time: 0.9s" in c for c in captions), captions
    assert any("Estimated API cost: $0.0200" in c for c in captions), captions


def test_hidden_flag_types_stay_hidden(fake_http):
    at = AppTest.from_file(APP)
    at.run()
    at.session_state["feature_flags"] = {
        "labor_check": False,
        "repair_replace": False,
    }
    _replay(at)

    subs = _texts(at.subheader)
    assert any("Visible damage not on the estimate" in s for s in subs), subs
    assert not any("Labor hours" in s for s in subs), subs
    assert not any("Repair vs replace" in s for s in subs), subs
    assert len(at.image) == 1  # only the missing_line crop remains

    # Re-enabling the flags brings the hidden types back.
    at.session_state["feature_flags"] = {
        "labor_check": True,
        "repair_replace": True,
    }
    at.run()
    subs = _texts(at.subheader)
    assert any("Labor hours" in s for s in subs), subs
    assert any("Repair vs replace" in s for s in subs), subs


def test_progress_view_done_renders_report(fake_http):
    _FakeClient.next_status = None
    at = AppTest.from_file(APP)
    at.run()
    at.session_state["car_id"] = "car-ui-1"
    at.session_state["run_id"] = "run-ui-1"
    at.session_state["page"] = "progress"
    at.run()
    at.run()  # process the st.rerun() into the report view
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["page"] == "report"
    subs = _texts(at.subheader)
    assert any("hood" in s for s in subs), subs


def test_progress_view_error_shows_cause(fake_http):
    _FakeClient.next_status = {
        "run_id": "run-ui-1",
        "car_id": "car-ui-1",
        "status": "error",
        "progress": 1.0,
        "error": "run failed",
        "cause": "MetaClientError: API call failed: Connection error.",
        "feature_flags": {"labor_check": True, "repair_replace": True},
    }
    try:
        at = AppTest.from_file(APP)
        at.run()
        at.session_state["car_id"] = "car-ui-1"
        at.session_state["run_id"] = "run-ui-1"
        at.session_state["page"] = "progress"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        errors = _texts(at.error)
        assert any("Connection error" in e for e in errors), errors
    finally:
        _FakeClient.next_status = None


def test_upload_gating_needs_no_network(fake_http):
    at = AppTest.from_file(APP)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    errors = _texts(at.error)
    assert any("4-10 photos" in e for e in errors), errors
