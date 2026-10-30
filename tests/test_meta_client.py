"""meta_client tests: validation, cache, retry, hooks (stubbed transport)."""

import logging

import pytest

from claimlens import config, db, meta_client
from claimlens.models import Flag

_MSGS = [{"role": "user", "content": "judge this"}]


def _memdb():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def _flag_data():
    return {
        "flag_type": "phantom_line",
        "canonical_part": "hood",
        "reason": "no damage visible",
        "confidence": 0.9,
        "evidence_crop_ids": ["c1"],
        "evidence_line_nos": [],
        "shop_questions": ["Can you show me the damage behind this line?"],
    }


def test_structured_validates_and_caches(caplog):
    calls = []

    def stub(payload):
        calls.append(payload)
        return {"data": _flag_data(), "usage": {"x": 1}}

    conn = _memdb()
    kwargs = dict(
        kind="flag",
        response_model=Flag,
        model="test-model",
        effort="low",
        messages=_MSGS,
        conn=conn,
        transport=stub,
    )
    with caplog.at_level(logging.INFO, logger="claimlens.meta_client"):
        first = meta_client.structured(**kwargs)
    assert isinstance(first, Flag)
    assert first.canonical_part == "hood"
    with caplog.at_level(logging.INFO, logger="claimlens.meta_client"):
        second = meta_client.structured(**kwargs)
    assert len(calls) == 1, "second identical call must be a cache hit"
    assert second == first
    assert "cache MISS" in caplog.text
    assert "cache HIT" in caplog.text
    conn.close()


def test_structured_schema_violation_after_failed_retry():
    calls = []

    def bad(payload):
        calls.append(payload)
        return {"data": {"nope": True}}

    with pytest.raises(meta_client.SchemaViolation):
        meta_client.structured(
            kind="flag",
            response_model=Flag,
            model="test-model",
            effort="low",
            messages=_MSGS,
            conn=_memdb(),
            transport=bad,
        )
    assert len(calls) == 2, "initial attempt plus one schema-repair retry"


def test_structured_repair_retry_can_succeed():
    responses = [{"data": {"nope": True}}, {"data": _flag_data()}]

    def flaky(payload):
        return responses.pop(0)

    obj = meta_client.structured(
        kind="flag",
        response_model=Flag,
        model="test-model",
        effort="low",
        messages=_MSGS,
        conn=_memdb(),
        transport=flaky,
    )
    assert obj.confidence == 0.9


def test_structured_usage_hook():
    seen = []

    def stub(payload):
        return {"data": _flag_data(), "usage": {"prompt": 5}}

    conn = _memdb()
    meta_client.structured(
        kind="flag",
        response_model=Flag,
        model="test-model",
        effort="low",
        messages=_MSGS,
        conn=conn,
        transport=stub,
        on_usage=seen.append,
    )
    meta_client.structured(
        kind="flag",
        response_model=Flag,
        model="test-model",
        effort="low",
        messages=_MSGS,
        conn=conn,
        transport=stub,
        on_usage=seen.append,
    )
    assert [s["cached"] for s in seen] == [False, True]
    assert seen[0]["usage"] == {"prompt": 5}
    conn.close()


def test_strict_schema_normalizes_all_objects():
    from claimlens.models import EstimateDoc

    schema = meta_client._strict_schema(EstimateDoc.model_json_schema())

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node["additionalProperties"] is False
                assert node["required"] == sorted(node["properties"])
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    check(schema)
    # $defs nesting is covered: EstimateLine lives there
    assert "EstimateLine" in schema["$defs"]


def test_segment_with_stub_transport():
    def stub(payload):
        assert payload["phrases"] == ["dent"]
        return {
            "masks": [
                {
                    "label": "dent",
                    "x0": 1,
                    "y0": 2,
                    "x1": 3,
                    "y1": 4,
                    "mask": b"m",
                }
            ]
        }

    out = meta_client.segment(b"img", ["dent"], transport=stub)
    assert len(out) == 1
    assert (out[0].label, out[0].x1, out[0].mask) == ("dent", 3, b"m")


def test_segment_default_transport_points_to_live_gate():
    with pytest.raises(NotImplementedError, match="Issue #1 gate"):
        meta_client.segment(b"img", ["dent"])


def test_get_client_requires_configuration(monkeypatch):
    monkeypatch.setattr(meta_client, "_client", None)
    monkeypatch.setattr(config, "META_MODEL_BASE_URL", "")
    monkeypatch.setattr(config, "META_MODEL_API_KEY", "")
    with pytest.raises(meta_client.MetaClientError):
        meta_client.get_client()
