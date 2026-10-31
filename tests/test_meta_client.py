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


def test_segment_live_requires_configuration(monkeypatch):
    monkeypatch.setattr(meta_client, "_client", None)
    monkeypatch.setattr(config, "META_MODEL_BASE_URL", "")
    monkeypatch.setattr(config, "META_MODEL_API_KEY", "")
    with pytest.raises(meta_client.MetaClientError):
        meta_client.segment(b"img", ["dent"])


def test_get_client_requires_configuration(monkeypatch):
    monkeypatch.setattr(meta_client, "_client", None)
    monkeypatch.setattr(config, "META_MODEL_BASE_URL", "")
    monkeypatch.setattr(config, "META_MODEL_API_KEY", "")
    with pytest.raises(meta_client.MetaClientError):
        meta_client.get_client()


# --- live SAM transport (recorded fixture, no network) ---

# Recorded 2026-10-31: model sam-3.1, phrase "car", one_bit encoding, four
# matches on a 637x419 photo (data/eval/real/real001/photo_1.png).
_SAM_CAR_FIXTURE = (
    "<0f>0<|box;x1=533;y1=90;x2=636;y2=197;w=637;h=419|>"
    "<|mask;x=0;y=0;data=108,104,!!!!!cBbz+ga69X@3O[Ld@/kFzw)ZjsE#PBvH!qyt)"
    "iWbS7PN3.HR5Ytl2(DX%'Hm'dm+0r/.GJ-CkvHD|>,"
    "1<|box;x1=0;y1=74;x2=17;y2=87;w=637;h=419|>"
    "<|mask;x=0;y=0;data=14,18,!!!!!2de!HW'm2o`crjRkk.YN|>,"
    "2<|box;x1=35;y1=74;x2=76;y2=89;w=637;h=419|>"
    "<|mask;x=0;y=0;data=16,42,!!!!!AH30S#gRC.A(C?K&^fn04ndo7tUMTDtx.TW|>,"
    "3<|box;x1=68;y1=88;x2=610;y2=286;w=637;h=419|>"
    "<|mask;x=0;y=0;data=199,543,!!!!$j@p-s9!lOkxDQ]k5dmN%.%kZd&1#4!JJh]DLAh"
    "KxIst/6T?0T*1v(Cf1s$OL9@3H!?56=VmDU@vyroeptu?[c$UXz3$42-E*FQrM%o6%w-&h"
    "@JG`DWd)()GZ&V-7bJ08!zwh&l_.dt8U92(D-l.%wua*J^R(f4=oWA%i]Kt^:3tjL*l%wp"
    "]V%Q{O'9BtZBp#VA3:g5qzK)Mt9NZ]IVq*_KL$g/qL^^uUx#rkl*`hS-eguI!4:hf/-AK8"
    "`#nfttmH^f6]?sjhxh+[FQ$jH0cbH%q3vmSrYY1.xS.k?bqH!!|>"
)


class _FakeUsage:
    def model_dump(self):
        return {"images_processed": 1}


class _FakeSAMResponse:
    def __init__(self, text):
        self.output_text = text
        self.usage = _FakeUsage()


class _FakeResponses:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSAMResponse(self.texts.pop(0))


class _FakeClient:
    def __init__(self, texts):
        self.responses = _FakeResponses(texts)


def test_segment_live_parses_recorded_response(monkeypatch):
    client = _FakeClient([_SAM_CAR_FIXTURE])
    monkeypatch.setattr(meta_client, "_client", client)
    seen = []
    out = meta_client.segment(
        b"\x89PNG\r\n\x1a\nfakepng", ["car"], on_usage=seen.append
    )
    # One request per concept phrase, with the documented request shape.
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["model"] == config.SAM_MODEL
    assert call["metadata"] == {"mask_encoding": "one_bit"}
    content = call["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "car"}
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    # Four records, labeled with the request phrase, boxes in source pixels
    # (wire x2/y2 are inclusive; the parser converts to exclusive ends).
    assert [(m.label, m.x0, m.y0, m.x1, m.y1) for m in out] == [
        ("car", 533, 90, 637, 198),
        ("car", 0, 74, 18, 88),
        ("car", 35, 74, 77, 90),
        ("car", 68, 88, 611, 287),
    ]
    for mask in out:
        assert len(mask.mask) == (mask.x1 - mask.x0) * (mask.y1 - mask.y0)
        assert any(mask.mask)  # real foreground, not an empty raster
    assert seen == [
        {
            "kind": "segment",
            "model": config.SAM_MODEL,
            "cached": False,
            "usage": {"images_processed": 1},
        }
    ]


def test_segment_live_one_request_per_phrase(monkeypatch):
    client = _FakeClient(["", _SAM_CAR_FIXTURE])
    monkeypatch.setattr(meta_client, "_client", client)
    seen = []
    out = meta_client.segment(
        b"\xff\xd8fakejpeg", ["dent", "car"], on_usage=seen.append
    )
    assert [c["input"][0]["content"][0]["text"] for c in client.responses.calls] == [
        "dent",
        "car",
    ]
    assert client.responses.calls[0]["input"][0]["content"][1][
        "image_url"
    ].startswith("data:image/jpeg;base64,")
    # "dent" matched nothing (empty lane is valid); "car" matched four.
    assert [m.label for m in out] == ["car"] * 4
    assert len(seen) == 2  # every request bills, matches or not
    assert all(e["kind"] == "segment" and not e["cached"] for e in seen)


def test_segment_live_api_error_surfaces_as_meta_client_error(monkeypatch):
    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    client = _FakeClient([])
    client.responses = _Boom()
    monkeypatch.setattr(meta_client, "_client", client)
    with pytest.raises(meta_client.MetaClientError, match="SAM call failed"):
        meta_client.segment(b"img", ["dent"])


def test_segment_stub_transport_emits_no_usage(monkeypatch):
    seen = []
    out = meta_client.segment(
        b"img",
        ["dent"],
        transport=lambda payload: {"masks": []},
        on_usage=seen.append,
    )
    assert out == []
    assert seen == []


def test_raster_to_box_mask_resizes_to_box():
    raster = bytes([1, 0, 0, 1])  # 2x2
    out = meta_client._raster_to_box_mask(raster, 2, 2, 4, 4)
    assert len(out) == 16
    assert out == bytes(
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    )
