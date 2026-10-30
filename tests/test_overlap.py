"""Vision pipeline tests: overlap rule, SAM call budget, DB rows (stubbed)."""

from __future__ import annotations

import io

from PIL import Image

from claimlens import config, db, taxonomy, vision
from claimlens.meta_client import SegmentMask

_CAR = "car-overlap-1"


def _mask(
    label: str, x0: int, y0: int, x1: int, y1: int, fill: int = 1
) -> SegmentMask:
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return SegmentMask(
        label=label, x0=x0, y0=y0, x1=x1, y1=y1, mask=bytes([fill]) * area
    )


def _photo_bytes(width: int = 100, height: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


def _memdb_with_photos(photo_ids: list[str]):
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO cars (car_id) VALUES (?)", [_CAR])
    img = _photo_bytes()
    for idx, pid in enumerate(photo_ids):
        conn.execute(
            "INSERT INTO photos (photo_id, car_id, photo_idx, width, height, img)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [pid, _CAR, idx, 100, 80, img],
        )
    return conn


def _listing_stub(parts: list[dict]):
    def stub(payload):
        stub.payloads.append(payload)
        return {
            "data": {"photo_id": "ignored", "parts": parts},
            "usage": {"stub": 1},
        }

    stub.payloads = []
    return stub


def _segment_stub(calls: list, part_box=(10, 10, 50, 50), dmg_box=(40, 40, 70, 60)):
    """Stub SAM: part batch echoes one full mask per phrase; damage batch one dent."""

    def stub(payload):
        calls.append(payload)
        phrases = payload["phrases"]
        if phrases == list(config.DAMAGE_PHRASES):
            x0, y0, x1, y1 = dmg_box
            masks = [
                {
                    "label": "dent",
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "mask": bytes([1]) * (x1 - x0) * (y1 - y0),
                }
            ]
        else:
            masks = []
            for phrase in phrases:
                x0, y0, x1, y1 = part_box
                masks.append(
                    {
                        "label": phrase,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "mask": bytes([1]) * (x1 - x0) * (y1 - y0),
                    }
                )
        return {"masks": masks}

    return stub


# --- is_damaged pure-rule cases ---


def test_is_damaged_identical_masks():
    part = _mask("hood", 5, 5, 15, 15)
    damage = _mask("dent", 5, 5, 15, 15)
    assert vision.is_damaged(part, damage) is True


def test_is_damaged_disjoint_masks():
    part = _mask("hood", 0, 0, 10, 10)
    damage = _mask("dent", 20, 20, 30, 30)
    assert vision.is_damaged(part, damage) is False


def test_is_damaged_touching_edges_not_damaged():
    part = _mask("hood", 0, 0, 10, 10)
    damage = _mask("dent", 10, 0, 20, 10)  # shares only the x=10 edge
    assert vision.is_damaged(part, damage) is False


def test_is_damaged_empty_mask_never_damaged():
    part = _mask("hood", 0, 0, 10, 10)
    all_bg = _mask("dent", 0, 0, 10, 10, fill=0)
    zero_len = SegmentMask(
        label="dent", x0=0, y0=0, x1=10, y1=10, mask=b""
    )
    assert vision.is_damaged(part, all_bg) is False
    assert vision.is_damaged(part, zero_len) is False
    assert vision.is_damaged(all_bg, part) is False


def test_is_damaged_box_overlap_without_shared_pixels():
    # Boxes overlap on x in [5, 10), but foreground pixels never coincide.
    part = SegmentMask(
        label="hood", x0=0, y0=0, x1=10, y1=1,
        mask=bytes([1] * 5 + [0] * 5),
    )
    damage = SegmentMask(
        label="dent", x0=5, y0=0, x1=15, y1=1,
        mask=bytes([0] * 5 + [1] * 5),
    )
    assert vision.is_damaged(part, damage) is False


def test_is_damaged_respects_min_intersect_threshold():
    part = _mask("hood", 0, 0, 10, 10)
    damage = SegmentMask(
        label="dent", x0=5, y0=5, x1=15, y1=15,
        mask=bytes([1] + [0] * 99),  # exactly 1 shared pixel at (5, 5)
    )
    assert vision.is_damaged(part, damage, min_intersect_px=1) is True
    assert vision.is_damaged(part, damage, min_intersect_px=2) is False


# --- listing: taxonomy block + confidence filter ---


def test_list_visible_parts_attaches_taxonomy_and_filters():
    conn = _memdb_with_photos(["p1"])
    stub = _listing_stub(
        [
            {"canonical_part": "hood", "side": None, "confidence": 0.9},
            {"canonical_part": "roof", "side": None, "confidence": 0.1},
        ]
    )
    visible = vision.list_visible_parts(conn, "p1", transport=stub)
    assert [p.canonical_part for p in visible.parts] == ["hood"]
    user_text = stub.payloads[0]["messages"][1]["content"]
    assert taxonomy.prompt_block() in user_text
    rows = conn.execute(
        "SELECT canonical_part, confidence FROM visible_parts"
        " WHERE photo_id = 'p1'"
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("hood", 0.9)]
    conn.close()


# --- SAM call budget ---


def test_segment_call_count_bound_per_car():
    photo_ids = ["p1", "p2", "p3"]
    conn = _memdb_with_photos(photo_ids)
    listing = _listing_stub(
        [{"canonical_part": "front bumper", "side": None, "confidence": 0.8}]
    )
    calls: list = []
    segment = _segment_stub(calls)
    for pid in photo_ids:
        vision.process_photo(
            conn, pid, list_transport=listing, segment_transport=segment
        )
    assert len(calls) <= len(photo_ids) * 2
    assert len(calls) == len(photo_ids) * 2  # exactly one part + one damage batch
    for pid_calls in (calls[i : i + 2] for i in range(0, len(calls), 2)):
        assert pid_calls[0]["phrases"] == ["front bumper"]
        assert pid_calls[1]["phrases"] == list(config.DAMAGE_PHRASES)
    conn.close()


# --- rows written to :memory: DuckDB ---


def test_pipeline_writes_masks_visible_parts_and_crops():
    conn = _memdb_with_photos(["p1"])
    listing = _listing_stub(
        [
            {"canonical_part": "hood", "side": None, "confidence": 0.9},
            {"canonical_part": "roof", "side": None, "confidence": 0.95},
        ]
    )
    calls: list = []

    def segment(payload):
        calls.append(payload)
        if payload["phrases"] == list(config.DAMAGE_PHRASES):
            masks = [
                {
                    "label": "dent", "x0": 40, "y0": 40, "x1": 70, "y1": 60,
                    "mask": bytes([1]) * 30 * 20,
                }
            ]
        else:
            assert payload["phrases"] == ["hood", "roof"]
            masks = [
                {
                    "label": "hood", "x0": 10, "y0": 10, "x1": 50, "y1": 50,
                    "mask": bytes([1]) * 40 * 40,  # overlaps the dent
                },
                {
                    "label": "roof", "x0": 60, "y0": 5, "x1": 95, "y1": 25,
                    "mask": bytes([1]) * 35 * 20,  # disjoint from the dent
                },
            ]
        return {"masks": masks}

    summary = vision.process_photo(
        conn, "p1", list_transport=listing, segment_transport=segment
    )
    assert summary["damaged_parts"] == ["hood"]
    assert len(summary["crop_ids"]) == 1

    parts = conn.execute(
        "SELECT canonical_part FROM visible_parts WHERE photo_id = 'p1'"
        " ORDER BY canonical_part"
    ).fetchall()
    assert [r[0] for r in parts] == ["hood", "roof"]

    masks = conn.execute(
        "SELECT kind, label FROM masks WHERE photo_id = 'p1' ORDER BY mask_id"
    ).fetchall()
    assert [(r[0], r[1]) for r in masks] == [
        ("part", "hood"),
        ("part", "roof"),
        ("damage", "dent"),
    ]
    assert all(
        len(row[0]) > 0
        for row in conn.execute(
            "SELECT mask FROM masks WHERE photo_id = 'p1'"
        ).fetchall()
    )

    crops = conn.execute(
        "SELECT crop_id, canonical_part, x0, y0, x1, y1, img FROM crops"
        " WHERE photo_id = 'p1'"
    ).fetchall()
    assert len(crops) == 1, "undamaged roof must get no crop"
    crop_id, part, x0, y0, x1, y1, blob = crops[0]
    assert (crop_id, part) == (summary["crop_ids"][0], "hood")
    # 40x40 box padded by 15% -> 6px per side: (4, 4, 56, 56).
    assert (x0, y0, x1, y1) == (4, 4, 56, 56)
    crop_img = Image.open(io.BytesIO(bytes(blob)))
    assert crop_img.size == (52, 52)
    conn.close()


def test_make_crops_clamps_to_photo_bounds():
    conn = _memdb_with_photos(["p1"])
    # 20x20 box in the bottom-right corner; padding would run past the edges.
    damaged = [_mask("rear bumper", 85, 65, 105, 85)]
    (crop_id,) = vision.make_crops(conn, "p1", damaged)
    (x0, y0, x1, y1, blob) = conn.execute(
        "SELECT x0, y0, x1, y1, img FROM crops WHERE crop_id = ?", [crop_id]
    ).fetchone()
    assert (x0, y0, x1, y1) == (82, 62, 100, 80)
    assert Image.open(io.BytesIO(bytes(blob))).size == (18, 18)
    conn.close()


def test_make_crops_writes_nothing_without_damage():
    conn = _memdb_with_photos(["p1"])
    assert vision.make_crops(conn, "p1", []) == []
    assert (
        conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0] == 0
    )
    conn.close()
