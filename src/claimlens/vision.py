"""Vision pipeline: visible-parts listing, SAM segmentation, overlap, crops.

Per-photo order (spec section 6): ``list_visible_parts`` -> ``segment_photo``
-> ``compute_overlaps`` -> ``make_crops``. ``process_photo`` runs the chain.

Mask encoding (v1): ``SegmentMask.mask`` is a row-major, 1-byte-per-pixel
foreground bitmap cropped to the mask box, so ``len(mask) ==
(x1 - x0) * (y1 - y0)`` and any nonzero byte is foreground. Stub transports in
tests produce this layout; the future live SAM transport must normalize to it.

v1 simplifications: one part phrase per distinct canonical part (no
side-qualified phrases); part-batch labels are treated as canonical parts.
"""

from __future__ import annotations

import io
from typing import Any

import duckdb
from PIL import Image

from . import config, meta_client, taxonomy
from .meta_client import FileAttachment, SegmentMask, Transport, UsageHook
from .models import VisiblePart, VisibleParts

_LISTING_SYSTEM = (
    "You list the car body parts visible in a damage photo. "
    "Use only canonical part names from the known-parts list. "
    "Assign each part a confidence between 0 and 1."
)


def _guess_mime(img: bytes) -> str:
    """Guess the photo MIME type from magic bytes (PNG or JPEG)."""
    if img.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return "image/jpeg"


def _photo_bytes(conn: duckdb.DuckDBPyConnection, photo_id: str) -> bytes:
    row = conn.execute(
        "SELECT img FROM photos WHERE photo_id = ?", [photo_id]
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown photo_id: {photo_id!r}")
    return bytes(row[0])


def list_visible_parts(
    conn: duckdb.DuckDBPyConnection,
    photo_id: str,
    *,
    transport: Transport | None = None,
    on_usage: UsageHook | None = None,
) -> VisibleParts:
    """List visible parts via Spark; write filtered ``visible_parts`` rows.

    Parts below ``MIN_PART_CONFIDENCE`` are dropped before SAM and omitted
    from both the return value and the database.
    """
    img = _photo_bytes(conn, photo_id)
    messages = [
        {"role": "system", "content": _LISTING_SYSTEM},
        {
            "role": "user",
            "content": (
                f"List the car parts visible in photo {photo_id}.\n\n"
                f"{taxonomy.prompt_block()}"
            ),
        },
    ]
    result = meta_client.structured(
        kind="visible_parts",
        response_model=VisibleParts,
        model=config.SPARK_MODEL,
        effort=config.EFFORT_LOW,
        messages=messages,
        files=[
            FileAttachment(
                content=img, mime_type=_guess_mime(img), name=photo_id
            )
        ],
        conn=conn,
        transport=transport,
        on_usage=on_usage,
    )
    assert isinstance(result, VisibleParts)
    kept = [
        p for p in result.parts if p.confidence >= config.MIN_PART_CONFIDENCE
    ]
    conn.execute("DELETE FROM visible_parts WHERE photo_id = ?", [photo_id])
    conn.executemany(
        "INSERT INTO visible_parts (photo_id, canonical_part, side, confidence)"
        " VALUES (?, ?, ?, ?)",
        [(photo_id, p.canonical_part, p.side, p.confidence) for p in kept],
    )
    return VisibleParts(photo_id=photo_id, parts=kept)


def segment_photo(
    conn: duckdb.DuckDBPyConnection,
    photo_id: str,
    parts: VisibleParts | list[VisiblePart] | list[str],
    *,
    transport: Transport | None = None,
) -> tuple[list[SegmentMask], list[SegmentMask]]:
    """Segment one photo; write ``masks`` rows.

    Makes exactly two ``segment()`` calls: one batch over the distinct part
    phrases plus one batch over ``DAMAGE_PHRASES``. With no part phrases the
    part batch is skipped (a single damage batch), keeping the per-car budget
    at or under photos x 2 batches.
    """
    items = parts.parts if isinstance(parts, VisibleParts) else list(parts)
    phrases: list[str] = []
    for item in items:
        phrase = item.canonical_part if isinstance(item, VisiblePart) else str(item)
        if phrase not in phrases:
            phrases.append(phrase)
    img = _photo_bytes(conn, photo_id)
    part_masks: list[SegmentMask] = []
    if phrases:
        part_masks = meta_client.segment(img, phrases, transport=transport)
    damage_masks = meta_client.segment(
        img, list(config.DAMAGE_PHRASES), transport=transport
    )
    conn.execute("DELETE FROM masks WHERE photo_id = ?", [photo_id])
    seq = 0
    for kind, masks in (("part", part_masks), ("damage", damage_masks)):
        for mask in masks:
            mask_id = f"{photo_id}:{seq:04d}:{kind}"
            seq += 1
            conn.execute(
                "INSERT INTO masks (mask_id, photo_id, kind, label,"
                " x0, y0, x1, y1, mask) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    mask_id,
                    photo_id,
                    kind,
                    mask.label,
                    mask.x0,
                    mask.y0,
                    mask.x1,
                    mask.y1,
                    bytes(mask.mask),
                ],
            )
    return part_masks, damage_masks


def _pixel_on(mask: SegmentMask, x: int, y: int) -> bool:
    """True if the v1 foreground bitmap covers global pixel (x, y)."""
    width = mask.x1 - mask.x0
    if width <= 0:
        return False
    idx = (y - mask.y0) * width + (x - mask.x0)
    if idx < 0 or idx >= len(mask.mask):
        return False
    return mask.mask[idx] != 0


def is_damaged(
    part_mask: SegmentMask,
    damage_mask: SegmentMask,
    *,
    min_intersect_px: int | None = None,
) -> bool:
    """Pure overlap rule: damaged iff shared foreground pixels >= threshold.

    Defaults to ``OVERLAP_MIN_INTERSECT_PX``. Touching or disjoint boxes, and
    empty (all-background) masks, never count as damaged.
    """
    threshold = (
        config.OVERLAP_MIN_INTERSECT_PX
        if min_intersect_px is None
        else min_intersect_px
    )
    ix0 = max(part_mask.x0, damage_mask.x0)
    iy0 = max(part_mask.y0, damage_mask.y0)
    ix1 = min(part_mask.x1, damage_mask.x1)
    iy1 = min(part_mask.y1, damage_mask.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    if threshold <= 0:
        return True
    hits = 0
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            if _pixel_on(part_mask, x, y) and _pixel_on(damage_mask, x, y):
                hits += 1
                if hits >= threshold:
                    return True
    return False


def compute_overlaps(
    conn: duckdb.DuckDBPyConnection,
    photo_id: str,
    *,
    min_intersect_px: int | None = None,
) -> list[SegmentMask]:
    """Return the part masks of ``photo_id`` that overlap any damage mask."""
    rows = conn.execute(
        "SELECT kind, label, x0, y0, x1, y1, mask FROM masks"
        " WHERE photo_id = ? ORDER BY mask_id",
        [photo_id],
    ).fetchall()
    parts: list[SegmentMask] = []
    damages: list[SegmentMask] = []
    for kind, label, x0, y0, x1, y1, blob in rows:
        mask = SegmentMask(
            label=label, x0=x0, y0=y0, x1=x1, y1=y1, mask=bytes(blob)
        )
        (parts if kind == "part" else damages).append(mask)
    return [
        part
        for part in parts
        if any(
            is_damaged(part, damage, min_intersect_px=min_intersect_px)
            for damage in damages
        )
    ]


def make_crops(
    conn: duckdb.DuckDBPyConnection,
    photo_id: str,
    damaged_parts: list[SegmentMask] | None = None,
    *,
    padding_frac: float | None = None,
) -> list[str]:
    """Write one padded crop per damaged part; return the crop ids.

    Each part box grows by ``CROP_PADDING_FRAC`` of its own width/height per
    side and is clamped to the photo bounds. Parts without damage get no crop.
    """
    if damaged_parts is None:
        damaged_parts = compute_overlaps(conn, photo_id)
    frac = config.CROP_PADDING_FRAC if padding_frac is None else padding_frac
    photo = Image.open(io.BytesIO(_photo_bytes(conn, photo_id))).convert("RGB")
    width, height = photo.size
    conn.execute("DELETE FROM crops WHERE photo_id = ?", [photo_id])
    crop_ids: list[str] = []
    seen: set[str] = set()
    for mask in damaged_parts:
        box_w, box_h = mask.x1 - mask.x0, mask.y1 - mask.y0
        if box_w <= 0 or box_h <= 0:
            continue
        dx, dy = int(round(frac * box_w)), int(round(frac * box_h))
        x0, y0 = max(0, mask.x0 - dx), max(0, mask.y0 - dy)
        x1, y1 = min(width, mask.x1 + dx), min(height, mask.y1 + dy)
        if x1 <= x0 or y1 <= y0:
            continue
        buf = io.BytesIO()
        photo.crop((x0, y0, x1, y1)).save(buf, format="PNG")
        base = f"{photo_id}:{mask.label}:{x0}-{y0}-{x1}-{y1}"
        crop_id, suffix = base, 2
        while crop_id in seen:
            crop_id, suffix = f"{base}#{suffix}", suffix + 1
        seen.add(crop_id)
        conn.execute(
            "INSERT INTO crops (crop_id, photo_id, canonical_part,"
            " x0, y0, x1, y1, img) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [crop_id, photo_id, mask.label, x0, y0, x1, y1, buf.getvalue()],
        )
        crop_ids.append(crop_id)
    return crop_ids


def process_photo(
    conn: duckdb.DuckDBPyConnection,
    photo_id: str,
    *,
    list_transport: Transport | None = None,
    segment_transport: Transport | None = None,
    on_usage: UsageHook | None = None,
) -> dict[str, Any]:
    """Run listing -> segment -> overlap -> crops for one photo."""
    visible = list_visible_parts(
        conn, photo_id, transport=list_transport, on_usage=on_usage
    )
    part_masks, damage_masks = segment_photo(
        conn, photo_id, visible, transport=segment_transport
    )
    damaged = compute_overlaps(conn, photo_id)
    crop_ids = make_crops(conn, photo_id, damaged)
    return {
        "photo_id": photo_id,
        "parts": [p.canonical_part for p in visible.parts],
        "part_mask_count": len(part_masks),
        "damage_mask_count": len(damage_masks),
        "damaged_parts": [m.label for m in damaged],
        "crop_ids": crop_ids,
    }
