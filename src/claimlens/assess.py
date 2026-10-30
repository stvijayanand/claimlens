"""Damage assessment: one Spark judgment per damaged-part crop.

Spec section 7a. ``assess_crop`` sends the crop bytes plus the parent photo
bytes (context) and the canonical part hint to Spark (low effort), validates
the response against ``DamageAssessment`` (spec invariant I1), and writes one
``damage_assessments`` row. ``assess_car`` maps the same flow over every crop
of a car.

Never-guess rules:
- On ``SchemaViolation`` (``structured()`` already retried once), the crop is
  recorded as ``visibility="not_visible"`` with ``confidence=0`` and a logged
  cause. Placeholder fields use ``"unknown"``/``0.0``, never invented damage.
- Below ``MIN_ASSESS_CONFIDENCE``, a returned assessment is forced to
  ``visibility="not_visible"``. The report renders these as "not visible in
  photos", never "undamaged".
- ``photo_id``/``crop_id``/``canonical_part`` are forced to the authoritative
  database values so the evidence pointer cannot drift from its crop.
"""

from __future__ import annotations

import logging

import duckdb

from . import config, meta_client
from .meta_client import FileAttachment, Transport, UsageHook
from .models import DamageAssessment

log = logging.getLogger(__name__)

_ASSESS_SYSTEM = (
    "You assess visible car damage in a close-up crop, using the full photo "
    "for context. Judge only what is actually visible in the crop: report "
    "the damage type, severity, whether it needs repair or replace, and your "
    "confidence. If the damage is unclear or not visible, say so with low "
    "confidence. Never invent damage you cannot see."
)


def _guess_mime(img: bytes) -> str:
    """Guess the photo MIME type from magic bytes (PNG or JPEG)."""
    if img.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return "image/jpeg"


def _crop_row(conn: duckdb.DuckDBPyConnection, crop_id: str) -> tuple[bytes, str, str]:
    """Return (crop bytes, photo_id, canonical_part) for ``crop_id``."""
    row = conn.execute(
        "SELECT img, photo_id, canonical_part FROM crops WHERE crop_id = ?",
        [crop_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown crop_id: {crop_id!r}")
    return bytes(row[0]), str(row[1]), str(row[2])


def _photo_bytes(conn: duckdb.DuckDBPyConnection, photo_id: str) -> bytes:
    row = conn.execute(
        "SELECT img FROM photos WHERE photo_id = ?", [photo_id]
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown photo_id: {photo_id!r}")
    return bytes(row[0])


def _not_visible(
    photo_id: str, crop_id: str, canonical_part: str, *, cause: str
) -> DamageAssessment:
    """Build the never-guess fallback assessment, logging the cause."""
    log.warning(
        "assess_crop %s: schema failure (%s); recording not_visible",
        crop_id,
        cause,
    )
    return DamageAssessment(
        photo_id=photo_id,
        crop_id=crop_id,
        canonical_part=canonical_part,
        side=None,
        damage_type="unknown",
        severity="unknown",
        repair_replace="unknown",
        likelihood=0.0,
        confidence=0.0,
        visibility="not_visible",
    )


def assess_crop(
    conn: duckdb.DuckDBPyConnection,
    crop_id: str,
    *,
    transport: Transport | None = None,
    on_usage: UsageHook | None = None,
) -> DamageAssessment:
    """Assess one crop; write its ``damage_assessments`` row; return it.

    ``transport`` overrides the network call (tests). Never raises on model
    output problems: schema failures become ``not_visible`` assessments.
    """
    crop_img, photo_id, canonical_part = _crop_row(conn, crop_id)
    photo_img = _photo_bytes(conn, photo_id)
    messages = [
        {"role": "system", "content": _ASSESS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Assess the damage to {canonical_part!r} in crop {crop_id} "
                f"of photo {photo_id}. The first attached image is the crop, "
                "the second is the full photo for context. Reply with "
                "photo_id, crop_id, canonical part, side, damage type, "
                "severity, repair-or-replace, likelihood, confidence, and "
                "visibility."
            ),
        },
    ]
    try:
        result = meta_client.structured(
            kind="damage_assessment",
            response_model=DamageAssessment,
            model=config.SPARK_MODEL,
            effort=config.EFFORT_LOW,
            messages=messages,
            files=[
                FileAttachment(
                    content=crop_img,
                    mime_type=_guess_mime(crop_img),
                    name=crop_id,
                ),
                FileAttachment(
                    content=photo_img,
                    mime_type=_guess_mime(photo_img),
                    name=photo_id,
                ),
            ],
            conn=conn,
            transport=transport,
            on_usage=on_usage,
        )
    except meta_client.SchemaViolation as exc:
        assessment = _not_visible(
            photo_id, crop_id, canonical_part, cause=str(exc)
        )
    else:
        assert isinstance(result, DamageAssessment)
        assessment = result.model_copy(
            update={
                "photo_id": photo_id,
                "crop_id": crop_id,
                "canonical_part": canonical_part,
            }
        )
        if assessment.confidence < config.MIN_ASSESS_CONFIDENCE:
            log.info(
                "assess_crop %s: confidence %.3f below %.3f; forcing not_visible",
                crop_id,
                assessment.confidence,
                config.MIN_ASSESS_CONFIDENCE,
            )
            assessment = assessment.model_copy(
                update={"visibility": "not_visible"}
            )
    conn.execute(
        "INSERT OR REPLACE INTO damage_assessments (crop_id, assessment)"
        " VALUES (?, ?)",
        [crop_id, assessment.model_dump_json()],
    )
    return assessment


def assess_car(
    conn: duckdb.DuckDBPyConnection,
    car_id: str,
    *,
    transport: Transport | None = None,
    on_usage: UsageHook | None = None,
) -> list[DamageAssessment]:
    """Assess every crop of ``car_id``; return the assessments in crop order."""
    rows = conn.execute(
        "SELECT c.crop_id FROM crops c"
        " JOIN photos p ON c.photo_id = p.photo_id"
        " WHERE p.car_id = ? ORDER BY c.crop_id",
        [car_id],
    ).fetchall()
    return [
        assess_crop(conn, row[0], transport=transport, on_usage=on_usage)
        for row in rows
    ]
