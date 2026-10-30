"""Pydantic data contracts for ClaimLens.

These models are the single source of truth: ``model_json_schema()`` output is
passed as the structured-output schema for the matching model call, and every
model response is parsed with ``model_validate`` before use (spec invariant I1).

String enums stay open (plain ``str``) in v1: the taxonomy table owns the
canonical vocabulary, and unknown values flow to unmatched/unparsed paths
instead of failing validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class VisiblePart(BaseModel):
    """One part the model lists as visible in a photo."""

    canonical_part: str
    side: str | None = None  # FL/FR/RL/RR/center; None = unknown
    confidence: float = Field(ge=0, le=1)


class VisibleParts(BaseModel):
    """All parts listed as visible in one photo."""

    photo_id: str
    parts: list[VisiblePart]


class DamageAssessment(BaseModel):
    """Damage judgment for one damaged-part crop."""

    photo_id: str
    crop_id: str  # evidence pointer (FK -> crops)
    canonical_part: str
    side: str | None = None
    damage_type: str  # dent | scratch | crack | shattered | ...
    severity: str  # minor | moderate | severe
    repair_replace: str  # repair | replace
    likelihood: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    visibility: str  # visible | not_visible


class EstimateLine(BaseModel):
    """One parsed line of the repair estimate."""

    line_no: int
    raw_part_text: str
    canonical_part: str | None = None  # filled by normalize_part_name
    side: str | None = None
    operation: str  # repair | replace | ri | refinish | ...
    labor_hours: float | None = None
    price: float | None = None


class EstimateDoc(BaseModel):
    """Full parsed estimate; unparsed lines are counted, never guessed."""

    lines: list[EstimateLine]
    unparsed_count: int = 0


class Flag(BaseModel):
    """One reconcile finding. Exists only if record_flag accepts it (I2)."""

    flag_type: str  # phantom_line | missing_line | repair_vs_replace |
    # duplicate | labor_out_of_range
    canonical_part: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    evidence_crop_ids: list[str] = []
    evidence_line_nos: list[int] = []
    shop_questions: list[str] = []

    @model_validator(mode="after")
    def _evidence_required(self) -> Flag:
        if not self.evidence_crop_ids and not self.evidence_line_nos:
            raise ValueError("flag needs >=1 evidence ref")
        return self


class Match(BaseModel):
    """Scored pairing of an estimate line with observed damage."""

    line_no: int
    crop_id: str
    score: float = Field(ge=0, le=1)
    basis: str  # e.g. "part+side+location"
