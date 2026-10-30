"""Contract tests: every model round-trips and generates its API schema."""

import pytest
from pydantic import ValidationError

from claimlens import models


def test_flag_requires_evidence():
    with pytest.raises(ValidationError):
        models.Flag(
            flag_type="phantom_line",
            canonical_part="hood",
            reason="no damage visible",
            confidence=0.9,
        )
    by_crop = models.Flag(
        flag_type="phantom_line",
        canonical_part="hood",
        reason="no damage visible",
        confidence=0.9,
        evidence_crop_ids=["c1"],
    )
    assert by_crop.evidence_crop_ids == ["c1"]
    by_line = models.Flag(
        flag_type="missing_line",
        canonical_part="hood",
        reason="damage has no line",
        confidence=0.8,
        evidence_line_nos=[3],
    )
    assert by_line.evidence_line_nos == [3]


def test_damage_assessment_round_trip():
    data = {
        "photo_id": "p1",
        "crop_id": "c1",
        "canonical_part": "front door",
        "side": "FL",
        "damage_type": "dent",
        "severity": "moderate",
        "repair_replace": "repair",
        "likelihood": 0.7,
        "confidence": 0.8,
        "visibility": "visible",
    }
    assert models.DamageAssessment.model_validate(data).model_dump() == data


def test_estimate_doc_defaults():
    doc = models.EstimateDoc(
        lines=[
            {
                "line_no": 1,
                "raw_part_text": "R&I frt bumper cover",
                "operation": "ri",
            }
        ]
    )
    assert doc.unparsed_count == 0
    assert doc.lines[0].canonical_part is None
    assert doc.lines[0].labor_hours is None


def test_visible_parts_confidence_bounds():
    with pytest.raises(ValidationError):
        models.VisiblePart(canonical_part="hood", confidence=1.5)


@pytest.mark.parametrize(
    ("model", "expected_props"),
    [
        (models.VisibleParts, {"photo_id", "parts"}),
        (
            models.DamageAssessment,
            {"crop_id", "damage_type", "severity", "repair_replace"},
        ),
        (models.EstimateDoc, {"lines", "unparsed_count"}),
        (models.Flag, {"flag_type", "evidence_crop_ids", "evidence_line_nos"}),
        (models.Match, {"line_no", "crop_id", "score", "basis"}),
    ],
)
def test_json_schemas_generate(model, expected_props):
    """Each contract yields the JSON schema the API call will enforce."""
    props = set(model.model_json_schema()["properties"])
    assert expected_props <= props
