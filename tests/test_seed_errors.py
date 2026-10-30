"""Seed-error transform tests: all five transforms + record correctness."""

from __future__ import annotations

import pytest

from claimlens.eval.seed_errors import (
    INFLATE_HOURS_FACTOR,
    add_phantom,
    apply_plan,
    drop_line,
    duplicate_line,
    inflate_hours,
    swap_repair_replace,
)
from claimlens.models import EstimateDoc, EstimateLine


def _fixture() -> EstimateDoc:
    return EstimateDoc(
        lines=[
            EstimateLine(
                line_no=1,
                raw_part_text="frt bpr cvr",
                canonical_part="front bumper cover",
                operation="replace",
                labor_hours=1.5,
                price=420.0,
            ),
            EstimateLine(
                line_no=2,
                raw_part_text="hood",
                canonical_part="hood",
                operation="repair",
                labor_hours=4.0,
                price=610.0,
            ),
            EstimateLine(
                line_no=3,
                raw_part_text="headlamp",
                canonical_part="headlight",
                side="FL",
                operation="replace",
                labor_hours=1.0,
            ),
        ]
    )


def test_add_phantom_appends_line_and_record():
    doc, error = add_phantom(
        _fixture(),
        raw_part_text="spoiler",
        canonical_part="spoiler",
        operation="replace",
        labor_hours=0.8,
    )
    assert [line.line_no for line in doc.lines] == [1, 2, 3, 4]
    phantom = doc.lines[-1]
    assert phantom.raw_part_text == "spoiler"
    assert phantom.canonical_part == "spoiler"
    assert error.error_type == "phantom_line"
    assert error.line_no == 4
    assert "spoiler" in error.detail


def test_drop_line_removes_line_and_record():
    doc, error = drop_line(_fixture(), 2)
    assert [line.line_no for line in doc.lines] == [1, 3]
    assert error.error_type == "missing_line"
    assert error.line_no == 2
    assert "hood" in error.detail


def test_drop_line_unknown_line_raises():
    with pytest.raises(ValueError, match="no line_no=9"):
        drop_line(_fixture(), 9)


def test_swap_repair_replace_flips_both_ways():
    doc, error = swap_repair_replace(_fixture(), 2)
    assert doc.lines[1].operation == "replace"
    assert error.error_type == "repair_vs_replace"
    assert error.line_no == 2
    assert "repair" in error.detail and "replace" in error.detail
    doc2, _ = swap_repair_replace(_fixture(), 1)
    assert doc2.lines[0].operation == "repair"


def test_swap_repair_replace_rejects_other_ops():
    doc = _fixture()
    doc.lines[0].operation = "refinish"
    with pytest.raises(ValueError, match="not repair/replace"):
        swap_repair_replace(doc, 1)


def test_duplicate_line_copies_with_new_line_no():
    doc, error = duplicate_line(_fixture(), 2)
    assert [line.line_no for line in doc.lines] == [1, 2, 3, 4]
    clone = doc.lines[-1]
    assert clone.raw_part_text == "hood"
    assert clone.canonical_part == "hood"
    assert clone.operation == "repair"
    assert clone.labor_hours == 4.0
    assert error.error_type == "duplicate"
    assert error.line_no == 4
    assert "2" in error.detail and "4" in error.detail


def test_inflate_hours_multiplies_by_2_5():
    assert INFLATE_HOURS_FACTOR == 2.5
    doc, error = inflate_hours(_fixture(), 1)
    assert doc.lines[0].labor_hours == pytest.approx(1.5 * 2.5)
    assert error.error_type == "labor_out_of_range"
    assert error.line_no == 1
    assert "1.5" in error.detail and "3.75" in error.detail


def test_inflate_hours_without_hours_raises():
    doc = _fixture()
    doc.lines[0].labor_hours = None
    with pytest.raises(ValueError, match="no labor_hours"):
        inflate_hours(doc, 1)


def test_transforms_do_not_mutate_input():
    fixture = _fixture()
    before = fixture.model_dump(mode="json")
    add_phantom(fixture, raw_part_text="spoiler")
    drop_line(fixture, 1)
    swap_repair_replace(fixture, 1)
    duplicate_line(fixture, 1)
    inflate_hours(fixture, 1)
    assert fixture.model_dump(mode="json") == before


def test_apply_plan_chains_in_order():
    doc, errors = apply_plan(
        _fixture(),
        [
            ("swap_repair_replace", {"line_no": 2}),
            ("inflate_hours", {"line_no": 1}),
        ],
    )
    assert [e.error_type for e in errors] == [
        "repair_vs_replace",
        "labor_out_of_range",
    ]
    assert doc.lines[1].operation == "replace"
    assert doc.lines[0].labor_hours == pytest.approx(3.75)


def test_apply_plan_unknown_transform_raises():
    with pytest.raises(ValueError, match="unknown seed transform"):
        apply_plan(_fixture(), [("explode", {})])
