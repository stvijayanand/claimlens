"""Unit tests for eval metric helpers (pure, no pipeline runs)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import (  # noqa: E402
    TARGET_TEXT,
    CarResult,
    box_iou,
    cost_of_usage,
    flag_matches_error,
    format_table,
    percentile_nearest_rank,
    score_extraction,
    score_split,
)


def test_iou_identical_is_one():
    assert box_iou((0, 0, 4, 4), (0, 0, 4, 4)) == 1.0


def test_iou_disjoint_is_zero():
    assert box_iou((0, 0, 2, 2), (5, 5, 7, 7)) == 0.0


def test_iou_partial_overlap():
    # inter 2, union 6
    assert box_iou((0, 0, 2, 2), (1, 0, 3, 2)) == 2 / 6


def test_iou_contained():
    assert box_iou((0, 0, 4, 4), (1, 1, 3, 3)) == 4 / 16


def test_iou_zero_area_is_zero():
    assert box_iou((1, 1, 1, 5), (0, 0, 4, 4)) == 0.0
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_percentile_nearest_rank():
    assert percentile_nearest_rank([10, 20, 30, 40], 50) == 20
    assert percentile_nearest_rank([10, 20, 30, 40], 95) == 40
    assert percentile_nearest_rank([7], 50) == 7
    assert percentile_nearest_rank([3, 1, 2], 50) == 2
    assert percentile_nearest_rank([], 50) is None


def _flag(flag_type, part="fender", lines=(), crops=()):
    return {
        "flag_type": flag_type,
        "canonical_part": part,
        "evidence_line_nos": list(lines),
        "evidence_crop_ids": list(crops),
    }


def test_match_phantom_duplicate_repair_labor_on_line_no():
    for error_type in (
        "phantom_line", "duplicate", "repair_vs_replace",
        "labor_out_of_range",
    ):
        assert flag_matches_error(
            _flag(error_type, lines=[2]),
            {"error_type": error_type, "line_no": 2},
            "fender",
        )
        assert not flag_matches_error(
            _flag(error_type, lines=[1]),
            {"error_type": error_type, "line_no": 2},
            "fender",
        )
        other = (
            "duplicate" if error_type == "phantom_line" else "phantom_line"
        )
        assert not flag_matches_error(
            _flag(other, lines=[2]),
            {"error_type": error_type, "line_no": 2},
            "fender",
        )


def test_match_missing_line_on_part_not_line_no():
    assert flag_matches_error(
        _flag("missing_line", part="Rear Door", crops=["c1"]),
        {"error_type": "missing_line", "line_no": 2},
        "rear door",
    )
    assert not flag_matches_error(
        _flag("missing_line", part="hood", crops=["c1"]),
        {"error_type": "missing_line", "line_no": 2},
        "rear door",
    )
    assert not flag_matches_error(
        _flag("missing_line", part="rear door", crops=["c1"]),
        {"error_type": "missing_line", "line_no": 2},
        None,
    )


def test_score_extraction_exact_and_mismatches():
    reference = [
        {"line_no": 1, "canonical_part": "Front Door",
         "operation": "repair", "labor_hours": 3.0},
        {"line_no": 2, "canonical_part": "fender",
         "operation": "replace", "labor_hours": None},
    ]
    db_lines = [
        {"line_no": 1, "canonical_part": "front door",
         "operation": "repair", "labor_hours": 3.0},
        {"line_no": 2, "canonical_part": "fender",
         "operation": "replace", "labor_hours": None},
        {"line_no": 3, "canonical_part": "mirror",
         "operation": "replace", "labor_hours": 0.5},
    ]
    assert score_extraction(db_lines, reference) == (2, 2)

    swapped = [dict(db_lines[0], operation="replace"), db_lines[1]]
    assert score_extraction(swapped, reference) == (1, 2)

    inflated = [dict(db_lines[0], labor_hours=7.5), db_lines[1]]
    assert score_extraction(inflated, reference) == (1, 2)

    dropped = [db_lines[1]]
    assert score_extraction(dropped, reference) == (1, 2)

    hours_missing = [dict(db_lines[0], labor_hours=None), db_lines[1]]
    assert score_extraction(hours_missing, reference) == (1, 2)


def test_cost_of_usage_cached_free_fresh_fallback(monkeypatch):
    from claimlens import config

    monkeypatch.setattr(config, "SPARK_USD_PER_CALL", 0.02)
    events = [
        {"cached": True, "usage": None},
        {"cached": False, "usage": None},
        {"cached": False, "usage": {"tokens": 5}},
    ]
    total, estimated = cost_of_usage(events)
    assert total == 0.04
    assert estimated is True

    total, estimated = cost_of_usage([{"cached": True, "usage": None}])
    assert (total, estimated) == (0.0, False)


def test_cost_of_usage_prices_live_segment_at_sam_rate(monkeypatch):
    from claimlens import config

    monkeypatch.setattr(config, "SPARK_USD_PER_CALL", 0.02)
    monkeypatch.setattr(config, "SAM_USD_PER_IMAGE", 0.0025)
    events = [
        {"kind": "segment", "cached": False, "usage": None},
        {"kind": "segment", "cached": True, "usage": None},
        {"kind": "visible_parts", "cached": False, "usage": None},
    ]
    total, estimated = cost_of_usage(events)
    assert total == 0.0225
    assert estimated is True


def _car(car_id, split="synthetic", **over):
    base = CarResult(car_id=car_id, split=split)
    for key, value in over.items():
        setattr(base, key, value)
    return base


def test_score_split_pools_counts():
    cars = [
        _car("a", injected=[{"error_type": "duplicate", "line_no": 2}],
             flags=[_flag("duplicate", lines=[1, 2])],
             matched_errors=1, matched_flags=1,
             extraction_matched=1, extraction_total=1,
             detection_hit=True, schema_passed=3, schema_failed=0,
             wall_ms=100.0, cost_usd=0.0, cache_misses=3),
        _car("b", injected=[{"error_type": "phantom_line", "line_no": 3}],
             flags=[_flag("phantom_line", lines=[3]),
                    _flag("phantom_line", lines=[1])],
             matched_errors=1, matched_flags=1,
             extraction_matched=1, extraction_total=2,
             detection_hit=False, schema_passed=4, schema_failed=0,
             wall_ms=300.0, cost_usd=0.0, cache_misses=4),
    ]
    scored = score_split("synthetic", cars)
    metrics = scored["metrics"]
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 2 / 3
    assert metrics["extraction_accuracy"] == 2 / 3
    assert metrics["part_detection"] == 0.5
    assert metrics["schema_pass_rate"] == 1.0
    assert metrics["time_p50_ms"] == 100.0
    assert metrics["time_p95_ms"] == 300.0
    assert metrics["cost_mean_usd"] == 0.0
    assert metrics["cache_misses"] == 7
    assert scored["cars_completed"] == 2


def test_score_split_precision_with_no_flags():
    cars = [_car("a", injected=[{"error_type": "duplicate", "line_no": 2}],
                 flags=[], matched_errors=0, matched_flags=0)]
    assert score_split("synthetic", cars)["metrics"]["precision"] == 0.0
    cars = [_car("a", injected=[], flags=[])]
    assert score_split("synthetic", cars)["metrics"]["precision"] == 1.0


def test_score_split_part_detection_only_for_synthetic():
    cars = [_car("a", split="real", detection_hit=None)]
    assert score_split("real", cars)["metrics"]["part_detection"] is None


def test_format_table_shows_all_seven_metrics_with_targets():
    cars = [_car("a", injected=[], flags=[],
                 extraction_matched=1, extraction_total=1,
                 detection_hit=True, schema_passed=2, schema_failed=0,
                 wall_ms=50.0, cost_usd=0.0)]
    table = format_table({"cold": False, "splits": {
        "synthetic": score_split("synthetic", cars)}})
    for name in (
        "recall", "precision", "extraction accuracy",
        "synthetic part detection", "schema pass rate", "time per car",
        "mean cost per car",
    ):
        assert name in table
    for target in TARGET_TEXT.values():
        assert target in table
    assert "cache:" in table
