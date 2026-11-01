"""Eval smoke + cache + regression-sensitivity tests (issue #7).

All runs use fixture-replay transports: no network, no API cost. Each test
uses throwaway ``--cache-db``/``--out`` paths under ``tmp_path`` so the
shared checkout is never touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claimlens.eval.datasets import load_car  # noqa: E402
from eval.run_eval import run_car, run_eval  # noqa: E402


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "eval/run_eval.py", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_smoke_two_car_synthetic_run(tmp_path):
    """Eval smoke: 2-car synthetic run completes and emits the table."""
    out = tmp_path / "eval_report.json"
    proc = _run_cli(
        "--split", "synthetic", "--max-cars", "2",
        "--out", str(out), "--cache-db", str(tmp_path / "cache.duckdb"),
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    for name in (
        "recall", "precision", "extraction accuracy",
        "synthetic part detection", "schema pass rate", "time per car",
        "mean cost per car",
    ):
        assert name in proc.stdout
    for target in (">=80%", ">=75%", ">=90%", ">=85%", "100%",
                   "p50<2min", "<$0.50"):
        assert target in proc.stdout
    assert "cache:" in proc.stdout and "miss" in proc.stdout

    report_data = json.loads(out.read_text(encoding="utf-8"))
    scored = report_data["splits"]["synthetic"]
    assert scored["cars_total"] == 2
    assert scored["cars_completed"] == 2
    assert scored["metrics"]["recall"] == 1.0
    assert scored["metrics"]["precision"] == 1.0
    assert scored["metrics"]["part_detection"] == 1.0


def test_cold_bypasses_model_cache(tmp_path):
    """--cold misses on every call; a warm rerun hits the persisted cache."""
    cache_db = str(tmp_path / "cache.duckdb")
    first = run_eval(["synthetic"], cache_db=cache_db, cold=False,
                     max_cars=2)["splits"]["synthetic"]["metrics"]
    assert first["cache_misses"] > 0
    assert first["cache_hits"] == 0

    second = run_eval(["synthetic"], cache_db=cache_db, cold=False,
                      max_cars=2)["splits"]["synthetic"]["metrics"]
    assert second["cache_hits"] == first["cache_misses"]
    assert second["cache_misses"] == 0

    cold = run_eval(["synthetic"], cache_db=cache_db, cold=True,
                    max_cars=2)["splits"]["synthetic"]["metrics"]
    assert cold["cache_hits"] == 0
    assert cold["cache_misses"] == first["cache_misses"]


def test_broken_matching_moves_the_metrics(monkeypatch, tmp_path):
    """Sanity: killing match_damage must move recall and precision.

    The break is a monkeypatch in this test only; shared pipeline files
    are never edited (reverts automatically).
    """
    import claimlens.tools

    def broken_match_damage(ctx, line_no, crop_id):
        return {"score": 0.0, "basis": "broken", "matchable": False}

    baseline = run_eval(
        ["synthetic"], cache_db=str(tmp_path / "cache.duckdb"),
        cold=True, max_cars=2,
    )["splits"]["synthetic"]["metrics"]

    monkeypatch.setattr(
        claimlens.tools, "match_damage", broken_match_damage
    )
    broken = run_eval(
        ["synthetic"], cache_db=str(tmp_path / "cache2.duckdb"),
        cold=True, max_cars=2,
    )["splits"]["synthetic"]["metrics"]

    assert baseline["recall"] == 1.0
    assert baseline["precision"] == 1.0
    assert broken["recall"] < baseline["recall"]
    assert broken["precision"] < baseline["precision"]


def test_extraction_scored_against_seeded_input():
    """Extraction accuracy measures transcribe fidelity vs the seeded input.

    The pipeline reads ``estimate_seeded.json``; scoring the extractor
    against ``estimate_correct.json`` punishes faithful transcription of
    seeded errors (a dropped line is absent from the input, so no
    extractor could ever reproduce it). syn003's swapped op and syn001's
    dropped line must therefore not count against extraction.
    """
    swapped = run_car(
        load_car(ROOT / "data" / "eval" / "synthetic" / "syn003",
                 split="synthetic"),
        None,  # cold: every replayed call misses, nothing saved back
    )
    assert swapped.ok, swapped.error
    assert (swapped.extraction_matched, swapped.extraction_total) == (1, 1)

    dropped = run_car(
        load_car(ROOT / "data" / "eval" / "synthetic" / "syn001",
                 split="synthetic"),
        None,
    )
    assert dropped.ok, dropped.error
    assert (dropped.extraction_matched, dropped.extraction_total) == (0, 0)


@pytest.mark.parametrize("split", ["real", "synthetic"])
def test_full_split_completes_with_separate_scoring(tmp_path, split):
    """Both staged splits run to completion and score separately."""
    report_data = run_eval(
        [split], cache_db=str(tmp_path / "cache.duckdb"), cold=True
    )
    assert list(report_data["splits"]) == [split]
    scored = report_data["splits"][split]
    assert scored["cars_total"] == 5
    assert scored["cars_completed"] == 5
    assert scored["metrics"]["recall"] == 1.0
    assert scored["metrics"]["precision"] == 1.0
    if split == "real":
        assert scored["metrics"]["part_detection"] is None
    else:
        assert scored["metrics"]["part_detection"] == 1.0
