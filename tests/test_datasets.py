"""Dataset loader tests: layout loading + synthetic rejection (I4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimlens.eval import datasets
from claimlens.eval.datasets import (
    SyntheticInRealPathError,
    load_car,
    load_real_car,
    load_synthetic_car,
)
from claimlens.eval.seed_errors import InjectedError, add_phantom
from claimlens.models import EstimateDoc, EstimateLine


def _doc() -> EstimateDoc:
    return EstimateDoc(
        lines=[
            EstimateLine(
                line_no=1,
                raw_part_text="hood",
                canonical_part="hood",
                operation="repair",
                labor_hours=4.0,
            )
        ]
    )


def _stage_real(car_dir: Path) -> None:
    car_dir.mkdir(parents=True)
    (car_dir / "photo_1.jpg").write_bytes(b"fake-jpeg")
    datasets.write_estimate(car_dir / "estimate_correct.json", _doc(), synthetic=False)
    seeded, error = add_phantom(_doc(), raw_part_text="spoiler")
    datasets.write_estimate(car_dir / "estimate_seeded.json", seeded, synthetic=False)
    datasets.write_injected_errors(
        car_dir / "injected_errors.json", [error], synthetic=False
    )


def _stage_synthetic(car_dir: Path) -> None:
    _stage_real(car_dir)
    for name in (
        "estimate_correct.json",
        "estimate_seeded.json",
        "injected_errors.json",
    ):
        payload = json.loads((car_dir / name).read_text(encoding="utf-8"))
        payload["synthetic"] = True
        (car_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (car_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "part": "hood",
                "damage_type": "dent",
                "box": {"x0": 10, "y0": 10, "x1": 60, "y1": 60},
            }
        ),
        encoding="utf-8",
    )


def test_load_real_car(tmp_path):
    car_dir = tmp_path / "car001"
    _stage_real(car_dir)
    car = load_real_car(car_dir)
    assert car.car_id == "car001"
    assert car.split == "real"
    assert car.synthetic is False
    assert [p.name for p in car.photos] == ["photo_1.jpg"]
    assert car.estimate_correct.lines[0].canonical_part == "hood"
    assert len(car.estimate_seeded.lines) == 2
    assert car.injected_errors[0].error_type == "phantom_line"


def test_load_real_car_rejects_synthetic_estimate(tmp_path):
    car_dir = tmp_path / "car001"
    _stage_real(car_dir)
    payload = json.loads(
        (car_dir / "estimate_seeded.json").read_text(encoding="utf-8")
    )
    payload["synthetic"] = True
    (car_dir / "estimate_seeded.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(SyntheticInRealPathError, match="synthetic"):
        load_real_car(car_dir)


def test_load_real_car_rejects_synthetic_ground_truth(tmp_path):
    car_dir = tmp_path / "car001"
    _stage_real(car_dir)
    (car_dir / "ground_truth.json").write_text(
        json.dumps({"synthetic": True, "part": "hood"}), encoding="utf-8"
    )
    with pytest.raises(SyntheticInRealPathError, match="synthetic"):
        load_real_car(car_dir)


def test_load_real_car_rejects_any_synthetic_json(tmp_path):
    car_dir = tmp_path / "car001"
    _stage_real(car_dir)
    (car_dir / "notes.json").write_text(
        json.dumps({"nested": [{"synthetic": True}]}), encoding="utf-8"
    )
    with pytest.raises(SyntheticInRealPathError):
        load_real_car(car_dir)


def test_load_real_car_requires_all_three_files(tmp_path):
    car_dir = tmp_path / "car001"
    _stage_real(car_dir)
    (car_dir / "injected_errors.json").unlink()
    with pytest.raises(datasets.EvalDataError, match="missing"):
        load_real_car(car_dir)


def test_load_synthetic_car(tmp_path):
    car_dir = tmp_path / "syn001"
    _stage_synthetic(car_dir)
    (car_dir / "photo_1.jpg").rename(car_dir / "photo.jpg")
    car = load_synthetic_car(car_dir)
    assert car.split == "synthetic"
    assert car.synthetic is True
    assert [p.name for p in car.photos] == ["photo.jpg"]
    assert car.ground_truth.part == "hood"
    assert car.ground_truth.damage_type == "dent"
    assert car.ground_truth.box.x1 == 60


def test_load_synthetic_car_requires_markers(tmp_path):
    car_dir = tmp_path / "syn001"
    _stage_synthetic(car_dir)
    payload = json.loads(
        (car_dir / "estimate_correct.json").read_text(encoding="utf-8")
    )
    payload["synthetic"] = False
    (car_dir / "estimate_correct.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(datasets.EvalDataError, match="synthetic"):
        load_synthetic_car(car_dir)


def test_load_car_dispatch_and_iter_cars(tmp_path):
    real_dir = tmp_path / "real" / "car001"
    _stage_real(real_dir)
    assert load_car(real_dir, split="real").split == "real"
    synth_dir = tmp_path / "synthetic" / "syn001"
    _stage_synthetic(synth_dir)
    assert load_car(synth_dir, split="synthetic").split == "synthetic"
    with pytest.raises(ValueError, match="unknown split"):
        load_car(real_dir, split="nope")
    cars = datasets.iter_cars(tmp_path, split="real")
    assert [c.car_id for c in cars] == ["car001"]


def test_injected_errors_round_trip(tmp_path):
    path = tmp_path / "injected_errors.json"
    errors = [InjectedError(error_type="duplicate", line_no=4, detail="d")]
    datasets.write_injected_errors(path, errors, synthetic=False)
    loaded, marker = datasets.read_injected_errors(path)
    assert marker is False
    assert loaded == errors
