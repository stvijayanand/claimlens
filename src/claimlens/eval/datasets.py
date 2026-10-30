"""Eval dataset loader with hard real/synthetic separation (spec 10.1, I4).

Layout::

    data/eval/real/<car_id>/{photo_*.jpg, estimate_correct.json,
                            estimate_seeded.json, injected_errors.json}
    data/eval/synthetic/<car_id>/{photo.jpg, ground_truth.json,
                            estimate_correct.json, estimate_seeded.json,
                            injected_errors.json}

Estimate files are ``EstimateDoc`` JSON plus a top-level ``"synthetic"``
bool; ``injected_errors.json`` is ``{"synthetic": bool, "errors": [...]}``.
All synthetic JSON files carry ``"synthetic": true``.

The real-evidence path (:func:`load_real_car`) scans every JSON file in the
car directory and raises :class:`SyntheticInRealPathError` if any
``"synthetic": true`` marker is found, so synthetic fixtures can never flow
into real-evidence scoring or display.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .. import config
from ..models import EstimateDoc
from .seed_errors import InjectedError

#: Directory holding the ``real/`` and ``synthetic/`` splits.
EVAL_ROOT = config.DATA_DIR / "eval"

ESTIMATE_CORRECT = "estimate_correct.json"
ESTIMATE_SEEDED = "estimate_seeded.json"
INJECTED_ERRORS = "injected_errors.json"
GROUND_TRUTH = "ground_truth.json"

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


class EvalDataError(ValueError):
    """A staged eval car directory is malformed or incomplete."""


class SyntheticInRealPathError(EvalDataError):
    """A synthetic-marked file reached the real-evidence path (I4)."""


class Box(BaseModel):
    """Pixel box, x0/y0 inclusive, x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int


class GroundTruth(BaseModel):
    """Known damage authored at synthetic-generation time."""

    synthetic: bool = Field(default=True)
    part: str
    side: str | None = None
    damage_type: str
    box: Box
    box_source: str = "nominal"  # nominal | author
    clean_photo: str = ""
    prompt: str = ""


@dataclass
class EvalCar:
    """One staged eval car, loaded and validated."""

    car_id: str
    split: str  # real | synthetic
    directory: Path
    photos: list[Path] = field(default_factory=list)
    estimate_correct: EstimateDoc | None = None
    estimate_seeded: EstimateDoc | None = None
    injected_errors: list[InjectedError] = field(default_factory=list)
    ground_truth: GroundTruth | None = None
    synthetic: bool = False


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalDataError(f"{path}: invalid JSON ({exc})") from exc


def _contains_synthetic_true(node: Any) -> bool:
    """Recursively detect a ``"synthetic": true`` marker."""
    if isinstance(node, dict):
        return any(
            (key == "synthetic" and value is True)
            or _contains_synthetic_true(value)
            for key, value in node.items()
        )
    if isinstance(node, list):
        return any(_contains_synthetic_true(item) for item in node)
    return False


def _assert_no_synthetic_marker(payload: Any, path: Path) -> None:
    if _contains_synthetic_true(payload):
        raise SyntheticInRealPathError(
            f"{path}: file carries \"synthetic\": true and cannot load "
            "through the real-evidence path (I4)"
        )


def _photo_files(car_dir: Path) -> list[Path]:
    found = list(car_dir.glob("photo_*")) + list(car_dir.glob("photo.*"))
    return sorted(
        {p for p in found if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES}
    )


def _load_estimate(path: Path) -> tuple[EstimateDoc, bool]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise EvalDataError(f"{path}: expected a JSON object")
    synthetic = payload.pop("synthetic", False)
    if not isinstance(synthetic, bool):
        raise EvalDataError(f"{path}: 'synthetic' must be a bool")
    try:
        doc = EstimateDoc.model_validate(payload)
    except Exception as exc:
        raise EvalDataError(f"{path}: invalid EstimateDoc ({exc})") from exc
    return doc, synthetic


def read_injected_errors(path: Path) -> tuple[list[InjectedError], bool]:
    """Read ``injected_errors.json`` -> (errors, synthetic_marker)."""
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("errors"), list
    ):
        raise EvalDataError(f"{path}: expected {{\"errors\": [...]}}")
    synthetic = payload.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise EvalDataError(f"{path}: 'synthetic' must be a bool")
    try:
        errors = [InjectedError.model_validate(e) for e in payload["errors"]]
    except Exception as exc:
        raise EvalDataError(f"{path}: invalid injected error ({exc})") from exc
    return errors, synthetic


def write_injected_errors(
    path: Path, errors: list[InjectedError], *, synthetic: bool
) -> None:
    """Write ``injected_errors.json`` in the canonical shape."""
    path.write_text(
        json.dumps(
            {
                "synthetic": synthetic,
                "errors": [e.model_dump(mode="json") for e in errors],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_estimate(path: Path, doc: EstimateDoc, *, synthetic: bool) -> None:
    """Write an estimate file (EstimateDoc JSON + ``synthetic`` marker)."""
    payload = {"synthetic": synthetic, **doc.model_dump(mode="json")}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_real_car(car_dir: str | Path) -> EvalCar:
    """Load a real car; HARD-fail on any synthetic marker (I4).

    Every ``*.json`` file in the directory is scanned for
    ``"synthetic": true`` before anything is returned.
    """
    directory = Path(car_dir)
    if not directory.is_dir():
        raise EvalDataError(f"{directory}: not a directory")
    for json_file in sorted(directory.glob("*.json")):
        _assert_no_synthetic_marker(_read_json(json_file), json_file)
    if (directory / GROUND_TRUTH).exists():
        raise EvalDataError(
            f"{directory}: {GROUND_TRUTH} only exists on the synthetic path"
        )
    correct_path = directory / ESTIMATE_CORRECT
    seeded_path = directory / ESTIMATE_SEEDED
    errors_path = directory / INJECTED_ERRORS
    for required in (correct_path, seeded_path, errors_path):
        if not required.is_file():
            raise EvalDataError(f"{directory}: missing {required.name}")
    correct, _ = _load_estimate(correct_path)
    seeded, _ = _load_estimate(seeded_path)
    errors, _ = read_injected_errors(errors_path)
    return EvalCar(
        car_id=directory.name,
        split="real",
        directory=directory,
        photos=_photo_files(directory),
        estimate_correct=correct,
        estimate_seeded=seeded,
        injected_errors=errors,
        synthetic=False,
    )


def load_synthetic_car(car_dir: str | Path) -> EvalCar:
    """Load a synthetic car; requires the ``synthetic: true`` markers."""
    directory = Path(car_dir)
    if not directory.is_dir():
        raise EvalDataError(f"{directory}: not a directory")
    truth_path = directory / GROUND_TRUTH
    if not truth_path.is_file():
        raise EvalDataError(f"{directory}: missing {GROUND_TRUTH}")
    try:
        truth = GroundTruth.model_validate(_read_json(truth_path))
    except Exception as exc:
        raise EvalDataError(f"{truth_path}: invalid ground truth ({exc})") from exc
    if truth.synthetic is not True:
        raise EvalDataError(f"{truth_path}: 'synthetic' must be true")
    correct_path = directory / ESTIMATE_CORRECT
    seeded_path = directory / ESTIMATE_SEEDED
    errors_path = directory / INJECTED_ERRORS
    for required in (correct_path, seeded_path, errors_path):
        if not required.is_file():
            raise EvalDataError(f"{directory}: missing {required.name}")
    correct, correct_synth = _load_estimate(correct_path)
    seeded, seeded_synth = _load_estimate(seeded_path)
    errors, errors_synth = read_injected_errors(errors_path)
    for name, marker in (
        (ESTIMATE_CORRECT, correct_synth),
        (ESTIMATE_SEEDED, seeded_synth),
        (INJECTED_ERRORS, errors_synth),
    ):
        if marker is not True:
            raise EvalDataError(f"{directory}: {name} lacks 'synthetic': true")
    return EvalCar(
        car_id=directory.name,
        split="synthetic",
        directory=directory,
        photos=_photo_files(directory),
        estimate_correct=correct,
        estimate_seeded=seeded,
        injected_errors=errors,
        ground_truth=truth,
        synthetic=True,
    )


def load_car(car_dir: str | Path, *, split: str) -> EvalCar:
    """Load a car directory from the named split (``real`` | ``synthetic``)."""
    if split == "real":
        return load_real_car(car_dir)
    if split == "synthetic":
        return load_synthetic_car(car_dir)
    raise ValueError(f"unknown split {split!r}: expected 'real'|'synthetic'")


def iter_cars(
    root: str | Path | None = None, *, split: str = "real"
) -> list[EvalCar]:
    """Load every car directory under ``<root>/<split>/``, sorted by id."""
    base = Path(root) if root is not None else EVAL_ROOT
    split_dir = base / split
    if not split_dir.is_dir():
        raise EvalDataError(f"{split_dir}: split directory not found")
    return [
        load_car(child, split=split)
        for child in sorted(split_dir.iterdir())
        if child.is_dir()
    ]
