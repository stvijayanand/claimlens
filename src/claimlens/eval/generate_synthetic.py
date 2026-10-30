"""Synthetic eval-car generation via Muse Image edits (spec 10.1).

CLI::

    python -m claimlens.eval.generate_synthetic <clean_photo>
        --damage dent --at 400,300 --out data/eval/synthetic/<car_id>/
        [--part "front door"] [--side FL] [--severity moderate]
        [--operation repair] [--seed swap_repair_replace --seed-line 1]

Writes ``photo.jpg`` (the damage edit), ``ground_truth.json``
(``synthetic: true``, part, damage type, box), an authored matching
``estimate_correct.json``, plus ``estimate_seeded.json`` and
``injected_errors.json`` from one deterministic seed transform.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

from .. import config, meta_client
from ..models import EstimateDoc, EstimateLine
from .datasets import (
    ESTIMATE_CORRECT,
    ESTIMATE_SEEDED,
    GROUND_TRUTH,
    INJECTED_ERRORS,
    Box,
    GroundTruth,
    write_estimate,
    write_injected_errors,
)
from .seed_errors import TRANSFORMS

#: Side of the nominal ground-truth box as a fraction of min(image w, h).
NOMINAL_BOX_FRAC = 0.25


def nominal_box(
    image_width: int, image_height: int, at: tuple[int, int]
) -> Box:
    """Center a nominal box on the requested edit point, clamped to bounds."""
    side = max(1, int(min(image_width, image_height) * NOMINAL_BOX_FRAC))
    half = side // 2
    x0 = min(max(at[0] - half, 0), max(image_width - side, 0))
    y0 = min(max(at[1] - half, 0), max(image_height - side, 0))
    return Box(x0=x0, y0=y0, x1=x0 + side, y1=y0 + side)


def build_prompt(
    *, damage: str, part: str, side: str | None, severity: str
) -> str:
    """Render the image-edit instruction for one damage injection."""
    where = f" on the car's {part}" + (f" ({side} side)" if side else "")
    return (
        f"Photorealistic edit of the car in the photo: add a {severity} "
        f"{damage}{where}, keeping everything else in the photo unchanged. "
        "Match lighting, angle, and paint color; damaged area only, no "
        "other modifications."
    )


def default_hours(part: str, operation: str) -> float:
    """Midpoint of the labor-range table, or a small fallback default."""
    import csv

    try:
        with config.LABOR_CSV.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (
                    row["canonical_part"].strip().lower() == part.lower()
                    and row["operation"].strip().lower() == operation.lower()
                ):
                    return (
                        float(row["low_hrs"]) + float(row["high_hrs"])
                    ) / 2
    except FileNotFoundError:
        pass
    return 2.0


def generate(
    clean_photo: bytes,
    *,
    damage: str,
    at: tuple[int, int],
    part: str,
    side: str | None = None,
    severity: str = "moderate",
    operation: str = "repair",
    seed: str = "add_phantom",
    seed_kwargs: dict[str, Any] | None = None,
    clean_name: str = "",
    model: str | None = None,
    transport: meta_client.Transport | None = None,
) -> tuple[bytes, GroundTruth, EstimateDoc, EstimateDoc, list]:
    """Run one synthetic edit; return photo bytes + authored eval records.

    ``transport`` stubs the image call in tests; without it the live
    ``edit_image`` default transport runs.
    """
    prompt = build_prompt(
        damage=damage, part=part, side=side, severity=severity
    )
    edited = meta_client.edit_image(
        clean_photo, prompt, model=model, transport=transport
    )
    from PIL import Image

    with Image.open(io.BytesIO(edited)) as img:
        width, height = img.size
    truth = GroundTruth(
        synthetic=True,
        part=part,
        side=side,
        damage_type=damage,
        box=nominal_box(width, height, at),
        box_source="nominal",
        clean_photo=clean_name,
        prompt=prompt,
    )
    correct = EstimateDoc(
        lines=[
            EstimateLine(
                line_no=1,
                raw_part_text=part,
                canonical_part=part,
                side=side,
                operation=operation,
                labor_hours=default_hours(part, operation),
            )
        ]
    )
    try:
        transform = TRANSFORMS[seed]
    except KeyError:
        raise ValueError(f"unknown seed transform {seed!r}") from None
    kwargs = dict(seed_kwargs or {})
    if seed == "add_phantom" and "raw_part_text" not in kwargs:
        kwargs.setdefault("raw_part_text", "headlight")
        kwargs.setdefault("canonical_part", "headlight")
        kwargs.setdefault("operation", "replace")
    seeded, error = transform(correct, **kwargs)
    return edited, truth, correct, seeded, [error]


def write_car(
    out_dir: Path,
    *,
    photo: bytes,
    truth: GroundTruth,
    correct: EstimateDoc,
    seeded: EstimateDoc,
    errors: list,
) -> None:
    """Write one synthetic car directory in the section 10.1 layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "photo.jpg").write_bytes(photo)
    (out_dir / GROUND_TRUTH).write_text(
        json.dumps(truth.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    write_estimate(out_dir / ESTIMATE_CORRECT, correct, synthetic=True)
    write_estimate(out_dir / ESTIMATE_SEEDED, seeded, synthetic=True)
    write_injected_errors(out_dir / INJECTED_ERRORS, errors, synthetic=True)


def _parse_at(raw: str) -> tuple[int, int]:
    try:
        x, y = (int(v) for v in raw.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--at expects 'x,y', got {raw!r}"
        ) from None
    return (x, y)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one synthetic eval car via Muse Image edit."
    )
    parser.add_argument("clean_photo", type=Path)
    parser.add_argument("--damage", required=True, help="e.g. dent")
    parser.add_argument("--at", dest="at", required=True, type=_parse_at)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--side", default=None)
    parser.add_argument("--severity", default="moderate")
    parser.add_argument("--operation", default="repair")
    parser.add_argument("--seed", default="add_phantom")
    parser.add_argument("--seed-line", type=int, default=1)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    clean_bytes = args.clean_photo.read_bytes()
    seed_kwargs: dict[str, Any] = (
        {"line_no": args.seed_line} if args.seed != "add_phantom" else {}
    )
    if args.seed == "add_phantom":
        seed_kwargs = {
            "raw_part_text": "headlight",
            "canonical_part": "headlight",
            "operation": "replace",
        }
    photo, truth, correct, seeded, errors = generate(
        clean_bytes,
        damage=args.damage,
        at=args.at,
        part=args.part,
        side=args.side,
        severity=args.severity,
        operation=args.operation,
        seed=args.seed,
        seed_kwargs=seed_kwargs,
        clean_name=args.clean_photo.name,
        model=args.model,
    )
    write_car(
        args.out,
        photo=photo,
        truth=truth,
        correct=correct,
        seeded=seeded,
        errors=errors,
    )
    print(f"wrote synthetic car to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
