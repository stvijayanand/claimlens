"""Deterministic error injection for eval estimates (spec 10.2).

Each transform takes an :class:`~claimlens.models.EstimateDoc`, returns a new
doc plus an :class:`InjectedError` record ``{error_type, line_no, detail}``.
Inputs are never mutated. Error types match the pipeline flag types so the
eval runner can compare flags against ``injected_errors.json`` directly.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from ..models import EstimateDoc, EstimateLine

#: Hours multiplier used by :func:`inflate_hours` (spec 10.2).
INFLATE_HOURS_FACTOR = 2.5


class InjectedError(BaseModel):
    """One known injected error: what was changed and where."""

    error_type: str  # phantom_line | missing_line | repair_vs_replace |
    # duplicate | labor_out_of_range
    line_no: int
    detail: str


def _copy_doc(doc: EstimateDoc) -> EstimateDoc:
    return EstimateDoc.model_validate(doc.model_dump(mode="json"))


def _next_line_no(doc: EstimateDoc) -> int:
    return max((line.line_no for line in doc.lines), default=0) + 1


def _get_line(doc: EstimateDoc, line_no: int) -> EstimateLine:
    for line in doc.lines:
        if line.line_no == line_no:
            return line
    raise ValueError(f"no line_no={line_no} in estimate")


def add_phantom(
    doc: EstimateDoc,
    *,
    raw_part_text: str,
    canonical_part: str | None = None,
    side: str | None = None,
    operation: str = "replace",
    labor_hours: float | None = None,
    price: float | None = None,
) -> tuple[EstimateDoc, InjectedError]:
    """Append a plausible line for an undamaged part (expect phantom_line)."""
    out = _copy_doc(doc)
    line_no = _next_line_no(out)
    out.lines.append(
        EstimateLine(
            line_no=line_no,
            raw_part_text=raw_part_text,
            canonical_part=canonical_part,
            side=side,
            operation=operation,
            labor_hours=labor_hours,
            price=price,
        )
    )
    return out, InjectedError(
        error_type="phantom_line",
        line_no=line_no,
        detail=f"added phantom line for undamaged part {raw_part_text!r}",
    )


def drop_line(doc: EstimateDoc, line_no: int) -> tuple[EstimateDoc, InjectedError]:
    """Remove the line covering real damage (expect missing_line)."""
    target = _get_line(doc, line_no)
    out = _copy_doc(doc)
    out.lines = [line for line in out.lines if line.line_no != line_no]
    return out, InjectedError(
        error_type="missing_line",
        line_no=line_no,
        detail=(
            f"dropped line covering {target.raw_part_text!r} "
            f"(op={target.operation})"
        ),
    )


def swap_repair_replace(
    doc: EstimateDoc, line_no: int
) -> tuple[EstimateDoc, InjectedError]:
    """Flip operation repair<->replace on one line (expect repair_vs_replace)."""
    target = _get_line(doc, line_no)
    flipped = {"repair": "replace", "replace": "repair"}.get(target.operation)
    if flipped is None:
        raise ValueError(
            f"line_no={line_no} op={target.operation!r} is not repair/replace"
        )
    out = _copy_doc(doc)
    _get_line(out, line_no).operation = flipped
    return out, InjectedError(
        error_type="repair_vs_replace",
        line_no=line_no,
        detail=f"swapped op {target.operation!r} -> {flipped!r}",
    )


def duplicate_line(doc: EstimateDoc, line_no: int) -> tuple[EstimateDoc, InjectedError]:
    """Copy a line under a new line_no (expect duplicate)."""
    target = _get_line(doc, line_no)
    out = _copy_doc(doc)
    new_no = _next_line_no(out)
    clone = EstimateLine.model_validate(target.model_dump(mode="json"))
    clone.line_no = new_no
    out.lines.append(clone)
    return out, InjectedError(
        error_type="duplicate",
        line_no=new_no,
        detail=f"duplicated line_no={line_no} as line_no={new_no}",
    )


def inflate_hours(
    doc: EstimateDoc, line_no: int, factor: float = INFLATE_HOURS_FACTOR
) -> tuple[EstimateDoc, InjectedError]:
    """Multiply one line's hours (expect labor_out_of_range)."""
    target = _get_line(doc, line_no)
    if target.labor_hours is None:
        raise ValueError(f"line_no={line_no} has no labor_hours to inflate")
    out = _copy_doc(doc)
    new_hours = target.labor_hours * factor
    _get_line(out, line_no).labor_hours = new_hours
    return out, InjectedError(
        error_type="labor_out_of_range",
        line_no=line_no,
        detail=f"inflated hours {target.labor_hours} -> {new_hours} (x{factor})",
    )


#: Registry mapping CLI/plan names to transforms. Each entry takes
#: ``(doc, **kwargs)`` and returns ``(new_doc, InjectedError)``.
TRANSFORMS: dict[str, Callable[..., tuple[EstimateDoc, InjectedError]]] = {
    "add_phantom": add_phantom,
    "drop_line": drop_line,
    "swap_repair_replace": swap_repair_replace,
    "duplicate_line": duplicate_line,
    "inflate_hours": inflate_hours,
}


def apply_plan(
    doc: EstimateDoc, plan: list[tuple[str, dict]]
) -> tuple[EstimateDoc, list[InjectedError]]:
    """Apply a list of ``(transform_name, kwargs)`` steps in order."""
    errors: list[InjectedError] = []
    current = doc
    for name, kwargs in plan:
        try:
            transform = TRANSFORMS[name]
        except KeyError:
            raise ValueError(f"unknown seed transform {name!r}") from None
        current, error = transform(current, **kwargs)
        errors.append(error)
    return current, errors
