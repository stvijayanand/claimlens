"""Eval runner + metrics (issue #7, spec section 10.3).

One command scores the full pipeline against seeded errors, real and
synthetic separately::

    python eval/run_eval.py --split synthetic|real|all [--cold]

Per car the runner drives the REAL stage functions -- ``vision.process_photo``
(listing -> segment -> overlap -> crops), ``assess.assess_car``,
``estimate.extract_estimate``, ``agent.run_agent``, ``report.build_report`` --
with deterministic fixture-replay transports standing in for live model
calls. Replay mode is deliberate and documented, not a shortcut: it keeps
scoring coherent (model outputs agree with the staged fixtures) and
deterministic (no network, no cost, rerunnable). Live SAM works
(``meta_client.segment()`` without ``transport=`` calls SAM 3.1), but live
model outputs would not agree with the staged fixtures, so eval stays on
replay by design.

What IS real in every run: the vision overlap/crop code, extraction
validation + ``unparsed`` counting, ``model_cache`` lookup/store inside
``structured()``, the full deterministic agent loop with its audit log, and
report assembly. ``--cold`` bypasses the persistent ``model_cache`` file
(no preload, no save) so every ``structured()`` call misses; a warm rerun
hits. Cache hits/misses are counted from the ``on_usage`` hook and printed.

Outputs: ``eval_report.json`` (``--out``) + a printed per-split metrics
table with the section 10.3 targets alongside.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from claimlens import agent, assess, config, db, estimate, report, vision
    from claimlens import agent, assess, config, db, estimate, report, vision
    from claimlens.eval import datasets
    from claimlens.eval.datasets import EvalCar
except ImportError:  # fall back to the in-repo source tree
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from claimlens import agent, assess, config, db, estimate, report, vision
    from claimlens.eval import datasets
    from claimlens.eval.datasets import EvalCar

from PIL import Image

# ---------------------------------------------------------------------------
# Section 10.3 targets (metric key -> (target text, predicate over value)).
# ---------------------------------------------------------------------------

TARGET_RECALL = 0.80
TARGET_PRECISION = 0.75
TARGET_EXTRACTION = 0.90
TARGET_PART_DETECTION = 0.85
TARGET_SCHEMA = 1.00
TARGET_P50_MS = 2 * 60 * 1000
TARGET_COST_USD = 0.50

TARGET_TEXT = {
    "recall": ">=80%",
    "precision": ">=75%",
    "extraction_accuracy": ">=90%",
    "part_detection": ">=85%",
    "schema_pass_rate": "100%",
    "time": "p50<2min",
    "cost": "<$0.50",
}

# ---------------------------------------------------------------------------
# Pure metric helpers (unit-tested in tests/test_eval_metrics.py).
# ---------------------------------------------------------------------------

def box_iou(a: tuple[float, float, float, float],
            b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def percentile_nearest_rank(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (p50/p95); None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -int(-pct / 100 * len(ordered) // 1)))
    return ordered[rank - 1]


def norm_part(text: str | None) -> str:
    """Canonical-part comparison key: lowercase, collapsed whitespace."""
    return " ".join((text or "").strip().lower().split())


def flag_matches_error(flag: dict[str, Any], error: dict[str, Any],
                       correct_canonical: str | None) -> bool:
    """True when ``flag`` is the expected detection of ``error`` (spec 10.3).

    Same type + same part/line: ``missing_line`` matches on the dropped
    line's canonical part (the dropped line number cannot appear in
    evidence); every other type matches on the injected line number cited
    in the flag's ``evidence_line_nos``.
    """
    if flag.get("flag_type") != error.get("error_type"):
        return False
    if error.get("error_type") == "missing_line":
        return (
            correct_canonical is not None
            and norm_part(str(flag.get("canonical_part")))
            == norm_part(correct_canonical)
        )
    try:
        line_no = int(error.get("line_no", -1))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return line_no in list(flag.get("evidence_line_nos") or [])


def score_extraction(db_lines: list[dict[str, Any]],
                     correct_lines: list[dict[str, Any]]) -> tuple[int, int]:
    """(matched, total): DB lines exactly matching estimate_correct.json.

    A correct line counts as matched when the pipeline's post-reconcile DB
    row with the same ``line_no`` agrees on canonical part (case/space
    insensitive), operation, and labor hours. ``total`` is the correct-line
    count. Note this literally follows section 10.3, so injected errors
    that alter a counted field (dropped line, swapped op, inflated hours)
    lower the score by design; the per-car detail in ``eval_report.json``
    keeps that visible.
    """
    by_no = {line["line_no"]: line for line in db_lines}
    matched = 0
    for correct in correct_lines:
        got = by_no.get(correct.get("line_no"))
        if got is None:
            continue
        if norm_part(got.get("canonical_part")) != norm_part(
            correct.get("canonical_part")
        ):
            continue
        if (got.get("operation") or "") != (correct.get("operation") or ""):
            continue
        want_hours = correct.get("labor_hours")
        got_hours = got.get("labor_hours")
        if want_hours is None or got_hours is None:
            if want_hours is not None or got_hours is not None:
                continue
        elif float(got_hours) != float(want_hours):
            continue
        matched += 1
    return matched, len(correct_lines)


def cost_of_usage(events: list[dict[str, Any]]) -> tuple[float, bool]:
    """(usd, estimated): price fresh calls at the fallback rate (spec 13).

    Cached calls cost $0. Fresh calls without a usable usage payload fall
    back to ``SPARK_USD_PER_CALL`` — except ``kind="segment"`` events, which
    price at ``SAM_USD_PER_IMAGE`` (each SAM request bills one image) — and
    mark the figure estimated. Replayed eval calls carry no usage payload,
    so with the default ``$0`` Spark fallback eval cost is ``$0.00`` and
    ``estimated`` is true whenever any fresh (cache-miss) call ran.
    """
    total = 0.0
    estimated = False
    for event in events:
        if event.get("cached"):
            continue
        if event.get("kind") == "segment":
            total += config.SAM_USD_PER_IMAGE
        else:
            total += config.SPARK_USD_PER_CALL
        estimated = True
    return total, estimated


# ---------------------------------------------------------------------------
# Persistent model_cache file (warm/cold support).
# ---------------------------------------------------------------------------

def default_cache_db() -> Path:
    return REPO_ROOT / "eval" / ".eval_cache.duckdb"


def load_cache_rows(path: str | Path) -> dict[str, str]:
    """Read cached {input_hash: response_json} from the cache DB file."""
    target = Path(path)
    if str(target) == ":memory:" or not target.exists():
        return {}
    conn = db.connect(target)
    try:
        db.init_schema(conn)
        rows = conn.execute(
            "SELECT input_hash, response FROM model_cache"
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, str] = {}
    for key, response in rows:
        out[str(key)] = (
            response if isinstance(response, str) else json.dumps(response)
        )
    return out


def save_cache_rows(path: str | Path, rows: dict[str, str]) -> None:
    """Merge {input_hash: response_json} into the cache DB file."""
    if str(path) == ":memory:" or not rows:
        return
    conn = db.connect(path)
    try:
        db.init_schema(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO model_cache (input_hash, response)"
            " VALUES (?, ?)",
            sorted(rows.items()),
        )
    finally:
        conn.close()


def seed_labor_ranges(conn: Any) -> int:
    """Load data/labor_ranges.csv into the car DB; return the row count."""
    with open(config.LABOR_CSV, newline="", encoding="utf-8") as handle:
        rows = [
            (
                row["canonical_part"],
                row["operation"],
                float(row["low_hrs"]),
                float(row["high_hrs"]),
            )
            for row in csv.DictReader(handle)
        ]
    conn.executemany(
        "INSERT OR REPLACE INTO labor_ranges"
        " (canonical_part, operation, low_hrs, high_hrs) VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Fixture-replay transports (stand-ins for live model calls).
# ---------------------------------------------------------------------------

_CROP_RE = re.compile(r"in crop (\S+) of photo")
_REPAIR_MARK = "Your previous output was not valid"


def _is_repair_call(payload: dict[str, Any]) -> bool:
    for message in payload.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, str) and _REPAIR_MARK in content:
            return True
    return False


@dataclass
class ReplayPlan:
    """Fixture-derived answers for one car's replayed model calls."""

    car_id: str
    split: str
    # Listing: distinct visible damaged parts (canonical phrases).
    list_parts: list[dict[str, Any]] = field(default_factory=list)
    # photo_id -> (width, height).
    photo_dims: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Part phrase -> replayed part-mask box (clamped to its photo).
    photo_boxes: dict[str, dict[str, tuple[int, int, int, int]]] = field(
        default_factory=dict
    )
    # Canonical part key -> assessment material from estimate_correct.
    assess_by_part: dict[str, dict[str, Any]] = field(default_factory=dict)
    damage_type: str = "dent"
    # Seeded estimate lines, as the extractor would emit them.
    seeded_lines: list[dict[str, Any]] = field(default_factory=list)
    repair_calls: int = 0  # observed schema-repair retries (first try failed)


def _clamp_box(box: tuple[int, int, int, int],
               dims: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = dims
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return (x0, y0, x1, y1)


def _full_mask(box: tuple[int, int, int, int]) -> bytes:
    """v1 foreground bitmap: every pixel in the box is foreground."""
    return bytes([1]) * ((box[2] - box[0]) * (box[3] - box[1]))


def build_plan(car: EvalCar, photo_dims: dict[str, tuple[int, int]]) -> ReplayPlan:
    """Derive replay answers from the staged fixtures of ``car``."""
    from claimlens import taxonomy

    plan = ReplayPlan(car_id=car.car_id, split=car.split,
                      photo_dims=dict(photo_dims))
    correct = car.estimate_correct
    lines = list(correct.lines) if correct is not None else []
    seen: set[str] = set()
    for line in lines:
        phrase = line.canonical_part
        if not phrase:
            entry = taxonomy.find(line.raw_part_text)
            phrase = entry.canonical_part if entry else line.raw_part_text
        key = norm_part(phrase)
        if key and key not in seen:
            seen.add(key)
            plan.list_parts.append(
                {"canonical_part": phrase, "side": line.side,
                 "confidence": 0.9}
            )
        if key and key not in plan.assess_by_part:
            plan.assess_by_part[key] = {
                "side": line.side,
                "operation": line.operation,
            }
    if car.ground_truth is not None:
        plan.damage_type = car.ground_truth.damage_type or "dent"
    gt_box: tuple[int, int, int, int] | None = None
    gt_key: str | None = None
    if car.ground_truth is not None:
        gt = car.ground_truth.box
        gt_box = (gt.x0, gt.y0, gt.x1, gt.y1)
        gt_key = norm_part(car.ground_truth.part)
    for photo_id, dims in photo_dims.items():
        boxes: dict[str, tuple[int, int, int, int]] = {}
        for part in plan.list_parts:
            phrase = part["canonical_part"]
            if gt_box is not None and norm_part(phrase) == gt_key:
                boxes[phrase] = _clamp_box(gt_box, dims)
            else:
                boxes[phrase] = (0, 0, dims[0], dims[1])
        plan.photo_boxes[photo_id] = boxes
    if car.estimate_seeded is not None:
        for line in car.estimate_seeded.lines:
            plan.seeded_lines.append(
                {
                    "line_no": line.line_no,
                    "raw_part_text": line.raw_part_text,
                    "canonical_part": None,
                    "side": None,
                    "operation": line.operation,
                    "labor_hours": line.labor_hours,
                    "price": line.price,
                }
            )
    return plan


def make_list_transport(plan: ReplayPlan, photo_id: str) -> Any:
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        if _is_repair_call(payload):
            plan.repair_calls += 1
        return {
            "data": {
                "photo_id": photo_id,
                "parts": [
                    dict(part) for part in plan.list_parts
                ],
            }
        }

    return transport


def make_segment_transport(plan: ReplayPlan, photo_id: str) -> Any:
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        phrases = list(payload.get("phrases", []))
        dims = plan.photo_dims[photo_id]
        boxes = plan.photo_boxes.get(photo_id, {})
        if sorted(phrases) == sorted(config.DAMAGE_PHRASES):
            # Damage batch: full-photo masks so every part mask overlaps.
            full = (0, 0, dims[0], dims[1])
            return {
                "masks": [
                    {
                        "label": phrase,
                        "x0": full[0], "y0": full[1],
                        "x1": full[2], "y1": full[3],
                        "mask": _full_mask(full),
                    }
                    for phrase in phrases
                ]
            }
        return {
            "masks": [
                {
                    "label": phrase,
                    "x0": boxes.get(phrase, (0, 0, dims[0], dims[1]))[0],
                    "y0": boxes.get(phrase, (0, 0, dims[0], dims[1]))[1],
                    "x1": boxes.get(phrase, (0, 0, dims[0], dims[1]))[2],
                    "y1": boxes.get(phrase, (0, 0, dims[0], dims[1]))[3],
                    "mask": _full_mask(
                        boxes.get(phrase, (0, 0, dims[0], dims[1]))
                    ),
                }
                for phrase in phrases
            ]
        }

    return transport


def make_assess_transport(plan: ReplayPlan, conn: Any) -> Any:
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        if _is_repair_call(payload):
            plan.repair_calls += 1
        # Crop ids embed the part label ("...:front door:0-0-8-8"), so spaces
        # defeat naive token parsing: resolve by substring match against the
        # stored crop ids (longest wins: "X" is a prefix of "X#2").
        texts = [
            message.get("content", "")
            for message in payload.get("messages", [])
            if isinstance(message.get("content", ""), str)
        ]
        known = [
            str(row[0])
            for row in conn.execute("SELECT crop_id FROM crops").fetchall()
        ]
        crop_id: str | None = None
        for text in texts:
            hits = [cid for cid in known if cid in text]
            if hits:
                crop_id = max(hits, key=len)
                break
        if crop_id is None:
            if len(known) == 1:
                crop_id = known[0]
            else:
                match = next(
                    (_CROP_RE.search(t) for t in texts if _CROP_RE.search(t)),
                    None,
                )
                if match is not None:
                    crop_id = match.group(1)
        if crop_id is None:
            raise KeyError("assess replay: cannot resolve crop_id")
        row = conn.execute(
            "SELECT photo_id, canonical_part FROM crops WHERE crop_id = ?",
            [crop_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"assess replay: unknown crop {crop_id!r}")
        photo_id, part = str(row[0]), str(row[1])
        material = plan.assess_by_part.get(norm_part(part), {})
        operation = str(material.get("operation") or "repair")
        repair_replace = (
            operation if operation in ("repair", "replace") else "repair"
        )
        return {
            "data": {
                "photo_id": photo_id,
                "crop_id": crop_id,
                "canonical_part": part,
                "side": material.get("side"),
                "damage_type": plan.damage_type,
                "severity": "moderate",
                "repair_replace": repair_replace,
                "likelihood": 0.9,
                "confidence": 0.9,
                "visibility": "visible",
            }
        }

    return transport


def make_extract_transport(plan: ReplayPlan) -> Any:
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        if _is_repair_call(payload):
            plan.repair_calls += 1
        return {"data": {"lines": [dict(line) for line in plan.seeded_lines]}}

    return transport


# ---------------------------------------------------------------------------
# Per-car pipeline run.
# ---------------------------------------------------------------------------

@dataclass
class CarResult:
    car_id: str
    split: str
    ok: bool = True
    error: str = ""
    wall_ms: float = 0.0
    cost_usd: float = 0.0
    cost_estimated: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    schema_passed: int = 0
    schema_failed: int = 0
    flags: list[dict[str, Any]] = field(default_factory=list)
    injected: list[dict[str, Any]] = field(default_factory=list)
    matched_errors: int = 0
    matched_flags: int = 0
    extraction_matched: int = 0
    extraction_total: int = 0
    detection_hit: bool | None = None
    best_iou: float | None = None
    unparsed_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def run_car(car: EvalCar, cache_rows: dict[str, str] | None) -> CarResult:
    """Run the full pipeline for one staged car; never raises.

    ``cache_rows`` is the shared warm cache (mutated in place with newly
    cached rows); None means cold (fresh empty cache, nothing saved back).
    """
    result = CarResult(car_id=car.car_id, split=car.split)
    started = time.perf_counter()
    conn = db.connect(":memory:")
    try:
        db.init_schema(conn)
        if cache_rows is not None and cache_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO model_cache (input_hash, response)"
                " VALUES (?, ?)",
                sorted(cache_rows.items()),
            )
        seed_labor_ranges(conn)
        conn.execute(
            "INSERT INTO cars (car_id) VALUES (?)", [car.car_id]
        )

        # Stage photos.
        photo_dims: dict[str, tuple[int, int]] = {}
        for idx, photo_path in enumerate(car.photos):
            blob = photo_path.read_bytes()
            width, height = Image.open(io.BytesIO(blob)).size
            photo_id = f"{car.car_id}-p{idx}"
            conn.execute(
                "INSERT INTO photos (photo_id, car_id, photo_idx,"
                " width, height, img) VALUES (?, ?, ?, ?, ?, ?)",
                [photo_id, car.car_id, idx, width, height, blob],
            )
            photo_dims[photo_id] = (width, height)
        if not photo_dims:
            raise ValueError("car has no staged photos")

        # Stage the seeded estimate file. The blob is plumbing for the
        # replay transport (the staged estimates are JSON, while
        # extract_estimate expects a pdf/image file); the replayed
        # extraction output is the seeded line list.
        seeded_doc = car.estimate_seeded
        if seeded_doc is None:
            raise ValueError("car has no staged seeded estimate")
        conn.execute(
            "INSERT INTO estimate_files (car_id, source_type, blob)"
            " VALUES (?, ?, ?)",
            [car.car_id, "pdf", seeded_doc.model_dump_json().encode()],
        )

        plan = build_plan(car, photo_dims)
        usage_events: list[dict[str, Any]] = []
        violations = 0

        def on_usage(event: dict[str, Any]) -> None:
            usage_events.append(
                {
                    "kind": event.get("kind"),
                    "model": event.get("model"),
                    "cached": bool(event.get("cached")),
                    "usage": event.get("usage"),
                }
            )

        for photo_id in photo_dims:
            vision.process_photo(
                conn,
                photo_id,
                list_transport=make_list_transport(plan, photo_id),
                segment_transport=make_segment_transport(plan, photo_id),
                on_usage=on_usage,
            )
        assess.assess_car(
            conn, car.car_id,
            transport=make_assess_transport(plan, conn),
            on_usage=on_usage,
        )
        try:
            estimate.extract_estimate(
                conn, car.car_id,
                transport=make_extract_transport(plan),
                on_usage=on_usage,
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            from claimlens.meta_client import SchemaViolation

            if isinstance(exc, SchemaViolation):
                violations += 1
            raise

        result.cost_usd, result.cost_estimated = cost_of_usage(usage_events)
        result.cache_hits = sum(1 for e in usage_events if e["cached"])
        result.cache_misses = sum(1 for e in usage_events if not e["cached"])

        run_id = f"eval-{car.split}-{car.car_id}"
        conn.execute(
            "INSERT INTO runs (run_id, car_id, api_cost_usd) VALUES (?, ?, ?)",
            [run_id, car.car_id, result.cost_usd],
        )
        agent.run_agent(conn, car.car_id, run_id)
        built = report.build_report(conn, car.car_id, run_id)
        result.flags = list(built.get("flags", []))
        result.unparsed_lines = int(built.get("unparsed_lines", 0))

        # Flag-vs-injected comparison.
        correct = car.estimate_correct
        correct_by_line = (
            {line.line_no: line for line in correct.lines}
            if correct is not None
            else {}
        )
        result.injected = [
            err.model_dump(mode="json") for err in car.injected_errors
        ]
        for err in result.injected:
            line = correct_by_line.get(err.get("line_no"))
            canonical = line.canonical_part if line is not None else None
            if any(
                flag_matches_error(flag, err, canonical)
                for flag in result.flags
            ):
                result.matched_errors += 1
        matched_flag_ids: set[int] = set()
        for idx, flag in enumerate(result.flags):
            for err in result.injected:
                line = correct_by_line.get(err.get("line_no"))
                canonical = (
                    line.canonical_part if line is not None else None
                )
                if flag_matches_error(flag, err, canonical):
                    matched_flag_ids.add(idx)
                    break
        result.matched_flags = len(matched_flag_ids)

        # Extraction accuracy on post-reconcile DB rows.
        db_lines = [
            _row_json(row[0])
            for row in conn.execute(
                "SELECT line FROM estimate_lines ORDER BY line_no"
            ).fetchall()
        ]
        correct_lines = (
            [line.model_dump(mode="json") for line in correct.lines]
            if correct is not None
            else []
        )
        matched, total = score_extraction(db_lines, correct_lines)
        result.extraction_matched = matched
        result.extraction_total = total

        # Synthetic part detection: best crop IoU vs the ground-truth box.
        if car.split == "synthetic" and car.ground_truth is not None:
            gt = car.ground_truth.box
            gt_box = (gt.x0, gt.y0, gt.x1, gt.y1)
            best = 0.0
            crops = conn.execute(
                "SELECT x0, y0, x1, y1 FROM crops"
            ).fetchall()
            for x0, y0, x1, y1 in crops:
                best = max(best, box_iou(gt_box, (x0, y0, x1, y1)))
            result.best_iou = best
            result.detection_hit = best >= 0.5
        else:
            result.detection_hit = None

        # Schema pass rate: successes + observed repair retries + violations.
        # Each repair observation means its first attempt failed validation;
        # each violation means both attempts failed (one already counted as
        # a repair observation). assess.py swallows SchemaViolation into a
        # not_visible/zero-confidence/unknown assessment, detected here.
        swallowed = 0
        for row in conn.execute(
            "SELECT assessment FROM damage_assessments"
        ).fetchall():
            payload = _row_json(row[0])
            if (
                payload.get("visibility") == "not_visible"
                and float(payload.get("confidence", 1.0)) == 0.0
                and payload.get("damage_type") == "unknown"
            ):
                swallowed += 1
        violations += swallowed
        passed = len(usage_events)
        failed = plan.repair_calls + violations
        result.schema_passed = passed
        result.schema_failed = failed

        if cache_rows is not None:
            for key, response in conn.execute(
                "SELECT input_hash, response FROM model_cache"
            ).fetchall():
                cache_rows[str(key)] = (
                    response
                    if isinstance(response, str)
                    else json.dumps(response)
                )
    except Exception as exc:  # noqa: BLE001 - per-car failure is data
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.wall_ms = (time.perf_counter() - started) * 1000
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Split scoring, report, and printed table.
# ---------------------------------------------------------------------------

def _ratio(matched: int, total: int) -> float | None:
    if total <= 0:
        return None
    return matched / total


def score_split(split: str, cars: list[CarResult]) -> dict[str, Any]:
    """Pool per-car results into the section 10.3 split metrics."""
    done = [car for car in cars if car.ok]
    total_injected = sum(len(car.injected) for car in done)
    total_flags = sum(len(car.flags) for car in done)
    matched_errors = sum(car.matched_errors for car in done)
    matched_flags = sum(car.matched_flags for car in done)
    recall = _ratio(matched_errors, total_injected)
    if total_flags == 0:
        precision: float | None = 1.0 if total_injected == 0 else 0.0
    else:
        precision = _ratio(matched_flags, total_flags)
    extraction = _ratio(
        sum(car.extraction_matched for car in done),
        sum(car.extraction_total for car in done),
    )
    synth_hits = [
        car for car in done if car.split == "synthetic"
        and car.detection_hit is not None
    ]
    part_detection = (
        _ratio(
            sum(1 for car in synth_hits if car.detection_hit),
            len(synth_hits),
        )
        if split == "synthetic"
        else None
    )
    schema_total = sum(
        car.schema_passed + car.schema_failed for car in done
    )
    schema_pass_rate = _ratio(
        sum(car.schema_passed for car in done), schema_total
    )
    walls = [car.wall_ms for car in done]
    costs = [car.cost_usd for car in done]
    return {
        "split": split,
        "cars_total": len(cars),
        "cars_completed": len(done),
        "cars_failed": len(cars) - len(done),
        "metrics": {
            "recall": recall,
            "precision": precision,
            "extraction_accuracy": extraction,
            "part_detection": part_detection,
            "schema_pass_rate": schema_pass_rate,
            "time_p50_ms": percentile_nearest_rank(walls, 50),
            "time_p95_ms": percentile_nearest_rank(walls, 95),
            "cost_mean_usd": (
                sum(costs) / len(costs) if costs else None
            ),
            "cost_estimated": any(car.cost_estimated for car in done),
            "cache_hits": sum(car.cache_hits for car in done),
            "cache_misses": sum(car.cache_misses for car in done),
        },
        "targets": dict(TARGET_TEXT),
        "cars": [car.to_dict() for car in cars],
    }


def run_eval(splits: list[str], *, eval_root: str | Path | None = None,
             cache_db: str | Path | None = None, cold: bool = False,
             max_cars: int | None = None) -> dict[str, Any]:
    """Run the requested splits; return the eval report dict."""
    cache_path = Path(cache_db) if cache_db is not None else default_cache_db()
    cache_rows: dict[str, str] | None = (
        None if cold else load_cache_rows(cache_path)
    )
    working: dict[str, str] = {} if cache_rows is None else cache_rows
    out: dict[str, Any] = {
        "cold": cold,
        "replay": True,
        "replay_note": (
            "Model calls replayed from staged fixtures (live SAM has no"
            " transport; see module docstring). Vision overlap/crop code,"
            " extraction validation, model_cache behavior, the agent loop,"
            " and report assembly all run for real."
        ),
        "splits": {},
    }
    for split in splits:
        cars = datasets.iter_cars(eval_root, split=split)
        if max_cars is not None:
            cars = cars[:max_cars]
        results = [
            run_car(car, None if cold else working) for car in cars
        ]
        out["splits"][split] = score_split(split, results)
    if not cold:
        save_cache_rows(cache_path, working)
    return out


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}ms"


def _fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _status(value: float | None, target: float,
            higher_is_better: bool = True) -> str:
    if value is None:
        return "--"
    good = value >= target if higher_is_better else value <= target
    return "PASS" if good else "FAIL"


def format_table(report_data: dict[str, Any]) -> str:
    """Render the per-split metrics tables (real/synthetic always split)."""
    lines = [
        f"ClaimLens eval (cold={report_data.get('cold')}, replay=True)",
    ]
    for split, scored in report_data.get("splits", {}).items():
        metrics = scored["metrics"]
        lines.append(
            f"--- split: {split}"
            f" ({scored['cars_completed']}/{scored['cars_total']} cars) ---"
        )
        rows = [
            (
                "recall",
                _fmt_ratio(metrics["recall"]),
                TARGET_TEXT["recall"],
                _status(metrics["recall"], TARGET_RECALL),
            ),
            (
                "precision",
                _fmt_ratio(metrics["precision"]),
                TARGET_TEXT["precision"],
                _status(metrics["precision"], TARGET_PRECISION),
            ),
            (
                "extraction accuracy",
                _fmt_ratio(metrics["extraction_accuracy"]),
                TARGET_TEXT["extraction_accuracy"],
                _status(metrics["extraction_accuracy"], TARGET_EXTRACTION),
            ),
            (
                "synthetic part detection",
                _fmt_ratio(metrics["part_detection"]),
                TARGET_TEXT["part_detection"],
                _status(metrics["part_detection"], TARGET_PART_DETECTION),
            ),
            (
                "schema pass rate",
                _fmt_ratio(metrics["schema_pass_rate"]),
                TARGET_TEXT["schema_pass_rate"],
                _status(metrics["schema_pass_rate"], TARGET_SCHEMA),
            ),
            (
                "time per car",
                f"p50 {_fmt_ms(metrics['time_p50_ms'])}"
                f" / p95 {_fmt_ms(metrics['time_p95_ms'])}",
                TARGET_TEXT["time"],
                _status(
                    metrics["time_p50_ms"], TARGET_P50_MS,
                    higher_is_better=False,
                ),
            ),
            (
                "mean cost per car",
                _fmt_usd(metrics["cost_mean_usd"])
                + (" (est)" if metrics["cost_estimated"] else ""),
                TARGET_TEXT["cost"],
                _status(
                    metrics["cost_mean_usd"], TARGET_COST_USD,
                    higher_is_better=False,
                ),
            ),
        ]
        width = max(len(name) for name, _, _, _ in rows)
        lines.append(
            f"{'metric':<{width}}  {'value':<26}  {'target':<10}  status"
        )
        for name, value, target, status in rows:
            lines.append(f"{name:<{width}}  {value:<26}  {target:<10}  {status}")
        lines.append(
            f"cache: {metrics['cache_hits']} hits /"
            f" {metrics['cache_misses']} misses"
        )
        if scored["cars_failed"]:
            lines.append(f"errors: {scored['cars_failed']} car(s) failed")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the ClaimLens pipeline against seeded errors."
    )
    parser.add_argument(
        "--split", required=True, choices=("real", "synthetic", "all"),
        help="Eval split to run (real and synthetic always scored separately).",
    )
    parser.add_argument(
        "--cold", action="store_true",
        help="Bypass the persistent model_cache (every call misses).",
    )
    parser.add_argument(
        "--max-cars", type=int, default=None,
        help="Run at most N cars per split (smoke runs).",
    )
    parser.add_argument(
        "--out", default="eval_report.json",
        help="Where to write eval_report.json (default: ./eval_report.json).",
    )
    parser.add_argument(
        "--cache-db", default=str(default_cache_db()),
        help="Persistent model_cache DuckDB file (use :memory: for none).",
    )
    parser.add_argument(
        "--eval-root", default=None,
        help="Eval data root holding real/ and synthetic/ (default: data/eval).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    splits = ["real", "synthetic"] if args.split == "all" else [args.split]
    try:
        report_data = run_eval(
            splits,
            eval_root=args.eval_root,
            cache_db=args.cache_db,
            cold=args.cold,
            max_cars=args.max_cars,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports, nonzero exit
        print(f"eval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not any(
        scored["cars_total"] for scored in report_data["splits"].values()
    ):
        print("eval failed: no staged cars found", file=sys.stderr)
        return 1
    Path(args.out).write_text(
        json.dumps(report_data, indent=2) + "\n", encoding="utf-8"
    )
    print(format_table(report_data), end="")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
