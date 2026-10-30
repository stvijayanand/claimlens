"""Part-name taxonomy: canonical vocabulary, alias lookup, side parsing.

The CSV owns the canonical part names. ``find()`` resolves raw text to a
canonical entry (exact match, then longest-substring fallback); ``combine_side()``
folds side tokens (LF, left/front, ...) into FL/FR/RL/RR/center.

Deliberate v1 simplifications (documented, revisit on eval evidence):
- ``rr`` reads as right-rear (RR). A bare positional "rear" reading only
  matters for parts whose canonical name already contains the position
  (e.g. "rear door"), which alias lookup handles.
- driver/passenger side words are intentionally unmapped (market-dependent).
- Partial side info (only left/right, only front/rear) yields None; the caller
  applies a confidence penalty instead of guessing.
- Bare "door" synonyms resolve to front/rear door only with a front/rear side
  (handled by normalize_part_name in Issue #4 using GENERIC_DOOR_ALIASES).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config

# Full-side tokens.
_FULL_SIDES = {
    "fl": "FL",
    "lf": "FL",
    "fr": "FR",
    "rf": "FR",
    "rl": "RL",
    "lr": "RL",
    "rr": "RR",
}
# Single-axis tokens: L/R/F/Re/C.
_AXIS_SIDES = {
    "l": "L",
    "lt": "L",
    "left": "L",
    "lh": "L",
    "r": "R",
    "rt": "R",
    "right": "R",
    "rh": "R",
    "f": "F",
    "frt": "F",
    "front": "F",
    "rear": "Re",
    "back": "Re",
    "bk": "Re",
    "center": "C",
    "ctr": "C",
    "middle": "C",
}

#: Bare door synonyms needing a front/rear side to resolve (see module docstring).
GENERIC_DOOR_ALIASES = frozenset(
    {"door", "door outer panel", "door shell", "door skin"}
)


@dataclass(frozen=True)
class TaxonomyEntry:
    canonical_part: str
    side_scope: str  # sided | unsided
    hidden: bool  # True = cannot be photographed (behind-panel internals)


def load_taxonomy(path: str | Path | None = None) -> dict[str, TaxonomyEntry]:
    """Load alias -> entry map from the taxonomy CSV."""
    target = Path(path or config.TAXONOMY_CSV)
    entries: dict[str, TaxonomyEntry] = {}
    with target.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entries[row["alias"].strip().lower()] = TaxonomyEntry(
                canonical_part=row["canonical_part"].strip(),
                side_scope=row["side_scope"].strip(),
                hidden=bool(int(row["hidden"])),
            )
    return entries


@lru_cache(maxsize=1)
def get_taxonomy() -> dict[str, TaxonomyEntry]:
    """Cached alias map from the configured taxonomy CSV."""
    return load_taxonomy()


def find(text: str) -> TaxonomyEntry | None:
    """Resolve raw text to a taxonomy entry.

    Exact alias match first, then longest-alias-substring fallback so
    "front bumper cover outer" still resolves. Returns None when unknown.
    """
    aliases = get_taxonomy()
    key = " ".join(text.strip().lower().split())
    if key in aliases:
        return aliases[key]
    best: TaxonomyEntry | None = None
    best_len = 0
    for alias, entry in aliases.items():
        if alias in key and len(alias) > best_len:
            best, best_len = entry, len(alias)
    return best


def combine_side(words: list[str]) -> str | None:
    """Fold side tokens into FL/FR/RL/RR/center, or None if unresolvable."""
    tokens = [w.strip().lower() for w in words]
    fulls = [_FULL_SIDES[t] for t in tokens if t in _FULL_SIDES]
    if fulls:
        return fulls[0] if len(set(fulls)) == 1 else None
    axes = {_AXIS_SIDES[t] for t in tokens if t in _AXIS_SIDES}
    if axes == {"C"}:
        return "center"
    lateral = ({"L"} <= axes, {"R"} <= axes)
    longitudinal = ({"F"} <= axes, {"Re"} <= axes)
    if lateral == (True, False) and longitudinal == (True, False):
        return "FL"
    if lateral == (False, True) and longitudinal == (True, False):
        return "FR"
    if lateral == (True, False) and longitudinal == (False, True):
        return "RL"
    if lateral == (False, True) and longitudinal == (False, True):
        return "RR"
    return None


@lru_cache(maxsize=1)
def prompt_block() -> str:
    """Render the taxonomy as a reusable (and cacheable) prompt block."""
    seen: dict[str, list[str]] = {}
    for alias, entry in sorted(get_taxonomy().items()):
        seen.setdefault(entry.canonical_part, []).append(alias)
    lines = []
    for canonical in sorted(seen):
        entry = get_taxonomy()[seen[canonical][0]]
        tags = [entry.side_scope] + (["hidden"] if entry.hidden else [])
        lines.append(
            f"- {canonical} ({', '.join(tags)}); also called: "
            + ", ".join(sorted(seen[canonical]))
        )
    return "Known car parts:\n" + "\n".join(lines)
