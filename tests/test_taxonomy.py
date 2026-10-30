"""Taxonomy tests: alias lookup, hidden markers, side folding."""

import pytest

from claimlens import taxonomy


@pytest.mark.parametrize(
    ("raw", "canonical", "scope", "hidden"),
    [
        ("frt bpr cvr", "front bumper cover", "unsided", False),
        ("FRt Bumper Cover", "front bumper cover", "unsided", False),
        ("front bumper cover outer", "front bumper cover", "unsided", False),
        ("headlamp", "headlight", "sided", False),
        ("qtr panel", "quarter panel", "sided", False),
        ("radiator support", "radiator support", "unsided", True),
        ("crash bar", "bumper reinforcement", "unsided", True),
    ],
)
def test_find_aliases(raw, canonical, scope, hidden):
    entry = taxonomy.find(raw)
    assert entry is not None
    assert entry.canonical_part == canonical
    assert entry.side_scope == scope
    assert entry.hidden is hidden


def test_find_unknown_returns_none():
    assert taxonomy.find("rocket engine") is None


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (["lf"], "FL"),
        (["left", "front"], "FL"),
        (["RF"], "FR"),
        (["right", "rear"], "RR"),
        (["rr"], "RR"),
        (["lr"], "RL"),
        (["center"], "center"),
        (["left"], None),  # partial info: caller penalizes, never guesses
        (["rear"], None),
        (["driver"], None),  # intentionally unmapped (market-dependent)
        (["left", "right"], None),  # conflicting tokens
        ([], None),
    ],
)
def test_combine_side(words, expected):
    assert taxonomy.combine_side(words) == expected


def test_prompt_block_lists_canonical_parts():
    block = taxonomy.prompt_block()
    assert "front bumper cover" in block
    assert "headlight (sided" in block
    assert "radiator support (unsided, hidden)" in block
