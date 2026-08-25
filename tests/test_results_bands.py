"""Tests for the banding helper the committed claims are written with.

Every threshold in ``results/waterfall.md`` goes through ``band``, so a bug
here would silently weaken or overstate a published claim. It is a pure
function, so it is cheap to pin exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The script lives outside the package, so import it by path rather than
# duplicating the helper into src/ just to make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_results import PROMPTS, REPLIES, band  # noqa: E402


def test_band_rounds_down_to_the_cleared_edge() -> None:
    assert band(0.78, (0.30, 0.50, 0.70)) == 0.70
    assert band(0.55, (0.30, 0.50, 0.70)) == 0.50
    assert band(0.31, (0.30, 0.50, 0.70)) == 0.30


def test_band_is_inclusive_at_an_edge() -> None:
    assert band(0.50, (0.30, 0.50, 0.70)) == 0.50


def test_band_never_overstates_the_measurement() -> None:
    """The published claim must always be true of the measured value."""
    edges = (0.30, 0.50, 0.70)
    for value in (0.30, 0.42, 0.5, 0.69, 0.71, 0.99):
        assert band(value, edges) <= value


def test_band_is_stable_across_small_perturbations() -> None:
    """Why banding exists: noise inside a band must not move the claim."""
    edges = (0.30, 0.50, 0.70)
    assert band(0.78, edges) == band(0.80, edges) == band(0.72, edges)


def test_band_raises_below_the_lowest_edge() -> None:
    # A value under every edge means the claim itself has broken, and the
    # script should fail rather than publish a threshold it cannot support.
    with pytest.raises(AssertionError, match="below the lowest band edge"):
        band(0.10, (0.30, 0.50))


def test_prompts_and_replies_line_up() -> None:
    # The stub selects a reply per prompt; a mismatch would silently reuse
    # one reply for several prompts and flatten the TTS measurement.
    assert len(PROMPTS) == len(REPLIES)


def test_every_reply_is_multi_sentence() -> None:
    """Piper chunks per sentence, so a one-sentence reply cannot stream.

    If a reply here were single-sentence, first-chunk and total synthesis
    time would coincide and the ADR's central measurement would silently
    become vacuous for that turn.
    """
    for reply in REPLIES:
        assert reply.count(".") + reply.count("?") >= 2, reply


def test_no_prompt_is_empty() -> None:
    assert all(p.strip() for p in PROMPTS)
