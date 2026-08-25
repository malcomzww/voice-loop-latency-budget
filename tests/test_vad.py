"""Tests for endpointing.

The endpointing decision is pure logic over a probability array, so it is
tested directly on synthetic probability sequences rather than on audio. That
keeps these tests fast, deterministic and independent of whether the ONNX
model is installed -- and it tests the part that actually has behaviour, since
the model call itself is one line.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_loop_latency_budget.vad import (
    FRAME_MS,
    FRAME_SAMPLES,
    VadConfig,
    endpoint_from_probabilities,
    frames,
)


def probs(*runs: tuple[float, int]) -> np.ndarray:
    """Build a probability array from (value, frame_count) runs."""
    return np.concatenate([np.full(n, v, dtype=np.float32) for v, n in runs])


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def test_frames_are_fixed_size() -> None:
    got = list(frames(np.zeros(FRAME_SAMPLES * 3, dtype=np.float32)))
    assert len(got) == 3
    assert all(len(f) == FRAME_SAMPLES for f in got)


def test_frames_drops_the_short_tail_rather_than_padding() -> None:
    """A zero-padded tail scores as silence and would fake an endpoint."""
    got = list(frames(np.zeros(FRAME_SAMPLES * 2 + 100, dtype=np.float32)))
    assert len(got) == 2


def test_frames_of_too_short_audio_is_empty() -> None:
    assert list(frames(np.zeros(10, dtype=np.float32))) == []


def test_frame_is_32ms() -> None:
    assert FRAME_MS == 32


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_config_rejects_threshold_outside_the_open_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        VadConfig(threshold=bad)


def test_config_rejects_negative_hangover() -> None:
    with pytest.raises(ValueError, match="hangover"):
        VadConfig(hangover_ms=-1)


# --------------------------------------------------------------------------
# Endpoint detection
# --------------------------------------------------------------------------


def test_silence_yields_no_endpoint() -> None:
    assert endpoint_from_probabilities(probs((0.01, 50)), VadConfig()) is None


def test_empty_input_yields_no_endpoint() -> None:
    assert endpoint_from_probabilities(np.zeros(0, np.float32), VadConfig()) is None


def test_detects_speech_bounds() -> None:
    # 10 silent frames, 20 speech frames, 30 silent.
    p = probs((0.01, 10), (0.9, 20), (0.01, 30))
    ep = endpoint_from_probabilities(p, VadConfig(hangover_ms=500))
    assert ep is not None
    assert ep.speech_start_s == pytest.approx(10 * FRAME_MS / 1000)
    assert ep.speech_end_s == pytest.approx(30 * FRAME_MS / 1000)
    assert ep.speech_duration_s == pytest.approx(20 * FRAME_MS / 1000)


def test_endpoint_wait_equals_the_hangover() -> None:
    """The VAD's real latency contribution, and the knob that changes it."""
    p = probs((0.01, 5), (0.9, 20), (0.01, 40))
    for hangover in (200, 500, 800):
        ep = endpoint_from_probabilities(p, VadConfig(hangover_ms=hangover))
        assert ep is not None
        assert ep.endpoint_s == pytest.approx(hangover / 1000, abs=1e-9)


def test_short_gap_does_not_end_the_turn() -> None:
    """A pause between words must not be mistaken for the end of a turn."""
    # 10 speech, 4 silent frames (128 ms) -- well under a 500 ms hangover --
    # then 10 more speech. One turn spanning the gap, not two.
    p = probs((0.9, 10), (0.01, 4), (0.9, 10), (0.01, 40))
    ep = endpoint_from_probabilities(p, VadConfig(hangover_ms=500))
    assert ep is not None
    assert ep.speech_end_s == pytest.approx(24 * FRAME_MS / 1000)


def test_long_gap_ends_the_turn_at_the_first_run() -> None:
    # 30 silent frames = 960 ms > 500 ms hangover, so the turn ends before
    # the second utterance begins.
    p = probs((0.9, 10), (0.01, 30), (0.9, 10), (0.01, 40))
    ep = endpoint_from_probabilities(p, VadConfig(hangover_ms=500))
    assert ep is not None
    assert ep.speech_end_s == pytest.approx(10 * FRAME_MS / 1000)


def test_a_blip_shorter_than_min_speech_is_not_a_turn() -> None:
    # One 32 ms frame of speech against a 100 ms minimum.
    p = probs((0.01, 10), (0.9, 1), (0.01, 40))
    assert endpoint_from_probabilities(p, VadConfig(min_speech_ms=100)) is None


def test_threshold_decides_what_counts_as_speech() -> None:
    p = probs((0.01, 5), (0.6, 20), (0.01, 40))
    assert endpoint_from_probabilities(p, VadConfig(threshold=0.5)) is not None
    assert endpoint_from_probabilities(p, VadConfig(threshold=0.8)) is None


def test_speech_running_to_the_end_still_endpoints() -> None:
    # No trailing silence: the decision still lands hangover after the last
    # speech frame, which is what a live loop would do on stream close.
    p = probs((0.01, 5), (0.9, 20))
    ep = endpoint_from_probabilities(p, VadConfig(hangover_ms=500))
    assert ep is not None
    assert ep.speech_end_s == pytest.approx(25 * FRAME_MS / 1000)


def test_compute_time_is_carried_through_unchanged() -> None:
    p = probs((0.01, 5), (0.9, 20), (0.01, 40))
    ep = endpoint_from_probabilities(p, VadConfig(), compute_s=0.004)
    assert ep is not None
    assert ep.compute_s == pytest.approx(0.004)


def test_endpoint_compute_and_wait_are_independent() -> None:
    """The point of splitting them: hangover dwarfs compute by ~100x."""
    p = probs((0.01, 5), (0.9, 20), (0.01, 40))
    ep = endpoint_from_probabilities(p, VadConfig(hangover_ms=500), compute_s=0.004)
    assert ep is not None
    assert ep.endpoint_s > ep.compute_s * 50
