"""Tests for the first-chunk accounting.

These use hand-built ``TtsResult`` objects rather than a real voice. The
arithmetic that the ADR rests on -- first chunk versus total, and the penalty
for buffering -- is exactly checkable, and doing it without a 60 MB ONNX
keeps the property under test visible.
"""

from __future__ import annotations

import pytest

from voice_loop_latency_budget.tts import AudioChunk, TtsResult, playback_start_s

SR = 22_050


def result(*chunks: tuple[float, float], total_s: float) -> TtsResult:
    """Build a result from (ready_s, duration_s) pairs."""
    return TtsResult(
        chunks=[
            AudioChunk(i, int(dur * SR), SR, ready_s=ready)
            for i, (ready, dur) in enumerate(chunks)
        ],
        total_compute_s=total_s,
    )


def test_chunk_duration_from_sample_count() -> None:
    assert AudioChunk(0, SR, SR, 0.1).duration_s == pytest.approx(1.0)


def test_first_chunk_is_the_first_yielded() -> None:
    r = result((0.2, 1.0), (0.5, 1.0), total_s=0.6)
    assert r.first_chunk_s == pytest.approx(0.2)


def test_audio_duration_sums_all_chunks() -> None:
    r = result((0.2, 1.5), (0.5, 2.5), total_s=0.6)
    assert r.audio_duration_s == pytest.approx(4.0)


def test_blocking_penalty_is_what_buffering_costs() -> None:
    """The ADR's headline: identical audio, later start, no benefit."""
    r = result((0.2, 2.0), (0.6, 2.0), total_s=0.6)
    assert r.blocking_penalty_s == pytest.approx(0.4)


def test_no_penalty_when_a_single_chunk_is_the_whole_utterance() -> None:
    # Streaming cannot help if the engine yields once at the very end.
    r = result((0.5, 3.0), total_s=0.5)
    assert r.blocking_penalty_s == pytest.approx(0.0)


def test_first_chunk_can_be_far_below_total_audio_duration() -> None:
    """Why total audio duration is the wrong latency number.

    6 s of speech, first chunk playable in 200 ms. Reporting the 6 s as
    latency would be off by 30x.
    """
    r = result((0.2, 3.0), (0.4, 3.0), total_s=0.45)
    assert r.audio_duration_s == pytest.approx(6.0)
    assert r.first_chunk_s == pytest.approx(0.2)
    assert r.first_chunk_s < r.audio_duration_s / 10


def test_real_time_factor_below_one_means_playback_outruns_synthesis() -> None:
    # 0.6 s of compute for 4 s of audio: chunk n plays while n+1 is made, so
    # streaming never underruns.
    r = result((0.2, 2.0), (0.6, 2.0), total_s=0.6)
    assert r.real_time_factor == pytest.approx(0.15)
    assert r.real_time_factor < 1.0


def test_empty_result_has_no_first_chunk() -> None:
    with pytest.raises(ValueError, match="no audio"):
        _ = TtsResult().first_chunk_s


def test_real_time_factor_of_empty_result_is_infinite() -> None:
    assert TtsResult().real_time_factor == float("inf")


def test_playback_starts_on_first_chunk_when_streaming() -> None:
    r = result((0.2, 2.0), (0.6, 2.0), total_s=0.6)
    assert playback_start_s(r, streaming=True) == pytest.approx(0.2)


def test_playback_waits_for_the_last_sample_when_blocking() -> None:
    r = result((0.2, 2.0), (0.6, 2.0), total_s=0.6)
    assert playback_start_s(r, streaming=False) == pytest.approx(0.6)


def test_streaming_is_never_slower_than_blocking() -> None:
    r = result((0.2, 2.0), (0.6, 2.0), total_s=0.6)
    assert playback_start_s(r, True) <= playback_start_s(r, False)
