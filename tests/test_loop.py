"""Tests for the loop's orchestration, ASR partial accounting, and the stub.

Driven entirely by fakes. Loading Whisper costs ~10 s and Piper ~1.5 s, and
none of the behaviour under test here is a property of those models -- it is
a property of how the loop sequences them and adds up what they report.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_loop_latency_budget.asr import AsrResult, Partial
from voice_loop_latency_budget.loop import SimulatedLlm, VoiceLoop
from voice_loop_latency_budget.tracer import SIMULATED_HOPS
from voice_loop_latency_budget.tts import AudioChunk, TtsResult
from voice_loop_latency_budget.vad import FRAME_SAMPLES, VadConfig

SR = 16_000


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeVad:
    """Reports speech for a fixed middle portion of the audio."""

    def __init__(self, speech_frames: int = 40, lead_frames: int = 10) -> None:
        self.speech_frames = speech_frames
        self.lead_frames = lead_frames

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio) // FRAME_SAMPLES
        p = np.full(n, 0.01, dtype=np.float32)
        p[self.lead_frames : self.lead_frames + self.speech_frames] = 0.95
        return p


class FakeAsr:
    """Returns fixed text, charging a fixed cost per partial."""

    def __init__(self, cost_s: float = 0.05) -> None:
        self.cost_s = cost_s
        self.calls = 0

    def stream(self, audio, partial_interval_s=0.5, clock=None) -> AsrResult:
        self.calls += 1
        total_s = len(audio) / SR
        r = AsrResult()
        cursor = partial_interval_s
        while cursor < total_s:
            r.partials.append(Partial("partial", cursor, self.cost_s, is_final=False))
            cursor += partial_interval_s
        r.partials.append(Partial("hello world", total_s, self.cost_s, is_final=True))
        return r


class FakeTts:
    """Yields two chunks, the first well before the last."""

    sample_rate = 22_050

    def synthesize(self, text, clock=None) -> tuple[TtsResult, bytes]:
        r = TtsResult(
            chunks=[
                AudioChunk(0, self.sample_rate * 2, self.sample_rate, ready_s=0.10),
                AudioChunk(1, self.sample_rate * 2, self.sample_rate, ready_s=0.30),
            ],
            total_compute_s=0.30,
        )
        return r, b"\x00\x00"


def audio_of(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def build_loop(**kw) -> VoiceLoop:
    kw.setdefault("vad", FakeVad())
    kw.setdefault("asr", FakeAsr())
    kw.setdefault("tts", FakeTts())
    kw.setdefault("llm", SimulatedLlm(ttft_s=0.0))  # keep the suite fast
    return VoiceLoop(**kw)


# --------------------------------------------------------------------------
# The simulated LLM
# --------------------------------------------------------------------------


def test_the_llm_hop_is_declared_simulated() -> None:
    # The honesty guarantee, asserted rather than documented.
    assert SimulatedLlm().simulated is True
    assert "llm" in SIMULATED_HOPS


def test_only_the_llm_hop_is_simulated() -> None:
    assert SIMULATED_HOPS == {"llm"}


def test_stub_total_accounts_for_every_token() -> None:
    llm = SimulatedLlm(ttft_s=0.2, per_token_s=0.01, tokens=11)
    assert llm.total_s() == pytest.approx(0.2 + 0.10)


def test_stub_first_token_is_independent_of_token_count() -> None:
    """Why TTFT is the number that matters: it does not grow with the reply."""
    a = SimulatedLlm(ttft_s=0.3, tokens=10).first_token_delay_s()
    b = SimulatedLlm(ttft_s=0.3, tokens=500).first_token_delay_s()
    assert a == b == pytest.approx(0.3)


def test_stub_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        SimulatedLlm(ttft_s=-0.1)


# --------------------------------------------------------------------------
# ASR partial accounting
# --------------------------------------------------------------------------


def test_partial_overhead_is_everything_but_the_final_pass() -> None:
    r = AsrResult(
        partials=[
            Partial("a", 0.5, 0.10, is_final=False),
            Partial("a b", 1.0, 0.15, is_final=False),
            Partial("a b c", 1.5, 0.20, is_final=True),
        ]
    )
    assert r.total_compute_s == pytest.approx(0.45)
    assert r.final_compute_s == pytest.approx(0.20)
    assert r.partial_overhead_s == pytest.approx(0.25)


def test_no_partials_means_no_overhead() -> None:
    r = AsrResult(partials=[Partial("a", 1.0, 0.2, is_final=True)])
    assert r.partial_overhead_s == pytest.approx(0.0)


def test_final_text_is_the_last_partial() -> None:
    r = AsrResult(
        partials=[
            Partial("what is", 0.5, 0.1, is_final=False),
            Partial("what is the time", 1.0, 0.1, is_final=True),
        ]
    )
    assert r.final_text == "what is the time"


def test_empty_asr_result_has_empty_text() -> None:
    assert AsrResult().final_text == ""


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_turn_records_every_hop() -> None:
    r = build_loop().run_turn(audio_of(3.0))
    assert set(r.trace.by_hop()) == {"vad", "asr", "llm", "tts", "playback"}


def test_turn_returns_the_transcript_and_a_reply() -> None:
    r = build_loop().run_turn(audio_of(3.0))
    assert r.transcript == "hello world"
    assert r.reply


def test_silent_audio_raises_rather_than_reporting_a_fake_turn() -> None:
    class SilentVad:
        def probabilities(self, audio):
            return np.zeros(len(audio) // FRAME_SAMPLES, dtype=np.float32)

    with pytest.raises(ValueError, match="no speech"):
        build_loop(vad=SilentVad()).run_turn(audio_of(3.0))


def test_perceived_latency_includes_the_endpoint_wait() -> None:
    """The user sits through the hangover, so it is part of what they feel."""
    fast = build_loop(vad_config=VadConfig(hangover_ms=200)).run_turn(audio_of(3.0))
    slow = build_loop(vad_config=VadConfig(hangover_ms=800)).run_turn(audio_of(3.0))
    assert slow.perceived_s - fast.perceived_s == pytest.approx(0.6, abs=0.05)


def test_streaming_beats_blocking_by_the_tts_penalty() -> None:
    r = build_loop().run_turn(audio_of(3.0))
    # FakeTts: first chunk 0.10 s, complete 0.30 s.
    assert r.streaming_saving_s == pytest.approx(0.20, abs=1e-6)
    assert r.perceived_s < r.perceived_blocking_s


def test_disabling_partials_costs_one_asr_pass_only() -> None:
    with_partials = build_loop(emit_partials=True).run_turn(audio_of(3.0))
    without = build_loop(emit_partials=False).run_turn(audio_of(3.0))
    assert len(without.asr.partials) == 1
    assert len(with_partials.asr.partials) > 1
    assert without.asr.total_compute_s < with_partials.asr.total_compute_s


def test_hop_durations_never_exceed_the_wall_clock() -> None:
    """Guards against a hop being timed on a different clock."""
    r = build_loop().run_turn(audio_of(3.0))
    total = r.trace.total_s()
    for name, dur in r.trace.by_hop().items():
        assert dur <= total + 1e-6, f"{name} exceeds wall clock"


def test_first_audio_mark_is_set() -> None:
    r = build_loop().run_turn(audio_of(3.0))
    assert r.trace.first_audio_s is not None
    assert r.trace.first_audio_s > 0
