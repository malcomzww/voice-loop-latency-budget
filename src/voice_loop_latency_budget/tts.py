"""Streaming TTS on Piper (VITS, ONNX), instrumented for first-chunk latency.

This module exists to measure one thing: **time to first audio chunk versus
time to complete audio**. Piper's ``synthesize`` is a generator that yields
one chunk per sentence, so the gap between the two is directly observable
rather than argued about.

Why the distinction is the whole point: the user's clock starts when they
stop speaking and stops when they hear the first phoneme. Everything
synthesised after that arrives while they are already listening, and audio
playing is audio the user is not waiting for. A loop that blocks until the
last sample is ready therefore pays the full synthesis cost in perceived
latency and gets nothing for it.

See ``docs/adr/0001-first-chunk-not-total-audio.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One synthesised chunk and the moment it became available."""

    index: int
    n_samples: int
    sample_rate: int
    # Seconds from the synthesis call to this chunk being yielded.
    ready_s: float

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass
class TtsResult:
    """Timings for one synthesis."""

    chunks: list[AudioChunk] = field(default_factory=list)
    total_compute_s: float = 0.0

    @property
    def first_chunk_s(self) -> float:
        """What the user perceives as the response latency."""
        if not self.chunks:
            raise ValueError("synthesis produced no audio")
        return self.chunks[0].ready_s

    @property
    def audio_duration_s(self) -> float:
        return sum(c.duration_s for c in self.chunks)

    @property
    def blocking_penalty_s(self) -> float:
        """Extra wait a non-streaming caller pays for identical audio.

        The ADR's headline number. Buffering the whole utterance before
        playback delays the first phoneme by exactly this much and improves
        nothing the user can hear.
        """
        return self.total_compute_s - self.first_chunk_s

    @property
    def real_time_factor(self) -> float:
        """Compute seconds per second of audio produced. Below 1.0 is faster
        than real time, so playback of chunk *n* covers synthesis of *n+1*."""
        audio = self.audio_duration_s
        return self.total_compute_s / audio if audio else float("inf")


class PiperTts:
    """Piper wrapper that timestamps every chunk as it is yielded."""

    def __init__(self, voice_path: str | Path) -> None:
        from piper import PiperVoice

        self.voice_path = Path(voice_path)
        if not self.voice_path.exists():
            raise FileNotFoundError(
                f"voice not found: {self.voice_path}. Run scripts/fetch_models.py."
            )
        self._voice = PiperVoice.load(str(self.voice_path))

    @property
    def sample_rate(self) -> int:
        return self._voice.config.sample_rate

    def synthesize(self, text: str, clock=time.perf_counter) -> tuple[TtsResult, bytes]:
        """Synthesise, timestamping each chunk at the instant it is yielded.

        Timing inside the generator loop is what makes the first-chunk claim a
        measurement: the clock is read before the audio is concatenated, so
        the number is when the chunk *became playable*, not when the whole
        buffer was assembled.
        """
        result = TtsResult()
        buffers: list[bytes] = []
        t0 = clock()
        for i, chunk in enumerate(self._voice.synthesize(text)):
            ready = clock() - t0
            audio = chunk.audio_int16_bytes
            buffers.append(audio)
            result.chunks.append(
                AudioChunk(
                    index=i,
                    n_samples=len(audio) // 2,  # int16
                    sample_rate=chunk.sample_rate,
                    ready_s=ready,
                )
            )
        result.total_compute_s = clock() - t0
        return result, b"".join(buffers)


def playback_start_s(result: TtsResult, streaming: bool) -> float:
    """When audio starts playing, under each buffering policy.

    Streaming starts on the first chunk; blocking waits for the last sample.
    Modelled rather than played through a sound card so the results run
    unattended in CI, where no audio device exists.
    """
    return result.first_chunk_s if streaming else result.total_compute_s
