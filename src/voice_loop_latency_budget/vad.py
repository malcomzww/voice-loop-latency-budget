"""Streaming voice activity detection and endpointing.

The VAD's compute cost is negligible -- Silero scores a 32 ms frame in well
under a millisecond -- but the VAD still sets the floor on the whole budget,
because **endpointing is a waiting decision, not a computing one**. The loop
cannot call the ASR until it believes the user stopped talking, and that
belief costs `hangover_ms` of silence no matter how fast the model is.

That distinction is why this module reports two separate numbers:

``compute_s``    time spent running the model. Genuinely tiny.
``endpoint_s``   time from the last speech frame to the endpoint decision.
                 This is ``hangover_ms`` by construction, and it is real
                 latency the user waits through.

Conflating them would hide the only VAD tuning knob that matters. A results
table showing "VAD: 3 ms" is true about compute and badly misleading about
the turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# Silero v6 is fixed at 16 kHz with a 512-sample window: 32 ms per frame.
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Endpointing parameters.

    ``hangover_ms`` is the whole trade-off: too short and the VAD cuts the
    user off mid-sentence, too long and every turn pays the difference. 500 ms
    is a common interactive default and is swept in the results.
    """

    threshold: float = 0.5
    hangover_ms: int = 500
    min_speech_ms: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.hangover_ms < 0:
            raise ValueError("hangover_ms must not be negative")


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Where a turn's speech began and ended, and what the decision cost."""

    speech_start_s: float
    speech_end_s: float
    # When the endpointer *declared* the turn over: speech_end_s + hangover.
    decision_s: float
    compute_s: float

    @property
    def speech_duration_s(self) -> float:
        return self.speech_end_s - self.speech_start_s

    @property
    def endpoint_s(self) -> float:
        """Silence the loop waited through before it could act."""
        return self.decision_s - self.speech_end_s


def frames(audio: np.ndarray) -> Iterator[np.ndarray]:
    """Split audio into fixed 512-sample frames, dropping any short tail.

    The tail is dropped rather than zero-padded because a padded frame scores
    as silence and would fabricate an endpoint the audio does not contain.
    """
    n = len(audio) // FRAME_SAMPLES
    for i in range(n):
        yield audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]


class SileroVad:
    """Frame-by-frame speech probabilities from Silero VAD.

    The model ships inside ``faster-whisper``, so it needs no separate
    download and no network access at test time.
    """

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Speech probability per 32 ms frame."""
        audio = np.asarray(audio, dtype=np.float32)
        usable = (len(audio) // FRAME_SAMPLES) * FRAME_SAMPLES
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._model(audio[:usable]), dtype=np.float32).ravel()


def endpoint_from_probabilities(
    probs: np.ndarray,
    config: VadConfig,
    compute_s: float = 0.0,
) -> Endpoint | None:
    """Find the turn's endpoint from per-frame speech probabilities.

    Split out from the model call so the endpointing logic -- the part with
    the interesting behaviour -- is testable without loading any ONNX at all.
    Returns ``None`` when no run of speech is long enough to count as a turn.
    """
    speech = probs >= config.threshold
    if not speech.any():
        return None

    min_frames = max(1, config.min_speech_ms // FRAME_MS)
    hangover_frames = config.hangover_ms // FRAME_MS

    # Walk runs of speech; a gap shorter than the hangover does not end the
    # turn, which is what stops the endpointer firing between words.
    start: int | None = None
    best: tuple[int, int] | None = None
    gap = 0
    for i, is_speech in enumerate(speech):
        if is_speech:
            if start is None:
                start = i
            gap = 0
            best = (start, i)
        elif start is not None:
            gap += 1
            if gap > hangover_frames:
                if best is not None and best[1] - best[0] + 1 >= min_frames:
                    break
                start, best, gap = None, None, 0

    if best is None or best[1] - best[0] + 1 < min_frames:
        return None

    first, last = best
    speech_start_s = first * FRAME_MS / 1000.0
    speech_end_s = (last + 1) * FRAME_MS / 1000.0
    return Endpoint(
        speech_start_s=speech_start_s,
        speech_end_s=speech_end_s,
        decision_s=speech_end_s + config.hangover_ms / 1000.0,
        compute_s=compute_s,
    )
