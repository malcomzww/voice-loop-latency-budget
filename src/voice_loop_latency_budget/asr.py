"""Streaming ASR with partial hypotheses, on faster-whisper (CTranslate2, int8).

Whisper is an offline encoder-decoder, not a streaming model: it consumes a
window and emits text for that window. A streaming *loop* is therefore built
on top of it by re-transcribing a growing buffer at intervals, and every
partial costs a full forward pass over everything said so far.

That has a consequence the budget must not hide. **Partials are not free, and
their cost grows with the utterance.** A loop emitting a partial every 500 ms
over a 5 s turn does roughly ten transcriptions, and the last ones are the
expensive ones. This module measures each partial separately so the results
can show the compute that partials add versus the final transcription alone.

The alternative -- a true streaming CTC or RNN-T model -- is discussed in the
README rather than implemented, because swapping the architecture would
change what the repo measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vad import SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class Partial:
    """One intermediate transcription of the buffer so far."""

    text: str
    audio_s: float  # how much audio had accumulated when this ran
    compute_s: float
    is_final: bool


@dataclass
class AsrResult:
    """Everything one turn of ASR produced."""

    partials: list[Partial] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return self.partials[-1].text if self.partials else ""

    @property
    def total_compute_s(self) -> float:
        return sum(p.compute_s for p in self.partials)

    @property
    def final_compute_s(self) -> float:
        """Cost of the final pass alone, ignoring partials."""
        for p in reversed(self.partials):
            if p.is_final:
                return p.compute_s
        return 0.0

    @property
    def partial_overhead_s(self) -> float:
        """Compute spent on partials that the final pass repeats anyway."""
        return self.total_compute_s - self.final_compute_s


class WhisperAsr:
    """faster-whisper wrapper that reports per-partial timings.

    ``int8`` on CPU is the point of the CTranslate2 backend: it is what makes
    a Whisper-family model viable in an interactive loop without a GPU.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        download_root: str | None = None,
        compute_type: str = "int8",
        beam_size: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.compute_type = compute_type
        # beam_size=1 (greedy) on purpose: beam search buys accuracy this repo
        # does not measure, at latency it does.
        self.beam_size = beam_size
        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language="en",
            beam_size=self.beam_size,
        )
        return "".join(s.text for s in segments).strip()

    def stream(
        self,
        audio: np.ndarray,
        partial_interval_s: float = 0.5,
        clock=None,
    ) -> AsrResult:
        """Re-transcribe a growing buffer, emitting partials then a final.

        Simulates arrival by slicing the array rather than sleeping: the
        budget is interested in the model's compute per partial, and sleeping
        through the audio in real time would add hours to the sweep without
        changing a single measured number.
        """
        import time

        clock = clock or time.perf_counter
        audio = np.asarray(audio, dtype=np.float32)
        total_s = len(audio) / SAMPLE_RATE
        result = AsrResult()

        cursor = partial_interval_s
        while cursor < total_s:
            chunk = audio[: int(cursor * SAMPLE_RATE)]
            t0 = clock()
            text = self.transcribe(chunk)
            result.partials.append(
                Partial(text, audio_s=cursor, compute_s=clock() - t0, is_final=False)
            )
            cursor += partial_interval_s

        t0 = clock()
        text = self.transcribe(audio)
        result.partials.append(
            Partial(text, audio_s=total_s, compute_s=clock() - t0, is_final=True)
        )
        return result
