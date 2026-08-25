"""The end-to-end loop: VAD -> ASR -> LLM (SIMULATED) -> TTS -> playback.

Read this first: **the LLM hop is a stub.** No API key is configured in the
environment this repo was built and measured in, so no live model was ever
called. The LLM hop is a synthetic delay with a configurable time-to-first-
token and inter-token rate, and it is labelled SIMULATED in every table it
appears in.

Stubbing it is not a shortcut around the measurement, because the question is
*which hop dominates*, and that question needs the LLM hop to be a knob
rather than a constant. A live endpoint would contribute one provider's
network variance on one afternoon -- unreproducible, and not a property of
this loop. A parameterised stub instead lets the results sweep TTFT across
the plausible range and show where the crossover between hops actually sits.
The three hops that are genuinely measured are measured for real.

What the stub does *not* establish is stated in the README Limitations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .asr import AsrResult, WhisperAsr
from .tracer import Trace, Tracer
from .tts import PiperTts, TtsResult
from .vad import SAMPLE_RATE, SileroVad, VadConfig, endpoint_from_probabilities


@dataclass(frozen=True, slots=True)
class SimulatedLlm:
    """A stand-in for a streaming LLM. NOT A REAL MODEL CALL.

    Two parameters, because those are the two a streaming loop feels:

    ``ttft_s``        time to first token. The loop can start TTS on the first
                      sentence, so this -- not total generation time -- is
                      what enters the perceived-latency path.
    ``per_token_s``   inter-token delay after the first.

    Defaults are ballpark figures for a small hosted model, chosen to be
    plausible rather than authoritative. The results sweep them precisely
    because no single value here can be defended as measured.
    """

    ttft_s: float = 0.30
    per_token_s: float = 0.01
    tokens: int = 24
    # Replies cycled through in order, so a sweep speaks varied text rather
    # than one memorised string. Fixed content: the repo measures timing, not
    # answer quality.
    replies: tuple[str, ...] = (
        "The forecast is mild with a light breeze from the south west. "
        "Expect clear skies through the afternoon.",
    )

    def __post_init__(self) -> None:
        if self.ttft_s < 0 or self.per_token_s < 0:
            raise ValueError("simulated latencies must not be negative")
        if not self.replies:
            raise ValueError("the stub needs at least one reply")

    @property
    def simulated(self) -> bool:
        """Always True. Present so callers can assert on it rather than
        remember it."""
        return True

    def first_token_delay_s(self) -> float:
        return self.ttft_s

    def total_s(self) -> float:
        return self.ttft_s + self.per_token_s * max(0, self.tokens - 1)

    def reply_text(self, prompt: str) -> str:
        """Pick a reply for this prompt. NOT a generated answer.

        Selection is a hash of the prompt rather than a counter, so it stays
        deterministic across runs and independent of call order -- a repeated
        sweep must produce the same reply for the same utterance, or the TTS
        timings would not be comparable between runs.
        """
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies[sum(map(ord, prompt)) % len(self.replies)]


@dataclass
class TurnResult:
    """One complete turn, with its trace and the per-hop evidence."""

    trace: Trace
    transcript: str
    reply: str
    asr: AsrResult | None = None
    tts: TtsResult | None = None
    # Perceived latency: user stops speaking -> first phoneme is audible.
    perceived_s: float = 0.0
    # Same turn if playback waited for complete audio instead of streaming.
    perceived_blocking_s: float = 0.0

    @property
    def streaming_saving_s(self) -> float:
        return self.perceived_blocking_s - self.perceived_s


class VoiceLoop:
    """Runs one turn end to end, timing every hop.

    Engines are injected rather than constructed here so the loop can be
    driven by fakes in tests: the orchestration logic is worth testing
    without paying a 10 s model load for each case.
    """

    def __init__(
        self,
        vad: SileroVad,
        asr: WhisperAsr,
        tts: PiperTts,
        llm: SimulatedLlm | None = None,
        vad_config: VadConfig | None = None,
        partial_interval_s: float = 0.5,
        emit_partials: bool = True,
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.llm = llm or SimulatedLlm()
        self.vad_config = vad_config or VadConfig()
        self.partial_interval_s = partial_interval_s
        self.emit_partials = emit_partials

    def run_turn(self, audio: np.ndarray, clock=time.perf_counter) -> TurnResult:
        """One turn: detect the endpoint, transcribe, reply, speak.

        The turn clock starts at the endpoint decision, not at the start of
        the audio. Time spent listening is the user talking, and nobody
        experiences their own speech as latency.
        """
        tracer = Tracer(clock=clock)

        # --- VAD ------------------------------------------------------
        with tracer.span("vad"):
            t0 = clock()
            probs = self.vad.probabilities(audio)
            compute_s = clock() - t0
            endpoint = endpoint_from_probabilities(probs, self.vad_config, compute_s)
        if endpoint is None:
            raise ValueError("no speech detected in the supplied audio")

        # --- ASR ------------------------------------------------------
        # Trimmed at the endpoint, not the end of the buffer: transcribing the
        # trailing silence would charge the ASR for audio the VAD already
        # ruled out, which is a measurement error, not a conservative one.
        speech = audio[: int(endpoint.speech_end_s * SAMPLE_RATE)]
        with tracer.span("asr"):
            if self.emit_partials:
                asr_result = self.asr.stream(
                    speech, partial_interval_s=self.partial_interval_s, clock=clock
                )
            else:
                asr_result = self.asr.stream(
                    speech, partial_interval_s=float("inf"), clock=clock
                )
        transcript = asr_result.final_text

        # --- LLM (SIMULATED) -----------------------------------------
        # Sleeping the stub's TTFT rather than adding a number afterwards
        # keeps the span on the same clock as every measured hop, so the
        # overlap arithmetic in `critical_path` stays valid.
        with tracer.span("llm"):
            time.sleep(self.llm.first_token_delay_s())
        reply = self.llm.reply_text(transcript)

        # --- TTS ------------------------------------------------------
        with tracer.span("tts"):
            tts_result, _audio_bytes = self.tts.synthesize(reply, clock=clock)

        first_audio_s = tracer.mark_first_audio()
        tracer.trace.audio_duration_s = tts_result.audio_duration_s

        # --- playback -------------------------------------------------
        # Buffer handoff only. No sound card is opened: CI has no audio
        # device, and the cost being measured is the buffering policy, not
        # the driver.
        with tracer.span("playback"):
            pass

        # Perceived latency = hangover + everything the loop then does before
        # the first phoneme is audible.
        #
        # The timeline this models, from the user's last word:
        #   [hangover silence] [VAD scoring] [ASR] [LLM stub] [TTS 1st chunk]
        #
        # The hangover is added rather than measured because this loop is fed a
        # complete buffer, not a live stream: it never actually waits out the
        # silence, so the wait has to be reintroduced from the endpointer's own
        # decision time or the number would flatter the loop by ~500 ms.
        #
        # `first_audio_s` runs from turn start, so it includes VAD compute.
        # That is deliberate -- it is real work happening while the user waits
        # -- and it does not double-count the hangover, which is wall-clock
        # silence the loop never spent.
        endpoint_wait = endpoint.endpoint_s
        perceived = endpoint_wait + first_audio_s
        blocking_extra = tts_result.blocking_penalty_s

        return TurnResult(
            trace=tracer.trace,
            transcript=transcript,
            reply=reply,
            asr=asr_result,
            tts=tts_result,
            perceived_s=perceived,
            perceived_blocking_s=perceived + blocking_extra,
        )
