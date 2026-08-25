# ADR 0001: Measure time-to-first-audio-chunk, not total audio

- **Status:** Accepted
- **Date:** 2026-08-25
- **Context:** `voice-loop-latency-budget`

## Context

A voice loop has to report a latency number, and there are several
defensible candidates:

1. **Total turn time** — user stops speaking to the last sample of the reply.
2. **Total synthesis time** — how long TTS took to produce the whole reply.
3. **Time to first audio chunk** — user stops speaking to the first phoneme
   they can hear.

The first two are what the libraries hand you. `PiperVoice.synthesize()` is a
generator, but the obvious way to use it is to drain it into a buffer and
return the buffer, at which point the only number available is the total. Most
TTS benchmarks report a real-time factor computed exactly that way.

Choosing between them is not bookkeeping. It changes which optimisations look
worthwhile, and it changes them in opposite directions.

## Decision

**The loop's headline latency metric is time to first audio chunk.** Total
synthesis time is recorded, and reported only as the counterfactual — what a
non-streaming caller would have paid for identical audio.

Concretely, in `src/voice_loop_latency_budget/tts.py` the clock is read
*inside* the generator loop, before any concatenation:

```python
for i, chunk in enumerate(self._voice.synthesize(text)):
    ready = clock() - t0        # when this chunk became playable
```

`TtsResult.blocking_penalty_s` is then `total_compute_s - first_chunk_s`: the
latency a buffering implementation adds and gets nothing for.

## Why

**The user's clock stops at the first phoneme, not the last.** Once audio is
playing, the user is listening rather than waiting. Everything synthesised
after playback begins is hidden behind sound that is already coming out of the
speaker — provided synthesis keeps ahead of playback.

**On this hardware it comfortably does.** The measured real-time factor is far
below 1.0 (see `results/waterfall.md`), meaning a second of audio costs a
small fraction of a second to synthesise. Playback of chunk *n* therefore
covers synthesis of chunk *n+1*, and the stream does not underrun. This is the
condition that makes the decision safe, and it is the thing to re-check on
different hardware rather than assume: if RTF ever exceeded 1.0, first-chunk
latency would become a promise the loop could not keep, and buffering would go
from wasteful to necessary.

**The two metrics rank optimisations differently.** This is the part that
matters. Under total-synthesis-time, a change that halves total compute looks
twice as good as one that halves time-to-first-chunk. Under the metric the
user actually experiences, work done after the first chunk is nearly free, so
the only TTS optimisations worth anything are the ones that pull the *first*
chunk earlier — sentence-level chunking, a smaller first-sentence model, or
simply not buffering. A team optimising the wrong metric will spend real
effort making the tail faster and ship no perceptible improvement.

**It also reframes the model-quality choice.** Both public voices stream, and
both clear real time by a wide margin (see the TTS table in
`results/waterfall-raw.md`). That is the whole answer under first-chunk
accounting: when every candidate's first chunk lands early and synthesis keeps
ahead of playback, the choice between them stops being a latency decision at
all and becomes a quality judgement.

An earlier draft of this ADR claimed the higher-quality `medium` voice costs
roughly twice what `low` does, and used that as the illustration. **That claim
did not survive being asserted.** Across repeated trials the ordering between
the two voices flipped — sometimes `low` measured slower — which makes sense
once you notice the two ONNX files are the same size and differ mainly in
sample rate. The apparent gap was CPU-contention noise. It is recorded here
because a results script that asserts its claims is what caught it, and
because the corrected version makes the same point more cleanly: the metric's
job is to tell you when latency has stopped being the deciding factor.

## Consequences

**Accepted:**

- Perceived latency in this repo means endpoint wait plus time to first
  audio chunk. It is the number in the README.
- `results/waterfall.md` asserts on every run that the first chunk precedes
  complete synthesis, that TTS yields more than one chunk, and that synthesis
  beats real time. If the engine ever stops streaming, the results fail loudly
  rather than quietly reporting a meaningless first-chunk figure.
- The loop hands buffers to a modelled playback stage rather than a sound
  card, so this runs unattended in CI where no audio device exists.

**Costs and limits:**

- **First-chunk latency hides tail risk.** A reply whose first chunk arrives
  in 60 ms but whose fourth chunk stalls would score well and sound broken.
  This repo does not model underrun, because measured RTF leaves a wide
  margin; a loop running near RTF 1.0 needs a jitter-buffer metric this
  budget does not provide.
- **Chunk granularity is the engine's, not ours.** Piper chunks per sentence,
  so a single-sentence reply cannot stream and first-chunk collapses onto
  total. The measurement is therefore partly a property of reply shape.
  Sub-sentence chunking would decouple them, at the cost of prosody across
  the boundary.
- **No sound card means no output-device latency.** Real playback adds buffer
  and driver latency (commonly 10–50 ms) that this budget omits entirely.

## Alternatives considered

**Report total turn time as the headline.** Honest and simple, and it is what
an end-to-end trace naturally produces. Rejected because it is dominated by
audio duration — a long reply scores worse than a short one even when it
starts speaking sooner, which inverts the ranking of the thing the user cares
about.

**Report real-time factor only.** Standard in TTS benchmarking and genuinely
useful for capacity planning: it answers how many streams a box can serve.
Rejected as the headline because it is a throughput metric. RTF 0.02 tells you
nothing about whether the first phoneme took 60 ms or 600 ms.

**Play through a real audio device and measure acoustically.** The most
faithful option, and the only one that would capture driver and device
latency. Rejected because it cannot run in CI, cannot be asserted on, and
would make the results unreproducible on any machine but one.
