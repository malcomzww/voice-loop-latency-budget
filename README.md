# voice-loop-latency-budget

An instrumented streaming voice loop — VAD → ASR → LLM → TTS → playback — built
to answer exactly one question: **which hop dominates perceived voice latency,
and what is the one optimisation that mattered?** The scope is deliberately
narrow. It measures latency on one machine, in English, with CPU-only public
models, and it does not measure transcription accuracy, audio quality, or any
real LLM provider. **The LLM hop is a stub with a synthetic delay — no API key
was configured and no live model was ever called.** Everything else is real
measured inference.

## The waterfall

Measured hops ranked by p50 cost, default configuration (partials on, greedy
decoding, `tiny.en`, `en_US-lessac-low`):

| rank | hop | measured or simulated |
|---|---|---|
| 1 | **ASR** — takes more than 70% of hop time at p50 | measured |
| 2 | TTS | measured |
| 3 | VAD — negligible *compute*, but see below | measured |
| — | LLM — swept across a TTFT range, not measured | **SIMULATED** |

Ranking rather than a percentage for every row, because each hop's share
depends on every other hop: ASR timing noise moves the denominator under the
whole table, and the stub LLM's share wandered across a band edge between runs
while the stub itself never changed. The order is what is stable.

**ASR dominates, and the one optimisation that mattered was turning off
partial hypotheses: it cuts p50 ASR compute by more than 50%.**

The reason is architectural. Whisper is an encoder-decoder that consumes a
window, not a streaming model. Partials are therefore implemented by
re-transcribing a growing buffer, so every partial is a full forward pass over
everything said so far — and the last partial costs nearly as much as the final
transcription it is about to duplicate. The majority of ASR compute in the
default configuration goes on work that the final pass repeats. No amount of
interval tuning fixes that; a true streaming model (CTC or RNN-T) would,
because it emits partials from a single pass over each new frame.

Two findings that a naive budget gets wrong:

- **The VAD's ~0% compute share is the most misleading number in the table.**
  Silero scores a 32 ms frame in well under a millisecond, but endpointing is a
  *waiting* decision, not a computing one: the loop cannot call the ASR until
  it believes the user stopped, and that belief costs the full hangover in
  silence. At a 500 ms hangover the VAD adds 500 ms to every turn while
  consuming almost no CPU. It is a turn-taking accuracy trade, not a compute
  trade.
- **Perceived latency is not the sum of the hops.** Streaming overlaps work, so
  summing span durations double-counts it and yields shares above 100%. The
  budget merges intervals into a critical path instead.

Full generated tables, including p50/p95 per hop and the simulated-TTFT
crossover: **[`results/waterfall.md`](results/waterfall.md)**.

## First chunk, not total audio

The judgement call the repo is built on:
**[`docs/adr/0001-first-chunk-not-total-audio.md`](docs/adr/0001-first-chunk-not-total-audio.md)**.

A TTS that streams its first chunk in 200 ms feels faster than one returning
complete audio in 400 ms, even at identical total duration — because the user's
clock stops at the first phoneme, not the last. Piper's `synthesize()` is a
generator that yields one chunk per sentence, so this repo timestamps chunks
*inside* the generator loop, before concatenation. The measured first chunk
arrives within 60% of total synthesis time, and synthesis runs several times
faster than real time (the generated file states the asserted multiple for the
run that produced it), so playback of chunk *n* covers synthesis of chunk
*n+1* and the stream never underruns.

That last condition is what makes the decision safe rather than merely
optimistic, and it is the thing to re-check on different hardware: if the
real-time factor ever exceeded 1.0, first-chunk latency would become a promise
the loop could not keep.

## Quickstart

```bash
uv sync --extra speech --extra dev
uv run python scripts/fetch_models.py      # public models into .models/ (gitignored)
uv run python scripts/generate_results.py  # regenerates results/
uv run pytest -q
```

The test suite needs neither the models nor the `speech` extra: the tracer and
the budget arithmetic are pure Python, the engines are imported lazily, and the
loop is tested against fakes. `uv sync --extra dev && uv run pytest -q` is
green on a machine that cannot fetch a 60 MB ONNX voice.

## What is measured and what is not

| hop | status | engine |
|---|---|---|
| VAD | **measured** | Silero VAD v6 (bundled with `faster-whisper`) |
| ASR | **measured** | `faster-whisper` `tiny.en`, CTranslate2, int8, CPU, greedy |
| LLM | **SIMULATED** | synthetic delay; configurable TTFT; **no live call** |
| TTS | **measured** | Piper `en_US-lessac-low` / `-medium` (VITS, ONNX) |
| playback | modelled | buffer handoff; no sound card is opened |

The LLM stub is not a shortcut around the measurement. The question is *which
hop dominates*, and answering it needs TTFT to be a knob that can be swept
rather than one provider's network variance on one afternoon. Section 6 of the
results uses it that way: it reports how fast a real TTFT would have to be
before the LLM displaced ASR, and the answer flips depending on whether
partials are on. It is labelled SIMULATED in every table it appears in, and a
test asserts it is the only hop that is.

## No collected audio

The repo ships **no recorded speech, no trained voice, and no speaker data**.
The test utterances are synthesised by Piper at measurement time from fixed
prompt strings, so there is no audio in the repository at all — nothing that
could identify a person. English only, on public models. Multilingual G2P is
written analysis with public references and no data:
[`docs/multilingual-g2p-notes.md`](docs/multilingual-g2p-notes.md).

## Provenance

Every number in this README comes from `scripts/generate_results.py`, which
writes two files:

- **`results/waterfall.md`** — committed. Only machine-independent claims:
  which hop dominates, the direction and rough size of each effect, and banded
  thresholds. The script **asserts each claim and exits non-zero if it
  breaks**, and CI fails if regenerating produces a diff.
- **`results/waterfall-raw.md`** — gitignored. Absolute millisecond timings,
  with date, hardware, model snapshot, seed, and reproduce command.

The split is the point. CPU inference speed is hardware-dependent, so
byte-comparing milliseconds across machines is a gate that fails for reasons
unrelated to the code. Committed claims are written as bounds the script
re-checks ("more than 50%") rather than as one run's observation ("78.4%") —
stable across machines, and a stronger statement.

Bands still move between *bands* on a heavily loaded machine: the real-time
factor claim lands on 10x when nothing else is running and 5x when three
measurement runs are competing for the same 24 cores. Both are true; the
committed file states whichever the run supported, and every prose claim here
is written to hold at the weaker bound. Run it on an idle machine if you want
the tighter numbers.

Reproduce: `python scripts/fetch_models.py && python scripts/generate_results.py`

## Limitations

What this repo does **not** establish:

- **Nothing about any real LLM's latency.** The LLM hop is simulated. The
  decomposition around it is real; the hop itself is a delay.
- **Synthetic speech in, synthetic speech out.** Piper output is unnaturally
  clean input for a VAD and an ASR. Real microphones bring noise, clipping and
  far-field reverb that would raise ASR cost and hurt endpointing. **These
  numbers are a floor, not a forecast.**
- **Accuracy is not measured at all.** Greedy decoding and `tiny.en` are
  latency choices with a real, unquantified WER cost. A faster loop that
  mistranscribes is not a better loop, and this repo cannot tell you which
  you have.
- **One machine, CPU only, no GPU.** A GPU changes the ASR/TTS balance and
  could plausibly move which hop dominates.
- **No barge-in, no diarization, no multi-turn context.** The user interrupting
  playback is a real turn-taking cost this budget ignores entirely.
- **English only.** See the G2P notes for why per-language cost differs and
  why measuring it properly needs data this repo deliberately does not collect.

## Concepts covered

See [`docs/inventory-coverage.md`](docs/inventory-coverage.md).

- Streaming ASR: Whisper-style vs CTC/RNN-T, partial hypotheses, endpointing
- VAD, turn detection, barge-in (discussed), diarization (out of scope)
- TTS architectures: VITS, neural vocoders, chunked streaming synthesis
- Voice loop latency budget: ASR partials → LLM TTFT → TTS first chunk
- Backpressure and partial responses in a streaming pipeline

## License

MIT
