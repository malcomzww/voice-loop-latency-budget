# Inventory coverage

Anchored to `Bucket_Concept_Inventory.md`, bucket B2 (Agentic & multimodal).

Split honestly into what the repo *measures*, what it *discusses*, and what it
deliberately leaves alone. A latency repo that claimed to cover voice cloning
or MOS evaluation would be overselling itself.

## Measured

- **Voice loop latency budget: ASR partial → LLM TTFT → TTS first chunk.**
  The repo's whole purpose. p50/p95 per hop with a merged critical path, in
  `results/waterfall.md`. The LLM hop is a labelled stub; the rest is real.
- **Streaming ASR: partial hypotheses and their cost.** Measured directly, and
  the source of the repo's main finding — partials are the majority of ASR
  compute because Whisper is not a streaming model.
- **VAD and endpointing.** Silero VAD per-frame, with compute and endpoint
  wait reported separately because they behave completely differently.
- **TTS streaming: first chunk vs complete audio.** Chunk-level timestamps
  inside the generator, and the ADR that rests on them.
- **Backpressure and partial responses.** Present as the streaming-vs-blocking
  playback comparison: the cost of buffering a whole response before emitting
  it is exactly what section 3 of the results measures.

## Discussed, not measured

Written analysis with public references. Called out as such rather than
implied.

- **CTC vs RNN-T vs Whisper-style architectures.** The partials finding is an
  argument about architecture — an encoder-decoder cannot emit cheap partials —
  so the comparison is reasoned about in the README and results, without a
  CTC/RNN-T model being installed and benchmarked.
- **TTS architectures: VITS, vocoders, neural codecs.** Piper is VITS, and the
  two quality tiers are measured, but no architectural comparison is run.
  Diffusion and flow-matching TTS are not touched.
- **Multilingual G2P.** `docs/multilingual-g2p-notes.md`, public references
  only, no data. Includes why measuring it properly needs speech data this
  repo does not collect.
- **Barge-in.** Named as a real turn-taking cost the budget ignores, in the
  README Limitations and the ADR.
- **SSE vs WebSocket.** The loop is in-process, so no transport is measured.
  The backpressure question it raises is covered by the buffering comparison.

## Deliberately out of scope

- **Voice cloning, speaker conditioning, prosody control.** Requires speaker
  data. The repo ships none, by constraint and by choice.
- **TTS evaluation: MOS, intelligibility, WER-via-ASR, speaker similarity.**
  All quality metrics. This repo measures latency and says plainly that it
  establishes nothing about accuracy or audio quality.
- **Diarization.** No multi-speaker audio, so nothing to diarize.
