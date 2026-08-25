# Second-language and G2P notes

**This is written analysis with public references. No multilingual data was
collected, and no model was trained or fine-tuned for it.** The repo's
measurements are English-only, on the public `en_US-lessac` Piper voices and
`faster-whisper`'s `tiny.en` / `base.en` snapshots. This document exists
because a latency budget that ignores language is quietly assuming a
monolingual deployment, and that assumption should be stated rather than
implied.

The scope constraint in the brief is deliberate and observed here: the
interesting multilingual question for a latency budget is *where in the
pipeline language changes the cost*, and that is answerable from public
documentation and architecture without collecting anyone's speech.

## Where language enters the budget

Language does not scale the loop uniformly. It hits three of the four hops in
different ways, and only one of them is a latency effect.

### VAD — essentially language-independent

Silero VAD operates on acoustic energy and spectral structure, not phonetics.
Its per-frame cost does not change with the language being spoken. The
`hangover_ms` *policy*, however, arguably should: languages and speaking
styles differ in typical inter-word pause length, and an endpoint threshold
tuned on one can truncate speakers of another. That is a turn-taking accuracy
question, not a compute one, and this repo measures neither across languages.

### ASR — a model-choice effect, not a per-language compute effect

The `.en` Whisper snapshots are English-only. Handling a second language means
the multilingual snapshot of the same size, and the relevant cost is that
multilingual Whisper spends decoder capacity on language identification and a
much larger token vocabulary. The compute per forward pass is a property of
the model size, so `tiny` multilingual is comparable in cost to `tiny.en`;
what changes is accuracy per unit of compute, which tends to be worse for the
same parameter count, pushing a deployment toward a larger — and therefore
slower — model to hold quality. That is a real latency consequence, arrived at
indirectly.

Whisper's reported per-language error rates vary by well over an order of
magnitude across its supported set (see the WER/CER tables in the Whisper
paper's appendix, Radford et al. 2022). A budget built on English numbers does
not transfer to a low-resource language, and the honest statement is that this
repo measures one language and does not establish the others.

### TTS and G2P — where the genuinely awkward work lives

Piper's front end converts graphemes to phonemes with eSpeak NG, then feeds
phoneme IDs to a VITS decoder. The split matters for latency reasoning: the
neural decode is language-agnostic once phonemes exist, but G2P quality is
entirely language-specific and is where the engineering effort goes.

The difficulty is not uniform across languages:

- **Shallow-orthography languages** (Spanish, Turkish, Finnish) map letters to
  sounds close to one-to-one. Rule-based G2P is accurate and cheap.
- **Deep-orthography languages** (English, French) need substantial exception
  handling; English `read` / `read` cannot be resolved without context.
- **Abjads** (Arabic, Hebrew) omit short vowels in normal writing. Piper ships
  a `tashkeel` diacritisation step for Arabic precisely because the vowels
  must be *predicted* before phonemisation — an extra model in the front end,
  and a latency cost English never pays.
- **Logographic and mixed scripts** (Chinese, Japanese) require word
  segmentation and pronunciation disambiguation; Japanese needs reading
  selection for kanji, which is a language-model-scale problem in itself.

The latency-relevant conclusion: **for some languages the TTS front end stops
being a lookup and becomes another inference step.** A budget that models TTS
as "phonemise, then decode" is accurate for English and understates Arabic or
Japanese. Since this repo measures English, its TTS numbers are a floor for
multilingual deployment, not a prediction of it.

## What this would take to measure properly

Stated so the gap is explicit rather than hand-waved:

1. Public multilingual TTS voices from the same family, to keep the decoder
   architecture constant while the front end varies.
2. Multilingual Whisper snapshots at matched sizes.
3. Public evaluation audio per language — and this is the constraint that
   stops the exercise here, because doing it credibly means real speech from
   real speakers, which is exactly the collected-audio and speaker-data
   territory this repo stays out of. Synthesising the test set with the same
   TTS family whose front end is under test would be circular.

The result would be a per-language waterfall showing G2P as a first-class hop.
That is a different repo, with a data-governance problem this one does not
have.

## References

Public sources; no data derived from them ships here.

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*
  (Whisper), 2022 — per-language WER/CER appendix.
- `faster-whisper` — CTranslate2 reimplementation of Whisper inference.
  <https://github.com/SYSTRAN/faster-whisper>
- Piper TTS, including its eSpeak NG phonemisation and Arabic `tashkeel`
  front-end step. <https://github.com/OHF-Voice/piper1-gpl>
- eSpeak NG — rule-based multilingual G2P and its per-language rule files.
  <https://github.com/espeak-ng/espeak-ng>
- Kim et al., *Conditional Variational Autoencoder with Adversarial Learning
  for End-to-End Text-to-Speech* (VITS), 2021 — the architecture behind Piper
  voices.
- Silero VAD. <https://github.com/snakers4/silero-vad>
