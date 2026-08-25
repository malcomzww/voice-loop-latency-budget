"""Download the public models the results need, into .models/ (gitignored).

Only public model artifacts are fetched, and nothing here is committed. This
repo ships no collected audio, no trained voice and no speaker data -- the
test utterances are themselves synthesised by Piper at measurement time, so
there is no recorded speech in the repository at all.

Run:  python scripts/fetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / ".models"

# en_US-lessac at two quality tiers. Both are public Piper voices from the
# rhasspy/piper-voices release. Two tiers, not one, because model size is the
# TTS knob the results sweep.
VOICES = ("en_US-lessac-low", "en_US-lessac-medium")
WHISPER_SIZES = ("tiny.en", "base.en")


def fetch_piper() -> None:
    from piper.download_voices import download_voice

    out = MODELS / "piper"
    out.mkdir(parents=True, exist_ok=True)
    for voice in VOICES:
        if (out / f"{voice}.onnx").exists():
            print(f"  piper {voice}: already present")
            continue
        print(f"  piper {voice}: downloading...")
        download_voice(voice, out)


def fetch_whisper() -> None:
    from faster_whisper import WhisperModel

    out = MODELS / "whisper"
    out.mkdir(parents=True, exist_ok=True)
    for size in WHISPER_SIZES:
        print(f"  whisper {size}: ensuring snapshot...")
        # Constructing the model is what triggers the download; it is cached
        # by size under download_root thereafter.
        WhisperModel(size, device="cpu", compute_type="int8", download_root=str(out))


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    print(f"fetching public models into {MODELS}")
    try:
        fetch_piper()
        fetch_whisper()
    except ImportError as exc:
        print(f"error: speech extras not installed ({exc}).", file=sys.stderr)
        print("Run: uv sync --extra speech --extra dev", file=sys.stderr)
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
