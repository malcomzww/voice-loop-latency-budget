"""A streaming ASR to LLM to TTS loop instrumented end-to-end, answering where
the perceived latency actually lives.

The tracer and the budget arithmetic import cleanly with no speech engines
installed; ``asr``, ``tts``, ``vad`` and ``loop`` import their engines lazily
so that a machine without the optional ``speech`` extra can still run the
arithmetic and its tests.
"""

from .tracer import (
    HOPS,
    SIMULATED_HOPS,
    HopStat,
    Span,
    Trace,
    Tracer,
    critical_path,
    dominant_hop,
    percentile,
    summarise,
)

__version__ = "0.1.0"

__all__ = [
    "HOPS",
    "SIMULATED_HOPS",
    "HopStat",
    "Span",
    "Trace",
    "Tracer",
    "critical_path",
    "dominant_hop",
    "percentile",
    "summarise",
    "__version__",
]
