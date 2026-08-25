"""Span tracing and the budget arithmetic built on top of it.

The whole repo rests on this file being right, so it is deliberately free of
audio, models and I/O: a span is a name, a start and an end, and everything
the results claim is arithmetic over a list of them. That makes the budget
exactly testable, which the engines themselves never are.

Two things here are easy to get wrong and are the reason the module exists.

**Wall-clock is not the sum of the hops.** A streaming loop overlaps work --
TTS starts on the first token while the LLM is still producing the rest -- so
adding up span durations double-counts the overlap and produces a budget that
exceeds the wall clock it is supposed to explain. :func:`critical_path` walks
the union of the intervals instead, so the parts always sum to the whole.

**The mean is the wrong statistic for a turn.** A voice loop is judged on its
bad turns, so every hop is reported p50 *and* p95. :func:`percentile` uses
nearest-rank on a sorted list rather than interpolating, because the reported
value should be a measurement that actually happened.
"""

from __future__ import annotations

import math
import time
from bisect import insort
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

# Hop names are fixed so the waterfall always renders in loop order rather
# than in dict-insertion order. `summarise` iterates this tuple, so a hop
# recorded under a misspelled name is dropped from the table rather than
# quietly appearing at the bottom of it.
HOPS: tuple[str, ...] = ("vad", "asr", "llm", "tts", "playback")

# Hops whose timing is produced by a stub rather than by a real engine. This
# constant is the single source of truth for the SIMULATED labelling, so the
# results script and the README cannot drift from what the code actually does.
SIMULATED_HOPS: frozenset[str] = frozenset({"llm"})


@dataclass(frozen=True, slots=True)
class Span:
    """One timed hop within a single turn.

    Times are monotonic seconds relative to the start of the turn, not epoch
    timestamps: the loop only ever asks about durations and overlaps, and a
    turn-relative origin makes traces from different turns directly
    comparable.
    """

    name: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(f"span {self.name!r} ends before it starts")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def simulated(self) -> bool:
        return self.name in SIMULATED_HOPS


@dataclass
class Trace:
    """All spans for one turn of the loop, plus the perceived-latency marks."""

    spans: list[Span] = field(default_factory=list)
    # Set by the loop when the first byte of audio is handed to playback. This
    # is the number the user feels; total audio duration is not.
    first_audio_s: float | None = None
    # Total duration of the synthesised audio, kept only so the results can
    # show how far first-chunk sits ahead of it.
    audio_duration_s: float | None = None

    def add(self, name: str, start_s: float, end_s: float) -> Span:
        span = Span(name, start_s, end_s)
        self.spans.append(span)
        return span

    def total_s(self) -> float:
        """Wall-clock span of the turn, first start to last end."""
        if not self.spans:
            return 0.0
        return max(s.end_s for s in self.spans) - min(s.start_s for s in self.spans)

    def by_hop(self) -> dict[str, float]:
        """Summed duration per hop.

        Sums, not the count of spans: ASR runs once per partial and the hop's
        cost is the total compute it demanded across the turn.
        """
        out: dict[str, float] = {}
        for s in self.spans:
            out[s.name] = out.get(s.name, 0.0) + s.duration_s
        return out


def merge_intervals(spans: list[Span]) -> list[tuple[float, float]]:
    """Union of the spans' intervals, sorted and non-overlapping."""
    if not spans:
        return []
    ordered = sorted((s.start_s, s.end_s) for s in spans)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # touching counts as overlapping
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def critical_path(spans: list[Span]) -> float:
    """Wall-clock time actually occupied by work, counting overlap once.

    This is the honest denominator for a share-of-budget table. Summing hop
    durations instead would let the shares total more than 100% as soon as the
    loop overlaps two hops, which a streaming loop does by design.
    """
    return sum(end - start for start, end in merge_intervals(spans))


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: always returns an observed value.

    Interpolating between two turns would report a latency that no turn
    experienced. For a budget whose job is to name the worst hop, reporting a
    real measurement matters more than a smooth curve.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    ordered = sorted(values)
    if q == 0.0:
        return ordered[0]
    # Rank is ceil(q * n), clamped into range: the smallest observed value
    # with at least q of the sample at or below it. p95 of 20 samples is the
    # 19th, not the maximum.
    rank = math.ceil(q * len(ordered))
    return ordered[min(max(rank, 1) - 1, len(ordered) - 1)]


@dataclass(frozen=True, slots=True)
class HopStat:
    """p50/p95 for one hop across many turns."""

    name: str
    p50_s: float
    p95_s: float
    share_p50: float
    simulated: bool


def summarise(traces: list[Trace]) -> list[HopStat]:
    """Per-hop p50/p95 across turns, ordered by :data:`HOPS`.

    Share is computed against the summed p50s rather than against wall clock,
    so the column reads as "of the time spent in hops, this is the fraction
    this hop owns" and totals 100% by construction.
    """
    if not traces:
        raise ValueError("no traces to summarise")
    per_hop: dict[str, list[float]] = {}
    for t in traces:
        totals = t.by_hop()
        for hop in HOPS:
            if hop in totals:
                per_hop.setdefault(hop, []).append(totals[hop])

    stats = [
        HopStat(
            name=hop,
            p50_s=percentile(samples, 0.50),
            p95_s=percentile(samples, 0.95),
            share_p50=0.0,
            simulated=hop in SIMULATED_HOPS,
        )
        for hop in HOPS
        if (samples := per_hop.get(hop))
    ]
    denom = sum(s.p50_s for s in stats)
    if denom <= 0:
        return stats
    return [
        HopStat(s.name, s.p50_s, s.p95_s, s.p50_s / denom, s.simulated) for s in stats
    ]


def dominant_hop(stats: list[HopStat]) -> HopStat:
    """The hop owning the largest p50 share -- the repo's one question."""
    if not stats:
        raise ValueError("no hop statistics")
    return max(stats, key=lambda s: s.p50_s)


class Tracer:
    """Collects spans for one turn.

    Times are recorded against a monotonic clock captured at construction, so
    a trace is immune to wall-clock adjustments mid-turn.
    """

    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self._origin = clock()
        self.trace = Trace()

    def now(self) -> float:
        return self._clock() - self._origin

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = self.now()
        try:
            yield
        finally:
            # Recorded in `finally` so a hop that raises still contributes its
            # cost. A failed turn that burned 900 ms burned 900 ms.
            self.trace.add(name, start, self.now())

    def mark_first_audio(self) -> float:
        """Stamp the moment the first audio chunk is ready for playback."""
        t = self.now()
        self.trace.first_audio_s = t
        return t


class Reservoir:
    """Keeps a sorted sample of turn latencies for cheap percentile reads.

    A plain list would do at this repo's scale; the sorted insert exists so
    that streaming many turns stays O(log n) per read rather than re-sorting
    on every query.
    """

    def __init__(self) -> None:
        self._values: list[float] = []

    def add(self, value: float) -> None:
        insort(self._values, value)

    def __len__(self) -> int:
        return len(self._values)

    def quantile(self, q: float) -> float:
        return percentile(self._values, q)
