"""Tests for the budget arithmetic.

These are exact-value tests on purpose. The tracer is the one part of a voice
loop whose correct answer is knowable in advance, so it is checked against
hand-computed numbers rather than against whatever it happened to produce.
"""

from __future__ import annotations

import pytest

from voice_loop_latency_budget.tracer import (
    HOPS,
    SIMULATED_HOPS,
    HopStat,
    Span,
    Trace,
    Tracer,
    critical_path,
    dominant_hop,
    merge_intervals,
    percentile,
    summarise,
)


class FakeClock:
    """A clock the test drives by hand, so timings are exact, not flaky."""

    def __init__(self) -> None:
        self.t = 100.0  # nonzero origin: catches code that assumes t0 == 0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------
# Span
# --------------------------------------------------------------------------


def test_span_duration() -> None:
    assert Span("asr", 1.0, 1.25).duration_s == pytest.approx(0.25)


def test_span_rejects_negative_duration() -> None:
    # A backwards span means the caller mixed up two clocks. Failing here is
    # far cheaper than a negative number appearing in a results table.
    with pytest.raises(ValueError, match="ends before it starts"):
        Span("asr", 2.0, 1.0)


def test_zero_duration_span_is_allowed() -> None:
    # A hop faster than the clock resolution is legitimate, not an error.
    assert Span("vad", 1.0, 1.0).duration_s == 0.0


def test_only_the_llm_hop_is_marked_simulated() -> None:
    # Guards the honesty claim: if a real engine is ever stubbed, or the stub
    # is ever presented as measured, this test is the thing that notices.
    assert SIMULATED_HOPS == {"llm"}
    assert Span("llm", 0.0, 0.1).simulated
    assert not Span("asr", 0.0, 0.1).simulated


# --------------------------------------------------------------------------
# Interval merging and the critical path
# --------------------------------------------------------------------------


def test_merge_disjoint_intervals_stay_separate() -> None:
    spans = [Span("a", 0.0, 1.0), Span("b", 2.0, 3.0)]
    assert merge_intervals(spans) == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_overlapping_intervals_collapse() -> None:
    spans = [Span("a", 0.0, 2.0), Span("b", 1.0, 3.0)]
    assert merge_intervals(spans) == [(0.0, 3.0)]


def test_merge_handles_a_span_contained_in_another() -> None:
    # The nested span must not truncate the outer one back to its own end.
    spans = [Span("outer", 0.0, 5.0), Span("inner", 1.0, 2.0)]
    assert merge_intervals(spans) == [(0.0, 5.0)]


def test_merge_treats_touching_intervals_as_contiguous() -> None:
    spans = [Span("a", 0.0, 1.0), Span("b", 1.0, 2.0)]
    assert merge_intervals(spans) == [(0.0, 2.0)]


def test_merge_is_insertion_order_independent() -> None:
    a, b = Span("a", 2.0, 3.0), Span("b", 0.0, 1.0)
    assert merge_intervals([a, b]) == merge_intervals([b, a])


def test_critical_path_counts_overlap_once() -> None:
    """The reason this module exists.

    Two 2 s hops overlapping by 1 s sum to 4 s but occupy 3 s of wall clock.
    Reporting 4 s would produce shares that exceed 100%.
    """
    spans = [Span("llm", 0.0, 2.0), Span("tts", 1.0, 3.0)]
    assert sum(s.duration_s for s in spans) == pytest.approx(4.0)
    assert critical_path(spans) == pytest.approx(3.0)


def test_critical_path_equals_sum_when_hops_are_sequential() -> None:
    spans = [Span("asr", 0.0, 1.0), Span("llm", 1.0, 2.5)]
    assert critical_path(spans) == pytest.approx(2.5)


def test_critical_path_of_nothing_is_zero() -> None:
    assert critical_path([]) == 0.0


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------


def test_percentile_returns_an_observed_value() -> None:
    # Nearest-rank never invents a value between two samples.
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) in values
    assert percentile(values, 0.95) in values


def test_percentile_nearest_rank_exact_values() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.2) == 10.0  # ceil(0.2*5)=1 -> first
    assert percentile(values, 0.5) == 30.0  # ceil(0.5*5)=3 -> third
    assert percentile(values, 1.0) == 50.0


def test_p95_of_twenty_samples_is_the_nineteenth() -> None:
    # ceil(0.95 * 20) == 19, so the p95 is the 19th smallest, not the max.
    values = [float(i) for i in range(1, 21)]
    assert percentile(values, 0.95) == 19.0


def test_percentile_is_input_order_independent() -> None:
    assert percentile([3.0, 1.0, 2.0], 0.5) == percentile([1.0, 2.0, 3.0], 0.5)


def test_percentile_of_single_sample() -> None:
    assert percentile([7.0], 0.95) == 7.0


def test_percentile_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        percentile([], 0.5)


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_percentile_rejects_out_of_range_quantile(q: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        percentile([1.0], q)


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------


def test_by_hop_sums_repeated_spans() -> None:
    """ASR runs once per partial; the hop's cost is all of them together."""
    t = Trace()
    t.add("asr", 0.0, 0.1)
    t.add("asr", 0.2, 0.5)
    assert t.by_hop()["asr"] == pytest.approx(0.4)


def test_total_is_wall_clock_not_the_sum_of_hops() -> None:
    t = Trace()
    t.add("llm", 0.0, 2.0)
    t.add("tts", 1.0, 3.0)
    assert t.total_s() == pytest.approx(3.0)


def test_total_of_empty_trace_is_zero() -> None:
    assert Trace().total_s() == 0.0


# --------------------------------------------------------------------------
# Tracer
# --------------------------------------------------------------------------


def test_tracer_records_span_durations_from_the_clock() -> None:
    clock = FakeClock()
    tr = Tracer(clock=clock)
    with tr.span("asr"):
        clock.advance(0.25)
    (span,) = tr.trace.spans
    assert span.name == "asr"
    assert span.duration_s == pytest.approx(0.25)


def test_tracer_times_are_relative_to_construction() -> None:
    # Origin is subtracted, so a nonzero clock start does not leak in.
    clock = FakeClock()
    tr = Tracer(clock=clock)
    clock.advance(1.0)
    with tr.span("llm"):
        clock.advance(0.5)
    (span,) = tr.trace.spans
    assert span.start_s == pytest.approx(1.0)
    assert span.end_s == pytest.approx(1.5)


def test_tracer_records_a_span_that_raised() -> None:
    # A turn that failed after 300 ms still cost 300 ms.
    clock = FakeClock()
    tr = Tracer(clock=clock)
    with pytest.raises(RuntimeError), tr.span("tts"):
        clock.advance(0.3)
        raise RuntimeError("engine died")
    (span,) = tr.trace.spans
    assert span.duration_s == pytest.approx(0.3)


def test_mark_first_audio_records_perceived_latency() -> None:
    clock = FakeClock()
    tr = Tracer(clock=clock)
    clock.advance(0.42)
    assert tr.mark_first_audio() == pytest.approx(0.42)
    assert tr.trace.first_audio_s == pytest.approx(0.42)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _trace(**hops: float) -> Trace:
    """Build a trace of sequential hops with the given durations."""
    t = Trace()
    cursor = 0.0
    for name, dur in hops.items():
        t.add(name, cursor, cursor + dur)
        cursor += dur
    return t


def test_summarise_reports_p50_and_p95_per_hop() -> None:
    traces = [_trace(asr=0.1 * i, llm=0.2) for i in range(1, 21)]
    stats = {s.name: s for s in summarise(traces)}
    # ASR samples are 0.1..2.0; nearest-rank p95 of 20 is the 19th = 1.9.
    assert stats["asr"].p95_s == pytest.approx(1.9)
    assert stats["llm"].p50_s == pytest.approx(0.2)


def test_summarise_orders_hops_in_loop_order_not_insertion_order() -> None:
    t = Trace()
    t.add("tts", 2.0, 3.0)
    t.add("vad", 0.0, 1.0)
    assert [s.name for s in summarise([t])] == ["vad", "tts"]


def test_summarise_shares_sum_to_one() -> None:
    stats = summarise([_trace(vad=0.01, asr=0.4, llm=0.3, tts=0.2)])
    assert sum(s.share_p50 for s in stats) == pytest.approx(1.0)


def test_summarise_omits_hops_that_never_ran() -> None:
    names = [s.name for s in summarise([_trace(asr=0.1)])]
    assert names == ["asr"]


def test_summarise_marks_the_llm_hop_simulated() -> None:
    stats = {s.name: s for s in summarise([_trace(asr=0.1, llm=0.2)])}
    assert stats["llm"].simulated
    assert not stats["asr"].simulated


def test_summarise_rejects_no_traces() -> None:
    with pytest.raises(ValueError, match="no traces"):
        summarise([])


def test_dominant_hop_picks_the_largest_p50() -> None:
    stats = summarise([_trace(vad=0.01, asr=0.5, llm=0.2, tts=0.1)])
    assert dominant_hop(stats).name == "asr"


def test_dominant_hop_rejects_empty_stats() -> None:
    with pytest.raises(ValueError, match="no hop statistics"):
        dominant_hop([])


def test_hop_names_are_in_loop_order() -> None:
    assert HOPS == ("vad", "asr", "llm", "tts", "playback")


def test_hopstat_is_hashable_and_immutable() -> None:
    s = HopStat("asr", 0.1, 0.2, 0.5, simulated=False)
    with pytest.raises((AttributeError, TypeError)):
        s.p50_s = 0.9  # type: ignore[misc]
