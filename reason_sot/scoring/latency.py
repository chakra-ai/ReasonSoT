"""Latency profiler — tracks and reports timing metrics.

Records per-turn: TTFT, total duration, token counts, cache hit rates,
model used, and reasoning mode. Aggregates into percentile reports.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from reason_sot.types import InterviewSession, InterviewTurn


@dataclass
class LatencyReport:
    """Aggregated latency metrics for a session or benchmark run."""

    # TTFT (time to first token)
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    ttft_mean_ms: float | None = None

    # Total turn duration
    total_p50_ms: float | None = None
    total_p95_ms: float | None = None
    total_mean_ms: float | None = None

    # Token efficiency
    avg_output_tokens: float = 0.0
    avg_thinking_tokens: float = 0.0
    thinking_to_output_ratio: float = 0.0

    # Cache performance
    avg_cache_hit_rate: float = 0.0

    # System breakdown
    s1_turns: int = 0
    s2_turns: int = 0
    s1_avg_ttft_ms: float | None = None
    s2_avg_ttft_ms: float | None = None

    # Latency target compliance
    s1_within_target: float = 0.0  # Fraction of S1 turns with TTFT < 500ms
    s2_within_target: float = 0.0  # Fraction of S2 turns with TTFT < 1000ms

    # Per-mode breakdown
    mode_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)


def profile_session(
    session: InterviewSession,
    s1_ttft_target_ms: float = 500.0,
    s2_ttft_target_ms: float = 1000.0,
) -> LatencyReport:
    """Profile latency metrics for a complete interview session."""
    if not session.turns:
        return LatencyReport()

    report = LatencyReport()

    # Collect TTFT and total times
    all_ttfts: list[float] = []
    all_totals: list[float] = []
    s1_ttfts: list[float] = []
    s2_ttfts: list[float] = []
    cache_rates: list[float] = []
    output_tokens: list[int] = []
    thinking_tokens: list[int] = []

    mode_data: dict[str, list[float]] = {}

    for turn in session.turns:
        ttft = turn.latency.ttft_ms
        total = turn.latency.total_ms

        if ttft is not None:
            all_ttfts.append(ttft)
            if turn.routing.system == 1:
                s1_ttfts.append(ttft)
            else:
                s2_ttfts.append(ttft)

            mode = turn.routing.reasoning_mode.value
            mode_data.setdefault(mode, []).append(ttft)

        if total is not None:
            all_totals.append(total)

        cache_rates.append(turn.usage.cache_hit_rate)
        output_tokens.append(turn.usage.output_tokens)
        thinking_tokens.append(turn.usage.thinking_tokens)

    # TTFT percentiles
    if all_ttfts:
        sorted_ttfts = sorted(all_ttfts)
        report.ttft_mean_ms = statistics.mean(sorted_ttfts)
        report.ttft_p50_ms = _percentile(sorted_ttfts, 50)
        report.ttft_p95_ms = _percentile(sorted_ttfts, 95)
        report.ttft_p99_ms = _percentile(sorted_ttfts, 99)

    # Total duration percentiles
    if all_totals:
        sorted_totals = sorted(all_totals)
        report.total_mean_ms = statistics.mean(sorted_totals)
        report.total_p50_ms = _percentile(sorted_totals, 50)
        report.total_p95_ms = _percentile(sorted_totals, 95)

    # Token efficiency
    if output_tokens:
        report.avg_output_tokens = statistics.mean(output_tokens)
    if thinking_tokens:
        report.avg_thinking_tokens = statistics.mean(thinking_tokens)
        if report.avg_output_tokens > 0:
            report.thinking_to_output_ratio = report.avg_thinking_tokens / report.avg_output_tokens

    # Cache performance
    if cache_rates:
        report.avg_cache_hit_rate = statistics.mean(cache_rates)

    # System breakdown
    report.s1_turns = len(s1_ttfts)
    report.s2_turns = len(s2_ttfts)
    report.s1_avg_ttft_ms = statistics.mean(s1_ttfts) if s1_ttfts else None
    report.s2_avg_ttft_ms = statistics.mean(s2_ttfts) if s2_ttfts else None

    # Target compliance
    if s1_ttfts:
        report.s1_within_target = sum(1 for t in s1_ttfts if t < s1_ttft_target_ms) / len(s1_ttfts)
    if s2_ttfts:
        report.s2_within_target = sum(1 for t in s2_ttfts if t < s2_ttft_target_ms) / len(s2_ttfts)

    # Per-mode breakdown
    for mode, ttfts in mode_data.items():
        report.mode_breakdown[mode] = {
            "count": len(ttfts),
            "avg_ttft_ms": statistics.mean(ttfts),
            "p95_ttft_ms": _percentile(sorted(ttfts), 95),
        }

    return report


def format_report(report: LatencyReport) -> str:
    """Format a latency report as a human-readable string."""
    lines = [
        "=== Latency Report ===",
        "",
        f"TTFT:  mean={_fmt(report.ttft_mean_ms)}  p50={_fmt(report.ttft_p50_ms)}  "
        f"p95={_fmt(report.ttft_p95_ms)}  p99={_fmt(report.ttft_p99_ms)}",
        f"Total: mean={_fmt(report.total_mean_ms)}  p50={_fmt(report.total_p50_ms)}  "
        f"p95={_fmt(report.total_p95_ms)}",
        "",
        f"System 1: {report.s1_turns} turns, avg TTFT={_fmt(report.s1_avg_ttft_ms)}, "
        f"within 500ms target: {report.s1_within_target:.0%}",
        f"System 2: {report.s2_turns} turns, avg TTFT={_fmt(report.s2_avg_ttft_ms)}, "
        f"within 1000ms target: {report.s2_within_target:.0%}",
        "",
        f"Tokens: avg output={report.avg_output_tokens:.0f}, "
        f"avg thinking={report.avg_thinking_tokens:.0f}, "
        f"think/output ratio={report.thinking_to_output_ratio:.1f}",
        f"Cache hit rate: {report.avg_cache_hit_rate:.0%}",
    ]

    if report.mode_breakdown:
        lines.append("")
        lines.append("Per-mode breakdown:")
        for mode, data in report.mode_breakdown.items():
            lines.append(
                f"  {mode}: {data['count']} turns, "
                f"avg TTFT={data['avg_ttft_ms']:.0f}ms, "
                f"p95={data['p95_ttft_ms']:.0f}ms"
            )

    return "\n".join(lines)


def _percentile(sorted_values: list[float], p: int) -> float:
    """Compute the p-th percentile from a sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def _fmt(value: float | None) -> str:
    """Format a millisecond value."""
    return f"{value:.0f}ms" if value is not None else "n/a"
