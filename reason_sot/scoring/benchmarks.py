"""Comparative benchmark runner — validates ReasonSoT against baselines.

Runs the same interview scenarios through multiple configurations:
  1. BASELINE: Direct Haiku call, no reasoning framework
  2. COT: Direct Sonnet call with "think step by step"
  3. TOT: Fixed beam Tree-of-Thought (beam=3, no adaptation)
  4. REASON_SOT: Full dual-system with routing, MoT, early exit, follow-up

Compares on: 4 quality metrics + latency.

The benchmark uses simulated candidate responses (pre-scripted or LLM-generated)
to ensure reproducibility and fair comparison.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reason_sot.persona.profiles import PersonaProfile
from reason_sot.scoring.latency import LatencyReport, profile_session
from reason_sot.scoring.metrics import ReasoningScores, score_session
from reason_sot.types import InterviewSession

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str
    description: str
    system_mode: str  # "s1_only", "s2_only", "s2_cot", "s2_tot", "reason_sot"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config_name: str
    scenario_name: str
    scores: ReasoningScores
    latency: LatencyReport
    session: InterviewSession
    run_time_seconds: float = 0.0


@dataclass
class BenchmarkComparison:
    """Side-by-side comparison of multiple benchmark runs."""

    scenario_name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""

    def to_table(self) -> str:
        """Format as a comparison table."""
        if not self.results:
            return "No results"

        header = (
            f"{'Config':<15} {'Depth':>6} {'Breadth':>7} {'Persona':>8} "
            f"{'Follow':>7} {'Overall':>8} {'TTFT p50':>9} {'TTFT p95':>9} "
            f"{'S1/S2':>6}"
        )
        separator = "-" * len(header)
        lines = [
            f"=== Benchmark: {self.scenario_name} ===",
            "",
            header,
            separator,
        ]

        for r in self.results:
            s = r.scores
            l = r.latency
            lines.append(
                f"{r.config_name:<15} "
                f"{s.depth_score:>6.2f} {s.breadth_score:>7.2f} "
                f"{s.persona_consistency_score:>8.2f} "
                f"{s.followup_relevance_score:>7.2f} {s.overall_score:>8.2f} "
                f"{l.ttft_p50_ms or 0:>8.0f}ms {l.ttft_p95_ms or 0:>8.0f}ms "
                f"{l.s1_turns:>2}/{l.s2_turns:<2}"
            )

        lines.append(separator)

        # Winner by overall score
        best = max(self.results, key=lambda r: r.scores.overall_score)
        lines.append(f"Best overall: {best.config_name} ({best.scores.overall_score:.2f})")

        # Winner by latency
        fastest = min(
            self.results,
            key=lambda r: r.latency.ttft_p50_ms or float("inf"),
        )
        lines.append(f"Fastest TTFT:  {fastest.config_name} ({fastest.latency.ttft_p50_ms:.0f}ms)")

        return "\n".join(lines)


# ── Predefined benchmark configs ──────────────────────────────────────────

BASELINE_CONFIGS: list[BenchmarkConfig] = [
    BenchmarkConfig(
        name="baseline",
        description="Direct Haiku call, no reasoning framework",
        system_mode="s1_only",
    ),
    BenchmarkConfig(
        name="cot_sonnet",
        description="Sonnet with 'think step by step' prompt",
        system_mode="s2_cot",
    ),
    BenchmarkConfig(
        name="reason_sot",
        description="Full ReasonSoT: dual-system, MoT, early exit, follow-up",
        system_mode="reason_sot",
    ),
]


# ── Scenario loading ──────────────────────────────────────────────────────

@dataclass
class BenchmarkScenario:
    """A scripted interview scenario for reproducible benchmarking."""

    name: str
    description: str
    persona_name: str
    candidate_responses: list[str]  # Pre-scripted candidate responses in order


def load_scenario(path: Path) -> BenchmarkScenario:
    """Load a benchmark scenario from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return BenchmarkScenario(**data)


def list_scenarios(scenarios_dir: Path | None = None) -> list[Path]:
    """List available benchmark scenario files."""
    d = scenarios_dir or Path(__file__).parent.parent.parent / "benchmarks" / "scenarios"
    return sorted(d.glob("*.json"))


# ── Scoring helper (works on any session) ─────────────────────────────────


def score_and_profile(
    session: InterviewSession,
    persona: PersonaProfile,
    config_name: str,
    scenario_name: str,
    run_time: float = 0.0,
) -> BenchmarkResult:
    """Score a completed session and produce a BenchmarkResult."""
    scores = score_session(session, persona)
    latency = profile_session(session)

    return BenchmarkResult(
        config_name=config_name,
        scenario_name=scenario_name,
        scores=scores,
        latency=latency,
        session=session,
        run_time_seconds=run_time,
    )


def save_comparison(
    comparison: BenchmarkComparison,
    output_dir: Path | None = None,
) -> Path:
    """Save benchmark comparison results to JSON."""
    d = output_dir or Path(__file__).parent.parent.parent / "benchmarks" / "results"
    d.mkdir(parents=True, exist_ok=True)

    filename = f"benchmark_{comparison.scenario_name}_{int(time.time())}.json"
    path = d / filename

    # Serialize (excluding raw session data for size)
    data = {
        "scenario": comparison.scenario_name,
        "timestamp": comparison.timestamp,
        "results": [
            {
                "config": r.config_name,
                "scores": {
                    "depth": r.scores.depth_score,
                    "breadth": r.scores.breadth_score,
                    "persona": r.scores.persona_consistency_score,
                    "followup": r.scores.followup_relevance_score,
                    "overall": r.scores.overall_score,
                },
                "latency": {
                    "ttft_p50_ms": r.latency.ttft_p50_ms,
                    "ttft_p95_ms": r.latency.ttft_p95_ms,
                    "s1_turns": r.latency.s1_turns,
                    "s2_turns": r.latency.s2_turns,
                    "cache_hit_rate": r.latency.avg_cache_hit_rate,
                },
                "run_time_seconds": r.run_time_seconds,
            }
            for r in comparison.results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved benchmark results to %s", path)
    return path
