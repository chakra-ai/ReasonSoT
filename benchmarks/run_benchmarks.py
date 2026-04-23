#!/usr/bin/env python3
"""Benchmark runner — compares ReasonSoT against baselines.

Runs the same pre-scripted interview scenario through multiple configurations
and produces a side-by-side comparison of quality metrics and latency.

Usage:
    python benchmarks/run_benchmarks.py                          # All scenarios
    python benchmarks/run_benchmarks.py --scenario technical_python
    python benchmarks/run_benchmarks.py --configs baseline,reason_sot
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from reason_sot.interview.engine import InterviewEngine
from reason_sot.llm.client import LLMClient
from reason_sot.persona.manager import load_persona
from reason_sot.scoring.benchmarks import (
    BASELINE_CONFIGS,
    BenchmarkComparison,
    BenchmarkConfig,
    BenchmarkScenario,
    load_scenario,
    list_scenarios,
    save_comparison,
    score_and_profile,
)
from reason_sot.scoring.latency import format_report, profile_session
from reason_sot.types import StreamDone, StreamError, TextDelta

logger = logging.getLogger(__name__)


async def run_scenario_with_config(
    client: LLMClient,
    scenario: BenchmarkScenario,
    config: BenchmarkConfig,
) -> InterviewEngine:
    """Run a scenario through a specific configuration."""
    persona = load_persona(scenario.persona_name)

    # Configure engine based on benchmark config
    if config.system_mode == "s1_only":
        # Force System 1 by setting impossibly high threshold
        engine = InterviewEngine(client=client, persona=persona, complexity_threshold=10.0)
    elif config.system_mode == "s2_cot":
        # Force System 2 by setting threshold to 0
        engine = InterviewEngine(client=client, persona=persona, complexity_threshold=0.0)
    elif config.system_mode == "reason_sot":
        # Normal ReasonSoT with default threshold
        engine = InterviewEngine(client=client, persona=persona)
    else:
        engine = InterviewEngine(client=client, persona=persona)

    # Run through pre-scripted responses
    for i, response in enumerate(scenario.candidate_responses):
        logger.info(
            "  [%s] Turn %d/%d: %s...",
            config.name,
            i + 1,
            len(scenario.candidate_responses),
            response[:50],
        )
        async for event in engine.process_turn(response):
            if isinstance(event, StreamError):
                logger.error("Error in turn %d: %s", i + 1, event.error)

    await engine.end_session()
    return engine


async def run_benchmark(
    scenario_path: Path,
    config_names: list[str] | None = None,
) -> BenchmarkComparison:
    """Run a complete benchmark comparison for one scenario."""
    cfg = get_config()
    if not cfg.ANTHROPIC_API_KEY or cfg.ANTHROPIC_API_KEY.startswith("sk-ant-..."):
        print("Error: Set ANTHROPIC_API_KEY in .env or environment.")
        sys.exit(1)

    scenario = load_scenario(scenario_path)
    client = LLMClient(api_key=cfg.ANTHROPIC_API_KEY)

    # Filter configs if specified
    configs = BASELINE_CONFIGS
    if config_names:
        configs = [c for c in configs if c.name in config_names]

    print(f"\n{'='*60}")
    print(f"  Benchmark: {scenario.name}")
    print(f"  Scenario: {scenario.description}")
    print(f"  Candidate turns: {len(scenario.candidate_responses)}")
    print(f"  Configs: {', '.join(c.name for c in configs)}")
    print(f"{'='*60}\n")

    comparison = BenchmarkComparison(
        scenario_name=scenario.name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    for config in configs:
        print(f"Running config: {config.name} ({config.description})...")
        start = time.monotonic()

        try:
            engine = await run_scenario_with_config(client, scenario, config)
            run_time = time.monotonic() - start

            persona = load_persona(scenario.persona_name)
            result = score_and_profile(
                session=engine.session,
                persona=persona,
                config_name=config.name,
                scenario_name=scenario.name,
                run_time=run_time,
            )
            comparison.results.append(result)

            # Print interim latency
            report = profile_session(engine.session)
            print(f"  Done in {run_time:.1f}s — "
                  f"TTFT p50={report.ttft_p50_ms or 0:.0f}ms, "
                  f"S1={report.s1_turns} S2={report.s2_turns}\n")

        except Exception as e:
            logger.error("Config %s failed: %s", config.name, e)
            print(f"  FAILED: {e}\n")

    return comparison


async def main() -> None:
    parser = argparse.ArgumentParser(description="ReasonSoT Benchmark Runner")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Specific scenario name to run (default: all)",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config names to compare (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    config_names = args.configs.split(",") if args.configs else None

    # Find scenarios
    scenario_paths = list_scenarios()
    if args.scenario:
        scenario_paths = [p for p in scenario_paths if p.stem == args.scenario]
        if not scenario_paths:
            print(f"Scenario '{args.scenario}' not found.")
            available = [p.stem for p in list_scenarios()]
            print(f"Available: {available}")
            sys.exit(1)

    for path in scenario_paths:
        comparison = await run_benchmark(path, config_names)

        # Print comparison table
        print(comparison.to_table())
        print()

        # Save results
        output_path = save_comparison(comparison)
        print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
