#!/usr/bin/env python3
"""Interactive CLI demo for ReasonSoT interview agent.

Usage:
    python demo.py                              # Default: technical_interviewer
    python demo.py --persona behavioral_interviewer
    python demo.py --verbose                    # Show latency metrics per turn
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from config import get_config
from reason_sot.interview.engine import InterviewEngine
from reason_sot.llm.client import LLMClient
from reason_sot.persona.manager import list_personas, load_persona
from reason_sot.types import StreamDone, StreamError, TextDelta


async def run_interview(persona_name: str, verbose: bool = False) -> None:
    cfg = get_config()

    if not cfg.ANTHROPIC_API_KEY or cfg.ANTHROPIC_API_KEY.startswith("sk-ant-..."):
        print("Error: Set ANTHROPIC_API_KEY in .env or environment.")
        print("  cp .env.example .env  # then edit with your key")
        sys.exit(1)

    # Load persona
    try:
        persona = load_persona(persona_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Available personas: {list_personas()}")
        sys.exit(1)

    # Initialize
    client = LLMClient(api_key=cfg.ANTHROPIC_API_KEY)
    engine = InterviewEngine(client=client, persona=persona)

    print(f"\n{'='*60}")
    print(f"  ReasonSoT Interview Agent v0.1")
    print(f"  Persona: {persona.name} ({persona.role})")
    print(f"  Domain: {persona.domain}")
    print(f"{'='*60}\n")

    # Opening message
    opening = engine.get_opening_message()
    print(f"Agent: {opening}\n")

    # Interview loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEnding interview...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nAgent: Thank you for your time today. Best of luck!\n")
            break

        # Stream agent response
        print("Agent: ", end="", flush=True)
        async for event in engine.process_turn(user_input):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, StreamDone):
                print()  # newline after streaming
                if verbose:
                    ttft = event.latency.ttft_ms
                    total = event.latency.total_ms
                    cache_rate = event.usage.cache_hit_rate * 100
                    # Get the latest turn for routing info
                    last_turn = engine.session.turns[-1] if engine.session.turns else None
                    system_label = f"S{last_turn.routing.system}" if last_turn else "?"
                    mode_label = last_turn.routing.reasoning_mode.value if last_turn else "?"
                    followup_label = last_turn.follow_up.action.value if last_turn and last_turn.follow_up else "n/a"
                    print(
                        f"  [{system_label}/{mode_label}] "
                        f"TTFT: {ttft:.0f}ms | Total: {total:.0f}ms | "
                        f"Tokens: {event.usage.output_tokens} | "
                        f"Cache: {cache_rate:.0f}% | "
                        f"Follow-up: {followup_label}"
                    )
            elif isinstance(event, StreamError):
                print(f"\n  [Error: {event.error}]")
        print()

    # Session summary
    session = await engine.end_session()
    stats = engine.get_session_stats()
    print(f"\n{'='*60}")
    print(f"  Session Summary")
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  System 1 turns: {stats['s1_turns']}")
    print(f"  System 2 turns: {stats['s2_turns']}")
    if stats['s1_avg_ttft_ms'] is not None:
        print(f"  S1 avg TTFT: {stats['s1_avg_ttft_ms']:.0f}ms")
    if stats['s2_avg_ttft_ms'] is not None:
        print(f"  S2 avg TTFT: {stats['s2_avg_ttft_ms']:.0f}ms")
    print(f"  Reasoning modes: {', '.join(stats['reasoning_modes_used'])}")

    # Knowledge graph stats
    if "kg_coverage" in stats:
        print(f"\n  Knowledge Graph:")
        print(f"    Nodes: {stats.get('kg_nodes', 0)}, Edges: {stats.get('kg_edges', 0)}")
        print(f"    Topic coverage:")
        for topic, score in stats["kg_coverage"].items():
            bar = "#" * int(score * 10) + "." * (10 - int(score * 10))
            print(f"      [{bar}] {score:.0%} {topic}")
        if stats.get("kg_gaps"):
            print(f"    Gaps: {', '.join(stats['kg_gaps'])}")

    if stats.get("persona_switch_suggested"):
        print(f"\n  Suggested persona switch: {stats['persona_switch_suggested']}")

    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ReasonSoT Interview Agent Demo")
    parser.add_argument(
        "--persona",
        default="technical_interviewer",
        help=f"Persona to use (available: {list_personas()})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show latency metrics per turn",
    )
    args = parser.parse_args()

    asyncio.run(run_interview(args.persona, args.verbose))


if __name__ == "__main__":
    main()
