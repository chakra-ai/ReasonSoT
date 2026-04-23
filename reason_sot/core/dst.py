"""Domain-Specialized Tree (DST) — confidence-based dynamic beam reasoning.

Based on the DST paper (2603.20267): unlike fixed-beam ToT, DST dynamically
adjusts search breadth based on real-time confidence:
  - HIGH confidence → near-greedy (beam=1), fast path
  - LOW confidence → expand beam (up to max), thorough exploration
  - 26-75% token savings vs fixed-beam approaches

Implementation: single-prompt approach that instructs the model to self-assess
confidence and adapt its exploration depth within the thinking budget.
The "beam search" is simulated by asking the model to explore N paths
and self-select, not by making N separate API calls.

For the interview context:
  - Confident about candidate's level → ask ONE targeted question (fast)
  - Uncertain about depth/breadth → explore 2-3 directions, pick best
  - Candidate gave surprising answer → full beam to explore the unexpected
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSTConfig:
    """Configuration for Domain-Specialized Tree reasoning."""

    min_beam: int = 1
    max_beam: int = 3
    confidence_threshold: float = 0.5  # Below this → expand beam
    initial_beam: int = 1  # Start greedy, expand if needed


def build_dst_prompt(
    config: DSTConfig | None = None,
    coverage_gaps: list[str] | None = None,
    current_topic: str = "",
) -> str:
    """Build a DST instruction prompt for the model's thinking.

    The prompt instructs the model to:
    1. Start with a greedy assessment (beam=1)
    2. Self-assess confidence
    3. If low confidence, expand to explore more paths
    4. Select the best path and synthesize a response
    """
    cfg = config or DSTConfig()
    gaps_context = ""
    if coverage_gaps:
        gaps_context = f"\nUncovered topics that need attention: {', '.join(coverage_gaps[:5])}"

    topic_context = ""
    if current_topic:
        topic_context = f"\nCurrent topic being explored: {current_topic}"

    return f"""[REASONING STRATEGY: Domain-Specialized Tree — adaptive beam search]
{topic_context}{gaps_context}

STEP 1 — INITIAL ASSESSMENT (beam=1, greedy):
Think about the candidate's response. What's your confidence level (0-1)?
  - HIGH confidence (>{cfg.confidence_threshold}): You clearly understand their level.
    → Go directly to STEP 3 with ONE follow-up direction.
  - LOW confidence (<={cfg.confidence_threshold}): The response is ambiguous, surprising, or incomplete.
    → Expand to STEP 2.

STEP 2 — EXPANDED SEARCH (beam={cfg.max_beam}, only if low confidence):
Generate {cfg.max_beam} distinct follow-up directions:
  Direction 1: [probe deeper on what they said]
  Direction 2: [test from a different angle]
  Direction 3: [explore an adjacent topic]
For each direction, briefly assess:
  - Information gain: how much would we learn?
  - Risk: could this dead-end?
  - Coverage: does this fill a gap in our knowledge map?

STEP 3 — SELECT & RESPOND:
Pick the single best direction. State why.
Generate a natural, conversational follow-up question (2-3 sentences max).

IMPORTANT: If you were confident in Step 1, skip Step 2 entirely — don't waste reasoning tokens.
Your spoken response should be natural — the tree reasoning happens in thinking only."""


def estimate_beam_from_context(
    previous_confidence: float = 1.0,
    followup_chain_depth: int = 0,
    coverage_gap_score: float = 0.0,
    config: DSTConfig | None = None,
) -> int:
    """Pre-estimate beam width from context signals.

    This helps set the thinking budget (wider beam = more budget needed).
    """
    cfg = config or DSTConfig()

    # Start at minimum
    beam = cfg.min_beam

    # Low previous confidence → wider beam
    if previous_confidence < cfg.confidence_threshold:
        beam += 1

    # Deep in follow-up chain → might need new direction
    if followup_chain_depth >= 2:
        beam += 1

    # Large coverage gaps → explore more broadly
    if coverage_gap_score > 0.5:
        beam += 1

    return min(beam, cfg.max_beam)
