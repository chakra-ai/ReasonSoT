"""Heuristic router — classifies complexity and routes to System 1 or System 2.

Zero LLM cost: uses regex patterns, utterance statistics, and conversation
state to make routing decisions in microseconds. The speculative CoT path
handles borderline cases where the router is uncertain.

Signals used:
  - Utterance length and word count
  - Question type detection (broad vs deep)
  - Technical jargon density
  - Conversation phase (early=rapport, mid=core, late=deep-dive)
  - Follow-up chain depth (consecutive clarifications)
  - Knowledge graph gap score (unexplored high-priority topics)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reason_sot.types import (
    InterviewPhase,
    ReasoningMode,
    RoutingDecision,
)


# ── Signal patterns ────────────────────────────────────────────────────────

# Patterns suggesting DEEP reasoning needed (System 2)
DEEP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhy\s+(specifically|exactly|do you think)\b",
        r"\bhow\s+would\s+you\s+(design|architect|implement|build|scale)\b",
        r"\bwalk\s+me\s+through\b",
        r"\bexplain\s+(the\s+)?trade-?offs?\b",
        r"\bwhat\s+(happens|would happen)\s+(if|when|under)\b",
        r"\bcompare\s+and\s+contrast\b",
        r"\bwhat\s+are\s+the\s+(pros|cons|advantages|disadvantages|limitations)\b",
        r"\bhow\s+does\s+(that|this|it)\s+work\s+under\s+the\s+hood\b",
        r"\bat\s+\d+x\s+scale\b",
        r"\bedge\s+case",
        r"\bwhat\s+if\b.*\bfail",
        r"\bdesign\s+(a|an|the)\b",
    ]
]

# Patterns suggesting SIMPLE response (System 1)
SIMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(yes|no|yeah|yep|nope|sure|ok|okay|right|got it|i see)\b",
        r"^(hi|hello|hey|thanks|thank you|good morning|good afternoon)\b",
        r"\bcan\s+you\s+repeat\b",
        r"\bwhat\s+was\s+the\s+question\b",
        r"^(i'?m\s+ready|let'?s\s+(start|go|begin))",
    ]
]

# Technical jargon that suggests deeper knowledge / deeper questions needed
TECHNICAL_TERMS: set[str] = {
    "async", "await", "coroutine", "thread", "mutex", "deadlock", "race condition",
    "gil", "garbage collection", "reference counting", "metaclass", "descriptor",
    "mro", "dunder", "decorator", "generator", "iterator", "protocol",
    "microservice", "monolith", "cqrs", "event sourcing", "saga",
    "sharding", "replication", "consensus", "raft", "paxos",
    "caching", "invalidation", "consistent hashing", "load balancing",
    "sql", "orm", "migration", "index", "b-tree", "transaction", "isolation",
    "rest", "grpc", "graphql", "websocket", "idempotent",
    "docker", "kubernetes", "ci/cd", "terraform", "observability",
    "big-o", "complexity", "amortized", "heap", "trie", "graph",
    "backpressure", "circuit breaker", "retry", "exponential backoff",
}


# ── Scoring functions ──────────────────────────────────────────────────────


@dataclass
class ComplexitySignals:
    """Collected signals for routing decision."""

    utterance_length: int = 0
    word_count: int = 0
    deep_pattern_matches: int = 0
    simple_pattern_matches: int = 0
    technical_term_count: int = 0
    technical_term_density: float = 0.0
    conversation_turn: int = 0
    phase: InterviewPhase = InterviewPhase.CORE
    followup_chain_depth: int = 0
    previous_confidence: float = 1.0


def _compute_signals(
    user_input: str,
    turn_number: int,
    phase: InterviewPhase,
    followup_chain_depth: int = 0,
    previous_confidence: float = 1.0,
) -> ComplexitySignals:
    """Compute all routing signals from user input and context."""
    words = user_input.lower().split()
    word_count = len(words)

    deep_matches = sum(1 for p in DEEP_PATTERNS if p.search(user_input))
    simple_matches = sum(1 for p in SIMPLE_PATTERNS if p.search(user_input))

    tech_count = sum(1 for w in words if w.strip(".,;:!?()") in TECHNICAL_TERMS)
    tech_density = tech_count / max(word_count, 1)

    return ComplexitySignals(
        utterance_length=len(user_input),
        word_count=word_count,
        deep_pattern_matches=deep_matches,
        simple_pattern_matches=simple_matches,
        technical_term_count=tech_count,
        technical_term_density=tech_density,
        conversation_turn=turn_number,
        phase=phase,
        followup_chain_depth=followup_chain_depth,
        previous_confidence=previous_confidence,
    )


def _compute_complexity_score(signals: ComplexitySignals) -> float:
    """Compute a 0-1 complexity score from signals.

    Higher score = more complex = more likely System 2.
    """
    score = 0.0

    # Length signal: longer responses from candidate may need deeper probing
    if signals.word_count > 50:
        score += 0.15
    elif signals.word_count < 5:
        score -= 0.2

    # Pattern matching — deep patterns are strong signals
    score += signals.deep_pattern_matches * 0.25
    score -= signals.simple_pattern_matches * 0.3

    # Multiple deep patterns compound: 2+ patterns strongly suggest S2
    if signals.deep_pattern_matches >= 2:
        score += 0.15

    # Technical density
    score += signals.technical_term_density * 0.5

    # Conversation phase
    phase_weights = {
        InterviewPhase.OPENING: -0.2,
        InterviewPhase.CORE: 0.0,
        InterviewPhase.DEEP_DIVE: 0.2,
        InterviewPhase.CLOSING: -0.15,
    }
    score += phase_weights.get(signals.phase, 0.0)

    # Follow-up chain: if we've been clarifying, go deeper
    if signals.followup_chain_depth >= 2:
        score += 0.15

    # Previous low confidence: escalate
    if signals.previous_confidence < 0.6:
        score += 0.2

    return max(0.0, min(1.0, score))


# ── Public API ─────────────────────────────────────────────────────────────


def route(
    user_input: str,
    turn_number: int = 1,
    phase: InterviewPhase = InterviewPhase.CORE,
    followup_chain_depth: int = 0,
    previous_confidence: float = 1.0,
    complexity_threshold: float = 0.6,
    s2_thinking_budget: int = 4096,
) -> RoutingDecision:
    """Route a user input to System 1 or System 2.

    Returns a RoutingDecision with the system, reasoning mode,
    thinking budget, and confidence level.

    Args:
        user_input: The candidate's response text.
        turn_number: Current turn in the interview.
        phase: Current interview phase.
        followup_chain_depth: How many consecutive follow-ups on the same topic.
        previous_confidence: The agent's confidence in its last response.
        complexity_threshold: Score above this routes to System 2.
        s2_thinking_budget: Default thinking budget for System 2.
    """
    signals = _compute_signals(
        user_input, turn_number, phase, followup_chain_depth, previous_confidence
    )
    complexity = _compute_complexity_score(signals)

    # Clear System 1: simple greetings, acknowledgments
    if signals.simple_pattern_matches > 0 and signals.deep_pattern_matches == 0:
        return RoutingDecision(
            system=1,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence=0.95,
            rationale=f"Simple pattern match (complexity={complexity:.2f})",
        )

    # Clear System 2: explicit deep-thinking patterns
    if complexity >= complexity_threshold + 0.15:
        # High complexity: use Matrix of Thought for breadth+depth
        mode = ReasoningMode.MOT if signals.deep_pattern_matches >= 2 else ReasoningMode.COT
        return RoutingDecision(
            system=2,
            reasoning_mode=mode,
            thinking_budget=s2_thinking_budget,
            confidence=min(0.95, 0.5 + complexity),
            rationale=f"High complexity={complexity:.2f}, deep_patterns={signals.deep_pattern_matches}",
        )

    # Above threshold: System 2 with standard CoT
    if complexity >= complexity_threshold:
        return RoutingDecision(
            system=2,
            reasoning_mode=ReasoningMode.COT,
            thinking_budget=s2_thinking_budget // 2,
            confidence=0.5 + (complexity - complexity_threshold),
            rationale=f"Above threshold (complexity={complexity:.2f})",
        )

    # Borderline zone (0.4 - threshold): speculative path
    if complexity >= complexity_threshold - 0.2:
        return RoutingDecision(
            system=1,
            reasoning_mode=ReasoningMode.SPECULATIVE,
            confidence=0.5,
            rationale=f"Borderline (complexity={complexity:.2f}), using speculative CoT",
        )

    # Default: System 1 fast path
    return RoutingDecision(
        system=1,
        reasoning_mode=ReasoningMode.DIRECT,
        confidence=max(0.6, 1.0 - complexity),
        rationale=f"Below threshold (complexity={complexity:.2f})",
    )
