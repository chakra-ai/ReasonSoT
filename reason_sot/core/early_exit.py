"""Early exit / adaptive compute — prevents overthinking and adjusts thinking budget.

Based on research showing 31-43% token reduction by detecting when the model
has reached sufficient confidence and stopping further reasoning.

Two modes:
1. **Pre-call budget sizing**: Estimates optimal thinking budget based on
   question complexity before the API call (since we can't interrupt mid-stream).
2. **Post-call analysis**: Analyzes thinking output to detect overthinking
   patterns, used to adjust budget for subsequent turns.

The Anthropic API doesn't support mid-stream thinking cancellation, so we
optimize the budget_tokens parameter upfront rather than interrupting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ThinkingBudgetEstimate:
    """Estimated thinking budget for a turn."""

    budget_tokens: int
    confidence: float  # How confident we are in this estimate (0-1)
    rationale: str = ""


@dataclass
class ThinkingAnalysis:
    """Post-hoc analysis of model thinking for budget adjustment."""

    total_thinking_words: int = 0
    repetition_score: float = 0.0  # 0=no repetition, 1=highly repetitive
    convergence_point: float = 1.0  # Fraction of thinking where conclusion was reached
    had_clear_conclusion: bool = False
    recommended_adjustment: float = 1.0  # Multiplier for next turn's budget


# ── Budget sizing (pre-call) ──────────────────────────────────────────────

# Complexity-to-budget mapping
BUDGET_TABLE: list[tuple[float, int]] = [
    (0.0, 1024),   # Trivial: minimal thinking
    (0.3, 2048),   # Simple: brief reasoning
    (0.5, 3072),   # Moderate: standard CoT
    (0.7, 4096),   # Complex: full reasoning
    (0.85, 6144),  # Very complex: extended exploration
    (1.0, 8192),   # Maximum: deep multi-path reasoning
]


def estimate_thinking_budget(
    complexity_score: float,
    reasoning_mode: str = "cot",
    max_budget: int = 10000,
    previous_analysis: ThinkingAnalysis | None = None,
) -> ThinkingBudgetEstimate:
    """Estimate the optimal thinking budget based on complexity and history.

    Args:
        complexity_score: Router's complexity score (0-1).
        reasoning_mode: The reasoning mode being used.
        max_budget: Maximum allowed thinking budget.
        previous_analysis: Analysis from the previous turn (for adjustment).

    Returns:
        ThinkingBudgetEstimate with recommended budget.
    """
    # Base budget from complexity
    base_budget = _interpolate_budget(complexity_score)

    # Mode multiplier: MoT needs more budget for multi-path exploration
    mode_multipliers = {
        "direct": 0.0,  # No thinking needed
        "cot": 1.0,
        "mot": 1.5,     # Multi-path needs more room
        "dst": 1.2,     # Dynamic beam needs moderate extra
        "speculative": 0.5,  # Speculative is meant to be fast
    }
    multiplier = mode_multipliers.get(reasoning_mode, 1.0)

    # Adjust based on previous turn's thinking analysis
    if previous_analysis:
        multiplier *= previous_analysis.recommended_adjustment

    budget = int(base_budget * multiplier)
    budget = max(1024, min(budget, max_budget))

    return ThinkingBudgetEstimate(
        budget_tokens=budget,
        confidence=0.7 if previous_analysis else 0.5,
        rationale=(
            f"complexity={complexity_score:.2f} mode={reasoning_mode} "
            f"multiplier={multiplier:.1f} → budget={budget}"
        ),
    )


def _interpolate_budget(complexity: float) -> int:
    """Linear interpolation from the budget table."""
    complexity = max(0.0, min(1.0, complexity))

    for i in range(len(BUDGET_TABLE) - 1):
        low_c, low_b = BUDGET_TABLE[i]
        high_c, high_b = BUDGET_TABLE[i + 1]
        if low_c <= complexity <= high_c:
            t = (complexity - low_c) / (high_c - low_c) if high_c > low_c else 0
            return int(low_b + t * (high_b - low_b))

    return BUDGET_TABLE[-1][1]


# ── Post-call analysis ────────────────────────────────────────────────────


def analyze_thinking(thinking_text: str) -> ThinkingAnalysis:
    """Analyze model thinking output to detect overthinking.

    Checks for:
    - Repetitive reasoning (restating the same conclusion)
    - Early convergence (reaching conclusion in first half)
    - Clear vs ambiguous conclusions
    """
    if not thinking_text:
        return ThinkingAnalysis()

    words = thinking_text.split()
    total_words = len(words)

    # Detect repetition: check if sentences in the second half repeat first-half content
    midpoint = total_words // 2
    first_half = set(w.lower() for w in words[:midpoint] if len(w) > 3)
    second_half = set(w.lower() for w in words[midpoint:] if len(w) > 3)

    if first_half and second_half:
        overlap = len(first_half & second_half) / max(len(second_half), 1)
    else:
        overlap = 0.0

    # Detect convergence point: where conclusion markers first appear
    conclusion_patterns = [
        r"\b(therefore|thus|so|in conclusion|the best|i('ll| will) ask|my question)\b",
        r"\b(selected|choosing|best approach|most informative)\b",
    ]
    convergence_point = 1.0
    for pattern in conclusion_patterns:
        match = re.search(pattern, thinking_text, re.IGNORECASE)
        if match:
            pos_fraction = match.start() / max(len(thinking_text), 1)
            convergence_point = min(convergence_point, pos_fraction)

    # Clear conclusion check
    has_conclusion = bool(re.search(
        r"\b(therefore|my question|i('ll| will) ask|the best follow-?up)\b",
        thinking_text,
        re.IGNORECASE,
    ))

    # Compute recommended adjustment for next turn
    adjustment = 1.0
    if convergence_point < 0.4 and overlap > 0.6:
        # Converged early AND repeated → was overthinking, reduce budget
        adjustment = 0.7
    elif convergence_point < 0.5:
        # Converged in first half → slightly reduce
        adjustment = 0.85
    elif not has_conclusion and total_words > 200:
        # Long thinking without clear conclusion → might need more budget
        adjustment = 1.15

    return ThinkingAnalysis(
        total_thinking_words=total_words,
        repetition_score=overlap,
        convergence_point=convergence_point,
        had_clear_conclusion=has_conclusion,
        recommended_adjustment=adjustment,
    )
