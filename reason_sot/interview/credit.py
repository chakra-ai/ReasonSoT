"""Turn-level credit assignment — scores each question's effectiveness post-interview.

Based on the Turn-Level Credit paper (2505.11821): instead of scoring the entire
interview as a unit, assigns credit to individual turns to identify which questions
were most productive. This creates training signal for future improvement.

Credit signals:
  1. Information gain: Did this question reveal new knowledge? (KG delta)
  2. Depth elicitation: Did the candidate go deeper after this question?
  3. Topic progression: Did this question advance the interview's coverage?
  4. Follow-up quality: Did this question's follow-up chain produce results?

Output: per-turn credit scores (0-1) used for:
  - Identifying the interviewer's best/worst questions
  - Training data for improving prompt engineering
  - Feedback to the persona's decision patterns
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reason_sot.types import (
    FollowUpAction,
    InterviewSession,
    InterviewTurn,
)


@dataclass
class TurnCredit:
    """Credit assignment for a single interview turn."""

    turn_number: int
    information_gain: float = 0.0  # New entities/topics revealed (0-1)
    depth_elicitation: float = 0.0  # Candidate went deeper (0-1)
    topic_progression: float = 0.0  # Advanced coverage (0-1)
    followup_quality: float = 0.0  # Follow-up chain was productive (0-1)
    overall_credit: float = 0.0  # Weighted combination
    rationale: str = ""


@dataclass
class CreditReport:
    """Full credit assignment for an interview session."""

    turn_credits: list[TurnCredit] = field(default_factory=list)
    top_turns: list[int] = field(default_factory=list)  # Highest-credit turn numbers
    bottom_turns: list[int] = field(default_factory=list)  # Lowest-credit turn numbers
    avg_credit: float = 0.0
    credit_variance: float = 0.0


def assign_credit(
    session: InterviewSession,
    kg_snapshots_per_turn: dict[int, dict[str, float]] | None = None,
    weights: dict[str, float] | None = None,
) -> CreditReport:
    """Assign credit to each turn in the interview session.

    Args:
        session: The completed interview session.
        kg_snapshots_per_turn: Optional per-turn KG coverage snapshots
            (turn_number -> {topic: coverage_score}). If not provided,
            information gain and topic progression are estimated heuristically.
        weights: Optional weights for combining sub-scores.

    Returns:
        CreditReport with per-turn scores and summary statistics.
    """
    if not session.turns:
        return CreditReport()

    w = weights or {
        "information_gain": 0.30,
        "depth_elicitation": 0.25,
        "topic_progression": 0.25,
        "followup_quality": 0.20,
    }

    credits: list[TurnCredit] = []

    for i, turn in enumerate(session.turns):
        tc = TurnCredit(turn_number=turn.turn_number)

        # 1. Information gain
        tc.information_gain = _score_information_gain(
            turn, i, session.turns, kg_snapshots_per_turn
        )

        # 2. Depth elicitation
        tc.depth_elicitation = _score_depth_elicitation(turn, i, session.turns)

        # 3. Topic progression
        tc.topic_progression = _score_topic_progression(
            turn, i, session.turns, kg_snapshots_per_turn
        )

        # 4. Follow-up quality
        tc.followup_quality = _score_followup_quality(turn, i, session.turns)

        # Weighted overall
        tc.overall_credit = (
            tc.information_gain * w["information_gain"]
            + tc.depth_elicitation * w["depth_elicitation"]
            + tc.topic_progression * w["topic_progression"]
            + tc.followup_quality * w["followup_quality"]
        )

        tc.rationale = _build_rationale(tc)
        credits.append(tc)

    # Build report
    report = CreditReport(turn_credits=credits)

    if credits:
        scores = [tc.overall_credit for tc in credits]
        report.avg_credit = sum(scores) / len(scores)
        report.credit_variance = (
            sum((s - report.avg_credit) ** 2 for s in scores) / len(scores)
        )

        # Top and bottom turns
        sorted_credits = sorted(credits, key=lambda tc: tc.overall_credit, reverse=True)
        report.top_turns = [tc.turn_number for tc in sorted_credits[:3]]
        report.bottom_turns = [tc.turn_number for tc in sorted_credits[-3:]]

    return report


# ── Sub-score functions ──────────────────────────────────────────────────


def _score_information_gain(
    turn: InterviewTurn,
    index: int,
    all_turns: list[InterviewTurn],
    kg_snapshots: dict[int, dict[str, float]] | None,
) -> float:
    """Score how much new information this turn revealed.

    If KG snapshots are available, measure delta in coverage.
    Otherwise, estimate from response content novelty.
    """
    if kg_snapshots:
        current = kg_snapshots.get(turn.turn_number, {})
        previous = kg_snapshots.get(turn.turn_number - 1, {})
        if current and previous:
            # Sum of coverage increases across all topics
            delta = sum(
                max(0, current.get(t, 0) - previous.get(t, 0))
                for t in current
            )
            return min(delta / 0.5, 1.0)  # Normalize: 0.5 total delta = 1.0

    # Heuristic: count unique content words not seen in previous turns
    prev_words = set()
    for prev in all_turns[:index]:
        prev_words.update(
            w.lower() for w in prev.user_input.split() if len(w) > 4
        )

    current_words = {
        w.lower() for w in turn.user_input.split() if len(w) > 4
    }
    new_words = current_words - prev_words

    if not current_words:
        return 0.0
    return min(len(new_words) / max(len(current_words) * 0.5, 1), 1.0)


def _score_depth_elicitation(
    turn: InterviewTurn,
    index: int,
    all_turns: list[InterviewTurn],
) -> float:
    """Score whether this question made the candidate go deeper.

    Signals:
    - Candidate's response length (longer = more depth)
    - Presence of depth markers (examples, trade-offs, specifics)
    - Candidate used technical terms
    """
    response = turn.user_input  # The candidate's response to the agent's question

    # Response length score
    word_count = len(response.split())
    length_score = min(word_count / 60.0, 1.0)

    # Depth markers
    depth_patterns = [
        r"\bfor example\b", r"\bspecifically\b", r"\btrade-?off\b",
        r"\bin my experience\b", r"\bthe reason\b", r"\bunder the hood\b",
        r"\bone time\b", r"\bwe (decided|chose|went with)\b",
        r"\bthe challenge was\b", r"\bwhat worked was\b",
    ]
    marker_hits = sum(
        1 for p in depth_patterns if re.search(p, response, re.IGNORECASE)
    )
    marker_score = min(marker_hits / 3.0, 1.0)

    # Compare to previous turn's response depth
    improvement = 0.0
    if index > 0:
        prev_words = len(all_turns[index - 1].user_input.split())
        if prev_words > 0:
            ratio = word_count / prev_words
            improvement = min(max(ratio - 1.0, 0) / 2.0, 0.3)

    return length_score * 0.3 + marker_score * 0.5 + improvement + 0.2 * min(marker_hits > 0, 1)


def _score_topic_progression(
    turn: InterviewTurn,
    index: int,
    all_turns: list[InterviewTurn],
    kg_snapshots: dict[int, dict[str, float]] | None,
) -> float:
    """Score whether this turn advanced the interview's overall coverage."""
    # If we have KG snapshots, use total coverage delta
    if kg_snapshots:
        current = kg_snapshots.get(turn.turn_number, {})
        previous = kg_snapshots.get(turn.turn_number - 1, {})
        if current:
            curr_avg = sum(current.values()) / max(len(current), 1)
            prev_avg = sum(previous.values()) / max(len(previous), 1) if previous else 0
            delta = curr_avg - prev_avg
            return min(max(delta / 0.1, 0), 1.0)  # 0.1 avg delta = 1.0

    # Heuristic: topic transitions are valuable for coverage
    if turn.follow_up and turn.follow_up.action == FollowUpAction.NEXT_TOPIC:
        return 0.7  # Moving to a new topic = good coverage progress

    if turn.follow_up and turn.follow_up.action == FollowUpAction.EXPLORE:
        return 0.5  # Exploring tangent = moderate coverage

    # Clarify on same topic = less coverage progression
    return 0.2


def _score_followup_quality(
    turn: InterviewTurn,
    index: int,
    all_turns: list[InterviewTurn],
) -> float:
    """Score the quality of the follow-up chain this turn participated in.

    A good follow-up chain:
    - Starts from a clarify/explore action
    - Results in deeper candidate responses
    - Eventually reaches sufficient depth (next_topic)
    """
    if not turn.follow_up:
        return 0.5  # No follow-up data

    # Did this turn start a productive chain?
    if turn.follow_up.action == FollowUpAction.CLARIFY:
        # Look ahead: did the chain eventually reach resolution?
        chain_resolved = False
        chain_length = 0
        for future in all_turns[index + 1:]:
            chain_length += 1
            if future.follow_up and future.follow_up.action == FollowUpAction.NEXT_TOPIC:
                chain_resolved = True
                break
            if chain_length > 4:
                break

        if chain_resolved and chain_length <= 3:
            return 0.9  # Good: clarified and resolved within 3 turns
        elif chain_resolved:
            return 0.6  # OK: resolved but took many turns
        else:
            return 0.3  # Poor: clarify chain didn't resolve

    if turn.follow_up.action == FollowUpAction.EXPLORE:
        return 0.7  # Exploring is generally productive

    if turn.follow_up.action == FollowUpAction.NEXT_TOPIC:
        return 0.5  # Neutral: moved on

    return 0.5


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_rationale(tc: TurnCredit) -> str:
    """Build a human-readable rationale for the credit assignment."""
    parts = []
    if tc.information_gain > 0.6:
        parts.append("high information gain")
    elif tc.information_gain < 0.2:
        parts.append("low new information")

    if tc.depth_elicitation > 0.6:
        parts.append("elicited deep response")
    elif tc.depth_elicitation < 0.2:
        parts.append("shallow response")

    if tc.topic_progression > 0.6:
        parts.append("advanced coverage")

    if tc.followup_quality > 0.7:
        parts.append("productive follow-up")
    elif tc.followup_quality < 0.3:
        parts.append("unresolved follow-up")

    return "; ".join(parts) if parts else "average turn"


def format_credit_report(report: CreditReport) -> str:
    """Format a credit report as a human-readable string."""
    if not report.turn_credits:
        return "No turns to score."

    lines = [
        "=== Turn-Level Credit Report ===",
        "",
        f"{'Turn':>5} {'Info':>6} {'Depth':>6} {'Topic':>6} {'F/U':>6} {'Overall':>8}  Rationale",
        "-" * 70,
    ]

    for tc in report.turn_credits:
        lines.append(
            f"{tc.turn_number:>5} "
            f"{tc.information_gain:>6.2f} {tc.depth_elicitation:>6.2f} "
            f"{tc.topic_progression:>6.2f} {tc.followup_quality:>6.2f} "
            f"{tc.overall_credit:>8.3f}  {tc.rationale}"
        )

    lines.extend([
        "-" * 70,
        f"Average credit: {report.avg_credit:.3f}  "
        f"Variance: {report.credit_variance:.4f}",
        f"Top turns: {report.top_turns}",
        f"Bottom turns: {report.bottom_turns}",
    ])

    return "\n".join(lines)
