"""Reasoning quality metrics — automated scoring from interview transcripts.

Four core metrics, each scored 0-1:
  1. Reasoning Depth: How deep does the interview probe on each topic?
  2. Reasoning Breadth: What fraction of expected topics were covered?
  3. Persona Consistency: Does the agent stay in character?
  4. Follow-Up Relevance: Are follow-ups actually connected to what was said?

All metrics are computed post-hoc from InterviewSession transcripts.
No LLM calls during scoring (for speed), except the optional LLM-as-judge.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reason_sot.persona.profiles import PersonaProfile
from reason_sot.types import (
    FollowUpAction,
    InterviewSession,
    InterviewTurn,
)


@dataclass
class ReasoningScores:
    """Complete scoring results for an interview session."""

    depth_score: float = 0.0
    breadth_score: float = 0.0
    persona_consistency_score: float = 0.0
    followup_relevance_score: float = 0.0
    overall_score: float = 0.0

    # Breakdown details
    depth_details: dict[str, Any] = field(default_factory=dict)
    breadth_details: dict[str, Any] = field(default_factory=dict)
    persona_details: dict[str, Any] = field(default_factory=dict)
    followup_details: dict[str, Any] = field(default_factory=dict)

    def compute_overall(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted overall score."""
        w = weights or {
            "depth": 0.30,
            "breadth": 0.25,
            "persona": 0.20,
            "followup": 0.25,
        }
        self.overall_score = (
            self.depth_score * w["depth"]
            + self.breadth_score * w["breadth"]
            + self.persona_consistency_score * w["persona"]
            + self.followup_relevance_score * w["followup"]
        )
        return self.overall_score


def score_session(
    session: InterviewSession,
    persona: PersonaProfile,
) -> ReasoningScores:
    """Score a complete interview session across all metrics."""
    scores = ReasoningScores()

    scores.depth_score, scores.depth_details = _score_depth(session)
    scores.breadth_score, scores.breadth_details = _score_breadth(session, persona)
    scores.persona_consistency_score, scores.persona_details = _score_persona_consistency(
        session, persona
    )
    scores.followup_relevance_score, scores.followup_details = _score_followup_relevance(
        session
    )
    scores.compute_overall()

    return scores


# ── 1. Reasoning Depth ────────────────────────────────────────────────────


def _score_depth(session: InterviewSession) -> tuple[float, dict]:
    """Measure how deeply the interview probes on topics.

    Signals:
    - Follow-up chain length (consecutive clarify actions)
    - Presence of depth markers in agent responses (why, how, under the hood)
    - Progression from surface → causal → structural understanding
    """
    if not session.turns:
        return 0.0, {"reason": "no turns"}

    # Track follow-up chains
    chains: list[int] = []
    current_chain = 0
    for turn in session.turns:
        if turn.follow_up and turn.follow_up.action == FollowUpAction.CLARIFY:
            current_chain += 1
        else:
            if current_chain > 0:
                chains.append(current_chain)
            current_chain = 0
    if current_chain > 0:
        chains.append(current_chain)

    # Max chain depth (normalized to 0-1, where 3+ = 1.0)
    max_chain = max(chains) if chains else 0
    chain_score = min(max_chain / 3.0, 1.0)

    # Depth markers in agent responses
    depth_markers = [
        r"\bwhy\b", r"\bhow\b.*\bunder the hood\b", r"\btrade-?off",
        r"\bspecifically\b", r"\bexample\b", r"\bedge case",
        r"\bwhat happens (if|when)\b", r"\blimitation", r"\bscale\b",
    ]
    agent_texts = " ".join(t.agent_response.lower() for t in session.turns)
    marker_hits = sum(1 for p in depth_markers if re.search(p, agent_texts))
    marker_score = min(marker_hits / 5.0, 1.0)

    # S2 usage ratio (deeper reasoning = more S2 turns)
    s2_ratio = sum(1 for t in session.turns if t.routing.system == 2) / len(session.turns)

    depth = chain_score * 0.4 + marker_score * 0.35 + s2_ratio * 0.25

    return depth, {
        "max_chain_depth": max_chain,
        "chain_score": chain_score,
        "depth_marker_hits": marker_hits,
        "marker_score": marker_score,
        "s2_ratio": s2_ratio,
        "chains": chains,
    }


# ── 2. Reasoning Breadth ──────────────────────────────────────────────────


def _score_breadth(
    session: InterviewSession, persona: PersonaProfile
) -> tuple[float, dict]:
    """Measure topic coverage completeness.

    A topic counts as "explored" if at least 2 turns addressed it
    (not just a single mention).
    """
    expected_topics = persona.get_topic_names()
    if not expected_topics:
        return 1.0, {"reason": "no topics defined"}

    # Count topic mentions across all turns
    topic_mention_counts: Counter[str] = Counter()
    all_text = " ".join(
        f"{t.user_input} {t.agent_response}".lower() for t in session.turns
    )

    for topic_name in expected_topics:
        # Check for topic keywords in conversation
        keywords = topic_name.lower().split()
        for keyword in keywords:
            if len(keyword) > 3:  # Skip short words
                count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", all_text))
                topic_mention_counts[topic_name] += count

    # A topic is "explored" if mentioned meaningfully (2+ keyword hits)
    explored = [t for t in expected_topics if topic_mention_counts.get(t, 0) >= 2]
    breadth = len(explored) / len(expected_topics)

    # Bonus for must-cover topics
    must_cover = persona.get_must_cover_topics()
    must_cover_names = [t.name for t in must_cover]
    must_covered = [t for t in must_cover_names if t in explored]
    must_cover_ratio = len(must_covered) / max(len(must_cover_names), 1)

    # Weight must-cover topics more heavily
    score = breadth * 0.4 + must_cover_ratio * 0.6

    return score, {
        "expected_topics": expected_topics,
        "explored_topics": explored,
        "topic_mentions": dict(topic_mention_counts),
        "breadth_ratio": breadth,
        "must_cover_ratio": must_cover_ratio,
    }


# ── 3. Persona Consistency ────────────────────────────────────────────────


def _score_persona_consistency(
    session: InterviewSession, persona: PersonaProfile
) -> tuple[float, dict]:
    """Measure how consistently the agent adheres to its persona.

    Checks:
    - Communication style adherence (keywords from persona definition)
    - Use of probing strategies defined in persona
    - No out-of-character behavior (being rude, off-topic, etc.)
    """
    if not session.turns:
        return 0.0, {"reason": "no turns"}

    agent_responses = [t.agent_response.lower() for t in session.turns]
    all_agent_text = " ".join(agent_responses)

    # Check probing strategy usage
    strategy_hits = 0
    strategy_keywords = []
    for strategy in persona.probing_strategies:
        # Extract key phrases from each strategy
        key_words = [w for w in strategy.lower().split() if len(w) > 4]
        strategy_keywords.extend(key_words)
        for kw in key_words[:3]:
            if kw in all_agent_text:
                strategy_hits += 1
                break

    strategy_score = min(
        strategy_hits / max(len(persona.probing_strategies), 1), 1.0
    )

    # Check communication style consistency
    style_keywords = [
        w.lower()
        for w in persona.communication_style.split()
        if len(w) > 4
    ]
    style_hits = sum(1 for kw in style_keywords if kw in all_agent_text)
    style_score = min(style_hits / max(len(style_keywords) // 3, 1), 1.0)

    # Check for consistency across turns (variance in response length, tone)
    lengths = [len(r.split()) for r in agent_responses]
    if len(lengths) >= 3:
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        cv = (variance ** 0.5) / max(avg_len, 1)
        consistency_score = max(0, 1.0 - cv)  # Lower CV = more consistent
    else:
        consistency_score = 0.7  # Not enough data

    score = strategy_score * 0.35 + style_score * 0.30 + consistency_score * 0.35

    return score, {
        "strategy_hits": strategy_hits,
        "strategy_score": strategy_score,
        "style_score": style_score,
        "response_length_cv": cv if len(lengths) >= 3 else None,
        "consistency_score": consistency_score,
    }


# ── 4. Follow-Up Relevance ───────────────────────────────────────────────


def _score_followup_relevance(session: InterviewSession) -> tuple[float, dict]:
    """Measure how relevant each follow-up is to the candidate's response.

    Uses TF-IDF-like overlap: for each (candidate_response, agent_followup) pair,
    check if the follow-up references specific content from the response.
    """
    if len(session.turns) < 2:
        return 0.5, {"reason": "not enough turns"}

    relevance_scores: list[float] = []

    for i in range(1, len(session.turns)):
        prev_response = session.turns[i - 1].user_input.lower()
        curr_question = session.turns[i].agent_response.lower()

        # Extract meaningful words (>3 chars, not stop words)
        stop_words = {
            "that", "this", "with", "from", "have", "been", "were", "will",
            "would", "could", "should", "about", "their", "there", "these",
            "those", "what", "when", "where", "which", "your", "some",
            "them", "they", "than", "very", "also", "just", "more",
        }
        prev_words = {
            w.strip(".,;:!?()\"'")
            for w in prev_response.split()
            if len(w) > 3 and w not in stop_words
        }
        curr_words = {
            w.strip(".,;:!?()\"'")
            for w in curr_question.split()
            if len(w) > 3 and w not in stop_words
        }

        if not prev_words:
            relevance_scores.append(0.5)
            continue

        # Overlap: what fraction of prev response's key words appear in follow-up
        overlap = len(prev_words & curr_words)
        relevance = min(overlap / max(len(prev_words) * 0.2, 1), 1.0)
        relevance_scores.append(relevance)

    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5

    return avg_relevance, {
        "per_turn_relevance": relevance_scores,
        "avg_relevance": avg_relevance,
        "num_pairs_scored": len(relevance_scores),
    }
