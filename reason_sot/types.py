"""Shared types and data models for ReasonSoT."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────


class ModelTier(str, Enum):
    """Which model tier to use."""

    FAST = "fast"  # Haiku — System 1
    DEEP = "deep"  # Sonnet — System 2


class ReasoningMode(str, Enum):
    """How the reasoning engine should think."""

    DIRECT = "direct"  # No chain-of-thought, immediate response
    COT = "cot"  # Standard chain-of-thought
    MOT = "mot"  # Matrix of Thought (breadth + depth)
    DST = "dst"  # Domain-Specialized Tree (dynamic beam)
    SPECULATIVE = "speculative"  # Draft (S1) then verify (S2)


class FollowUpAction(str, Enum):
    """What to do after receiving a user response."""

    NEXT_TOPIC = "next_topic"  # Answer sufficient, move on
    CLARIFY = "clarify"  # Answer vague, probe deeper
    EXPLORE = "explore"  # Interesting tangent, explore breadth


class InterviewPhase(str, Enum):
    """Current phase of the interview."""

    OPENING = "opening"  # Rapport building, introductions
    CORE = "core"  # Main interview questions
    DEEP_DIVE = "deep_dive"  # Deep probing on specific topics
    CLOSING = "closing"  # Wrap up, final questions


# ── Routing ────────────────────────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """Output of the heuristic router."""

    system: int = Field(description="1 for System 1 (fast), 2 for System 2 (deep)")
    reasoning_mode: ReasoningMode = ReasoningMode.DIRECT
    thinking_budget: int = Field(default=0, description="Thinking tokens for S2")
    beam_width: int = Field(default=1, description="Beam width for DST mode")
    confidence: float = Field(
        default=1.0, description="Router's confidence in this decision (0-1)"
    )
    rationale: str = Field(default="", description="Why this routing was chosen")


# ── LLM Response Tracking ──────────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token usage from a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens

    @property
    def cache_hit_rate(self) -> float:
        total = self.input_tokens + self.cache_read_input_tokens
        if total == 0:
            return 0.0
        return self.cache_read_input_tokens / total


class LatencyMetrics(BaseModel):
    """Timing metrics for a single LLM call."""

    start_time: float = Field(default_factory=time.monotonic)
    first_token_time: float | None = None
    end_time: float | None = None

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    @property
    def total_ms(self) -> float | None:
        """Total call duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


# ── Stream Events ──────────────────────────────────────────────────────────


class TextDelta(BaseModel):
    """A chunk of response text."""

    text: str
    event_type: str = "text_delta"


class ThinkingDelta(BaseModel):
    """A chunk of thinking/reasoning text (internal, not shown to user)."""

    text: str
    event_type: str = "thinking_delta"


class StreamDone(BaseModel):
    """Stream completed."""

    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    stop_reason: str = "end_turn"
    event_type: str = "done"


class StreamError(BaseModel):
    """Stream error."""

    error: str
    event_type: str = "error"


StreamEvent = TextDelta | ThinkingDelta | StreamDone | StreamError


# ── Follow-Up ──────────────────────────────────────────────────────────────


class FollowUpDecision(BaseModel):
    """Output of the follow-up classifier."""

    action: FollowUpAction
    reason: str = ""
    suggested_question: str = ""
    topic: str = ""


# ── Interview Turn ─────────────────────────────────────────────────────────


class InterviewTurn(BaseModel):
    """A single turn in the interview conversation."""

    turn_number: int
    user_input: str
    agent_response: str
    routing: RoutingDecision
    follow_up: FollowUpDecision | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    phase: InterviewPhase = InterviewPhase.CORE
    model_used: str = ""


class InterviewSession(BaseModel):
    """A complete interview session."""

    session_id: str = ""
    persona_name: str = ""
    turns: list[InterviewTurn] = Field(default_factory=list)
    knowledge_graph_snapshot: dict[str, Any] = Field(default_factory=dict)
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def avg_ttft_ms(self) -> float | None:
        ttfts = [t.latency.ttft_ms for t in self.turns if t.latency.ttft_ms is not None]
        if not ttfts:
            return None
        return sum(ttfts) / len(ttfts)


# ── Knowledge Graph Types ──────────────────────────────────────────────────


class KGNode(BaseModel):
    """A node in the knowledge graph."""

    id: str
    label: str
    node_type: str  # "topic", "entity", "skill", "experience"
    properties: dict[str, Any] = Field(default_factory=dict)
    turn_discovered: int = 0


class KGEdge(BaseModel):
    """An edge in the knowledge graph."""

    source: str
    target: str
    relation: str  # "related_to", "used_in", "led_to", "part_of"
    weight: float = 1.0


class KGCluster(BaseModel):
    """A thematic cluster of KG nodes."""

    cluster_id: str
    theme: str
    node_ids: list[str] = Field(default_factory=list)
    coverage_score: float = 0.0  # How well this theme has been explored (0-1)
