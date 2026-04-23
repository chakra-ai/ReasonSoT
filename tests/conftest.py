"""Shared test fixtures — MockLLMClient, sample sessions, persona fixtures."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from reason_sot.llm.client import LLMClient
from reason_sot.persona.profiles import (
    DecisionPattern,
    PersonaProfile,
    TopicArea,
)
from reason_sot.types import (
    FollowUpAction,
    FollowUpDecision,
    InterviewSession,
    InterviewTurn,
    LatencyMetrics,
    ModelTier,
    RoutingDecision,
    StreamDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
)


# ── Mock LLM Client ─────────────────────────────────────────────────────


class MockLLMClient:
    """Drop-in replacement for LLMClient that returns canned responses.

    Usage:
        client = MockLLMClient(responses=["Hello!", "Tell me more."])
        # Each call to stream_message/complete_message pops the next response
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = "That's a great point. Can you tell me more about your experience with that?",
        thinking_text: str = "",
        latency_ms: float = 100.0,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._default_response = default_response
        self._thinking_text = thinking_text
        self._latency_ms = latency_ms
        self._call_count = 0
        self._call_log: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return self._call_log

    def _next_response(self) -> str:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return self._default_response

    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]],
        model_tier: ModelTier = ModelTier.FAST,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]:
        self._call_log.append({
            "method": "stream_message",
            "model_tier": model_tier,
            "thinking_budget": thinking_budget,
            "messages_count": len(messages),
        })

        start = time.monotonic()
        response = self._next_response()

        # Yield thinking if S2
        if thinking_budget and self._thinking_text:
            yield ThinkingDelta(text=self._thinking_text)

        # Yield text in chunks
        words = response.split()
        for i, word in enumerate(words):
            prefix = " " if i > 0 else ""
            yield TextDelta(text=prefix + word)

        end = time.monotonic()
        yield StreamDone(
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=len(words) * 2,
                cache_read_input_tokens=50,
            ),
            latency=LatencyMetrics(
                start_time=start,
                first_token_time=start + (self._latency_ms / 1000),
                end_time=end,
            ),
        )

    async def complete_message(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]],
        model_tier: ModelTier = ModelTier.FAST,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, TokenUsage, LatencyMetrics]:
        self._call_log.append({
            "method": "complete_message",
            "model_tier": model_tier,
            "thinking_budget": thinking_budget,
        })

        response = self._next_response()
        start = time.monotonic()
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=len(response.split()) * 2,
            cache_read_input_tokens=50,
        )
        latency = LatencyMetrics(
            start_time=start,
            first_token_time=start + (self._latency_ms / 1000),
            end_time=start + (self._latency_ms * 2 / 1000),
        )
        return response, usage, latency


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def mock_client_with_responses():
    """Factory fixture for mock clients with custom responses."""
    def _factory(responses: list[str], **kwargs) -> MockLLMClient:
        return MockLLMClient(responses=responses, **kwargs)
    return _factory


@pytest.fixture
def sample_persona() -> PersonaProfile:
    """A minimal persona for testing."""
    return PersonaProfile(
        name="Test Interviewer",
        role="Technical Interviewer",
        domain="Backend Engineering",
        personality_traits=["Direct", "Technical"],
        communication_style="Conversational and direct, asking probing questions",
        opening_message="Welcome! Let's talk about your technical experience.",
        topic_coverage=[
            TopicArea(
                name="Python Fundamentals",
                description="Core language features, iterators, decorators, generators",
                priority=1,
                min_depth=2,
                example_questions=[
                    "How do generators work in Python?",
                    "Explain decorators and give an example.",
                ],
            ),
            TopicArea(
                name="System Design",
                description="Architecture, scalability, distributed systems",
                priority=1,
                min_depth=3,
                example_questions=[
                    "How would you design a URL shortener?",
                ],
            ),
            TopicArea(
                name="Testing",
                description="Unit testing, integration testing, test strategies",
                priority=2,
                min_depth=1,
                example_questions=[
                    "What's your testing approach?",
                ],
            ),
        ],
        decision_patterns=[
            DecisionPattern(
                situation="Candidate gives a vague answer",
                behavior="Ask for a specific example",
                rationale="Concrete examples reveal real understanding",
            ),
        ],
        probing_strategies=[
            "Can you give me a specific example?",
            "What would happen if that failed?",
            "How does that work under the hood?",
        ],
        depth_thresholds={"Python Fundamentals": 0.5, "System Design": 0.4},
    )


@pytest.fixture
def sample_session() -> InterviewSession:
    """A sample 5-turn interview session for scoring tests."""
    turns = []
    responses = [
        ("Hi, I'm a Python developer with 3 years experience.", "direct", 1),
        (
            "I use generators for lazy evaluation. For example, I wrote a generator to process large CSV files line by line.",
            "direct", 1,
        ),
        (
            "I think decorators wrap functions. Like @login_required in Flask.",
            "cot", 2,
        ),
        (
            "For system design I'd use microservices with message queues. The trade-off is complexity vs scalability. Specifically, I'd use Redis for caching and PostgreSQL for persistence.",
            "mot", 2,
        ),
        (
            "Testing — I use pytest with fixtures and mocking. In my experience the key trade-off is test speed vs coverage.",
            "direct", 1,
        ),
    ]

    for i, (user_input, mode, system) in enumerate(responses):
        follow_action = FollowUpAction.CLARIFY if i < 3 else FollowUpAction.NEXT_TOPIC
        t = InterviewTurn(
            turn_number=i + 1,
            user_input=user_input,
            agent_response=f"Good answer. Can you tell me more about the trade-offs and how that works under the hood?",
            routing=RoutingDecision(system=system, thinking_budget=4096 if system == 2 else 0),
            follow_up=FollowUpDecision(action=follow_action),
            latency=LatencyMetrics(
                start_time=0,
                first_token_time=0.3 if system == 1 else 0.8,
                end_time=1.0 if system == 1 else 2.0,
            ),
            usage=TokenUsage(
                input_tokens=500,
                output_tokens=100,
                cache_read_input_tokens=300,
            ),
        )
        turns.append(t)

    return InterviewSession(
        session_id="test-session-001",
        persona_name="Test Interviewer",
        turns=turns,
    )
