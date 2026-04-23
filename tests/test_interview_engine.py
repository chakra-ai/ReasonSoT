"""Tests for the interview engine orchestrator."""

import pytest

from reason_sot.interview.engine import InterviewEngine
from reason_sot.types import (
    InterviewPhase,
    StreamDone,
    StreamError,
    TextDelta,
)


@pytest.fixture
def engine(mock_client_with_responses, sample_persona):
    """Create an engine with mock client that returns follow-up classifier JSON."""
    responses = [
        # Responses for S1 processing + follow-up classification
        "That's interesting. Can you tell me more about your experience?",
        '{"action": "clarify", "reason": "vague", "suggested_question": "", "topic": "Python"}',
    ] * 10  # Enough for multiple turns
    client = mock_client_with_responses(responses)
    return InterviewEngine(
        client=client,
        persona=sample_persona,
        enable_kg=True,
        enable_speculative=False,
    )


@pytest.fixture
def simple_engine(mock_client, sample_persona):
    """Engine with default mock responses (no KG, no speculative)."""
    return InterviewEngine(
        client=mock_client,
        persona=sample_persona,
        enable_kg=False,
        enable_speculative=False,
    )


class TestEngineInit:
    def test_initial_state(self, simple_engine):
        assert simple_engine.turn_count == 0
        assert simple_engine.current_phase == InterviewPhase.OPENING
        assert simple_engine.session.total_turns == 0

    def test_opening_message(self, simple_engine):
        msg = simple_engine.get_opening_message()
        assert len(msg) > 0


class TestProcessTurn:
    @pytest.mark.asyncio
    async def test_single_turn(self, simple_engine):
        events = []
        async for event in simple_engine.process_turn("Hi, I'm ready."):
            events.append(event)

        text_events = [e for e in events if isinstance(e, TextDelta)]
        done_events = [e for e in events if isinstance(e, StreamDone)]

        assert len(text_events) > 0
        assert len(done_events) == 1
        assert simple_engine.turn_count == 1
        assert len(simple_engine.session.turns) == 1

    @pytest.mark.asyncio
    async def test_multiple_turns(self, simple_engine):
        for text in ["Hello", "I use Python", "Generators yield values"]:
            async for _ in simple_engine.process_turn(text):
                pass

        assert simple_engine.turn_count == 3
        assert len(simple_engine.session.turns) == 3

    @pytest.mark.asyncio
    async def test_turn_records_routing(self, simple_engine):
        async for _ in simple_engine.process_turn("How would you design a distributed system?"):
            pass

        turn = simple_engine.session.turns[0]
        assert turn.routing is not None
        assert turn.routing.system in (1, 2)

    @pytest.mark.asyncio
    async def test_turn_records_response(self, simple_engine):
        async for _ in simple_engine.process_turn("I use pytest for testing."):
            pass

        turn = simple_engine.session.turns[0]
        assert len(turn.agent_response) > 0
        assert turn.user_input == "I use pytest for testing."


class TestPhaseProgression:
    @pytest.mark.asyncio
    async def test_opening_phase(self, simple_engine):
        async for _ in simple_engine.process_turn("Hi there"):
            pass
        assert simple_engine.current_phase == InterviewPhase.OPENING

    @pytest.mark.asyncio
    async def test_core_phase_after_opening(self, simple_engine):
        for text in ["Hi", "Ready", "I use Python extensively"]:
            async for _ in simple_engine.process_turn(text):
                pass
        assert simple_engine.current_phase == InterviewPhase.CORE


class TestKnowledgeGraphIntegration:
    @pytest.mark.asyncio
    async def test_kg_updates_on_turn(self, engine):
        async for _ in engine.process_turn("I use Python generators and decorators a lot."):
            pass

        kg = engine.knowledge_graph
        assert kg is not None
        assert len(kg._nodes) > 0

    @pytest.mark.asyncio
    async def test_kg_coverage_changes(self, engine):
        kg = engine.knowledge_graph
        gap_before = kg.get_coverage_gap_score()

        async for _ in engine.process_turn("I use Python generators and decorators extensively."):
            pass

        gap_after = kg.get_coverage_gap_score()
        assert gap_after <= gap_before


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_end_session(self, simple_engine):
        async for _ in simple_engine.process_turn("Hello"):
            pass

        session = await simple_engine.end_session()
        assert session.ended_at is not None
        assert session.total_turns == 1

    @pytest.mark.asyncio
    async def test_session_stats(self, simple_engine):
        async for _ in simple_engine.process_turn("Hello"):
            pass
        async for _ in simple_engine.process_turn("I use Python"):
            pass

        stats = simple_engine.get_session_stats()
        assert stats["total_turns"] == 2
        assert "reasoning_modes_used" in stats
