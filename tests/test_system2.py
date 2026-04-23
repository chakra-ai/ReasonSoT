"""Tests for System 2 (deep reasoning path)."""

import pytest

from reason_sot.core.system2 import _augment_messages_for_mode, generate
from reason_sot.types import (
    ModelTier,
    ReasoningMode,
    RoutingDecision,
    StreamDone,
    TextDelta,
    ThinkingDelta,
)


class TestMessageAugmentation:
    def test_cot_adds_instruction(self):
        messages = [{"role": "user", "content": "original question"}]
        routing = RoutingDecision(system=2, reasoning_mode=ReasoningMode.COT)
        augmented = _augment_messages_for_mode(messages, routing)

        content = augmented[0]["content"]
        assert "step-by-step" in content
        assert "original question" in content

    def test_mot_adds_instruction(self):
        messages = [{"role": "user", "content": "tell me about design"}]
        routing = RoutingDecision(system=2, reasoning_mode=ReasoningMode.MOT)
        augmented = _augment_messages_for_mode(messages, routing)

        content = augmented[0]["content"]
        assert "Matrix of Thought" in content or "REASONING STRATEGY" in content

    def test_dst_adds_instruction(self):
        messages = [{"role": "user", "content": "test input"}]
        routing = RoutingDecision(system=2, reasoning_mode=ReasoningMode.DST)
        augmented = _augment_messages_for_mode(messages, routing)
        assert "beam" in augmented[0]["content"].lower() or "direction" in augmented[0]["content"].lower()

    def test_direct_mode_no_augmentation(self):
        messages = [{"role": "user", "content": "hello"}]
        routing = RoutingDecision(system=2, reasoning_mode=ReasoningMode.DIRECT)
        augmented = _augment_messages_for_mode(messages, routing)
        assert augmented[0]["content"] == "hello"

    def test_empty_messages(self):
        result = _augment_messages_for_mode([], RoutingDecision(system=2))
        assert result == []


@pytest.mark.asyncio
async def test_generate_streams_text_only(mock_client_with_responses):
    """System 2 should consume thinking and only yield text."""
    client = mock_client_with_responses(
        ["This is a deep analysis with follow-up question."],
        thinking_text="Let me think about this step by step...",
    )
    routing = RoutingDecision(
        system=2,
        reasoning_mode=ReasoningMode.COT,
        thinking_budget=4096,
    )

    events = []
    async for event in generate(
        client=client,
        messages=[{"role": "user", "content": "complex question"}],
        system="test",
        routing=routing,
    ):
        events.append(event)

    # Should have text events and done, but no thinking events
    text_events = [e for e in events if isinstance(e, TextDelta)]
    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    done_events = [e for e in events if isinstance(e, StreamDone)]

    assert len(text_events) > 0
    assert len(thinking_events) == 0  # Consumed internally
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_generate_uses_deep_model(mock_client):
    routing = RoutingDecision(
        system=2, reasoning_mode=ReasoningMode.COT, thinking_budget=4096,
    )
    async for _ in generate(
        client=mock_client,
        messages=[{"role": "user", "content": "test"}],
        system="test",
        routing=routing,
    ):
        pass

    assert mock_client.call_log[0]["model_tier"] == ModelTier.DEEP
    assert mock_client.call_log[0]["thinking_budget"] == 4096
