"""Tests for System 1 (fast path)."""

import pytest

from reason_sot.core import system1
from reason_sot.types import ModelTier, StreamDone, TextDelta


@pytest.mark.asyncio
async def test_generate_streams_text(mock_client):
    events = []
    async for event in system1.generate(
        client=mock_client,
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a test agent.",
    ):
        events.append(event)

    text_events = [e for e in events if isinstance(e, TextDelta)]
    done_events = [e for e in events if isinstance(e, StreamDone)]

    assert len(text_events) > 0
    assert len(done_events) == 1
    assert done_events[0].usage.output_tokens > 0


@pytest.mark.asyncio
async def test_generate_uses_fast_model(mock_client):
    async for _ in system1.generate(
        client=mock_client,
        messages=[{"role": "user", "content": "test"}],
        system="test",
    ):
        pass

    assert mock_client.call_count == 1
    assert mock_client.call_log[0]["model_tier"] == ModelTier.FAST
    assert mock_client.call_log[0]["thinking_budget"] is None


@pytest.mark.asyncio
async def test_generate_complete(mock_client):
    result = await system1.generate_complete(
        client=mock_client,
        messages=[{"role": "user", "content": "test"}],
        system="test",
    )
    assert isinstance(result, str)
    assert len(result) > 0
