"""System 1 — Fast path using Haiku for low-latency responses.

Handles simple questions, acknowledgments, topic transitions, and
serves as the draft generator for speculative CoT.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from reason_sot.llm.client import LLMClient
from reason_sot.types import ModelTier, StreamEvent

logger = logging.getLogger(__name__)


async def generate(
    client: LLMClient,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]],
    max_tokens: int = 512,
) -> AsyncIterator[StreamEvent]:
    """Generate a System 1 (fast) response via Haiku streaming.

    This is the primary fast path — no extended thinking, no complex
    reasoning structures. Optimized for sub-500ms TTFT.
    """
    async for event in client.stream_message(
        messages=messages,
        system=system,
        model_tier=ModelTier.FAST,
        max_tokens=max_tokens,
        thinking_budget=None,
        temperature=1.0,
    ):
        yield event


async def generate_complete(
    client: LLMClient,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]],
    max_tokens: int = 512,
) -> str:
    """Non-streaming System 1 call. Returns the full text response."""
    text, _, _ = await client.complete_message(
        messages=messages,
        system=system,
        model_tier=ModelTier.FAST,
        max_tokens=max_tokens,
    )
    return text
