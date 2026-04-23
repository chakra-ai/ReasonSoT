"""Async Anthropic SDK wrapper with streaming, prefix caching, and model routing."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from reason_sot.llm.cache import build_messages_with_cache, build_system_blocks
from reason_sot.llm.models import get_model
from reason_sot.types import (
    LatencyMetrics,
    ModelTier,
    StreamDone,
    StreamError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around the Anthropic API with streaming and caching."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]],
        model_tier: ModelTier = ModelTier.FAST,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a message from the Anthropic API.

        Yields StreamEvent objects: TextDelta, ThinkingDelta, StreamDone, or StreamError.
        """
        spec = get_model(model_tier)
        effective_max_tokens = max_tokens or spec.max_output_tokens

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": effective_max_tokens,
            "system": system,
            "messages": messages,
        }

        # Extended thinking for System 2
        if thinking_budget and spec.supports_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # When thinking is enabled, temperature must be 1
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        latency = LatencyMetrics(start_time=time.monotonic())
        usage = TokenUsage()
        first_token_seen = False

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    # Track first token time
                    if not first_token_seen and hasattr(event, "type"):
                        if event.type in (
                            "content_block_delta",
                        ):
                            first_token_seen = True
                            latency.first_token_time = time.monotonic()

                    # Yield appropriate event types
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield TextDelta(text=event.delta.text)
                        elif hasattr(event.delta, "thinking"):
                            yield ThinkingDelta(text=event.delta.thinking)

                # Get final message for usage stats
                final_message = await stream.get_final_message()
                if final_message and final_message.usage:
                    u = final_message.usage
                    usage.input_tokens = u.input_tokens
                    usage.output_tokens = u.output_tokens
                    usage.cache_creation_input_tokens = getattr(
                        u, "cache_creation_input_tokens", 0
                    ) or 0
                    usage.cache_read_input_tokens = getattr(
                        u, "cache_read_input_tokens", 0
                    ) or 0

            latency.end_time = time.monotonic()

            logger.debug(
                "LLM call complete: model=%s ttft=%.0fms total=%.0fms "
                "in_tokens=%d out_tokens=%d cache_hit=%.1f%%",
                spec.model_id,
                latency.ttft_ms or 0,
                latency.total_ms or 0,
                usage.total_input,
                usage.output_tokens,
                usage.cache_hit_rate * 100,
            )

            yield StreamDone(
                usage=usage,
                latency=latency,
                stop_reason="end_turn",
            )

        except anthropic.APIError as e:
            latency.end_time = time.monotonic()
            logger.error("Anthropic API error: %s", e)
            yield StreamError(error=str(e))

        except Exception as e:
            latency.end_time = time.monotonic()
            logger.error("Unexpected error in LLM call: %s", e)
            yield StreamError(error=str(e))

    async def complete_message(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]],
        model_tier: ModelTier = ModelTier.FAST,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, TokenUsage, LatencyMetrics]:
        """Non-streaming convenience method. Returns (text, usage, latency)."""
        chunks: list[str] = []
        usage = TokenUsage()
        latency = LatencyMetrics()

        async for event in self.stream_message(
            messages=messages,
            system=system,
            model_tier=model_tier,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            temperature=temperature,
        ):
            if isinstance(event, TextDelta):
                chunks.append(event.text)
            elif isinstance(event, StreamDone):
                usage = event.usage
                latency = event.latency
            elif isinstance(event, StreamError):
                raise RuntimeError(f"LLM error: {event.error}")

        return "".join(chunks), usage, latency
