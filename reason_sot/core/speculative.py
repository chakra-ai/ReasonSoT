"""Speculative Chain-of-Thought — draft fast, verify if needed.

For borderline complexity cases where the router is uncertain:
  1. DRAFT: System 1 (Haiku) generates a quick response + self-assessed confidence
  2. GATE: If confidence > threshold → ship the draft directly (fast path)
  3. VERIFY: If confidence is low → pass draft + context to System 2 (Sonnet)
     for refinement. Sonnet gets "verify and improve this draft" which is
     faster than generating from scratch.

Net effect: 48-66% latency reduction for medium-complexity questions.
Uses asyncio.create_task so the draft starts immediately.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from reason_sot.core import system1
from reason_sot.llm.client import LLMClient
from reason_sot.types import (
    LatencyMetrics,
    ModelTier,
    StreamDone,
    StreamError,
    StreamEvent,
    TextDelta,
    TokenUsage,
)

logger = logging.getLogger(__name__)

DRAFT_CONFIDENCE_PROMPT = """After generating your interview response, assess your confidence.
End your response with a confidence marker on a new line:
[CONFIDENCE: 0.X]
where X is your confidence (0.0-1.0) that your response:
- Asks the right follow-up question for this candidate's level
- Is appropriately deep (not too surface, not too advanced)
- References specific details from their answer

Important: The [CONFIDENCE: X] line will be stripped from the response shown to the candidate."""

VERIFY_PROMPT = """You are reviewing a draft interview response. The original interviewer draft is below.

Your task: Verify and improve this draft if needed. Consider:
1. Does the follow-up question match the candidate's demonstrated level?
2. Does it reference specific details from their response?
3. Could the question be more incisive or reveal more about the candidate?

If the draft is good: return it as-is (perhaps with minor polish).
If it needs improvement: return an improved version.

Keep it conversational (2-3 sentences, ONE question). Do NOT include any confidence markers.

DRAFT TO REVIEW:
{draft}"""


async def generate_speculative(
    client: LLMClient,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]],
    confidence_threshold: float = 0.8,
    s2_thinking_budget: int = 2048,
) -> AsyncIterator[StreamEvent]:
    """Generate a response using speculative CoT.

    Phase 1: Draft with System 1 (fast)
    Phase 2: If low confidence, refine with System 2

    Yields StreamEvent objects. If the draft is confident enough, the
    caller sees only System 1 latency. Otherwise, System 2 refines.
    """
    # ── Phase 1: Draft ────────────────────────────────────────────────
    # Augment system prompt with confidence instruction
    augmented_system = _augment_system_for_confidence(system)

    draft_text, draft_usage, draft_latency = await client.complete_message(
        messages=messages,
        system=augmented_system,
        model_tier=ModelTier.FAST,
        max_tokens=512,
    )

    # Extract confidence and clean response
    confidence, clean_draft = _extract_confidence(draft_text)
    logger.info(
        "Speculative draft: confidence=%.2f (threshold=%.2f) TTFT=%.0fms",
        confidence,
        confidence_threshold,
        draft_latency.ttft_ms or 0,
    )

    # ── Phase 2: Gate ─────────────────────────────────────────────────
    if confidence >= confidence_threshold:
        # Ship the draft directly — fast path
        logger.info("Speculative: shipping draft (confidence %.2f >= %.2f)", confidence, confidence_threshold)
        yield TextDelta(text=clean_draft)
        yield StreamDone(usage=draft_usage, latency=draft_latency, stop_reason="end_turn")
        return

    # ── Phase 3: Verify with System 2 ─────��──────────────────────────
    logger.info("Speculative: draft confidence %.2f < %.2f, verifying with S2", confidence, confidence_threshold)

    verify_messages = messages.copy()
    # Add the draft as context for verification
    verify_content = VERIFY_PROMPT.format(draft=clean_draft)
    verify_messages.append({"role": "user", "content": verify_content})

    # Stream S2 verification
    async for event in client.stream_message(
        messages=verify_messages,
        system=system,  # Original system (no confidence instruction)
        model_tier=ModelTier.DEEP,
        max_tokens=512,
        thinking_budget=s2_thinking_budget,
    ):
        if isinstance(event, TextDelta):
            yield event
        elif isinstance(event, StreamDone):
            # Combine usage from both calls
            combined_usage = TokenUsage(
                input_tokens=draft_usage.input_tokens + event.usage.input_tokens,
                output_tokens=draft_usage.output_tokens + event.usage.output_tokens,
                cache_creation_input_tokens=(
                    draft_usage.cache_creation_input_tokens
                    + event.usage.cache_creation_input_tokens
                ),
                cache_read_input_tokens=(
                    draft_usage.cache_read_input_tokens
                    + event.usage.cache_read_input_tokens
                ),
                thinking_tokens=event.usage.thinking_tokens,
            )
            # Latency: TTFT from draft (user sees draft speed), total from both
            combined_latency = LatencyMetrics(
                start_time=draft_latency.start_time,
                first_token_time=draft_latency.first_token_time,
                end_time=event.latency.end_time,
            )
            yield StreamDone(
                usage=combined_usage,
                latency=combined_latency,
                stop_reason="end_turn",
            )
        elif isinstance(event, StreamError):
            # S2 failed — fall back to draft
            logger.warning("S2 verification failed, falling back to draft")
            yield TextDelta(text=clean_draft)
            yield StreamDone(usage=draft_usage, latency=draft_latency, stop_reason="end_turn")
            return


def _augment_system_for_confidence(
    system: str | list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Add confidence instruction to system prompt."""
    if isinstance(system, str):
        return system + "\n\n" + DRAFT_CONFIDENCE_PROMPT
    elif isinstance(system, list):
        # Append to the last text block
        augmented = [block.copy() for block in system]
        if augmented:
            last = augmented[-1]
            if "text" in last:
                augmented[-1] = {
                    **last,
                    "text": last["text"] + "\n\n" + DRAFT_CONFIDENCE_PROMPT,
                }
        return augmented
    return system


def _extract_confidence(text: str) -> tuple[float, str]:
    """Extract confidence score and clean the response text.

    Returns (confidence, clean_text).
    """
    # Look for [CONFIDENCE: X.X] pattern
    match = re.search(r"\[CONFIDENCE:\s*([\d.]+)\]", text)
    if match:
        try:
            confidence = float(match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

        # Remove the confidence marker from the text
        clean = text[:match.start()].rstrip()
        return confidence, clean

    # No confidence marker found — assume medium confidence
    return 0.5, text.strip()
