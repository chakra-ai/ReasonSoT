"""System 2 — Deep reasoning path using Sonnet with extended thinking.

Handles complex questions that require multi-step reasoning, system design
discussions, trade-off analysis, and deep probing. Selects reasoning strategy
(CoT, MoT, DST) based on the RoutingDecision.

Streams responses: thinking tokens are consumed internally, only the final
response text streams to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from reason_sot.core.mot import build_mot_prompt, parse_mot_response
from reason_sot.llm.client import LLMClient
from reason_sot.types import (
    ModelTier,
    ReasoningMode,
    RoutingDecision,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
)

logger = logging.getLogger(__name__)


# ── Prompt augmentation per reasoning mode ────────────────────────────────

COT_INSTRUCTION = """
Before responding, think step-by-step about:
1. What is the candidate actually saying? (comprehension)
2. What does this reveal about their understanding level? (assessment)
3. What's the most valuable next question to ask? (strategy)

Then give a natural, conversational response with ONE clear follow-up question.
"""

DST_INSTRUCTION = """
Before responding, consider multiple possible follow-up directions:
- If you're confident about the candidate's level: ask ONE targeted question (fast path).
- If you're uncertain: briefly consider 2-3 possible directions, pick the most informative one.
- If the answer was surprising: explore the unexpected angle.

Respond naturally with ONE question.
"""


def _augment_messages_for_mode(
    messages: list[dict[str, Any]],
    routing: RoutingDecision,
) -> list[dict[str, Any]]:
    """Add reasoning-mode-specific instructions to the latest user message.

    Injects as a system-like prefix within the user message content to guide
    the model's extended thinking without changing the system prompt (which
    would bust the prefix cache).
    """
    if not messages:
        return messages

    augmented = [m.copy() for m in messages]

    if routing.reasoning_mode == ReasoningMode.MOT:
        # MoT uses its own prompt structure via mot.py
        # Inject MoT exploration instruction
        mot_prompt = build_mot_prompt(
            context="the candidate's response and the interview context",
        )
        instruction = mot_prompt
    elif routing.reasoning_mode == ReasoningMode.DST:
        instruction = DST_INSTRUCTION
    elif routing.reasoning_mode == ReasoningMode.COT:
        instruction = COT_INSTRUCTION
    else:
        return augmented

    # Prepend instruction to the last user message
    last_idx = len(augmented) - 1
    for i in range(last_idx, -1, -1):
        if augmented[i].get("role") == "user":
            content = augmented[i].get("content", "")
            if isinstance(content, str):
                augmented[i]["content"] = f"{instruction}\n\n{content}"
            elif isinstance(content, list):
                augmented[i]["content"] = [
                    {"type": "text", "text": instruction}
                ] + content
            break

    return augmented


# ── Public API ─────────────────────────────────────────────────────────────


async def generate(
    client: LLMClient,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]],
    routing: RoutingDecision,
    max_tokens: int = 2048,
) -> AsyncIterator[StreamEvent]:
    """Generate a System 2 (deep) response via Sonnet with extended thinking.

    - Augments messages with reasoning-mode-specific instructions
    - Uses extended thinking with the budget from RoutingDecision
    - Streams only TextDelta events (thinking is consumed internally)
    - Logs thinking token usage for analysis
    """
    augmented = _augment_messages_for_mode(messages, routing)
    thinking_budget = routing.thinking_budget or 4096

    thinking_tokens_consumed = 0

    async for event in client.stream_message(
        messages=augmented,
        system=system,
        model_tier=ModelTier.DEEP,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        temperature=1.0,  # Required when thinking is enabled
    ):
        if isinstance(event, ThinkingDelta):
            # Consume thinking internally — don't stream to user
            thinking_tokens_consumed += len(event.text.split())
            continue

        # Pass through text deltas, done, and error events
        yield event

    logger.debug(
        "System 2 thinking consumed ~%d words (budget=%d tokens)",
        thinking_tokens_consumed,
        thinking_budget,
    )
