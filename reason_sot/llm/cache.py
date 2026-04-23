"""Prefix cache manager for maximizing Anthropic API cache hits.

Anthropic supports up to 4 cache breakpoints per request. We place them
strategically on: (1) system prompt, (2) persona definition,
(3) conversation history, (4) knowledge graph summary.
"""

from __future__ import annotations

from typing import Any


def _add_cache_control(block: dict[str, Any]) -> dict[str, Any]:
    """Add ephemeral cache_control to a content block."""
    return {**block, "cache_control": {"type": "ephemeral"}}


def build_system_blocks(
    base_system: str,
    persona_prompt: str,
    kg_summary: str | None = None,
) -> list[dict[str, Any]]:
    """Build system prompt blocks with cache breakpoints.

    Returns a list of content blocks for the system parameter,
    with cache_control on each segment boundary.
    """
    blocks: list[dict[str, Any]] = []

    # Breakpoint 1: base system instructions (rarely changes)
    blocks.append(
        _add_cache_control({"type": "text", "text": base_system})
    )

    # Breakpoint 2: persona definition (changes on persona switch)
    blocks.append(
        _add_cache_control({"type": "text", "text": persona_prompt})
    )

    # Breakpoint 3: knowledge graph summary (changes each turn)
    if kg_summary:
        blocks.append(
            _add_cache_control({"type": "text", "text": kg_summary})
        )

    return blocks


def build_messages_with_cache(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add cache breakpoint to the last user message in conversation history.

    This ensures the full conversation prefix is cached, and only the
    newest turn requires fresh computation.
    """
    if not conversation:
        return []

    messages = [msg.copy() for msg in conversation]

    # Find the last user message and mark it for caching
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content", "")
            if isinstance(content, str):
                messages[i]["content"] = [
                    _add_cache_control({"type": "text", "text": content})
                ]
            elif isinstance(content, list) and content:
                # Mark the last block in the content list
                last_block = content[-1].copy()
                last_block["cache_control"] = {"type": "ephemeral"}
                messages[i]["content"] = content[:-1] + [last_block]
            break

    return messages
