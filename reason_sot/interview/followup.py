"""Adaptive follow-up classifier — decides what to do after each candidate response.

Three possible actions (from AI Interviewer paper):
  1. NEXT_TOPIC: Answer is sufficient, move to the next topic
  2. CLARIFY: Answer is vague/partial, probe deeper on the same point
  3. EXPLORE: Answer reveals an interesting tangent worth exploring

Uses System 1 (Haiku) with structured output for fast classification (<300ms).
Falls back to heuristic classification if LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from reason_sot.llm.client import LLMClient
from reason_sot.types import FollowUpAction, FollowUpDecision, ModelTier

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are an interview follow-up classifier. Given the interviewer's question and the candidate's response, classify the response and suggest a next action.

RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation):
{
  "action": "next_topic" | "clarify" | "explore",
  "reason": "brief explanation (1 sentence)",
  "suggested_question": "the follow-up question if clarify/explore, empty string if next_topic",
  "topic": "the topic area being discussed"
}

CLASSIFICATION RULES:
- "next_topic": The response demonstrates clear understanding. The candidate gave specific examples, explained trade-offs, or showed depth. Move to a new topic.
- "clarify": The response is vague, surface-level, or incomplete. The candidate used buzzwords without explanation, gave a textbook answer without examples, or dodged the question. Probe deeper on the SAME point.
- "explore": The response mentions something unexpected or interesting that's worth pursuing. The candidate revealed expertise in an adjacent area, contradicted a common assumption, or hinted at deep experience. Follow the thread.

BIAS TOWARD ACTION:
- Default to "clarify" when uncertain — it's better to probe too much than too little.
- Only use "next_topic" when you're genuinely satisfied with the depth of the answer.
- Use "explore" sparingly — only when the tangent is clearly more valuable than continuing the current line."""


async def classify_followup(
    client: LLMClient,
    agent_question: str,
    candidate_response: str,
    conversation_summary: str = "",
    followup_chain_depth: int = 0,
    max_chain: int = 3,
) -> FollowUpDecision:
    """Classify the candidate's response and decide the next action.

    Uses System 1 (Haiku) for speed. Falls back to heuristics on failure.

    Args:
        client: LLM client for the classification call.
        agent_question: The question the interviewer just asked.
        candidate_response: The candidate's response.
        conversation_summary: Brief context of the interview so far.
        followup_chain_depth: How many consecutive follow-ups on current topic.
        max_chain: Maximum follow-ups before forced topic change.
    """
    # Force topic change if we've hit the follow-up chain limit
    if followup_chain_depth >= max_chain:
        return FollowUpDecision(
            action=FollowUpAction.NEXT_TOPIC,
            reason=f"Reached max follow-up depth ({max_chain})",
            topic="",
        )

    try:
        return await _llm_classify(
            client, agent_question, candidate_response, conversation_summary
        )
    except Exception as e:
        logger.warning("LLM follow-up classification failed: %s. Using heuristic.", e)
        return _heuristic_classify(candidate_response, followup_chain_depth)


async def _llm_classify(
    client: LLMClient,
    agent_question: str,
    candidate_response: str,
    conversation_summary: str,
) -> FollowUpDecision:
    """Classify using System 1 LLM call with structured output."""
    user_content = (
        f"INTERVIEWER'S QUESTION:\n{agent_question}\n\n"
        f"CANDIDATE'S RESPONSE:\n{candidate_response}"
    )
    if conversation_summary:
        user_content = f"CONTEXT:\n{conversation_summary}\n\n{user_content}"

    text, _, _ = await client.complete_message(
        messages=[{"role": "user", "content": user_content}],
        system=CLASSIFIER_PROMPT,
        model_tier=ModelTier.FAST,
        max_tokens=256,
    )

    return _parse_classification(text)


def _parse_classification(text: str) -> FollowUpDecision:
    """Parse the LLM's JSON classification response."""
    # Try to extract JSON from the response
    text = text.strip()

    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Handle bare JSON
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    data = json.loads(text)

    action_map = {
        "next_topic": FollowUpAction.NEXT_TOPIC,
        "clarify": FollowUpAction.CLARIFY,
        "explore": FollowUpAction.EXPLORE,
    }
    action = action_map.get(data.get("action", ""), FollowUpAction.CLARIFY)

    return FollowUpDecision(
        action=action,
        reason=data.get("reason", ""),
        suggested_question=data.get("suggested_question", ""),
        topic=data.get("topic", ""),
    )


# ── Heuristic fallback ────────────────────────────────────────────────────

# Vague/surface indicators
VAGUE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi think\b.*\bmaybe\b",
        r"\bsomething like\b",
        r"\bi'?m not (sure|certain)\b",
        r"\bgenerally\b.*\bit depends\b",
        r"\bi'?ve heard\b",
        r"^(yes|no|it depends|i think so)[.!]?$",
    ]
]

# Depth/expertise indicators
DEPTH_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bfor example\b",
        r"\bin my experience\b",
        r"\bthe trade-?off\s+(is|was|between)\b",
        r"\bspecifically\b",
        r"\bunder the hood\b",
        r"\bone time\s+(i|we)\b",
        r"\bthe reason\s+(is|was)\b",
    ]
]


def _heuristic_classify(
    candidate_response: str,
    followup_chain_depth: int,
) -> FollowUpDecision:
    """Heuristic fallback when LLM classification fails."""
    words = candidate_response.split()
    word_count = len(words)

    vague_hits = sum(1 for p in VAGUE_PATTERNS if p.search(candidate_response))
    depth_hits = sum(1 for p in DEPTH_PATTERNS if p.search(candidate_response))

    # Very short response → clarify
    if word_count < 10:
        return FollowUpDecision(
            action=FollowUpAction.CLARIFY,
            reason="Response too brief",
        )

    # Strong depth signals → next topic
    if depth_hits >= 2 and word_count > 30:
        return FollowUpDecision(
            action=FollowUpAction.NEXT_TOPIC,
            reason="Response shows depth with examples",
        )

    # Vague signals → clarify
    if vague_hits >= 1:
        return FollowUpDecision(
            action=FollowUpAction.CLARIFY,
            reason="Response appears vague or uncertain",
        )

    # Long response with some substance → explore or next
    if word_count > 80:
        return FollowUpDecision(
            action=FollowUpAction.EXPLORE if followup_chain_depth < 1 else FollowUpAction.NEXT_TOPIC,
            reason="Substantial response, worth exploring further" if followup_chain_depth < 1 else "Sufficient depth reached",
        )

    # Default: clarify
    return FollowUpDecision(
        action=FollowUpAction.CLARIFY,
        reason="Heuristic default — probe for more detail",
    )
