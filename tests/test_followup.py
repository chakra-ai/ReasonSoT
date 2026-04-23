"""Tests for adaptive follow-up classifier."""

import pytest

from reason_sot.interview.followup import (
    _heuristic_classify,
    _parse_classification,
    classify_followup,
)
from reason_sot.types import FollowUpAction


class TestHeuristicClassify:
    def test_short_response_clarify(self):
        result = _heuristic_classify("I'm not sure.", followup_chain_depth=0)
        assert result.action == FollowUpAction.CLARIFY

    def test_deep_response_next_topic(self):
        result = _heuristic_classify(
            "For example in my experience building distributed systems, "
            "the trade-off is between consistency and availability. "
            "The reason we chose eventual consistency was because our "
            "read workload was 100x our write workload. Specifically, "
            "we used DynamoDB with read replicas.",
            followup_chain_depth=0,
        )
        assert result.action == FollowUpAction.NEXT_TOPIC

    def test_vague_response_clarify(self):
        result = _heuristic_classify(
            "I think maybe it depends on the situation and what you're looking for.",
            followup_chain_depth=0,
        )
        assert result.action == FollowUpAction.CLARIFY

    def test_long_response_explore(self):
        long_response = " ".join(["I built a system that " + str(i) for i in range(30)])
        result = _heuristic_classify(long_response, followup_chain_depth=0)
        assert result.action == FollowUpAction.EXPLORE

    def test_long_response_after_chain_next_topic(self):
        long_response = " ".join(["I built a system that " + str(i) for i in range(30)])
        result = _heuristic_classify(long_response, followup_chain_depth=2)
        assert result.action == FollowUpAction.NEXT_TOPIC


class TestParseClassification:
    def test_parse_valid_json(self):
        text = '{"action": "clarify", "reason": "vague", "suggested_question": "Can you elaborate?", "topic": "Python"}'
        result = _parse_classification(text)
        assert result.action == FollowUpAction.CLARIFY
        assert result.reason == "vague"
        assert result.topic == "Python"

    def test_parse_json_in_markdown(self):
        text = '```json\n{"action": "next_topic", "reason": "good answer"}\n```'
        result = _parse_classification(text)
        assert result.action == FollowUpAction.NEXT_TOPIC

    def test_parse_explore(self):
        text = '{"action": "explore", "reason": "interesting tangent", "topic": "DevOps"}'
        result = _parse_classification(text)
        assert result.action == FollowUpAction.EXPLORE
        assert result.topic == "DevOps"

    def test_parse_invalid_action_defaults_to_clarify(self):
        text = '{"action": "unknown", "reason": "test"}'
        result = _parse_classification(text)
        assert result.action == FollowUpAction.CLARIFY


@pytest.mark.asyncio
async def test_classify_max_chain_forces_next_topic(mock_client):
    result = await classify_followup(
        client=mock_client,
        agent_question="Tell me about testing.",
        candidate_response="I use pytest.",
        followup_chain_depth=3,
        max_chain=3,
    )
    assert result.action == FollowUpAction.NEXT_TOPIC


@pytest.mark.asyncio
async def test_classify_falls_back_to_heuristic(mock_client_with_responses):
    """If LLM returns invalid JSON, heuristic kicks in."""
    client = mock_client_with_responses(["This is not valid JSON at all!"])
    result = await classify_followup(
        client=client,
        agent_question="Tell me more.",
        candidate_response="I'm not sure what to say.",
        followup_chain_depth=0,
    )
    # Should still return a valid FollowUpDecision via heuristic
    assert result.action in (FollowUpAction.CLARIFY, FollowUpAction.NEXT_TOPIC, FollowUpAction.EXPLORE)
