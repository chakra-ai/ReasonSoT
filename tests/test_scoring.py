"""Tests for scoring metrics, latency profiling, and credit assignment."""

import pytest

from reason_sot.interview.credit import (
    CreditReport,
    TurnCredit,
    assign_credit,
    format_credit_report,
)
from reason_sot.scoring.latency import (
    LatencyReport,
    _percentile,
    format_report,
    profile_session,
)
from reason_sot.scoring.metrics import (
    ReasoningScores,
    _score_breadth,
    _score_depth,
    _score_followup_relevance,
    _score_persona_consistency,
    score_session,
)
from reason_sot.types import InterviewSession


class TestReasoningScores:
    def test_compute_overall(self):
        scores = ReasoningScores(
            depth_score=0.8,
            breadth_score=0.6,
            persona_consistency_score=0.7,
            followup_relevance_score=0.5,
        )
        overall = scores.compute_overall()
        assert 0 < overall < 1
        # Weighted: 0.8*0.3 + 0.6*0.25 + 0.7*0.2 + 0.5*0.25 = 0.24+0.15+0.14+0.125 = 0.655
        assert abs(overall - 0.655) < 0.01


class TestScoreSession:
    def test_score_session_returns_all_metrics(self, sample_session, sample_persona):
        scores = score_session(sample_session, sample_persona)
        assert 0 <= scores.depth_score <= 1
        assert 0 <= scores.breadth_score <= 1
        assert 0 <= scores.persona_consistency_score <= 1
        assert 0 <= scores.followup_relevance_score <= 1
        assert 0 < scores.overall_score <= 1

    def test_score_empty_session(self, sample_persona):
        empty = InterviewSession()
        scores = score_session(empty, sample_persona)
        assert scores.depth_score == 0.0

    def test_depth_detects_followup_chains(self, sample_session):
        score, details = _score_depth(sample_session)
        assert "chains" in details
        assert details["max_chain_depth"] > 0

    def test_breadth_checks_topic_coverage(self, sample_session, sample_persona):
        score, details = _score_breadth(sample_session, sample_persona)
        assert "expected_topics" in details
        assert len(details["expected_topics"]) > 0

    def test_persona_consistency(self, sample_session, sample_persona):
        score, details = _score_persona_consistency(sample_session, sample_persona)
        assert "strategy_score" in details
        assert "style_score" in details


class TestLatencyProfiler:
    def test_profile_empty_session(self):
        report = profile_session(InterviewSession())
        assert report.ttft_p50_ms is None

    def test_profile_session(self, sample_session):
        report = profile_session(sample_session)
        assert report.ttft_p50_ms is not None
        assert report.s1_turns + report.s2_turns == len(sample_session.turns)
        assert report.avg_cache_hit_rate > 0

    def test_target_compliance(self, sample_session):
        report = profile_session(sample_session)
        # Our mock latencies are 300ms (S1) and 800ms (S2)
        assert report.s1_within_target == 1.0  # All S1 within 500ms
        assert report.s2_within_target == 1.0  # All S2 within 1000ms

    def test_format_report(self, sample_session):
        report = profile_session(sample_session)
        text = format_report(report)
        assert "Latency Report" in text
        assert "System 1" in text
        assert "System 2" in text

    def test_percentile(self):
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        assert _percentile(values, 50) == 300.0
        assert _percentile(values, 0) == 100.0


class TestCreditAssignment:
    def test_assign_credit_empty(self):
        report = assign_credit(InterviewSession())
        assert len(report.turn_credits) == 0

    def test_assign_credit_basic(self, sample_session):
        report = assign_credit(sample_session)
        assert len(report.turn_credits) == len(sample_session.turns)
        assert report.avg_credit > 0
        assert len(report.top_turns) > 0

    def test_credit_scores_bounded(self, sample_session):
        report = assign_credit(sample_session)
        for tc in report.turn_credits:
            assert 0 <= tc.information_gain <= 1
            assert 0 <= tc.depth_elicitation <= 1
            assert 0 <= tc.topic_progression <= 1
            assert 0 <= tc.followup_quality <= 1
            assert 0 <= tc.overall_credit <= 1

    def test_format_credit_report(self, sample_session):
        report = assign_credit(sample_session)
        text = format_credit_report(report)
        assert "Credit Report" in text
        assert "Average credit" in text

    def test_credit_with_kg_snapshots(self, sample_session):
        snapshots = {
            1: {"Python Fundamentals": 0.0, "System Design": 0.0},
            2: {"Python Fundamentals": 0.3, "System Design": 0.0},
            3: {"Python Fundamentals": 0.3, "System Design": 0.1},
            4: {"Python Fundamentals": 0.5, "System Design": 0.4},
            5: {"Python Fundamentals": 0.6, "System Design": 0.4},
        }
        report = assign_credit(sample_session, kg_snapshots_per_turn=snapshots)
        # Turn 2 should have higher info gain (Python went from 0 to 0.3)
        turn2 = report.turn_credits[1]
        turn5 = report.turn_credits[4]
        assert turn2.information_gain > turn5.information_gain
