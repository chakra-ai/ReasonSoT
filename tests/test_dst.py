"""Tests for Domain-Specialized Tree (DST) reasoning."""

import pytest

from reason_sot.core.dst import DSTConfig, build_dst_prompt, estimate_beam_from_context


class TestDSTConfig:
    def test_defaults(self):
        cfg = DSTConfig()
        assert cfg.min_beam == 1
        assert cfg.max_beam == 3
        assert cfg.confidence_threshold == 0.5


class TestBuildPrompt:
    def test_basic_prompt(self):
        prompt = build_dst_prompt()
        assert "Domain-Specialized Tree" in prompt
        assert "INITIAL ASSESSMENT" in prompt
        assert "EXPANDED SEARCH" in prompt
        assert "SELECT & RESPOND" in prompt

    def test_prompt_with_coverage_gaps(self):
        prompt = build_dst_prompt(coverage_gaps=["Testing", "DevOps"])
        assert "Testing" in prompt
        assert "DevOps" in prompt
        assert "Uncovered topics" in prompt

    def test_prompt_with_current_topic(self):
        prompt = build_dst_prompt(current_topic="System Design")
        assert "System Design" in prompt

    def test_custom_config(self):
        cfg = DSTConfig(max_beam=5, confidence_threshold=0.7)
        prompt = build_dst_prompt(config=cfg)
        assert "beam=5" in prompt
        assert "0.7" in prompt


class TestBeamEstimation:
    def test_high_confidence_min_beam(self):
        beam = estimate_beam_from_context(previous_confidence=0.9)
        assert beam == 1

    def test_low_confidence_widens_beam(self):
        beam = estimate_beam_from_context(previous_confidence=0.3)
        assert beam > 1

    def test_deep_followup_widens_beam(self):
        beam = estimate_beam_from_context(
            previous_confidence=0.9, followup_chain_depth=3,
        )
        assert beam >= 2

    def test_high_coverage_gap_widens_beam(self):
        beam = estimate_beam_from_context(
            previous_confidence=0.9, coverage_gap_score=0.8,
        )
        assert beam >= 2

    def test_all_signals_max_beam(self):
        beam = estimate_beam_from_context(
            previous_confidence=0.2,
            followup_chain_depth=3,
            coverage_gap_score=0.9,
        )
        assert beam == 3  # Capped at max_beam

    def test_custom_config(self):
        cfg = DSTConfig(min_beam=2, max_beam=5)
        beam = estimate_beam_from_context(
            previous_confidence=0.2,
            followup_chain_depth=3,
            coverage_gap_score=0.9,
            config=cfg,
        )
        assert beam >= 2
        assert beam <= 5
