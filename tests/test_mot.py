"""Tests for Matrix of Thought (MoT) reasoning."""

import pytest

from reason_sot.core.mot import (
    MoTConfig,
    MoTResult,
    build_mot_prompt,
    parse_mot_response,
)


class TestMoTConfig:
    def test_broad_config(self):
        cfg = MoTConfig.broad()
        assert cfg.rows == 4
        assert cfg.cols == 1
        assert cfg.mode == "broad"

    def test_deep_config(self):
        cfg = MoTConfig.deep()
        assert cfg.rows == 1
        assert cfg.cols == 4
        assert cfg.mode == "deep"

    def test_balanced_config(self):
        cfg = MoTConfig.balanced()
        assert cfg.rows == 3
        assert cfg.cols == 2
        assert cfg.mode == "balanced"

    def test_from_signals_high_gap(self):
        cfg = MoTConfig.from_signals(topic_coverage_gap=0.7)
        assert cfg.mode == "broad"

    def test_from_signals_deep_patterns(self):
        cfg = MoTConfig.from_signals(deep_pattern_count=3)
        assert cfg.mode == "deep"

    def test_from_signals_default(self):
        cfg = MoTConfig.from_signals()
        assert cfg.mode == "balanced"


class TestBuildPrompt:
    def test_balanced_prompt(self):
        prompt = build_mot_prompt(context="the candidate's response")
        assert "Matrix of Thought" in prompt
        assert "BREADTH" in prompt
        assert "DEPTH" in prompt
        assert "SYNTHESIS" in prompt

    def test_deep_prompt(self):
        cfg = MoTConfig.deep()
        prompt = build_mot_prompt(config=cfg)
        assert "Deep Chain" in prompt
        assert "Level 1" in prompt

    def test_broad_prompt(self):
        cfg = MoTConfig.broad()
        prompt = build_mot_prompt(config=cfg)
        assert "Broad Exploration" in prompt
        assert "angles" in prompt.lower() or "directions" in prompt.lower()


class TestParseMoTResponse:
    def test_parse_with_paths_and_depth(self):
        thinking = """
        Angle 1: Technical depth — check their understanding of decorators
        Level 1: They mentioned decorators but couldn't explain closures
        Level 2: This suggests surface-level knowledge

        Angle 2: Practical experience — ask about real projects
        Level 1: They mentioned Flask but no details

        Angle 3: Problem-solving — give a debugging scenario
        Level 1: Worth exploring

        Comparing the three paths, selecting Angle 1 as the best.
        """
        result = parse_mot_response(thinking)
        assert result.paths_explored >= 2
        assert result.depth_reached >= 1
        assert result.synthesis == "present"

    def test_parse_empty_thinking(self):
        result = parse_mot_response("")
        assert result.paths_explored == 1
        assert result.depth_reached == 1

    def test_parse_no_synthesis(self):
        thinking = "Just a single thought about the question."
        result = parse_mot_response(thinking)
        assert result.synthesis == "absent"
