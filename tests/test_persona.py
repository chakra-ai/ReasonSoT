"""Tests for persona management."""

import pytest

from reason_sot.persona.manager import (
    list_personas,
    load_persona,
    render_system_prompt,
    suggest_persona_switch,
)
from reason_sot.persona.profiles import PersonaProfile, TopicArea


class TestLoadPersona:
    def test_load_technical_interviewer(self):
        persona = load_persona("technical_interviewer")
        assert persona.name
        assert persona.role
        assert len(persona.topic_coverage) > 0

    def test_load_behavioral_interviewer(self):
        persona = load_persona("behavioral_interviewer")
        assert "Behavioral" in persona.role or "behavioral" in persona.domain.lower()

    def test_load_career_coach(self):
        persona = load_persona("career_coach")
        assert "Coach" in persona.role or "coach" in persona.name.lower()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_persona("nonexistent_persona")


class TestListPersonas:
    def test_list_personas(self):
        personas = list_personas()
        assert len(personas) >= 3
        assert "technical_interviewer" in personas
        assert "behavioral_interviewer" in personas
        assert "career_coach" in personas


class TestRenderSystemPrompt:
    def test_render_produces_two_blocks(self, sample_persona):
        blocks = render_system_prompt(sample_persona)
        assert len(blocks) == 2
        assert isinstance(blocks[0], str)
        assert isinstance(blocks[1], str)

    def test_base_prompt_has_principles(self, sample_persona):
        blocks = render_system_prompt(sample_persona)
        base = blocks[0]
        assert "CORE PRINCIPLES" in base
        assert "FOLLOW-UP RULES" in base

    def test_persona_block_has_topics(self, sample_persona):
        blocks = render_system_prompt(sample_persona)
        persona_section = blocks[1]
        assert "TOPICS TO COVER" in persona_section
        assert sample_persona.name in persona_section

    def test_render_includes_decision_patterns(self, sample_persona):
        blocks = render_system_prompt(sample_persona)
        persona_section = blocks[1]
        assert "DECISION PATTERNS" in persona_section

    def test_render_includes_probing_strategies(self, sample_persona):
        blocks = render_system_prompt(sample_persona)
        persona_section = blocks[1]
        assert "PROBING STRATEGIES" in persona_section


class TestPersonaProfile:
    def test_get_must_cover_topics(self, sample_persona):
        must_cover = sample_persona.get_must_cover_topics()
        assert all(t.priority == 1 for t in must_cover)
        assert len(must_cover) == 2  # Python Fundamentals + System Design

    def test_get_topic_names(self, sample_persona):
        names = sample_persona.get_topic_names()
        assert "Python Fundamentals" in names
        assert "System Design" in names
        assert "Testing" in names


class TestPersonaSwitching:
    def test_no_switch_when_topics_uncovered(self, sample_persona):
        result = suggest_persona_switch(
            current_persona=sample_persona,
            conversation_signals={"topic_coverage": {}},
        )
        assert result is None

    def test_switch_when_all_covered(self):
        persona = load_persona("technical_interviewer")
        coverage = {t.name: 0.8 for t in persona.get_must_cover_topics()}
        result = suggest_persona_switch(
            current_persona=persona,
            conversation_signals={"topic_coverage": coverage},
        )
        assert result == "behavioral_interviewer"
