"""Pydantic models for structured persona definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicArea(BaseModel):
    """A topic the persona should cover during the interview."""

    name: str
    description: str = ""
    priority: int = Field(default=1, ge=1, le=5, description="1=must-cover, 5=optional")
    min_depth: int = Field(default=1, ge=1, description="Minimum follow-up depth expected")
    example_questions: list[str] = Field(default_factory=list)


class DecisionPattern(BaseModel):
    """A behavioral decision pattern for the persona (Stanford approach)."""

    situation: str  # When this pattern activates
    behavior: str  # What the persona does
    rationale: str = ""  # Why this behavior is appropriate


class PersonaProfile(BaseModel):
    """Full persona definition for an interview agent."""

    name: str
    role: str  # e.g., "Technical Interviewer"
    domain: str  # e.g., "Backend Engineering"
    personality_traits: list[str] = Field(default_factory=list)
    communication_style: str = ""
    opening_message: str = ""
    topic_coverage: list[TopicArea] = Field(default_factory=list)
    decision_patterns: list[DecisionPattern] = Field(default_factory=list)
    probing_strategies: list[str] = Field(default_factory=list)
    depth_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Per-topic threshold: when response score < threshold, probe deeper",
    )
    transition_triggers: list[str] = Field(
        default_factory=list,
        description="Signals that should cause a persona switch or topic transition",
    )

    def get_must_cover_topics(self) -> list[TopicArea]:
        """Return topics with priority 1 (must-cover)."""
        return [t for t in self.topic_coverage if t.priority == 1]

    def get_topic_names(self) -> list[str]:
        return [t.name for t in self.topic_coverage]
