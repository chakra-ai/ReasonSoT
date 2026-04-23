"""Persona manager — loads, validates, and renders persona definitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from reason_sot.persona.profiles import PersonaProfile

logger = logging.getLogger(__name__)

DEFINITIONS_DIR = Path(__file__).parent / "definitions"

# Base system instructions shared across all personas
BASE_SYSTEM_PROMPT = """You are an expert interview agent powered by the ReasonSoT reasoning engine.
Your role is to conduct adaptive, intelligent interviews that go both broad (covering all relevant topics)
and deep (probing for genuine understanding, not just surface knowledge).

CORE PRINCIPLES:
- Listen actively: reference specific details from the candidate's previous answers
- Adapt dynamically: adjust your questioning based on the candidate's demonstrated level
- Think before asking: every question should serve a clear purpose
- Be human: acknowledge good answers, be encouraging, maintain conversational flow
- Probe strategically: use follow-ups to distinguish real understanding from memorized answers

FOLLOW-UP RULES:
- After each response, decide: is this SUFFICIENT (move on), needs CLARIFICATION (dig deeper), or reveals an INTERESTING TANGENT (explore breadth)?
- Never ask more than 3 consecutive follow-ups on the same narrow point
- When you probe deeper, reference what the candidate just said specifically

RESPONSE FORMAT:
- Keep responses conversational and natural (this is a voice interview)
- Aim for 2-4 sentences per turn
- Ask ONE clear question at a time
"""


def load_persona(name: str) -> PersonaProfile:
    """Load a persona from a YAML file by name or filename.

    Searches for a matching file in the definitions directory.
    """
    # Try exact filename match
    yaml_path = DEFINITIONS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        # Try to find by persona name in files
        yaml_path = _find_persona_file(name)

    if not yaml_path or not yaml_path.exists():
        available = list_personas()
        raise FileNotFoundError(
            f"Persona '{name}' not found. Available: {available}"
        )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    persona = PersonaProfile(**data)
    logger.info("Loaded persona: %s (%s)", persona.name, persona.role)
    return persona


def _find_persona_file(name: str) -> Path | None:
    """Search for a persona file matching the given name."""
    name_lower = name.lower()
    for yaml_file in DEFINITIONS_DIR.glob("*.yaml"):
        if yaml_file.stem == "_schema":
            continue
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        if data and data.get("name", "").lower() == name_lower:
            return yaml_file
    return None


def list_personas() -> list[str]:
    """List available persona filenames (without extension)."""
    return [
        f.stem
        for f in DEFINITIONS_DIR.glob("*.yaml")
        if f.stem != "_schema"
    ]


def render_system_prompt(persona: PersonaProfile) -> list[str]:
    """Render a persona into system prompt blocks for the LLM client.

    Returns [base_system, persona_prompt] for cache.build_system_blocks().
    The base system prompt and persona prompt are separated for caching.
    """
    persona_section = _build_persona_section(persona)
    return [BASE_SYSTEM_PROMPT, persona_section]


# ── Dynamic persona switching ─────────────────────────────────────────────

_ALL_PERSONAS: dict[str, PersonaProfile] = {}


def preload_all_personas() -> dict[str, PersonaProfile]:
    """Load all available personas into memory for fast switching."""
    global _ALL_PERSONAS
    for name in list_personas():
        try:
            _ALL_PERSONAS[name] = load_persona(name)
        except Exception as e:
            logger.warning("Failed to preload persona '%s': %s", name, e)
    logger.info("Preloaded %d personas: %s", len(_ALL_PERSONAS), list(_ALL_PERSONAS.keys()))
    return _ALL_PERSONAS


def get_preloaded_persona(name: str) -> PersonaProfile | None:
    """Get a preloaded persona by name (fast, no disk I/O)."""
    return _ALL_PERSONAS.get(name)


def suggest_persona_switch(
    current_persona: PersonaProfile,
    conversation_signals: dict[str, Any],
) -> str | None:
    """Suggest a persona switch based on conversation signals.

    Returns the suggested persona filename or None if no switch needed.

    Signals considered:
    - Topic drift: if conversation has moved to a different domain
    - Explicit request: if the user/admin requests a different style
    - Coverage completion: if current persona's topics are fully covered
    """
    # Check if current persona's must-cover topics are all covered
    coverage = conversation_signals.get("topic_coverage", {})
    must_cover = current_persona.get_must_cover_topics()

    if must_cover and coverage:
        covered = sum(
            1 for t in must_cover
            if coverage.get(t.name, 0) >= 0.7
        )
        if covered >= len(must_cover):
            # All must-cover topics done — suggest switching
            current_domain = current_persona.domain.lower()
            current_role = current_persona.role.lower()
            if any(kw in current_domain or kw in current_role for kw in ("technical", "engineering", "backend", "frontend")):
                return "behavioral_interviewer"
            elif any(kw in current_domain or kw in current_role for kw in ("behavioral", "leadership")):
                return "career_coach"

    # Check for domain drift signals
    dominant_topics = conversation_signals.get("dominant_topics", [])
    if dominant_topics:
        current_topics = set(t.name.lower() for t in current_persona.topic_coverage)
        drift_count = sum(
            1 for t in dominant_topics
            if t.lower() not in " ".join(current_topics)
        )
        if drift_count > 2:
            logger.info("Possible domain drift detected: %s", dominant_topics)

    return None


def _build_persona_section(persona: PersonaProfile) -> str:
    """Build the persona-specific section of the system prompt."""
    lines = [
        f"\n--- PERSONA: {persona.name} ---",
        f"Role: {persona.role}",
        f"Domain: {persona.domain}",
    ]

    if persona.personality_traits:
        lines.append(f"Personality: {', '.join(persona.personality_traits)}")

    if persona.communication_style:
        lines.append(f"Communication Style: {persona.communication_style.strip()}")

    # Topic coverage
    lines.append("\nTOPICS TO COVER (in priority order):")
    for topic in sorted(persona.topic_coverage, key=lambda t: t.priority):
        priority_label = {1: "MUST", 2: "SHOULD", 3: "OPTIONAL"}.get(
            topic.priority, "OPTIONAL"
        )
        lines.append(f"  [{priority_label}] {topic.name}: {topic.description}")
        if topic.example_questions:
            lines.append(f"    Example questions: {topic.example_questions[0]}")

    # Decision patterns
    if persona.decision_patterns:
        lines.append("\nDECISION PATTERNS:")
        for dp in persona.decision_patterns:
            lines.append(f"  When: {dp.situation}")
            lines.append(f"  Do: {dp.behavior}")

    # Probing strategies
    if persona.probing_strategies:
        lines.append("\nPROBING STRATEGIES:")
        for strategy in persona.probing_strategies:
            lines.append(f"  - {strategy}")

    return "\n".join(lines)
