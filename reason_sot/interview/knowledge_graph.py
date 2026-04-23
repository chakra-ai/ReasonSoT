"""In-memory knowledge graph — tracks interview context as a structured graph.

Based on the Mind-Map approach from the Agentic Reasoning paper (2502.04644):
  - Nodes: topics, entities, skills, experiences mentioned by the candidate
  - Edges: relationships (related_to, used_in, led_to, part_of)
  - Clusters: automatic theme grouping for coverage tracking

Two extraction modes:
  1. Heuristic (zero-cost, ~0ms): regex-based keyword extraction from conversation
  2. LLM-assisted (S1 call, ~300ms): structured entity/relation extraction

The graph is serialized to a summary string and injected into the system prompt
(prefix-cached) so the model has structured context about what's been discussed.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

from reason_sot.llm.client import LLMClient
from reason_sot.persona.profiles import PersonaProfile, TopicArea
from reason_sot.types import KGCluster, KGEdge, KGNode, ModelTier

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph for an interview session."""

    def __init__(self, persona: PersonaProfile) -> None:
        self._persona = persona
        self._nodes: dict[str, KGNode] = {}
        self._edges: list[KGEdge] = []
        self._clusters: dict[str, KGCluster] = {}
        self._turn_count = 0

        # Initialize clusters from persona topic areas
        for topic in persona.topic_coverage:
            cluster_id = self._normalize_id(topic.name)
            self._clusters[cluster_id] = KGCluster(
                cluster_id=cluster_id,
                theme=topic.name,
                coverage_score=0.0,
            )

    # ── Node/Edge management ──────────────────────────────────────────

    def add_node(
        self,
        label: str,
        node_type: str = "entity",
        properties: dict[str, Any] | None = None,
        cluster_id: str | None = None,
    ) -> str:
        """Add a node to the graph. Returns the node ID."""
        node_id = self._normalize_id(label)
        if node_id in self._nodes:
            # Update existing node
            if properties:
                self._nodes[node_id].properties.update(properties)
            return node_id

        node = KGNode(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties or {},
            turn_discovered=self._turn_count,
        )
        self._nodes[node_id] = node

        # Add to cluster if specified
        if cluster_id and cluster_id in self._clusters:
            if node_id not in self._clusters[cluster_id].node_ids:
                self._clusters[cluster_id].node_ids.append(node_id)

        return node_id

    def add_edge(
        self, source_label: str, target_label: str, relation: str = "related_to"
    ) -> None:
        """Add a relationship between two nodes."""
        source_id = self._normalize_id(source_label)
        target_id = self._normalize_id(target_label)

        # Ensure both nodes exist
        if source_id not in self._nodes:
            self.add_node(source_label)
        if target_id not in self._nodes:
            self.add_node(target_label)

        # Check for duplicate edge
        for e in self._edges:
            if e.source == source_id and e.target == target_id and e.relation == relation:
                e.weight += 0.5  # Strengthen existing edge
                return

        self._edges.append(KGEdge(
            source=source_id,
            target=target_id,
            relation=relation,
        ))

    # ── Extraction ────────────────────────────────────────────────────

    def extract_from_turn(
        self, user_input: str, agent_response: str, turn_number: int
    ) -> None:
        """Extract entities and relationships from a conversation turn (heuristic)."""
        self._turn_count = turn_number
        combined = f"{user_input} {agent_response}".lower()

        # Match against persona topic keywords
        for topic in self._persona.topic_coverage:
            cluster_id = self._normalize_id(topic.name)
            topic_keywords = self._get_topic_keywords(topic)

            hits = 0
            for keyword in topic_keywords:
                if keyword in combined:
                    hits += 1
                    node_id = self.add_node(
                        label=keyword,
                        node_type="topic",
                        properties={"source_turn": turn_number},
                        cluster_id=cluster_id,
                    )

            # Update coverage score
            if hits > 0 and cluster_id in self._clusters:
                current = self._clusters[cluster_id].coverage_score
                increment = min(hits * 0.15, 0.4)  # Diminishing returns
                self._clusters[cluster_id].coverage_score = min(1.0, current + increment)

        # Extract technical entities (proper nouns, tools, frameworks)
        tech_entities = self._extract_technical_entities(combined)
        for entity in tech_entities:
            self.add_node(entity, node_type="entity", properties={"turn": turn_number})

        # Create edges between entities mentioned in the same turn
        turn_nodes = [
            nid for nid, n in self._nodes.items()
            if n.turn_discovered == turn_number or n.properties.get("source_turn") == turn_number
        ]
        for i, n1 in enumerate(turn_nodes):
            for n2 in turn_nodes[i + 1:]:
                self.add_edge(
                    self._nodes[n1].label,
                    self._nodes[n2].label,
                    relation="co_mentioned",
                )

    async def extract_from_turn_llm(
        self,
        client: LLMClient,
        user_input: str,
        agent_response: str,
        turn_number: int,
    ) -> None:
        """Extract entities and relationships using LLM (higher quality, ~300ms)."""
        self._turn_count = turn_number

        # Always do heuristic extraction first (it's free)
        self.extract_from_turn(user_input, agent_response, turn_number)

        # LLM extraction for richer structure
        try:
            prompt = (
                "Extract key entities and relationships from this interview exchange.\n"
                "Return ONLY a JSON object:\n"
                '{"entities": [{"label": "...", "type": "skill|experience|tool|concept"}], '
                '"relationships": [{"source": "...", "target": "...", "relation": "used_in|led_to|related_to|part_of"}]}\n\n'
                f"Interviewer: {agent_response}\n"
                f"Candidate: {user_input}"
            )

            text, _, _ = await client.complete_message(
                messages=[{"role": "user", "content": prompt}],
                system="You are an entity extraction engine. Return only valid JSON.",
                model_tier=ModelTier.FAST,
                max_tokens=256,
            )

            data = self._parse_extraction(text)
            for entity in data.get("entities", []):
                self.add_node(
                    label=entity["label"],
                    node_type=entity.get("type", "entity"),
                    properties={"turn": turn_number, "source": "llm"},
                )
            for rel in data.get("relationships", []):
                self.add_edge(rel["source"], rel["target"], rel.get("relation", "related_to"))

        except Exception as e:
            logger.debug("LLM extraction failed (using heuristic only): %s", e)

    # ── Coverage & Gap Analysis ───────────────────────────────────────

    def get_coverage(self) -> dict[str, float]:
        """Get coverage scores for all topic clusters."""
        return {c.theme: c.coverage_score for c in self._clusters.values()}

    def get_uncovered_topics(self, threshold: float = 0.3) -> list[str]:
        """Get topics that haven't been sufficiently explored."""
        return [
            c.theme for c in self._clusters.values()
            if c.coverage_score < threshold
        ]

    def get_most_covered_topic(self) -> str | None:
        """Get the topic with highest coverage."""
        if not self._clusters:
            return None
        return max(self._clusters.values(), key=lambda c: c.coverage_score).theme

    def get_coverage_gap_score(self) -> float:
        """Overall coverage gap (0 = all covered, 1 = nothing covered)."""
        if not self._clusters:
            return 0.0
        scores = [c.coverage_score for c in self._clusters.values()]
        return 1.0 - (sum(scores) / len(scores))

    def get_suggested_next_topic(self) -> str | None:
        """Suggest the next topic to explore based on coverage gaps.

        Prioritizes: (1) must-cover topics with low coverage,
        (2) any topic with zero coverage, (3) lowest coverage overall.
        """
        must_cover = {t.name for t in self._persona.get_must_cover_topics()}

        # Find uncovered must-cover topics
        uncovered_must = [
            (c.theme, c.coverage_score)
            for c in self._clusters.values()
            if c.theme in must_cover and c.coverage_score < 0.5
        ]
        if uncovered_must:
            return min(uncovered_must, key=lambda x: x[1])[0]

        # Find any uncovered topics
        uncovered = [
            (c.theme, c.coverage_score)
            for c in self._clusters.values()
            if c.coverage_score < 0.3
        ]
        if uncovered:
            return min(uncovered, key=lambda x: x[1])[0]

        return None

    # ── Serialization ─────────────────────────────────────────────────

    def to_summary(self, max_length: int = 800) -> str:
        """Serialize the graph to a concise summary for the system prompt.

        This summary is injected into the system prompt (prefix-cached)
        so the model knows what's been discussed and what gaps remain.
        """
        lines = ["\n--- INTERVIEW KNOWLEDGE MAP ---"]

        # Coverage overview
        lines.append("TOPIC COVERAGE:")
        for cluster in sorted(self._clusters.values(), key=lambda c: c.coverage_score):
            bar = "█" * int(cluster.coverage_score * 10) + "░" * (10 - int(cluster.coverage_score * 10))
            lines.append(f"  {cluster.theme}: [{bar}] {cluster.coverage_score:.0%}")

        # Key entities discovered
        if self._nodes:
            entities_by_type: dict[str, list[str]] = defaultdict(list)
            for node in self._nodes.values():
                entities_by_type[node.node_type].append(node.label)

            lines.append("\nKEY TOPICS DISCUSSED:")
            for ntype, labels in entities_by_type.items():
                unique = list(dict.fromkeys(labels))[:8]  # Dedupe, limit
                lines.append(f"  {ntype}: {', '.join(unique)}")

        # Suggested next direction
        suggested = self.get_suggested_next_topic()
        if suggested:
            lines.append(f"\nSUGGESTED NEXT: Explore '{suggested}' (lowest coverage)")

        # Gaps
        gaps = self.get_uncovered_topics()
        if gaps:
            lines.append(f"GAPS: {', '.join(gaps[:5])}")

        summary = "\n".join(lines)

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."

        return summary

    def to_snapshot(self) -> dict[str, Any]:
        """Full serialization for session storage."""
        return {
            "nodes": {nid: n.model_dump() for nid, n in self._nodes.items()},
            "edges": [e.model_dump() for e in self._edges],
            "clusters": {cid: c.model_dump() for cid, c in self._clusters.items()},
            "coverage": self.get_coverage(),
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _normalize_id(self, label: str) -> str:
        """Normalize a label to a node ID."""
        return re.sub(r"[^a-z0-9_]", "_", label.lower().strip())[:50]

    def _get_topic_keywords(self, topic: TopicArea) -> list[str]:
        """Extract searchable keywords from a topic definition."""
        keywords: list[str] = []
        # From topic name
        for word in topic.name.lower().split():
            if len(word) > 3:
                keywords.append(word)
        # From description
        for word in topic.description.lower().split():
            if len(word) > 4:
                keywords.append(word)
        # From example questions
        for q in topic.example_questions:
            for word in q.lower().split():
                if len(word) > 5:
                    keywords.append(word)
        return list(set(keywords))

    def _extract_technical_entities(self, text: str) -> list[str]:
        """Extract technical entities using pattern matching."""
        patterns = [
            r"\b(python|java|golang|rust|typescript|javascript)\b",
            r"\b(flask|django|fastapi|express|spring)\b",
            r"\b(postgresql|mysql|mongodb|redis|elasticsearch)\b",
            r"\b(docker|kubernetes|terraform|aws|gcp|azure)\b",
            r"\b(pytest|unittest|jest|mocha)\b",
            r"\b(git|github|gitlab|ci/cd)\b",
            r"\b(rest|grpc|graphql|websocket)\b",
            r"\b(sqlalchemy|alembic|prisma|sequelize)\b",
        ]
        entities = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.add(match.group(0).lower())
        return list(entities)

    def _parse_extraction(self, text: str) -> dict[str, Any]:
        """Parse LLM extraction response."""
        text = text.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"entities": [], "relationships": []}
