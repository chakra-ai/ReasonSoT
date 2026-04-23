"""Tests for the in-memory knowledge graph."""

import pytest

from reason_sot.interview.knowledge_graph import KnowledgeGraph


class TestNodeManagement:
    def test_add_node(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        node_id = kg.add_node("Python", node_type="topic")
        assert node_id == "python"
        assert "python" in kg._nodes

    def test_add_duplicate_node_updates(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.add_node("Python", properties={"v": 1})
        kg.add_node("Python", properties={"v": 2})
        assert kg._nodes["python"].properties["v"] == 2
        assert len(kg._nodes) == 1

    def test_add_node_to_cluster(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        cluster_id = kg._normalize_id("Python Fundamentals")
        kg.add_node("generators", node_type="topic", cluster_id=cluster_id)
        assert "generators" in kg._clusters[cluster_id].node_ids


class TestEdgeManagement:
    def test_add_edge(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.add_node("Python")
        kg.add_node("Flask")
        kg.add_edge("Python", "Flask", "used_in")
        assert len(kg._edges) == 1
        assert kg._edges[0].relation == "used_in"

    def test_add_edge_creates_missing_nodes(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.add_edge("Redis", "Caching")
        assert "redis" in kg._nodes
        assert "caching" in kg._nodes

    def test_duplicate_edge_strengthens(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.add_edge("Python", "Flask", "used_in")
        kg.add_edge("Python", "Flask", "used_in")
        assert len(kg._edges) == 1
        assert kg._edges[0].weight == 1.5


class TestExtraction:
    def test_extract_from_turn_heuristic(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.extract_from_turn(
            user_input="I use Python generators and decorators a lot.",
            agent_response="Tell me more about decorators.",
            turn_number=1,
        )
        assert len(kg._nodes) > 0

    def test_extract_technical_entities(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.extract_from_turn(
            user_input="I built it with Django and PostgreSQL, deployed on Docker and Kubernetes.",
            agent_response="Interesting architecture choices.",
            turn_number=1,
        )
        labels = {n.label for n in kg._nodes.values()}
        assert "django" in labels or "postgresql" in labels

    def test_coverage_increases_with_extraction(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        before = kg.get_coverage_gap_score()

        kg.extract_from_turn(
            user_input="I love Python generators, decorators, and iterators. They're core language features.",
            agent_response="Good. Explain how generators work.",
            turn_number=1,
        )
        after = kg.get_coverage_gap_score()
        # Gap should decrease (coverage increased)
        assert after <= before


class TestCoverage:
    def test_initial_coverage_zero(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        coverage = kg.get_coverage()
        assert all(v == 0.0 for v in coverage.values())

    def test_get_uncovered_topics(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        uncovered = kg.get_uncovered_topics()
        assert len(uncovered) == len(sample_persona.topic_coverage)

    def test_get_suggested_next_topic(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        suggested = kg.get_suggested_next_topic()
        # Should suggest a must-cover topic
        must_cover_names = [t.name for t in sample_persona.get_must_cover_topics()]
        assert suggested in must_cover_names

    def test_coverage_gap_score(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        assert kg.get_coverage_gap_score() == 1.0  # Nothing covered


class TestSerialization:
    def test_to_summary(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.extract_from_turn("I use Python and pytest", "Good.", turn_number=1)
        summary = kg.to_summary()
        assert "TOPIC COVERAGE" in summary
        assert "KNOWLEDGE MAP" in summary

    def test_to_snapshot(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        kg.add_node("test_node")
        kg.add_edge("test_node", "other_node")
        snapshot = kg.to_snapshot()
        assert "nodes" in snapshot
        assert "edges" in snapshot
        assert "clusters" in snapshot
        assert "coverage" in snapshot

    def test_summary_truncation(self, sample_persona):
        kg = KnowledgeGraph(sample_persona)
        summary = kg.to_summary(max_length=50)
        assert len(summary) <= 50 + 3  # +3 for "..."
