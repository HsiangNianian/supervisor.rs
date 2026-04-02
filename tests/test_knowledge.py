"""Tests for the knowledge graph module (Phase 7.3)."""

import json
import pytest

from supervisor.knowledge import (
    KnowledgeGraph,
    KnowledgeGraphExtension,
    Triple,
)


class TestTriple:
    """Tests for the Triple dataclass."""

    def test_create_triple(self):
        t = Triple("Python", "is_a", "language")
        assert t.subject == "Python"
        assert t.predicate == "is_a"
        assert t.object == "language"

    def test_triple_equality(self):
        t1 = Triple("A", "rel", "B")
        t2 = Triple("A", "rel", "B")
        assert t1 == t2

    def test_triple_inequality(self):
        t1 = Triple("A", "rel", "B")
        t2 = Triple("A", "rel", "C")
        assert t1 != t2

    def test_triple_hash(self):
        t1 = Triple("A", "rel", "B")
        t2 = Triple("A", "rel", "B")
        assert hash(t1) == hash(t2)

    def test_metadata(self):
        t = Triple("A", "rel", "B", metadata={"confidence": 0.9})
        assert t.metadata["confidence"] == 0.9

    def test_frozen(self):
        t = Triple("A", "rel", "B")
        with pytest.raises(AttributeError):
            t.subject = "C"


class TestKnowledgeGraph:
    """Tests for the KnowledgeGraph class."""

    def test_empty_graph(self):
        kg = KnowledgeGraph()
        assert kg.size == 0
        assert len(kg.entities()) == 0

    def test_add_triple(self):
        kg = KnowledgeGraph()
        kg.add(Triple("Python", "is_a", "language"))
        assert kg.size == 1

    def test_add_duplicate(self):
        kg = KnowledgeGraph()
        t = Triple("Python", "is_a", "language")
        kg.add(t)
        kg.add(t)
        assert kg.size == 1  # No duplicate

    def test_remove_triple(self):
        kg = KnowledgeGraph()
        t = Triple("Python", "is_a", "language")
        kg.add(t)
        assert kg.remove(t) is True
        assert kg.size == 0

    def test_remove_nonexistent(self):
        kg = KnowledgeGraph()
        assert kg.remove(Triple("A", "B", "C")) is False

    def test_query_by_subject(self):
        kg = KnowledgeGraph()
        kg.add(Triple("Python", "is_a", "language"))
        kg.add(Triple("Python", "has_feature", "dynamic typing"))
        kg.add(Triple("Rust", "is_a", "language"))
        results = kg.query(subject="Python")
        assert len(results) == 2

    def test_query_by_predicate(self):
        kg = KnowledgeGraph()
        kg.add(Triple("Python", "is_a", "language"))
        kg.add(Triple("Rust", "is_a", "language"))
        kg.add(Triple("Python", "has_feature", "GC"))
        results = kg.query(predicate="is_a")
        assert len(results) == 2

    def test_query_by_object(self):
        kg = KnowledgeGraph()
        kg.add(Triple("Python", "is_a", "language"))
        kg.add(Triple("Rust", "is_a", "language"))
        results = kg.query(object="language")
        assert len(results) == 2

    def test_query_combined(self):
        kg = KnowledgeGraph()
        kg.add(Triple("Python", "is_a", "language"))
        kg.add(Triple("Rust", "is_a", "language"))
        kg.add(Triple("Python", "is_a", "dynamic"))
        results = kg.query(subject="Python", predicate="is_a")
        assert len(results) == 2
        results = kg.query(subject="Python", object="language")
        assert len(results) == 1

    def test_query_all(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "r1", "B"))
        kg.add(Triple("C", "r2", "D"))
        results = kg.query()
        assert len(results) == 2

    def test_entities(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "rel", "B"))
        kg.add(Triple("B", "rel", "C"))
        entities = kg.entities()
        assert entities == {"A", "B", "C"}

    def test_predicates(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "is_a", "B"))
        kg.add(Triple("C", "has", "D"))
        preds = kg.predicates()
        assert preds == {"is_a", "has"}

    def test_neighbors(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "knows", "B"))
        kg.add(Triple("C", "knows", "A"))
        neighbors = kg.neighbors("A")
        assert neighbors == {"B", "C"}

    def test_shortest_path(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "knows", "B"))
        kg.add(Triple("B", "knows", "C"))
        kg.add(Triple("C", "knows", "D"))
        path = kg.shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_shortest_path_same_node(self):
        kg = KnowledgeGraph()
        path = kg.shortest_path("A", "A")
        assert path == ["A"]

    def test_shortest_path_no_path(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "knows", "B"))
        kg.add(Triple("C", "knows", "D"))
        path = kg.shortest_path("A", "D")
        assert path == []

    def test_shortest_path_bidirectional(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "knows", "B"))
        kg.add(Triple("B", "knows", "C"))
        # Can also go backwards
        path = kg.shortest_path("C", "A")
        assert path == ["C", "B", "A"]

    def test_clear(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "rel", "B"))
        kg.clear()
        assert kg.size == 0
        assert len(kg.entities()) == 0

    def test_json_serialization(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "rel", "B"))
        kg.add(Triple("C", "rel", "D"))
        json_str = kg.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["triples"]) == 2

    def test_json_round_trip(self):
        kg = KnowledgeGraph()
        kg.add(Triple("A", "rel", "B"))
        kg.add(Triple("C", "rel", "D", metadata={"weight": 1.0}))
        json_str = kg.to_json()
        kg2 = KnowledgeGraph.from_json(json_str)
        assert kg2.size == 2
        results = kg2.query(subject="A")
        assert len(results) == 1

    def test_file_persistence(self, tmp_path):
        kg = KnowledgeGraph()
        kg.add(Triple("X", "r", "Y"))
        filepath = tmp_path / "kg.json"
        kg.save(str(filepath))
        kg2 = KnowledgeGraph.load(str(filepath))
        assert kg2.size == 1


class TestKnowledgeGraphExtension:
    """Tests for the KnowledgeGraphExtension."""

    def test_extension_name(self):
        ext = KnowledgeGraphExtension()
        assert ext.name == "knowledge_graph"

    def test_manual_add(self):
        ext = KnowledgeGraphExtension()
        ext.graph.add(Triple("A", "rel", "B"))
        assert ext.graph.size == 1

    def test_auto_extract(self):
        from supervisor._core import Message

        ext = KnowledgeGraphExtension(auto_extract=True)

        class FakeAgent:
            name = "test"

        msg = Message("user", "test", "Python|is_a|language")
        ext.on_message(FakeAgent(), msg)
        assert ext.graph.size == 1
        results = ext.query(subject="Python")
        assert len(results) == 1

    def test_no_extract_without_flag(self):
        from supervisor._core import Message

        ext = KnowledgeGraphExtension(auto_extract=False)

        class FakeAgent:
            name = "test"

        msg = Message("user", "test", "A|rel|B")
        ext.on_message(FakeAgent(), msg)
        assert ext.graph.size == 0

    def test_invalid_format_ignored(self):
        from supervisor._core import Message

        ext = KnowledgeGraphExtension(auto_extract=True)

        class FakeAgent:
            name = "test"

        msg = Message("user", "test", "not a triple")
        ext.on_message(FakeAgent(), msg)
        assert ext.graph.size == 0

    def test_query_delegation(self):
        ext = KnowledgeGraphExtension()
        ext.graph.add(Triple("A", "r1", "B"))
        ext.graph.add(Triple("A", "r2", "C"))
        results = ext.query(subject="A")
        assert len(results) == 2

    def test_returns_none(self):
        from supervisor._core import Message

        ext = KnowledgeGraphExtension()

        class FakeAgent:
            name = "test"

        result = ext.on_message(FakeAgent(), Message("a", "b", "c"))
        assert result is None
