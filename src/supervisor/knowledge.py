"""Knowledge graph support for the supervisor framework.

Provides a lightweight in-memory knowledge graph with entity-relation triples,
querying, and optional persistence.  Designed to work with agent workflows
for knowledge extraction and retrieval.

Example::

    from supervisor.knowledge import KnowledgeGraph, Triple

    kg = KnowledgeGraph()
    kg.add(Triple("Python", "is_a", "programming language"))
    kg.add(Triple("Rust", "is_a", "programming language"))
    kg.add(Triple("PyO3", "bridges", "Python"))
    kg.add(Triple("PyO3", "bridges", "Rust"))

    results = kg.query(subject="PyO3")
    # [Triple("PyO3", "bridges", "Python"), Triple("PyO3", "bridges", "Rust")]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from supervisor.ext import Extension

if TYPE_CHECKING:
    from supervisor._core import Message
    from supervisor.agent import Agent


@dataclass(frozen=True)
class Triple:
    """A knowledge graph triple (subject, predicate, object).

    Attributes:
        subject: The entity or concept.
        predicate: The relationship.
        object: The target entity or value.
        metadata: Optional metadata dictionary.
    """

    subject: str
    predicate: str
    object: str
    metadata: Optional[Dict[str, Any]] = field(default=None, hash=False, compare=False)


class KnowledgeGraph:
    """In-memory knowledge graph backed by triple storage.

    Supports CRUD operations, querying by subject/predicate/object,
    shortest path finding, and JSON persistence.

    Example::

        kg = KnowledgeGraph()
        kg.add(Triple("Alice", "knows", "Bob"))
        kg.add(Triple("Bob", "knows", "Charlie"))

        path = kg.shortest_path("Alice", "Charlie")
        # ["Alice", "Bob", "Charlie"]
    """

    def __init__(self) -> None:
        self._triples: Set[Triple] = set()
        self._by_subject: Dict[str, Set[Triple]] = {}
        self._by_predicate: Dict[str, Set[Triple]] = {}
        self._by_object: Dict[str, Set[Triple]] = {}

    def add(self, triple: Triple) -> None:
        """Add a triple to the graph.

        Args:
            triple: The triple to add. Duplicates are ignored.
        """
        if triple in self._triples:
            return
        self._triples.add(triple)
        self._by_subject.setdefault(triple.subject, set()).add(triple)
        self._by_predicate.setdefault(triple.predicate, set()).add(triple)
        self._by_object.setdefault(triple.object, set()).add(triple)

    def remove(self, triple: Triple) -> bool:
        """Remove a triple from the graph.

        Args:
            triple: The triple to remove.

        Returns:
            ``True`` if the triple was found and removed.
        """
        if triple not in self._triples:
            return False
        self._triples.discard(triple)
        self._by_subject.get(triple.subject, set()).discard(triple)
        self._by_predicate.get(triple.predicate, set()).discard(triple)
        self._by_object.get(triple.object, set()).discard(triple)
        return True

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Triple]:
        """Query triples by subject, predicate, and/or object.

        All provided filters are combined with AND logic.  Omitted filters
        match everything.

        Args:
            subject: Filter by subject.
            predicate: Filter by predicate.
            object: Filter by object.

        Returns:
            List of matching triples.
        """
        candidates: Optional[Set[Triple]] = None

        if subject is not None:
            candidates = set(self._by_subject.get(subject, set()))
        if predicate is not None:
            pred_set = self._by_predicate.get(predicate, set())
            candidates = pred_set if candidates is None else candidates & pred_set
        if object is not None:
            obj_set = self._by_object.get(object, set())
            candidates = obj_set if candidates is None else candidates & obj_set

        if candidates is None:
            return sorted(self._triples, key=lambda t: (t.subject, t.predicate, t.object))
        return sorted(candidates, key=lambda t: (t.subject, t.predicate, t.object))

    def entities(self) -> Set[str]:
        """Return all unique entities (subjects and objects)."""
        result: Set[str] = set()
        for t in self._triples:
            result.add(t.subject)
            result.add(t.object)
        return result

    def predicates(self) -> Set[str]:
        """Return all unique predicates."""
        return set(self._by_predicate.keys())

    def neighbors(self, entity: str) -> Set[str]:
        """Return all entities directly connected to *entity*.

        Args:
            entity: The entity to find neighbors for.

        Returns:
            Set of connected entity names.
        """
        result: Set[str] = set()
        for t in self._by_subject.get(entity, set()):
            result.add(t.object)
        for t in self._by_object.get(entity, set()):
            result.add(t.subject)
        return result

    def shortest_path(self, start: str, end: str) -> List[str]:
        """Find the shortest path between two entities using BFS.

        Args:
            start: Starting entity.
            end: Target entity.

        Returns:
            List of entities forming the path, or empty list if none found.
        """
        if start == end:
            return [start]

        visited: Set[str] = {start}
        queue: List[Tuple[str, List[str]]] = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self.neighbors(current):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    @property
    def size(self) -> int:
        """Return the number of triples in the graph."""
        return len(self._triples)

    def clear(self) -> None:
        """Remove all triples from the graph."""
        self._triples.clear()
        self._by_subject.clear()
        self._by_predicate.clear()
        self._by_object.clear()

    def to_json(self) -> str:
        """Serialize the graph to JSON.

        Returns:
            JSON string representation of all triples.
        """
        triples = [
            {
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "metadata": t.metadata,
            }
            for t in sorted(
                self._triples, key=lambda t: (t.subject, t.predicate, t.object)
            )
        ]
        return json.dumps({"triples": triples}, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "KnowledgeGraph":
        """Deserialize a graph from JSON.

        Args:
            data: JSON string as produced by :meth:`to_json`.

        Returns:
            A new :class:`KnowledgeGraph` instance.
        """
        parsed = json.loads(data)
        kg = cls()
        for t in parsed.get("triples", []):
            kg.add(
                Triple(
                    subject=t["subject"],
                    predicate=t["predicate"],
                    object=t["object"],
                    metadata=t.get("metadata"),
                )
            )
        return kg

    def save(self, path: str | Path) -> None:
        """Save the graph to a JSON file.

        Args:
            path: File path.
        """
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        """Load a graph from a JSON file.

        Args:
            path: File path.

        Returns:
            A new :class:`KnowledgeGraph` instance.
        """
        data = Path(path).read_text(encoding="utf-8")
        return cls.from_json(data)


class KnowledgeGraphExtension(Extension):
    """Extension that provides knowledge graph capabilities to an agent.

    The extension maintains a :class:`KnowledgeGraph` and can optionally
    auto-extract triples from incoming messages.

    Args:
        graph: Optional pre-existing graph to use.
        auto_extract: Whether to auto-extract triples from messages.

    Example::

        kg_ext = KnowledgeGraphExtension()
        agent.use(kg_ext)

        # Add knowledge manually
        kg_ext.graph.add(Triple("supervisor.rs", "written_in", "Rust"))
    """

    name = "knowledge_graph"

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        auto_extract: bool = False,
    ) -> None:
        self.graph = graph or KnowledgeGraph()
        self.auto_extract = auto_extract

    def on_message(self, agent: "Agent", msg: "Message") -> Optional["Message"]:
        """Optionally extract triples from incoming messages.

        When ``auto_extract`` is enabled, simple ``subject|predicate|object``
        patterns in message content are parsed and added to the graph.
        """
        if self.auto_extract and msg.content:
            parts = msg.content.split("|")
            if len(parts) == 3:
                self.graph.add(
                    Triple(
                        subject=parts[0].strip(),
                        predicate=parts[1].strip(),
                        object=parts[2].strip(),
                    )
                )
        return None

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Triple]:
        """Query the knowledge graph.

        Args:
            subject: Filter by subject.
            predicate: Filter by predicate.
            object: Filter by object.

        Returns:
            Matching triples.
        """
        return self.graph.query(subject=subject, predicate=predicate, object=object)


__all__ = [
    "KnowledgeGraph",
    "KnowledgeGraphExtension",
    "Triple",
]
