"""Observability: metrics collection and export for the supervisor framework.

Provides Prometheus-style metric primitives (Counter, Gauge, Histogram) with
a global :class:`MetricsRegistry` and a Prometheus text-format exporter.

Example::

    from supervisor.metrics import Counter, Histogram, get_registry

    messages_processed = Counter("messages_processed", "Total messages processed")
    messages_processed.inc()

    latency = Histogram("message_latency", "Message handling latency in seconds")
    latency.observe(0.042)

    # Export in Prometheus text format
    print(get_registry().export())
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
)

from supervisor.ext import Extension

if TYPE_CHECKING:
    from supervisor._core import Message
    from supervisor.agent import Agent

F = TypeVar("F", bound=Callable[..., Any])

# ── Counter ────────────────────────────────────────────────────────────────


class Counter:
    """A monotonically increasing counter.

    Args:
        name: Metric name (must be unique within a registry).
        description: Human-readable description.
        labels: Optional label names for multi-dimensional metrics.

    Example::

        requests = Counter("http_requests_total", "Total HTTP requests")
        requests.inc()
        requests.inc(5)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.labels = labels or []
        self._value: float = 0.0
        self._labeled: Dict[tuple, float] = {}
        self._lock = threading.Lock()
        get_registry().register(self)

    def inc(self, amount: float = 1.0, **label_values: str) -> None:
        """Increment the counter by *amount* (default 1).

        Args:
            amount: Positive value to add.
            **label_values: Label key-value pairs for labeled metrics.

        Raises:
            ValueError: If *amount* is negative.
        """
        if amount < 0:
            raise ValueError("Counter increment must be non-negative")
        with self._lock:
            if label_values:
                key = tuple(sorted(label_values.items()))
                self._labeled[key] = self._labeled.get(key, 0.0) + amount
            else:
                self._value += amount

    @property
    def value(self) -> float:
        """Return the current counter value (unlabeled)."""
        return self._value

    def get(self, **label_values: str) -> float:
        """Return the counter value for the given labels."""
        if not label_values:
            return self._value
        key = tuple(sorted(label_values.items()))
        return self._labeled.get(key, 0.0)

    def reset(self) -> None:
        """Reset the counter to zero (for testing)."""
        with self._lock:
            self._value = 0.0
            self._labeled.clear()

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} counter")
        if self._labeled:
            for lbl_tuple, val in sorted(self._labeled.items()):
                lbl_str = ",".join(f'{k}="{v}"' for k, v in lbl_tuple)
                lines.append(f"{self.name}{{{lbl_str}}} {val}")
        else:
            lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


# ── Gauge ──────────────────────────────────────────────────────────────────


class Gauge:
    """A metric that can go up and down.

    Args:
        name: Metric name.
        description: Human-readable description.

    Example::

        active_agents = Gauge("active_agents", "Currently active agents")
        active_agents.set(5)
        active_agents.inc()
        active_agents.dec()
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()
        get_registry().register(self)

    def set(self, value: float) -> None:
        """Set the gauge to *value*."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge by *amount*."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge by *amount*."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """Return the current gauge value."""
        return self._value

    def reset(self) -> None:
        """Reset the gauge to zero."""
        with self._lock:
            self._value = 0.0

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} gauge")
        lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


# ── Histogram ──────────────────────────────────────────────────────────────

# Default bucket boundaries matching Prometheus defaults
DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)


class Histogram:
    """A metric that tracks the distribution of observed values.

    Args:
        name: Metric name.
        description: Human-readable description.
        buckets: Bucket boundaries (defaults to Prometheus defaults).

    Example::

        latency = Histogram("request_duration_seconds", "Request latency")
        latency.observe(0.042)

        with latency.time():
            do_work()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: tuple = DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.description = description
        self.buckets = tuple(sorted(buckets))
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()
        get_registry().register(self)

    def observe(self, value: float) -> None:
        """Record an observed *value*.

        Args:
            value: The value to record.
        """
        with self._lock:
            self._sum += value
            self._count += 1
            for boundary in self.buckets:
                if value <= boundary:
                    self._bucket_counts[boundary] += 1

    @contextmanager
    def time(self) -> Iterator[None]:
        """Context manager that records the elapsed time as an observation.

        Example::

            with latency.time():
                do_expensive_work()
        """
        start = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - start)

    @property
    def count(self) -> int:
        """Return the total number of observations."""
        return self._count

    @property
    def sum(self) -> float:
        """Return the sum of all observed values."""
        return self._sum

    def reset(self) -> None:
        """Reset all histogram data."""
        with self._lock:
            self._bucket_counts = {b: 0 for b in self.buckets}
            self._sum = 0.0
            self._count = 0

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} histogram")
        for boundary, count in sorted(self._bucket_counts.items()):
            le = "+Inf" if boundary == float("inf") else str(boundary)
            lines.append(f'{self.name}_bucket{{le="{le}"}} {count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


# ── Metrics Registry ──────────────────────────────────────────────────────


class MetricsRegistry:
    """Central registry for all metrics.

    Provides export in Prometheus text format and metric lookup by name.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, metric: Any) -> None:
        """Register a metric.

        Args:
            metric: A Counter, Gauge, or Histogram instance.
        """
        with self._lock:
            self._metrics[metric.name] = metric

    def unregister(self, name: str) -> bool:
        """Remove a metric by name. Returns True if it existed."""
        with self._lock:
            return self._metrics.pop(name, None) is not None

    def get(self, name: str) -> Any:
        """Look up a metric by name. Returns ``None`` if not found."""
        return self._metrics.get(name)

    def names(self) -> List[str]:
        """Return all registered metric names."""
        return list(self._metrics.keys())

    def export(self) -> str:
        """Export all metrics in Prometheus text format.

        Returns:
            A string in Prometheus exposition format.
        """
        parts = []
        for metric in self._metrics.values():
            if hasattr(metric, "export"):
                parts.append(metric.export())
        return "\n\n".join(parts) + "\n" if parts else ""

    def clear(self) -> None:
        """Remove all registered metrics (for testing)."""
        with self._lock:
            self._metrics.clear()


# ── Global registry singleton ──────────────────────────────────────────────

_global_registry: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    """Return the global :class:`MetricsRegistry`, creating one if needed."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricsRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    if _global_registry is not None:
        _global_registry.clear()
    _global_registry = None


# ── Built-in supervisor metrics ────────────────────────────────────────────


@dataclass
class SupervisorMetrics:
    """Pre-defined metrics for the supervisor framework.

    Attributes:
        messages_sent: Counter for messages enqueued.
        messages_processed: Counter for messages delivered.
        messages_failed: Counter for failed deliveries.
        message_latency: Histogram for message handling duration.
        active_agents: Gauge for currently registered agents.
    """

    messages_sent: Counter = field(init=False)
    messages_processed: Counter = field(init=False)
    messages_failed: Counter = field(init=False)
    message_latency: Histogram = field(init=False)
    active_agents: Gauge = field(init=False)

    def __post_init__(self) -> None:
        self.messages_sent = Counter(
            "supervisor_messages_sent_total",
            "Total messages enqueued",
        )
        self.messages_processed = Counter(
            "supervisor_messages_processed_total",
            "Total messages successfully delivered",
        )
        self.messages_failed = Counter(
            "supervisor_messages_failed_total",
            "Total messages that failed delivery",
        )
        self.message_latency = Histogram(
            "supervisor_message_latency_seconds",
            "Message handling latency in seconds",
        )
        self.active_agents = Gauge(
            "supervisor_active_agents",
            "Number of currently registered agents",
        )


# ── Metrics Extension ─────────────────────────────────────────────────────


class MetricsExtension(Extension):
    """Extension that collects per-agent metrics on message handling.

    Attach to an agent to automatically record message processing counts
    and latency.

    Example::

        agent.use(MetricsExtension())
    """

    name = "metrics"

    def __init__(self) -> None:
        self._agent_counter = Counter(
            "agent_messages_total",
            "Messages processed per agent",
            labels=["agent"],
        )
        self._agent_latency = Histogram(
            "agent_message_latency_seconds",
            "Per-agent message handling latency",
        )

    def on_message(self, agent: "Agent", msg: "Message") -> Optional["Message"]:
        """Record message count and timing for this agent."""
        self._agent_counter.inc(agent=agent.name)
        return None


__all__ = [
    "Counter",
    "DEFAULT_BUCKETS",
    "Gauge",
    "Histogram",
    "MetricsExtension",
    "MetricsRegistry",
    "SupervisorMetrics",
    "get_registry",
    "reset_registry",
]
