"""Observability: tracing and span management for the supervisor framework.

Provides a lightweight tracing system with context manager and decorator APIs
compatible with OpenTelemetry concepts.  Spans track agent execution, message
routing, and custom operations with timing and metadata.

Example::

    from supervisor.tracing import trace, get_tracer, TracingExtension

    # Context-manager style
    with trace("agent_execution") as span:
        span.set_attribute("agent", "greeter")
        agent.handle_message(msg)

    # Decorator style
    @trace("process_data")
    def process(data):
        return transform(data)

    # Extension: auto-trace every message dispatch
    agent.use(TracingExtension())
"""

from __future__ import annotations

import contextvars
import functools
import logging
import time
import uuid
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
    Union,
)

from supervisor.ext import Extension

if TYPE_CHECKING:
    from supervisor._core import Message
    from supervisor.agent import Agent

logger = logging.getLogger("supervisor.tracing")

F = TypeVar("F", bound=Callable[..., Any])

# ── Context variable for current span ──────────────────────────────────────

_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "current_span", default=None
)


# ── Span ───────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """A single unit of traced work.

    Attributes:
        name: Human-readable operation name.
        trace_id: Unique identifier for the overall trace.
        span_id: Unique identifier for this span.
        parent_id: Span id of the parent span (if any).
        start_time: Timestamp when the span started (seconds since epoch).
        end_time: Timestamp when the span ended, or ``None`` if still open.
        attributes: Key-value metadata attached to the span.
        status: ``"ok"`` or ``"error"``.
        events: Timestamped log entries within the span.
    """

    name: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Return the span duration in seconds, or ``None`` if not ended."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a key-value attribute on this span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record a timestamped event within this span."""
        self.events.append(
            {"name": name, "timestamp": time.time(), "attributes": attributes or {}}
        )

    def set_status(self, status: str, description: str = "") -> None:
        """Set the span status to ``'ok'`` or ``'error'``."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self) -> None:
        """End the span and record its end time."""
        if self.end_time is None:
            self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Export the span as a plain dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "attributes": dict(self.attributes),
            "status": self.status,
            "events": list(self.events),
        }


# ── Span Exporter ──────────────────────────────────────────────────────────


class SpanExporter:
    """Base class for span exporters.

    Subclass and override :meth:`export` to send spans to an external system.
    """

    def export(self, spans: List[Span]) -> None:
        """Export a batch of completed spans."""

    def shutdown(self) -> None:
        """Clean up resources."""


class InMemoryExporter(SpanExporter):
    """Stores spans in memory for testing and debugging.

    Attributes:
        spans: List of all exported spans.
    """

    def __init__(self) -> None:
        self.spans: List[Span] = []

    def export(self, spans: List[Span]) -> None:
        """Append spans to the in-memory list."""
        self.spans.extend(spans)

    def clear(self) -> None:
        """Remove all stored spans."""
        self.spans.clear()

    def shutdown(self) -> None:
        """No-op for in-memory exporter."""


class ConsoleExporter(SpanExporter):
    """Prints spans to the console as structured logs."""

    def export(self, spans: List[Span]) -> None:
        """Print each span to the logger."""
        for span in spans:
            logger.info(
                "Span(name=%s, duration=%.4fs, status=%s, attributes=%s)",
                span.name,
                span.duration or 0.0,
                span.status,
                span.attributes,
            )


# ── Tracer ─────────────────────────────────────────────────────────────────


class Tracer:
    """Manages span creation, context propagation, and export.

    Parameters:
        service_name: Logical name for this service/component.
        exporter: Span exporter to use.

    Example::

        tracer = Tracer("my-service", InMemoryExporter())
        with tracer.start_span("work") as span:
            span.set_attribute("key", "value")
    """

    def __init__(
        self,
        service_name: str = "supervisor",
        exporter: Optional[SpanExporter] = None,
    ) -> None:
        self.service_name = service_name
        self.exporter = exporter or InMemoryExporter()
        self._spans: List[Span] = []

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        """Create and manage a span as a context manager.

        The span is automatically ended when the context exits, and any
        exception sets the span status to ``'error'``.

        Args:
            name: Operation name for the span.
            attributes: Optional initial attributes.

        Yields:
            The created :class:`Span`.
        """
        parent = _current_span.get()
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
        )
        if attributes:
            span.attributes.update(attributes)
        span.set_attribute("service.name", self.service_name)

        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.set_status("error", str(exc))
            raise
        finally:
            span.end()
            _current_span.reset(token)
            self._spans.append(span)
            self.exporter.export([span])

    def get_completed_spans(self) -> List[Span]:
        """Return all completed spans tracked by this tracer."""
        return list(self._spans)

    def clear(self) -> None:
        """Clear all tracked spans."""
        self._spans.clear()

    def shutdown(self) -> None:
        """Shut down the exporter."""
        self.exporter.shutdown()


# ── Global tracer singleton ────────────────────────────────────────────────

_global_tracer: Optional[Tracer] = None


def get_tracer(
    service_name: str = "supervisor",
    exporter: Optional[SpanExporter] = None,
) -> Tracer:
    """Return the global :class:`Tracer` instance, creating one if needed.

    Args:
        service_name: Service name for the tracer.
        exporter: Optional exporter override.

    Returns:
        The global tracer instance.
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(service_name, exporter)
    return _global_tracer


def reset_tracer() -> None:
    """Reset the global tracer (useful for tests)."""
    global _global_tracer
    if _global_tracer is not None:
        _global_tracer.shutdown()
    _global_tracer = None


def current_span() -> Optional[Span]:
    """Return the currently active span, or ``None``."""
    return _current_span.get()


# ── trace() convenience ───────────────────────────────────────────────────


def trace(
    name_or_func: Union[str, Callable[..., Any], None] = None,
    **attrs: Any,
) -> Any:
    """Trace a block of code or decorate a function.

    Can be used three ways:

    1. As a context manager::

        with trace("operation") as span:
            do_work()

    2. As a decorator with name::

        @trace("process")
        def process(x):
            return x * 2

    3. As a bare decorator::

        @trace
        def process(x):
            return x * 2

    Args:
        name_or_func: Operation name (str) or function to decorate.
        **attrs: Attributes to attach to the span.
    """
    tracer = get_tracer()

    if callable(name_or_func):
        # @trace (without arguments)
        func = name_or_func
        op_name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_span(op_name, attributes=attrs) as span:
                span.set_attribute("function", func.__name__)
                return func(*args, **kwargs)

        return wrapper

    # @trace("name") or with trace("name")
    op_name = name_or_func or "unnamed"

    class _TraceContext:
        """Dual-purpose: acts as both decorator and context manager."""

        def __call__(self, func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_span(op_name, attributes=attrs) as span:
                    span.set_attribute("function", func.__name__)
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        def __enter__(self) -> Span:
            self._cm = tracer.start_span(op_name, attributes=attrs)
            return self._cm.__enter__()

        def __exit__(self, *exc_info: Any) -> None:
            self._cm.__exit__(*exc_info)

    return _TraceContext()


# ── Tracing Extension ─────────────────────────────────────────────────────


class TracingExtension(Extension):
    """Extension that auto-traces every message dispatch.

    Attach to an agent to automatically create spans for each message
    handled by the agent.

    Example::

        agent.use(TracingExtension())
    """

    name = "tracing"

    def __init__(self, tracer: Optional[Tracer] = None) -> None:
        self._tracer = tracer

    @property
    def tracer(self) -> Tracer:
        """The tracer instance used by this extension."""
        return self._tracer or get_tracer()

    def on_message(self, agent: "Agent", msg: "Message") -> Optional["Message"]:
        """Wrap the message in a tracing span before dispatch.

        The span records agent name, sender, recipient, and message content
        length as attributes.
        """
        with self.tracer.start_span(
            f"agent.{agent.name}.handle_message",
            attributes={
                "agent.name": agent.name,
                "message.sender": msg.sender,
                "message.recipient": msg.recipient,
                "message.content_length": len(msg.content),
            },
        ):
            pass  # The actual handling happens after this hook returns
        return None


__all__ = [
    "ConsoleExporter",
    "InMemoryExporter",
    "Span",
    "SpanExporter",
    "Tracer",
    "TracingExtension",
    "current_span",
    "get_tracer",
    "reset_tracer",
    "trace",
]
