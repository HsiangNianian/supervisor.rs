"""Tests for the tracing module (Phase 5.1 & 5.2)."""

import time

import pytest

from supervisor.tracing import (
    ConsoleExporter,
    InMemoryExporter,
    Span,
    SpanExporter,
    Tracer,
    TracingExtension,
    current_span,
    get_tracer,
    reset_tracer,
    trace,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset the global tracer between tests."""
    reset_tracer()
    yield
    reset_tracer()


# ── Span tests ─────────────────────────────────────────────────────────────


class TestSpan:
    """Tests for the Span dataclass."""

    def test_create_span(self):
        span = Span(name="test_op")
        assert span.name == "test_op"
        assert span.status == "ok"
        assert span.end_time is None
        assert span.duration is None

    def test_end_span(self):
        span = Span(name="test_op")
        time.sleep(0.01)
        span.end()
        assert span.end_time is not None
        assert span.duration is not None
        assert span.duration >= 0.0

    def test_set_attribute(self):
        span = Span(name="test_op")
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_add_event(self):
        span = Span(name="test_op")
        span.add_event("event1", {"detail": "info"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "event1"
        assert span.events[0]["attributes"]["detail"] == "info"

    def test_set_status(self):
        span = Span(name="test_op")
        span.set_status("error", "something failed")
        assert span.status == "error"
        assert span.attributes["status_description"] == "something failed"

    def test_to_dict(self):
        span = Span(name="test_op")
        span.set_attribute("k", "v")
        span.end()
        d = span.to_dict()
        assert d["name"] == "test_op"
        assert d["attributes"]["k"] == "v"
        assert d["duration"] is not None
        assert d["status"] == "ok"

    def test_double_end_is_idempotent(self):
        span = Span(name="test_op")
        span.end()
        first_end = span.end_time
        span.end()
        assert span.end_time == first_end


# ── Exporter tests ─────────────────────────────────────────────────────────


class TestExporters:
    """Tests for span exporters."""

    def test_in_memory_exporter(self):
        exp = InMemoryExporter()
        span = Span(name="op")
        span.end()
        exp.export([span])
        assert len(exp.spans) == 1
        assert exp.spans[0].name == "op"

    def test_in_memory_clear(self):
        exp = InMemoryExporter()
        exp.export([Span(name="op")])
        exp.clear()
        assert len(exp.spans) == 0

    def test_console_exporter(self, caplog):
        exp = ConsoleExporter()
        span = Span(name="op")
        span.end()
        import logging

        with caplog.at_level(logging.INFO, logger="supervisor.tracing"):
            exp.export([span])
        assert "op" in caplog.text

    def test_base_exporter_no_op(self):
        exp = SpanExporter()
        exp.export([Span(name="op")])  # Should not raise
        exp.shutdown()


# ── Tracer tests ───────────────────────────────────────────────────────────


class TestTracer:
    """Tests for the Tracer class."""

    def test_start_span_context_manager(self):
        exp = InMemoryExporter()
        tracer = Tracer("test-service", exp)
        with tracer.start_span("operation") as span:
            span.set_attribute("key", "value")
        assert len(exp.spans) == 1
        assert exp.spans[0].name == "operation"
        assert exp.spans[0].duration is not None

    def test_nested_spans_share_trace_id(self):
        exp = InMemoryExporter()
        tracer = Tracer("test-service", exp)
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.trace_id == parent.trace_id
                assert child.parent_id == parent.span_id

    def test_span_error_status(self):
        exp = InMemoryExporter()
        tracer = Tracer("test-service", exp)
        with pytest.raises(ValueError):
            with tracer.start_span("failing") as span:
                raise ValueError("boom")
        assert exp.spans[0].status == "error"

    def test_service_name_attribute(self):
        exp = InMemoryExporter()
        tracer = Tracer("my-service", exp)
        with tracer.start_span("op") as span:
            pass
        assert exp.spans[0].attributes["service.name"] == "my-service"

    def test_get_completed_spans(self):
        tracer = Tracer("svc")
        with tracer.start_span("op"):
            pass
        assert len(tracer.get_completed_spans()) == 1

    def test_clear(self):
        tracer = Tracer("svc")
        with tracer.start_span("op"):
            pass
        tracer.clear()
        assert len(tracer.get_completed_spans()) == 0


# ── Global tracer tests ───────────────────────────────────────────────────


class TestGlobalTracer:
    """Tests for the global tracer singleton."""

    def test_get_tracer_singleton(self):
        t1 = get_tracer("svc")
        t2 = get_tracer("svc")
        assert t1 is t2

    def test_reset_tracer(self):
        t1 = get_tracer("svc")
        reset_tracer()
        t2 = get_tracer("svc2")
        assert t1 is not t2

    def test_current_span_none_by_default(self):
        assert current_span() is None

    def test_current_span_inside_trace(self):
        tracer = get_tracer()
        with tracer.start_span("op") as span:
            assert current_span() is span
        assert current_span() is None


# ── trace() convenience tests ─────────────────────────────────────────────


class TestTrace:
    """Tests for the trace() convenience function."""

    def test_trace_context_manager(self):
        tracer = get_tracer()
        with trace("test_op") as span:
            assert span.name == "test_op"
        assert len(tracer.get_completed_spans()) >= 1

    def test_trace_decorator_with_name(self):
        @trace("my_func")
        def work():
            return 42

        result = work()
        assert result == 42
        tracer = get_tracer()
        spans = tracer.get_completed_spans()
        assert any(s.name == "my_func" for s in spans)

    def test_trace_bare_decorator(self):
        @trace
        def another_func():
            return "hello"

        result = another_func()
        assert result == "hello"
        tracer = get_tracer()
        spans = tracer.get_completed_spans()
        names = [s.name for s in spans]
        assert any("another_func" in n for n in names)


# ── TracingExtension tests ────────────────────────────────────────────────


class TestTracingExtension:
    """Tests for the TracingExtension."""

    def test_extension_name(self):
        ext = TracingExtension()
        assert ext.name == "tracing"

    def test_extension_creates_spans(self):
        from supervisor._core import Message

        exp = InMemoryExporter()
        tracer = Tracer("test", exp)
        ext = TracingExtension(tracer=tracer)

        class FakeAgent:
            name = "test_agent"

        msg = Message("alice", "test_agent", "hello")
        ext.on_message(FakeAgent(), msg)

        assert len(exp.spans) == 1
        assert "test_agent" in exp.spans[0].name
        assert exp.spans[0].attributes["agent.name"] == "test_agent"

    def test_extension_on_message_returns_none(self):
        from supervisor._core import Message

        ext = TracingExtension()

        class FakeAgent:
            name = "a"

        result = ext.on_message(FakeAgent(), Message("a", "b", "c"))
        assert result is None
