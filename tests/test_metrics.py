"""Tests for the metrics module (Phase 5.3)."""

import time
import threading

import pytest

from supervisor.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsExtension,
    MetricsRegistry,
    SupervisorMetrics,
    get_registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset the global metrics registry between tests."""
    reset_registry()
    yield
    reset_registry()


# ── Counter tests ──────────────────────────────────────────────────────────


class TestCounter:
    """Tests for the Counter metric."""

    def test_create_counter(self):
        c = Counter("test_counter", "A test counter")
        assert c.name == "test_counter"
        assert c.value == 0.0

    def test_increment(self):
        c = Counter("test_inc", "inc test")
        c.inc()
        assert c.value == 1.0

    def test_increment_by_amount(self):
        c = Counter("test_inc_amount", "amount test")
        c.inc(5.0)
        assert c.value == 5.0

    def test_negative_increment_raises(self):
        c = Counter("test_neg", "neg test")
        with pytest.raises(ValueError, match="non-negative"):
            c.inc(-1)

    def test_labeled_counter(self):
        c = Counter("test_labeled", "labeled test", labels=["agent"])
        c.inc(agent="alice")
        c.inc(agent="alice")
        c.inc(agent="bob")
        assert c.get(agent="alice") == 2.0
        assert c.get(agent="bob") == 1.0
        assert c.value == 0.0  # unlabeled stays 0

    def test_reset(self):
        c = Counter("test_reset_c", "reset test")
        c.inc(10)
        c.reset()
        assert c.value == 0.0

    def test_export_prometheus(self):
        c = Counter("http_requests_total", "Total HTTP requests")
        c.inc(42)
        text = c.export()
        assert "# HELP http_requests_total Total HTTP requests" in text
        assert "# TYPE http_requests_total counter" in text
        assert "http_requests_total 42" in text

    def test_export_labeled(self):
        c = Counter("req_total", "reqs", labels=["method"])
        c.inc(method="GET")
        c.inc(2, method="POST")
        text = c.export()
        assert 'method="GET"' in text
        assert 'method="POST"' in text

    def test_thread_safety(self):
        c = Counter("thread_counter", "thread safe")
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [c.inc() for _ in range(100)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert c.value == 1000.0


# ── Gauge tests ────────────────────────────────────────────────────────────


class TestGauge:
    """Tests for the Gauge metric."""

    def test_create_gauge(self):
        g = Gauge("test_gauge", "A test gauge")
        assert g.value == 0.0

    def test_set(self):
        g = Gauge("test_set_g", "set test")
        g.set(42.0)
        assert g.value == 42.0

    def test_inc_dec(self):
        g = Gauge("test_inc_dec_g", "inc/dec test")
        g.inc()
        g.inc(2.0)
        assert g.value == 3.0
        g.dec()
        assert g.value == 2.0

    def test_reset(self):
        g = Gauge("test_reset_g", "reset test")
        g.set(100)
        g.reset()
        assert g.value == 0.0

    def test_export_prometheus(self):
        g = Gauge("active_connections", "Active conns")
        g.set(5)
        text = g.export()
        assert "# TYPE active_connections gauge" in text
        assert "active_connections 5" in text


# ── Histogram tests ────────────────────────────────────────────────────────


class TestHistogram:
    """Tests for the Histogram metric."""

    def test_create_histogram(self):
        h = Histogram("test_hist", "A test histogram")
        assert h.count == 0
        assert h.sum == 0.0

    def test_observe(self):
        h = Histogram("test_observe_h", "observe test")
        h.observe(0.5)
        h.observe(1.5)
        assert h.count == 2
        assert h.sum == 2.0

    def test_time_context_manager(self):
        h = Histogram("test_time_h", "time test")
        with h.time():
            time.sleep(0.01)
        assert h.count == 1
        assert h.sum > 0.0

    def test_bucket_counting(self):
        h = Histogram("test_buckets_h", "bucket test", buckets=(0.1, 0.5, 1.0, float("inf")))
        h.observe(0.05)  # <= 0.1, 0.5, 1.0, inf
        h.observe(0.3)   # <= 0.5, 1.0, inf
        h.observe(0.8)   # <= 1.0, inf
        h.observe(2.0)   # <= inf
        text = h.export()
        assert 'le="0.1"} 1' in text
        assert 'le="0.5"} 2' in text
        assert 'le="1.0"} 3' in text
        assert 'le="+Inf"} 4' in text

    def test_reset(self):
        h = Histogram("test_reset_h", "reset test")
        h.observe(1.0)
        h.reset()
        assert h.count == 0
        assert h.sum == 0.0

    def test_export_prometheus(self):
        h = Histogram("req_duration", "Request duration", buckets=(0.1, 1.0, float("inf")))
        h.observe(0.05)
        text = h.export()
        assert "# TYPE req_duration histogram" in text
        assert "req_duration_sum" in text
        assert "req_duration_count 1" in text


# ── Registry tests ─────────────────────────────────────────────────────────


class TestMetricsRegistry:
    """Tests for the MetricsRegistry."""

    def test_auto_registration(self):
        c = Counter("auto_reg_c", "auto")
        registry = get_registry()
        assert registry.get("auto_reg_c") is c

    def test_export_all(self):
        Counter("export_c", "counter")
        Gauge("export_g", "gauge")
        registry = get_registry()
        text = registry.export()
        assert "export_c" in text
        assert "export_g" in text

    def test_unregister(self):
        Counter("unreg_c", "unreg")
        registry = get_registry()
        assert registry.unregister("unreg_c")
        assert registry.get("unreg_c") is None

    def test_names(self):
        Counter("names_c1", "c1")
        Counter("names_c2", "c2")
        registry = get_registry()
        names = registry.names()
        assert "names_c1" in names
        assert "names_c2" in names

    def test_clear(self):
        Counter("clear_c", "clear")
        registry = get_registry()
        registry.clear()
        assert len(registry.names()) == 0

    def test_reset_registry(self):
        Counter("reset_reg_c", "reset")
        reset_registry()
        registry = get_registry()
        assert len(registry.names()) == 0


# ── SupervisorMetrics tests ───────────────────────────────────────────────


class TestSupervisorMetrics:
    """Tests for built-in supervisor metrics."""

    def test_create_supervisor_metrics(self):
        sm = SupervisorMetrics()
        assert sm.messages_sent.value == 0
        assert sm.messages_processed.value == 0
        assert sm.messages_failed.value == 0
        assert sm.active_agents.value == 0

    def test_increment_metrics(self):
        sm = SupervisorMetrics()
        sm.messages_sent.inc()
        sm.messages_processed.inc()
        sm.active_agents.set(3)
        assert sm.messages_sent.value == 1
        assert sm.messages_processed.value == 1
        assert sm.active_agents.value == 3

    def test_latency_histogram(self):
        sm = SupervisorMetrics()
        sm.message_latency.observe(0.042)
        assert sm.message_latency.count == 1


# ── MetricsExtension tests ───────────────────────────────────────────────


class TestMetricsExtension:
    """Tests for the MetricsExtension."""

    def test_extension_name(self):
        ext = MetricsExtension()
        assert ext.name == "metrics"

    def test_extension_counts_messages(self):
        from supervisor._core import Message

        ext = MetricsExtension()

        class FakeAgent:
            name = "test_agent"

        msg = Message("alice", "test_agent", "hello")
        ext.on_message(FakeAgent(), msg)
        ext.on_message(FakeAgent(), msg)

        assert ext._agent_counter.get(agent="test_agent") == 2.0

    def test_extension_returns_none(self):
        from supervisor._core import Message

        ext = MetricsExtension()

        class FakeAgent:
            name = "a"

        result = ext.on_message(FakeAgent(), Message("a", "b", "c"))
        assert result is None
