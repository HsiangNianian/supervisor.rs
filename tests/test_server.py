"""Tests for the server, config, and CLI modules (Phase 6)."""

import json
import pytest

from supervisor.config import (
    AgentConfig,
    LoggingConfig,
    MetricsConfig,
    ServerConfig,
    SupervisorConfig,
    TracingConfig,
)


# ── Config tests ───────────────────────────────────────────────────────────


class TestServerConfig:
    """Tests for the ServerConfig model."""

    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.cors_origins == ["*"]
        assert cfg.websocket_enabled is True

    def test_custom_values(self):
        cfg = ServerConfig(host="127.0.0.1", port=9000)
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9000


class TestLoggingConfig:
    """Tests for the LoggingConfig model."""

    def test_defaults(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == "json"


class TestTracingConfig:
    """Tests for the TracingConfig model."""

    def test_defaults(self):
        cfg = TracingConfig()
        assert cfg.enabled is False
        assert cfg.exporter == "console"
        assert cfg.service_name == "supervisor"


class TestMetricsConfig:
    """Tests for the MetricsConfig model."""

    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.enabled is False
        assert cfg.endpoint == "/metrics"


class TestAgentConfig:
    """Tests for the AgentConfig model."""

    def test_minimal(self):
        cfg = AgentConfig(name="test")
        assert cfg.name == "test"
        assert cfg.class_path == ""
        assert cfg.extensions == []
        assert cfg.settings == {}

    def test_full(self):
        cfg = AgentConfig(
            name="chat",
            class_path="myapp.ChatAgent",
            extensions=["supervisor.ext.rag.RAGExtension"],
            settings={"temperature": 0.7},
        )
        assert cfg.class_path == "myapp.ChatAgent"
        assert len(cfg.extensions) == 1


class TestSupervisorConfig:
    """Tests for the SupervisorConfig model."""

    def test_defaults(self):
        cfg = SupervisorConfig()
        assert cfg.name == "supervisor"
        assert cfg.agents == []
        assert cfg.server.port == 8000

    def test_from_dict(self):
        data = {
            "name": "my-supervisor",
            "server": {"host": "localhost", "port": 9000},
            "agents": [{"name": "agent1", "class_path": "myapp.Agent1"}],
        }
        cfg = SupervisorConfig.from_dict(data)
        assert cfg.name == "my-supervisor"
        assert cfg.server.port == 9000
        assert len(cfg.agents) == 1
        assert cfg.agents[0].name == "agent1"

    def test_to_dict(self):
        cfg = SupervisorConfig(name="test")
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert "server" in d
        assert "agents" in d

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SUPERVISOR_SERVER_PORT", "3000")
        cfg = SupervisorConfig.from_dict({"name": "test"})
        assert cfg.server.port == 3000

    def test_env_override_host(self, monkeypatch):
        monkeypatch.setenv("SUPERVISOR_SERVER_HOST", "192.168.1.1")
        cfg = SupervisorConfig.from_dict({"name": "test"})
        assert cfg.server.host == "192.168.1.1"

    def test_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SupervisorConfig.from_yaml("/nonexistent/file.yaml")

    def test_yaml_loading(self, tmp_path):
        """Test YAML config loading (requires pyyaml)."""
        yaml_content = """
name: test-supervisor
server:
  host: localhost
  port: 9999
agents:
  - name: echo
    class_path: supervisor.__main__.EchoAgent
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        try:
            cfg = SupervisorConfig.from_yaml(str(yaml_file))
            assert cfg.name == "test-supervisor"
            assert cfg.server.port == 9999
            assert len(cfg.agents) == 1
        except ImportError:
            pytest.skip("PyYAML not installed")


# ── Server models tests ───────────────────────────────────────────────────


class TestServerModels:
    """Tests for server request/response models."""

    def test_message_request(self):
        from supervisor.server import MessageRequest

        req = MessageRequest(sender="alice", recipient="bob", content="hello")
        assert req.sender == "alice"

    def test_message_response(self):
        from supervisor.server import MessageResponse

        resp = MessageResponse(status="queued", detail="ok")
        assert resp.status == "queued"

    def test_agent_info(self):
        from supervisor.server import AgentInfo

        info = AgentInfo(name="test", pending=5)
        assert info.name == "test"
        assert info.pending == 5

    def test_status_response(self):
        from supervisor.server import StatusResponse

        resp = StatusResponse(status="ok", agent_count=3)
        assert resp.agent_count == 3

    def test_run_response(self):
        from supervisor.server import RunResponse

        resp = RunResponse(processed=10)
        assert resp.processed == 10


# ── ConnectionManager tests ──────────────────────────────────────────────


class TestConnectionManager:
    """Tests for the WebSocket ConnectionManager."""

    def test_disconnect(self):
        from supervisor.server import ConnectionManager

        mgr = ConnectionManager()
        # Disconnect a connection that was never added (should not raise)
        mgr.disconnect("fake")
        assert len(mgr.active_connections) == 0


# ── Server app creation tests ────────────────────────────────────────────


class TestCreateApp:
    """Tests for the create_app factory."""

    def test_create_app_default(self):
        """Test that create_app returns a FastAPI app."""
        try:
            from supervisor.server import create_app

            app = create_app()
            assert app is not None
            # Check routes exist
            routes = [r.path for r in app.routes]
            assert "/health" in routes
            assert "/agents" in routes
            assert "/send" in routes
            assert "/run" in routes
            assert "/metrics" in routes
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_create_app_with_supervisor(self):
        """Test app creation with an existing supervisor."""
        try:
            from supervisor._core import Supervisor
            from supervisor.server import create_app

            sup = Supervisor()
            app = create_app(supervisor=sup)
            assert app.state.supervisor is sup
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_create_app_with_config(self):
        """Test app creation with config."""
        try:
            from supervisor.server import create_app

            cfg = SupervisorConfig(
                server=ServerConfig(cors_origins=["http://localhost:3000"])
            )
            app = create_app(config=cfg)
            assert app.state.config is cfg
        except ImportError:
            pytest.skip("FastAPI not installed")


# ── HTTP endpoint tests (using TestClient) ────────────────────────────────


class TestHTTPEndpoints:
    """Integration tests for HTTP endpoints using FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient

            from supervisor._core import Supervisor
            from supervisor.server import create_app

            sup = Supervisor()
            sup.register("echo", lambda msg: None)
            app = create_app(supervisor=sup)
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["agent_count"] == 1

    def test_list_agents(self, client):
        resp = client.get("/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert len(agents) == 1
        assert agents[0]["name"] == "echo"

    def test_send_message(self, client):
        resp = client.post(
            "/send",
            json={"sender": "test", "recipient": "echo", "content": "hi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"

    def test_send_to_unknown_agent(self, client):
        resp = client.post(
            "/send",
            json={"sender": "test", "recipient": "unknown", "content": "hi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    def test_run_once(self, client):
        # Send a message first
        client.post(
            "/send",
            json={"sender": "test", "recipient": "echo", "content": "hi"},
        )
        resp = client.post("/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["processed"] >= 1

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


# ── WebSocket tests ──────────────────────────────────────────────────────


class TestWebSocket:
    """Tests for the WebSocket endpoint."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient

            from supervisor._core import Supervisor
            from supervisor.server import create_app

            sup = Supervisor()
            sup.register("echo", lambda msg: None)
            app = create_app(supervisor=sup)
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_websocket_send(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "send",
                        "sender": "ws_client",
                        "recipient": "echo",
                        "content": "hello from ws",
                    }
                )
            )
            data = ws.receive_text()
            parsed = json.loads(data)
            assert parsed["type"] == "message_sent"

    def test_websocket_run(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "run"}))
            data = ws.receive_text()
            parsed = json.loads(data)
            assert parsed["type"] == "run_complete"

    def test_websocket_invalid_json(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            data = ws.receive_text()
            parsed = json.loads(data)
            assert parsed["type"] == "error"
