"""Configuration system for the supervisor framework.

Loads configuration from YAML files or dictionaries and provides typed access
to supervisor, agent, and server settings.

Example::

    from supervisor.config import SupervisorConfig

    config = SupervisorConfig.from_yaml("supervisor.yaml")
    print(config.server.host, config.server.port)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for a single agent.

    Attributes:
        name: Unique agent name.
        class_path: Dotted Python path to the agent class.
        extensions: List of extension class paths to load.
        settings: Arbitrary key-value settings for the agent.
    """

    name: str
    class_path: str = ""
    extensions: List[str] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)


class ServerConfig(BaseModel):
    """HTTP/WebSocket server configuration.

    Attributes:
        host: Bind address.
        port: Bind port.
        cors_origins: Allowed CORS origins.
        websocket_enabled: Whether to enable WebSocket endpoint.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    websocket_enabled: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log format string. ``"json"`` for structured JSON logs.
    """

    level: str = "INFO"
    format: str = "json"


class TracingConfig(BaseModel):
    """Tracing / OpenTelemetry configuration.

    Attributes:
        enabled: Whether tracing is active.
        exporter: Exporter type (``"console"``, ``"memory"``, ``"otlp"``).
        endpoint: OTLP collector endpoint URL.
        service_name: Logical service name.
    """

    enabled: bool = False
    exporter: str = "console"
    endpoint: str = ""
    service_name: str = "supervisor"


class MetricsConfig(BaseModel):
    """Metrics configuration.

    Attributes:
        enabled: Whether metrics collection is active.
        endpoint: Path for the Prometheus metrics endpoint.
    """

    enabled: bool = False
    endpoint: str = "/metrics"


class SupervisorConfig(BaseModel):
    """Top-level configuration for the supervisor framework.

    Attributes:
        name: Supervisor instance name.
        agents: List of agent configurations.
        server: HTTP server settings.
        logging: Logging settings.
        tracing: Tracing settings.
        metrics: Metrics settings.
        env_prefix: Environment variable prefix for overrides.
    """

    name: str = "supervisor"
    agents: List[AgentConfig] = Field(default_factory=list)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    env_prefix: str = "SUPERVISOR_"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SupervisorConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A validated :class:`SupervisorConfig` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config. Install with: pip install pyyaml"
            )

        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SupervisorConfig":
        """Create configuration from a dictionary.

        Environment variables with the configured ``env_prefix`` override
        dict values.  For example, ``SUPERVISOR_SERVER_PORT=9000`` overrides
        ``server.port``.

        Args:
            data: Configuration dictionary.

        Returns:
            A validated :class:`SupervisorConfig` instance.
        """
        env_prefix = data.get("env_prefix", "SUPERVISOR_")
        config = cls(**data)
        config = _apply_env_overrides(config, env_prefix)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a plain dictionary."""
        return self.model_dump()


def _apply_env_overrides(config: SupervisorConfig, prefix: str) -> SupervisorConfig:
    """Override config values from environment variables.

    Supports flat env vars like ``SUPERVISOR_SERVER_PORT=9000`` which maps
    to ``config.server.port``.
    """
    port_env = os.environ.get(f"{prefix}SERVER_PORT")
    if port_env is not None:
        config.server.port = int(port_env)

    host_env = os.environ.get(f"{prefix}SERVER_HOST")
    if host_env is not None:
        config.server.host = host_env

    log_level_env = os.environ.get(f"{prefix}LOG_LEVEL")
    if log_level_env is not None:
        config.logging.level = log_level_env.upper()

    name_env = os.environ.get(f"{prefix}NAME")
    if name_env is not None:
        config.name = name_env

    return config


__all__ = [
    "AgentConfig",
    "LoggingConfig",
    "MetricsConfig",
    "ServerConfig",
    "SupervisorConfig",
    "TracingConfig",
]
