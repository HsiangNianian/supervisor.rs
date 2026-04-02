"""Command-line interface for the supervisor framework.

Provides commands for running supervisors, starting the HTTP server,
validating configuration, and displaying system information.

Example::

    # Start the HTTP server
    supervisor-cli serve --config supervisor.yaml

    # Validate a configuration file
    supervisor-cli config validate supervisor.yaml

    # Show system information
    supervisor-cli info
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional


def _get_app() -> "Any":
    """Create the Typer CLI application.

    Returns:
        A Typer application instance.

    Raises:
        ImportError: If typer is not installed.
    """
    try:
        import typer
    except ImportError:
        raise ImportError(
            "typer is required for the CLI. Install with: pip install typer"
        )

    app = typer.Typer(
        name="supervisor-cli",
        help="Supervisor Agent Framework CLI",
        add_completion=False,
    )

    @app.command()
    def serve(
        config: Optional[str] = typer.Option(
            None, "--config", "-c", help="Path to YAML config file"
        ),
        host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
        port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    ) -> None:
        """Start the HTTP/WebSocket server."""
        from supervisor.config import SupervisorConfig
        from supervisor.server import create_app, run_server

        if config:
            cfg = SupervisorConfig.from_yaml(config)
        else:
            cfg = SupervisorConfig(server={"host": host, "port": port})

        cfg.server.host = host
        cfg.server.port = port

        typer.echo(f"Starting supervisor server on {host}:{port}")
        app_instance = create_app(config=cfg)
        run_server(app_instance, host=host, port=port, config=cfg)

    @app.command("config")
    def config_cmd(
        action: str = typer.Argument(
            "validate", help="Action: validate, show"
        ),
        path: Optional[str] = typer.Argument(None, help="Config file path"),
    ) -> None:
        """Validate or display a configuration file."""
        if action == "validate":
            if not path:
                typer.echo("Error: config file path required", err=True)
                raise typer.Exit(1)
            try:
                from supervisor.config import SupervisorConfig

                cfg = SupervisorConfig.from_yaml(path)
                typer.echo(f"✓ Configuration valid: {path}")
                typer.echo(f"  Name: {cfg.name}")
                typer.echo(f"  Agents: {len(cfg.agents)}")
                typer.echo(f"  Server: {cfg.server.host}:{cfg.server.port}")
            except Exception as e:
                typer.echo(f"✗ Configuration error: {e}", err=True)
                raise typer.Exit(1)
        elif action == "show":
            if not path:
                typer.echo("Error: config file path required", err=True)
                raise typer.Exit(1)
            from supervisor.config import SupervisorConfig

            cfg = SupervisorConfig.from_yaml(path)
            typer.echo(json.dumps(cfg.to_dict(), indent=2))
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(1)

    @app.command()
    def info() -> None:
        """Display system and package information."""
        typer.echo("Supervisor Agent Framework")
        typer.echo(f"  Python: {sys.version}")
        typer.echo(f"  Platform: {sys.platform}")

        try:
            import supervisor._core as _core

            typer.echo("  Rust core: available")
        except ImportError:
            typer.echo("  Rust core: NOT available")

        # Check optional dependencies
        for pkg in ["fastapi", "uvicorn", "typer", "pyyaml", "pydantic"]:
            try:
                mod = importlib.import_module(pkg.replace("-", "_"))
                version = getattr(mod, "__version__", "unknown")
                typer.echo(f"  {pkg}: {version}")
            except ImportError:
                typer.echo(f"  {pkg}: not installed")

    @app.command()
    def run(
        config: str = typer.Option(
            ..., "--config", "-c", help="Path to YAML config file"
        ),
    ) -> None:
        """Load agents from config and run a processing cycle."""
        from supervisor._core import Message
        from supervisor._core import Supervisor as _CoreSupervisor
        from supervisor.config import SupervisorConfig

        cfg = SupervisorConfig.from_yaml(config)
        sup = _CoreSupervisor()

        for agent_cfg in cfg.agents:
            if agent_cfg.class_path:
                module_path, class_name = agent_cfg.class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                agent_cls = getattr(module, class_name)
                agent = agent_cls(agent_cfg.name)
                agent.register(sup)
                typer.echo(f"  Registered: {agent_cfg.name} ({agent_cfg.class_path})")

        processed = sup.run_once()
        typer.echo(f"Processed {processed} messages")

    return app


def main() -> None:
    """Entry point for the supervisor-cli command."""
    app = _get_app()
    app()


__all__ = ["main"]
