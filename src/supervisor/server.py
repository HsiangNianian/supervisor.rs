"""HTTP and WebSocket server for the supervisor framework.

Provides a FastAPI-based HTTP server with REST endpoints and optional
WebSocket support for real-time message streaming.

Example::

    from supervisor.server import create_app, run_server

    app = create_app()
    run_server(app, host="0.0.0.0", port=8000)
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger("supervisor.server")


# ── Request/Response models ────────────────────────────────────────────────


class MessageRequest(BaseModel):
    """Request body for sending a message.

    Attributes:
        sender: Name of the sending agent.
        recipient: Name of the target agent.
        content: Message payload.
    """

    sender: str
    recipient: str
    content: str


class MessageResponse(BaseModel):
    """Response after sending a message.

    Attributes:
        status: Delivery status (``"queued"`` or ``"error"``).
        detail: Human-readable detail string.
    """

    status: str = "queued"
    detail: str = ""


class AgentInfo(BaseModel):
    """Information about a registered agent.

    Attributes:
        name: Agent name.
        pending: Number of pending messages.
    """

    name: str
    pending: int = 0


class StatusResponse(BaseModel):
    """Server health/status response.

    Attributes:
        status: Overall status (``"ok"``).
        agent_count: Number of registered agents.
        agents: List of agent information.
    """

    status: str = "ok"
    agent_count: int = 0
    agents: List[AgentInfo] = Field(default_factory=list)


class RunResponse(BaseModel):
    """Response after running one processing cycle.

    Attributes:
        processed: Number of messages processed.
    """

    processed: int = 0


# ── WebSocket connection manager ──────────────────────────────────────────


class ConnectionManager:
    """Manages WebSocket connections for broadcasting real-time updates.

    Thread-safe management of active WebSocket connections with support
    for broadcasting messages to all connected clients.
    """

    def __init__(self) -> None:
        self.active_connections: Set[Any] = set()

    async def connect(self, websocket: Any) -> None:
        """Accept and track a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept.
        """
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: Any) -> None:
        """Remove a WebSocket connection from tracking.

        Args:
            websocket: The connection to remove.
        """
        self.active_connections.discard(websocket)

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected WebSocket clients.

        Args:
            message: JSON string to broadcast.
        """
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)
        self.active_connections -= disconnected


# ── App factory ────────────────────────────────────────────────────────────


def create_app(
    supervisor: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Any:
    """Create a FastAPI application for the supervisor framework.

    Args:
        supervisor: An optional :class:`~supervisor.Supervisor` instance.
            If ``None``, a new one is created.
        config: An optional :class:`~supervisor.config.SupervisorConfig`.

    Returns:
        A FastAPI application instance.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import PlainTextResponse
        from starlette.websockets import WebSocketDisconnect
    except ImportError:
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install fastapi uvicorn"
        )

    from supervisor._core import Message
    from supervisor._core import Supervisor as _CoreSupervisor

    if supervisor is None:
        supervisor = _CoreSupervisor()

    manager = ConnectionManager()

    app = FastAPI(
        title="Supervisor Agent Framework",
        description="HTTP and WebSocket API for the supervisor agent framework",
        version="0.1.0",
    )

    # Store supervisor in app state
    app.state.supervisor = supervisor
    app.state.config = config

    # CORS
    cors_origins = ["*"]
    if config and hasattr(config, "server"):
        cors_origins = config.server.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=StatusResponse)
    async def health() -> StatusResponse:
        """Health check endpoint returning server and agent status."""
        sup = app.state.supervisor
        agents = []
        for name in sup.agent_names():
            pending = sup.pending_count(name)
            agents.append(AgentInfo(name=name, pending=pending or 0))
        return StatusResponse(
            status="ok",
            agent_count=sup.agent_count(),
            agents=agents,
        )

    @app.get("/agents", response_model=List[AgentInfo])
    async def list_agents() -> List[AgentInfo]:
        """List all registered agents with their pending message counts."""
        sup = app.state.supervisor
        result = []
        for name in sup.agent_names():
            pending = sup.pending_count(name)
            result.append(AgentInfo(name=name, pending=pending or 0))
        return result

    @app.post("/send", response_model=MessageResponse)
    async def send_message(req: MessageRequest) -> MessageResponse:
        """Send a message to a registered agent.

        The message is enqueued for delivery on the next ``run_once`` cycle.
        """
        sup = app.state.supervisor
        try:
            sup.send(Message(req.sender, req.recipient, req.content))
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "message_sent",
                        "sender": req.sender,
                        "recipient": req.recipient,
                        "content": req.content,
                    }
                )
            )
            return MessageResponse(status="queued", detail="Message enqueued")
        except KeyError as e:
            return MessageResponse(status="error", detail=str(e))

    @app.post("/run", response_model=RunResponse)
    async def run_once() -> RunResponse:
        """Process all pending messages in a single cycle."""
        sup = app.state.supervisor
        processed = sup.run_once()
        await manager.broadcast(
            json.dumps({"type": "run_complete", "processed": processed})
        )
        return RunResponse(processed=processed)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint() -> str:
        """Export metrics in Prometheus text format."""
        try:
            from supervisor.metrics import get_registry

            return get_registry().export()
        except ImportError:
            return ""

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time message streaming.

        Clients connect and receive JSON-formatted event notifications
        for message sends and processing cycles.
        """
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg_data = json.loads(data)
                    if msg_data.get("type") == "send":
                        sup = app.state.supervisor
                        sup.send(
                            Message(
                                msg_data["sender"],
                                msg_data["recipient"],
                                msg_data["content"],
                            )
                        )
                        await manager.broadcast(
                            json.dumps(
                                {
                                    "type": "message_sent",
                                    "sender": msg_data["sender"],
                                    "recipient": msg_data["recipient"],
                                    "content": msg_data["content"],
                                }
                            )
                        )
                    elif msg_data.get("type") == "run":
                        sup = app.state.supervisor
                        processed = sup.run_once()
                        await manager.broadcast(
                            json.dumps(
                                {
                                    "type": "run_complete",
                                    "processed": processed,
                                }
                            )
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": str(e)})
                    )
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app


def run_server(
    app: Optional[Any] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    config: Optional[Any] = None,
) -> None:
    """Start the HTTP server (blocking).

    Args:
        app: FastAPI app instance. If ``None``, one is created.
        host: Bind address.
        port: Bind port.
        config: Optional :class:`~supervisor.config.SupervisorConfig`.

    Raises:
        ImportError: If uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install uvicorn"
        )

    if app is None:
        app = create_app(config=config)

    if config is not None:
        host = config.server.host
        port = config.server.port

    uvicorn.run(app, host=host, port=port)


__all__ = [
    "AgentInfo",
    "ConnectionManager",
    "MessageRequest",
    "MessageResponse",
    "RunResponse",
    "StatusResponse",
    "create_app",
    "run_server",
]
