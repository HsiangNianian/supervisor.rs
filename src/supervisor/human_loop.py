"""Human-in-the-loop collaboration for agent workflows.

Provides mechanisms for agents to pause execution and request human
input, approval, or review before continuing.

Example::

    from supervisor.human_loop import HumanApproval, HumanInput

    # Create an approval gate
    gate = HumanApproval(prompt="Deploy to production?")
    result = gate.request()
    if result.approved:
        deploy()

    # Create an input request
    inp = HumanInput(prompt="Enter target URL:")
    response = inp.request()
    print(response.value)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from supervisor.ext import Extension


class ReviewStatus(str, Enum):
    """Status of a human review request.

    Attributes:
        PENDING: Waiting for human response.
        APPROVED: Approved by human.
        REJECTED: Rejected by human.
        TIMEOUT: Timed out waiting.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class HumanResponse:
    """Response from a human reviewer.

    Attributes:
        request_id: Unique identifier for the original request.
        status: Review status.
        value: Human-provided value (for input requests).
        comment: Optional reviewer comment.
        reviewer: Name or ID of the reviewer.
    """

    request_id: str
    status: ReviewStatus = ReviewStatus.PENDING
    value: str = ""
    comment: str = ""
    reviewer: str = ""


@dataclass
class HumanRequest:
    """A request for human input or approval.

    Attributes:
        request_id: Unique identifier.
        request_type: Type of request (``"approval"``, ``"input"``, ``"review"``).
        prompt: Message displayed to the human.
        context: Additional context data.
        options: Available choices (for approval: ``["approve", "reject"]``).
        response: The human's response, once provided.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    request_type: str = "approval"
    prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=lambda: ["approve", "reject"])
    response: Optional[HumanResponse] = None

    @property
    def is_resolved(self) -> bool:
        """Whether the request has been answered."""
        return self.response is not None and self.response.status != ReviewStatus.PENDING


class HumanApproval:
    """Gate that pauses until a human approves or rejects.

    Args:
        prompt: Message to display to the reviewer.
        context: Additional context data for the reviewer.

    Example::

        gate = HumanApproval("Deploy to production?")
        result = gate.request()
        if result.approved:
            deploy()
    """

    def __init__(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.prompt = prompt
        self.context = context or {}
        self._pending: Dict[str, HumanRequest] = {}

    def request(self) -> HumanRequest:
        """Create and register a new approval request.

        Returns:
            A :class:`HumanRequest` in pending state.
        """
        req = HumanRequest(
            request_type="approval",
            prompt=self.prompt,
            context=self.context,
            options=["approve", "reject"],
        )
        self._pending[req.request_id] = req
        return req

    def respond(self, request_id: str, approved: bool, comment: str = "", reviewer: str = "") -> HumanResponse:
        """Respond to a pending approval request.

        Args:
            request_id: The ID of the request to respond to.
            approved: Whether to approve.
            comment: Optional comment.
            reviewer: Name of the reviewer.

        Returns:
            The :class:`HumanResponse`.

        Raises:
            KeyError: If the request ID is not found.
        """
        if request_id not in self._pending:
            raise KeyError(f"Request {request_id} not found")

        status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        response = HumanResponse(
            request_id=request_id,
            status=status,
            comment=comment,
            reviewer=reviewer,
        )
        self._pending[request_id].response = response
        return response

    @property
    def pending_requests(self) -> List[HumanRequest]:
        """Return all unresolved requests."""
        return [r for r in self._pending.values() if not r.is_resolved]


class HumanInput:
    """Gate that pauses until a human provides text input.

    Args:
        prompt: Message displayed to the human.
        context: Additional context data.

    Example::

        inp = HumanInput("Enter the target URL:")
        req = inp.request()
        # ... later, when human responds:
        inp.respond(req.request_id, "https://example.com")
    """

    def __init__(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.prompt = prompt
        self.context = context or {}
        self._pending: Dict[str, HumanRequest] = {}

    def request(self) -> HumanRequest:
        """Create and register a new input request.

        Returns:
            A :class:`HumanRequest` in pending state.
        """
        req = HumanRequest(
            request_type="input",
            prompt=self.prompt,
            context=self.context,
            options=[],
        )
        self._pending[req.request_id] = req
        return req

    def respond(self, request_id: str, value: str, reviewer: str = "") -> HumanResponse:
        """Provide input for a pending request.

        Args:
            request_id: The ID of the request.
            value: The human-provided input value.
            reviewer: Name of the person providing input.

        Returns:
            The :class:`HumanResponse`.

        Raises:
            KeyError: If the request ID is not found.
        """
        if request_id not in self._pending:
            raise KeyError(f"Request {request_id} not found")

        response = HumanResponse(
            request_id=request_id,
            status=ReviewStatus.APPROVED,
            value=value,
            reviewer=reviewer,
        )
        self._pending[request_id].response = response
        return response

    @property
    def pending_requests(self) -> List[HumanRequest]:
        """Return all unresolved requests."""
        return [r for r in self._pending.values() if not r.is_resolved]


class HumanReview:
    """Gate for content review with approve/reject/edit options.

    Args:
        prompt: Review prompt message.
        content: The content to be reviewed.

    Example::

        review = HumanReview("Review this draft:", draft_text)
        req = review.request()
        review.respond(req.request_id, approved=True, comment="Looks good!")
    """

    def __init__(self, prompt: str, content: str = "") -> None:
        self.prompt = prompt
        self.content = content
        self._pending: Dict[str, HumanRequest] = {}

    def request(self) -> HumanRequest:
        """Create a review request.

        Returns:
            A :class:`HumanRequest` in pending state.
        """
        req = HumanRequest(
            request_type="review",
            prompt=self.prompt,
            context={"content": self.content},
            options=["approve", "reject", "edit"],
        )
        self._pending[req.request_id] = req
        return req

    def respond(
        self,
        request_id: str,
        approved: bool,
        comment: str = "",
        edited_content: str = "",
        reviewer: str = "",
    ) -> HumanResponse:
        """Respond to a review request.

        Args:
            request_id: The request ID.
            approved: Whether to approve.
            comment: Reviewer comment.
            edited_content: Revised content (if editing).
            reviewer: Reviewer name.

        Returns:
            The :class:`HumanResponse`.
        """
        if request_id not in self._pending:
            raise KeyError(f"Request {request_id} not found")

        status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        response = HumanResponse(
            request_id=request_id,
            status=status,
            value=edited_content or self.content,
            comment=comment,
            reviewer=reviewer,
        )
        self._pending[request_id].response = response
        return response

    @property
    def pending_requests(self) -> List[HumanRequest]:
        """Return all unresolved requests."""
        return [r for r in self._pending.values() if not r.is_resolved]


class HumanLoopExtension(Extension):
    """Extension for human-in-the-loop agent workflows.

    Provides a callback-based mechanism for agents to request human input
    during message processing.

    Args:
        callback: Optional callback function invoked on human requests.

    Example::

        def on_human_request(request):
            if request.request_type == "approval":
                return True  # auto-approve

        agent.use(HumanLoopExtension(callback=on_human_request))
    """

    name = "human_loop"

    def __init__(self, callback: Optional[Callable] = None) -> None:
        self.callback = callback
        self.requests: List[HumanRequest] = []

    def create_approval(self, prompt: str, **context: Any) -> HumanRequest:
        """Create an approval request within an agent workflow.

        Args:
            prompt: Approval prompt.
            **context: Additional context.

        Returns:
            A :class:`HumanRequest`.
        """
        req = HumanRequest(
            request_type="approval",
            prompt=prompt,
            context=context,
        )
        self.requests.append(req)
        if self.callback:
            result = self.callback(req)
            if result is True:
                req.response = HumanResponse(
                    request_id=req.request_id,
                    status=ReviewStatus.APPROVED,
                )
            elif result is False:
                req.response = HumanResponse(
                    request_id=req.request_id,
                    status=ReviewStatus.REJECTED,
                )
        return req

    def create_input(self, prompt: str, **context: Any) -> HumanRequest:
        """Create an input request within an agent workflow.

        Args:
            prompt: Input prompt.
            **context: Additional context.

        Returns:
            A :class:`HumanRequest`.
        """
        req = HumanRequest(
            request_type="input",
            prompt=prompt,
            context=context,
            options=[],
        )
        self.requests.append(req)
        if self.callback:
            result = self.callback(req)
            if isinstance(result, str):
                req.response = HumanResponse(
                    request_id=req.request_id,
                    status=ReviewStatus.APPROVED,
                    value=result,
                )
        return req

    @property
    def pending(self) -> List[HumanRequest]:
        """Return all unresolved requests."""
        return [r for r in self.requests if not r.is_resolved]


__all__ = [
    "HumanApproval",
    "HumanInput",
    "HumanLoopExtension",
    "HumanRequest",
    "HumanResponse",
    "HumanReview",
    "ReviewStatus",
]
