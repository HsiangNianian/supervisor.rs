"""Tests for the human-in-the-loop module (Phase 7.2)."""

import pytest

from supervisor.human_loop import (
    HumanApproval,
    HumanInput,
    HumanLoopExtension,
    HumanRequest,
    HumanResponse,
    HumanReview,
    ReviewStatus,
)


class TestReviewStatus:
    """Tests for the ReviewStatus enum."""

    def test_values(self):
        assert ReviewStatus.PENDING == "pending"
        assert ReviewStatus.APPROVED == "approved"
        assert ReviewStatus.REJECTED == "rejected"
        assert ReviewStatus.TIMEOUT == "timeout"


class TestHumanRequest:
    """Tests for the HumanRequest dataclass."""

    def test_create_request(self):
        req = HumanRequest(prompt="Approve?")
        assert req.request_id != ""
        assert req.request_type == "approval"
        assert req.prompt == "Approve?"
        assert not req.is_resolved

    def test_resolved_when_responded(self):
        req = HumanRequest(prompt="Approve?")
        req.response = HumanResponse(
            request_id=req.request_id,
            status=ReviewStatus.APPROVED,
        )
        assert req.is_resolved

    def test_not_resolved_when_pending(self):
        req = HumanRequest(prompt="Approve?")
        req.response = HumanResponse(
            request_id=req.request_id,
            status=ReviewStatus.PENDING,
        )
        assert not req.is_resolved


class TestHumanApproval:
    """Tests for the HumanApproval gate."""

    def test_create_request(self):
        gate = HumanApproval("Deploy to production?")
        req = gate.request()
        assert req.request_type == "approval"
        assert req.prompt == "Deploy to production?"
        assert not req.is_resolved

    def test_approve(self):
        gate = HumanApproval("Deploy?")
        req = gate.request()
        resp = gate.respond(req.request_id, approved=True, comment="LGTM")
        assert resp.status == ReviewStatus.APPROVED
        assert resp.comment == "LGTM"
        assert req.is_resolved

    def test_reject(self):
        gate = HumanApproval("Deploy?")
        req = gate.request()
        resp = gate.respond(req.request_id, approved=False)
        assert resp.status == ReviewStatus.REJECTED

    def test_unknown_request(self):
        gate = HumanApproval("Deploy?")
        with pytest.raises(KeyError):
            gate.respond("nonexistent", approved=True)

    def test_pending_requests(self):
        gate = HumanApproval("Deploy?")
        req1 = gate.request()
        req2 = gate.request()
        assert len(gate.pending_requests) == 2
        gate.respond(req1.request_id, approved=True)
        assert len(gate.pending_requests) == 1

    def test_context_in_request(self):
        gate = HumanApproval("Deploy?", context={"env": "production"})
        req = gate.request()
        assert req.context["env"] == "production"


class TestHumanInput:
    """Tests for the HumanInput gate."""

    def test_create_request(self):
        inp = HumanInput("Enter URL:")
        req = inp.request()
        assert req.request_type == "input"
        assert req.options == []

    def test_provide_input(self):
        inp = HumanInput("Enter URL:")
        req = inp.request()
        resp = inp.respond(req.request_id, "https://example.com")
        assert resp.status == ReviewStatus.APPROVED
        assert resp.value == "https://example.com"

    def test_unknown_request(self):
        inp = HumanInput("Enter URL:")
        with pytest.raises(KeyError):
            inp.respond("nonexistent", "value")

    def test_pending_requests(self):
        inp = HumanInput("Enter value:")
        req = inp.request()
        assert len(inp.pending_requests) == 1
        inp.respond(req.request_id, "hello")
        assert len(inp.pending_requests) == 0


class TestHumanReview:
    """Tests for the HumanReview gate."""

    def test_create_review(self):
        review = HumanReview("Review draft:", "Hello world")
        req = review.request()
        assert req.request_type == "review"
        assert req.context["content"] == "Hello world"
        assert "approve" in req.options

    def test_approve_review(self):
        review = HumanReview("Review:", "content")
        req = review.request()
        resp = review.respond(req.request_id, approved=True, comment="Good")
        assert resp.status == ReviewStatus.APPROVED
        assert resp.value == "content"

    def test_reject_review(self):
        review = HumanReview("Review:", "content")
        req = review.request()
        resp = review.respond(req.request_id, approved=False, comment="Needs work")
        assert resp.status == ReviewStatus.REJECTED

    def test_edit_review(self):
        review = HumanReview("Review:", "draft content")
        req = review.request()
        resp = review.respond(
            req.request_id, approved=True, edited_content="final content"
        )
        assert resp.value == "final content"

    def test_unknown_request(self):
        review = HumanReview("Review:", "content")
        with pytest.raises(KeyError):
            review.respond("nonexistent", approved=True)


class TestHumanLoopExtension:
    """Tests for the HumanLoopExtension."""

    def test_extension_name(self):
        ext = HumanLoopExtension()
        assert ext.name == "human_loop"

    def test_create_approval_without_callback(self):
        ext = HumanLoopExtension()
        req = ext.create_approval("Approve?")
        assert req.request_type == "approval"
        assert not req.is_resolved
        assert len(ext.pending) == 1

    def test_create_approval_with_auto_approve(self):
        ext = HumanLoopExtension(callback=lambda req: True)
        req = ext.create_approval("Approve?")
        assert req.is_resolved
        assert req.response.status == ReviewStatus.APPROVED

    def test_create_approval_with_auto_reject(self):
        ext = HumanLoopExtension(callback=lambda req: False)
        req = ext.create_approval("Approve?")
        assert req.is_resolved
        assert req.response.status == ReviewStatus.REJECTED

    def test_create_input_without_callback(self):
        ext = HumanLoopExtension()
        req = ext.create_input("Enter value:")
        assert req.request_type == "input"
        assert not req.is_resolved

    def test_create_input_with_callback(self):
        ext = HumanLoopExtension(callback=lambda req: "auto-value")
        req = ext.create_input("Enter value:")
        assert req.is_resolved
        assert req.response.value == "auto-value"

    def test_pending_tracking(self):
        ext = HumanLoopExtension()
        ext.create_approval("A1")
        ext.create_approval("A2")
        ext.create_input("I1")
        assert len(ext.pending) == 3

    def test_pending_with_resolved(self):
        ext = HumanLoopExtension(callback=lambda req: True)
        ext.create_approval("A1")
        ext.create_approval("A2")
        assert len(ext.pending) == 0  # All auto-approved
