# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for placement_spillover create-path retry policy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from cwsandbox import PlacementMode, PlacementSpillover, Sandbox
from cwsandbox._error_info import (
    CWSANDBOX_BACKEND_UNAVAILABLE,
    CWSANDBOX_NO_SUITABLE_RUNNER,
    CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
    CWSANDBOX_PLACEMENT_REJECTED,
    CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED,
    CWSANDBOX_RUNNER_OVERLOADED,
    CWSANDBOX_RUNNER_UNAVAILABLE,
    CWSANDBOX_SERVERLESS_NOT_ALLOWED,
    SPILLOVER_ELIGIBLE_REASONS,
)
from cwsandbox._proto import sandbox_pb2
from cwsandbox._sandbox import (
    SandboxStatus,
    _create_attempt_definitely_rejected,
    _is_spillover_eligible,
    _Terminal,
    _translate_rpc_error,
)
from cwsandbox.exceptions import (
    SandboxError,
    SandboxNotRunningError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxUnavailableError,
)


def _create_sandbox_response(sandbox_id: str = "spill-ok") -> sandbox_pb2.Sandbox:
    return sandbox_pb2.Sandbox(
        sandbox_id=sandbox_id,
        status=sandbox_pb2.SandboxStatus(state=sandbox_pb2.STATE_PENDING),
    )


class _MockRpcErrorWithDetails(grpc.RpcError):
    """RpcError carrying AIP-193 ErrorInfo in trailing metadata."""

    def __init__(
        self,
        code: grpc.StatusCode,
        details: str = "err",
        *,
        reason: str | None = None,
        domain: str = "cwsandbox.com",
        metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._code = code
        self._details = details
        self._trailing: list[tuple[str, bytes]] = []
        if reason is not None:
            from google.protobuf import any_pb2
            from google.rpc import error_details_pb2, status_pb2

            status = status_pb2.Status(code=2, message=details)
            info = error_details_pb2.ErrorInfo(
                reason=reason, domain=domain, metadata=metadata or {}
            )
            packed = any_pb2.Any()
            packed.Pack(info)
            status.details.append(packed)
            self._trailing = [("grpc-status-details-bin", status.SerializeToString())]

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def trailing_metadata(self) -> list[tuple[str, bytes]]:
        return self._trailing


def _capacity_error(
    reason: str = CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED,
) -> _MockRpcErrorWithDetails:
    return _MockRpcErrorWithDetails(
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "capacity exhausted",
        reason=reason,
    )


def _run_with_create_side_effect(
    side_effect: Any,
    **kwargs: Any,
) -> tuple[Sandbox | None, MagicMock, BaseException | None]:
    """Start via Sandbox.run with a mocked CreateSandbox side_effect."""
    mock_stub = MagicMock()
    mock_stub.CreateSandbox = AsyncMock(side_effect=side_effect)
    mock_stub.CreateSandboxFromTemplate = AsyncMock(
        return_value=_create_sandbox_response("template-id")
    )

    async def ensure_client(sandbox: Sandbox) -> None:
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = mock_stub

    err: BaseException | None = None
    sandbox: Sandbox | None = None
    with patch.object(Sandbox, "_ensure_client", ensure_client):
        try:
            sandbox = Sandbox.run("sleep", "infinity", **kwargs)
        except BaseException as e:
            err = e
    if sandbox is not None:
        sandbox._state = _Terminal(
            sandbox_id=sandbox.sandbox_id or "spill-ok",
            status=SandboxStatus.COMPLETED,
        )
    return sandbox, mock_stub, err


class TestSpilloverEligibility:
    def test_eligible_reason_set_is_pinned(self) -> None:
        assert SPILLOVER_ELIGIBLE_REASONS == frozenset(
            {
                CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED,
                CWSANDBOX_PLACEMENT_REJECTED,
                CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
                CWSANDBOX_NO_SUITABLE_RUNNER,
                CWSANDBOX_RUNNER_OVERLOADED,
                CWSANDBOX_RUNNER_UNAVAILABLE,
            }
        )

    def test_capacity_reason_is_eligible(self) -> None:
        exc = _translate_rpc_error(_capacity_error())
        assert _is_spillover_eligible(exc)

    def test_placement_rejected_is_eligible(self) -> None:
        exc = _translate_rpc_error(
            _capacity_error(CWSANDBOX_PLACEMENT_REJECTED),
        )
        assert _is_spillover_eligible(exc)

    def test_constraint_unsatisfied_is_eligible(self) -> None:
        exc = _translate_rpc_error(
            _capacity_error(CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED),
        )
        assert _is_spillover_eligible(exc)

    def test_bare_resource_exhausted_is_not_eligible(self) -> None:
        exc = SandboxResourceExhaustedError("quota")
        assert exc.reason is None
        assert not _is_spillover_eligible(exc)

    def test_serverless_not_allowed_is_not_eligible(self) -> None:
        exc = _translate_rpc_error(
            _MockRpcErrorWithDetails(
                grpc.StatusCode.FAILED_PRECONDITION,
                "serverless blocked",
                reason=CWSANDBOX_SERVERLESS_NOT_ALLOWED,
            )
        )
        assert isinstance(exc, SandboxError)
        assert not _is_spillover_eligible(exc)

    def test_auth_style_error_is_not_eligible(self) -> None:
        exc = SandboxError("nope", reason="PERMISSION_DENIED")
        assert not _is_spillover_eligible(exc)

    def test_no_suitable_runner_is_eligible(self) -> None:
        exc = _translate_rpc_error(
            _MockRpcErrorWithDetails(
                grpc.StatusCode.FAILED_PRECONDITION,
                "no runner",
                reason=CWSANDBOX_NO_SUITABLE_RUNNER,
            )
        )
        assert _is_spillover_eligible(exc)

    def test_runner_overloaded_is_eligible(self) -> None:
        exc = _translate_rpc_error(
            _capacity_error(CWSANDBOX_RUNNER_OVERLOADED),
        )
        assert _is_spillover_eligible(exc)

    def test_runner_unavailable_is_eligible(self) -> None:
        exc = _translate_rpc_error(
            _MockRpcErrorWithDetails(
                grpc.StatusCode.UNAVAILABLE,
                "runner down",
                reason=CWSANDBOX_RUNNER_UNAVAILABLE,
            )
        )
        assert isinstance(exc, SandboxUnavailableError)
        assert _is_spillover_eligible(exc)


class TestSpilloverValidation:
    def test_cks_then_serverless_rejects_serverless_primary(self) -> None:
        with pytest.raises(ValueError, match="cks_then_serverless"):
            Sandbox(
                placement_mode=PlacementMode.SERVERLESS,
                placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
            )

    def test_serverless_then_cks_rejects_cks_primary(self) -> None:
        with pytest.raises(ValueError, match="serverless_then_cks"):
            Sandbox(
                placement_mode=PlacementMode.CKS,
                placement_spillover=PlacementSpillover.SERVERLESS_THEN_CKS,
            )

    def test_template_rejects_non_strict_spillover(self) -> None:
        with pytest.raises(ValueError, match="STRICT for template"):
            Sandbox(
                template_id="tmpl-1",
                placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
            )

    def test_cks_then_serverless_unset_mode_resolves_to_cks(self) -> None:
        sb = Sandbox(placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS)
        assert sb._placement_mode == PlacementMode.CKS
        assert sb._placement_spillover == PlacementSpillover.CKS_THEN_SERVERLESS

    def test_serverless_then_cks_unset_mode_resolves_to_serverless(self) -> None:
        sb = Sandbox(placement_spillover=PlacementSpillover.SERVERLESS_THEN_CKS)
        assert sb._placement_mode == PlacementMode.SERVERLESS

    def test_serverless_then_cks_rejects_runner_ids(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined with runner_ids"):
            Sandbox(
                placement_spillover=PlacementSpillover.SERVERLESS_THEN_CKS,
                runner_ids=["runner-a"],
            )


class TestSpilloverCreatePath:
    def test_strict_does_not_retry_on_capacity(self) -> None:
        sandbox, stub, err = _run_with_create_side_effect(
            [_capacity_error()],
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.STRICT,
            runner_ids=["runner-a"],
        )
        assert sandbox is None
        assert isinstance(err, SandboxResourceExhaustedError)
        assert stub.CreateSandbox.call_count == 1

    def test_cks_then_serverless_retries_clears_runner_ids_new_request_id(self) -> None:
        first_err = _capacity_error(CWSANDBOX_PLACEMENT_REJECTED)
        ok = _create_sandbox_response("spilled")
        sandbox, stub, err = _run_with_create_side_effect(
            [first_err, ok],
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
            runner_ids=["runner-a"],
        )
        assert err is None
        assert sandbox is not None
        assert stub.CreateSandbox.call_count == 2

        req1 = stub.CreateSandbox.call_args_list[0].args[0]
        req2 = stub.CreateSandbox.call_args_list[1].args[0]
        assert req1.sandbox.spec.mode == sandbox_pb2.SANDBOX_MODE_CKS
        assert list(req1.sandbox.spec.runner_ids) == ["runner-a"]
        assert req2.sandbox.spec.mode == sandbox_pb2.SANDBOX_MODE_SERVERLESS
        assert list(req2.sandbox.spec.runner_ids) == []
        assert req1.request_id != req2.request_id
        assert sandbox._placement_mode == PlacementMode.SERVERLESS
        assert sandbox._runner_ids is None

    def test_serverless_then_cks_retries_into_cks(self) -> None:
        first_err = _capacity_error()
        ok = _create_sandbox_response("spilled-cks")
        sandbox, stub, err = _run_with_create_side_effect(
            [first_err, ok],
            placement_mode=PlacementMode.SERVERLESS,
            placement_spillover=PlacementSpillover.SERVERLESS_THEN_CKS,
        )
        assert err is None
        assert sandbox is not None
        assert stub.CreateSandbox.call_count == 2
        req1 = stub.CreateSandbox.call_args_list[0].args[0]
        req2 = stub.CreateSandbox.call_args_list[1].args[0]
        assert req1.sandbox.spec.mode == sandbox_pb2.SANDBOX_MODE_SERVERLESS
        assert req2.sandbox.spec.mode == sandbox_pb2.SANDBOX_MODE_CKS
        assert sandbox._placement_mode == PlacementMode.CKS

    def test_non_spill_reason_does_not_retry(self) -> None:
        blocked = _MockRpcErrorWithDetails(
            grpc.StatusCode.FAILED_PRECONDITION,
            "serverless not allowed",
            reason=CWSANDBOX_SERVERLESS_NOT_ALLOWED,
        )
        sandbox, stub, err = _run_with_create_side_effect(
            [blocked],
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
        )
        assert sandbox is None
        assert isinstance(err, SandboxError)
        assert err.reason == CWSANDBOX_SERVERLESS_NOT_ALLOWED
        assert stub.CreateSandbox.call_count == 1

    def test_attempt_two_failure_chains_cause(self) -> None:
        first = _capacity_error(CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED)
        second = _MockRpcErrorWithDetails(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "still full",
            reason=CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
        )
        sandbox, stub, err = _run_with_create_side_effect(
            [first, second],
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
        )
        assert sandbox is None
        assert isinstance(err, SandboxResourceExhaustedError)
        assert err.reason == CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED
        assert isinstance(err.__cause__, SandboxResourceExhaustedError)
        assert err.__cause__.reason == CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED
        assert any(
            CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED in note for note in getattr(err, "__notes__", [])
        )
        assert stub.CreateSandbox.call_count == 2

    def test_bare_resource_exhausted_does_not_spill(self) -> None:
        bare = _MockRpcErrorWithDetails(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "throttled",
        )
        sandbox, stub, err = _run_with_create_side_effect(
            [bare],
            placement_spillover="cks_then_serverless",
        )
        assert sandbox is None
        assert isinstance(err, SandboxResourceExhaustedError)
        assert stub.CreateSandbox.call_count == 1

    def test_restore_on_failed_spill_second_start_retries_primary(self) -> None:
        first = _capacity_error(CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED)
        second = _MockRpcErrorWithDetails(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "still full",
            reason=CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
        )
        sandbox = Sandbox(
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
            runner_ids=["runner-a"],
        )
        mock_stub = MagicMock()
        mock_stub.CreateSandbox = AsyncMock(side_effect=[first, second])

        async def ensure_client(sb: Sandbox) -> None:
            sb._channel = MagicMock()
            sb._channel.close = AsyncMock()
            sb._stub = mock_stub

        with patch.object(Sandbox, "_ensure_client", ensure_client):
            with pytest.raises(SandboxResourceExhaustedError):
                sandbox.start().result()

        first_req = mock_stub.CreateSandbox.call_args_list[0].args[0]
        second_req = mock_stub.CreateSandbox.call_args_list[1].args[0]
        assert first_req.request_id != second_req.request_id
        assert sandbox._placement_mode == PlacementMode.CKS
        assert sandbox._runner_ids == ["runner-a"]
        assert sandbox._create_request_id == first_req.request_id
        assert mock_stub.CreateSandbox.call_count == 2

        ok = _create_sandbox_response("retry-ok")
        mock_stub.CreateSandbox = AsyncMock(return_value=ok)
        sandbox.start().result()
        req = mock_stub.CreateSandbox.call_args.args[0]
        assert req.sandbox.spec.mode == sandbox_pb2.SANDBOX_MODE_CKS
        assert list(req.sandbox.spec.runner_ids) == ["runner-a"]
        assert req.request_id == first_req.request_id
        sandbox._state = _Terminal(sandbox_id="retry-ok", status=SandboxStatus.COMPLETED)

    def test_ambiguous_spill_second_keeps_spilled_request_id(self) -> None:
        first = _capacity_error(CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED)
        second = _MockRpcErrorWithDetails(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            "maybe committed",
        )
        sandbox = Sandbox(
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
            runner_ids=["runner-a"],
        )
        mock_stub = MagicMock()
        mock_stub.CreateSandbox = AsyncMock(side_effect=[first, second])

        async def ensure_client(sb: Sandbox) -> None:
            sb._channel = MagicMock()
            sb._channel.close = AsyncMock()
            sb._stub = mock_stub

        with patch.object(Sandbox, "_ensure_client", ensure_client):
            with pytest.raises(SandboxRequestTimeoutError):
                sandbox.start().result()

        first_req = mock_stub.CreateSandbox.call_args_list[0].args[0]
        second_req = mock_stub.CreateSandbox.call_args_list[1].args[0]
        assert first_req.request_id != second_req.request_id
        assert sandbox._create_request_id == second_req.request_id
        assert sandbox._placement_mode == PlacementMode.SERVERLESS
        assert sandbox._runner_ids is None

        ok = _create_sandbox_response("retry-ok")
        mock_stub.CreateSandbox = AsyncMock(return_value=ok)
        sandbox.start().result()
        req = mock_stub.CreateSandbox.call_args.args[0]
        assert req.request_id == second_req.request_id
        assert req.sandbox.spec.mode == sandbox_pb2.SANDBOX_MODE_SERVERLESS
        sandbox._state = _Terminal(sandbox_id="retry-ok", status=SandboxStatus.COMPLETED)

    def test_later_start_same_mode_does_not_mint_new_request_id(self) -> None:
        first = _capacity_error(CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED)
        second = _MockRpcErrorWithDetails(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            "maybe committed",
        )
        sandbox = Sandbox(
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
            runner_ids=["runner-a"],
        )
        mock_stub = MagicMock()
        mock_stub.CreateSandbox = AsyncMock(side_effect=[first, second])

        async def ensure_client(sb: Sandbox) -> None:
            sb._channel = MagicMock()
            sb._channel.close = AsyncMock()
            sb._stub = mock_stub

        with patch.object(Sandbox, "_ensure_client", ensure_client):
            with pytest.raises(SandboxRequestTimeoutError):
                sandbox.start().result()

        spilled_id = sandbox._create_request_id
        later = _capacity_error(CWSANDBOX_PLACEMENT_REJECTED)
        mock_stub.CreateSandbox = AsyncMock(side_effect=[later])
        with patch.object(Sandbox, "_ensure_client", ensure_client):
            with pytest.raises(SandboxResourceExhaustedError):
                sandbox.start().result()
        assert mock_stub.CreateSandbox.call_count == 1
        assert mock_stub.CreateSandbox.call_args.args[0].request_id == spilled_id
        sandbox._state = _Terminal(sandbox_id="spill-id", status=SandboxStatus.COMPLETED)

    def test_create_attempt_reject_classification(self) -> None:
        assert _create_attempt_definitely_rejected(
            SandboxResourceExhaustedError("full", reason=CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED)
        )
        assert not _create_attempt_definitely_rejected(SandboxRequestTimeoutError("late"))
        assert not _create_attempt_definitely_rejected(SandboxUnavailableError("down"))
        assert not _create_attempt_definitely_rejected(
            SandboxUnavailableError("backend", reason=CWSANDBOX_BACKEND_UNAVAILABLE)
        )
        assert not _create_attempt_definitely_rejected(SandboxResourceExhaustedError("quota"))
        assert not _create_attempt_definitely_rejected(SandboxError("internal"))
        assert not _create_attempt_definitely_rejected(SandboxNotRunningError("cancelled"))
        assert _create_attempt_definitely_rejected(
            SandboxError("bad request", reason="CWSANDBOX_INVALID_REQUEST")
        )
        assert _create_attempt_definitely_rejected(
            SandboxUnavailableError("runner down", reason=CWSANDBOX_RUNNER_UNAVAILABLE)
        )
