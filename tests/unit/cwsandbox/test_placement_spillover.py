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
    CWSANDBOX_INVALID_REQUEST,
    CWSANDBOX_NO_SUITABLE_RUNNER,
    CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
    CWSANDBOX_PLACEMENT_REJECTED,
    CWSANDBOX_RESOURCE_CEILING_EXCEEDED,
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
    CWSandboxError,
    SandboxError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxUnavailableError,
    SandboxValidationError,
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
        field_violations: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__()
        self._code = code
        self._details = details
        self._trailing: list[tuple[str, bytes]] = []
        if reason is not None:
            from google.protobuf import any_pb2
            from google.rpc import error_details_pb2, status_pb2

            status = status_pb2.Status(code=code.value[0], message=details)
            info = error_details_pb2.ErrorInfo(
                reason=reason, domain=domain, metadata=metadata or {}
            )
            packed = any_pb2.Any()
            packed.Pack(info)
            status.details.append(packed)
            if field_violations:
                bad_request = error_details_pb2.BadRequest()
                for field, description in field_violations:
                    bad_request.field_violations.add(field=field, description=description)
                packed = any_pb2.Any()
                packed.Pack(bad_request)
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


def _ceiling_error(
    details: str = "memory exceeds the policy per-container maximum",
) -> _MockRpcErrorWithDetails:
    return _MockRpcErrorWithDetails(
        grpc.StatusCode.INVALID_ARGUMENT,
        details,
        reason=CWSANDBOX_RESOURCE_CEILING_EXCEEDED,
        field_violations=(("spec.containers[0].resources.memory", details),),
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
                CWSANDBOX_RESOURCE_CEILING_EXCEEDED,
            }
        )

    def test_capacity_reason_is_eligible(self) -> None:
        assert _is_spillover_eligible(_capacity_error())

    def test_placement_rejected_is_eligible(self) -> None:
        assert _is_spillover_eligible(_capacity_error(CWSANDBOX_PLACEMENT_REJECTED))

    def test_constraint_unsatisfied_is_eligible(self) -> None:
        assert _is_spillover_eligible(_capacity_error(CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED))

    def test_bare_resource_exhausted_is_not_eligible(self) -> None:
        error = _MockRpcErrorWithDetails(grpc.StatusCode.RESOURCE_EXHAUSTED, "quota")
        assert not _is_spillover_eligible(error)

    def test_serverless_not_allowed_is_not_eligible(self) -> None:
        assert not _is_spillover_eligible(
            _MockRpcErrorWithDetails(
                grpc.StatusCode.FAILED_PRECONDITION,
                "serverless blocked",
                reason=CWSANDBOX_SERVERLESS_NOT_ALLOWED,
            )
        )

    def test_auth_style_error_is_not_eligible(self) -> None:
        error = _MockRpcErrorWithDetails(grpc.StatusCode.PERMISSION_DENIED, "nope")
        assert not _is_spillover_eligible(error)

    def test_no_suitable_runner_is_eligible(self) -> None:
        assert _is_spillover_eligible(
            _MockRpcErrorWithDetails(
                grpc.StatusCode.FAILED_PRECONDITION,
                "no runner",
                reason=CWSANDBOX_NO_SUITABLE_RUNNER,
            )
        )

    def test_runner_overloaded_is_eligible(self) -> None:
        assert _is_spillover_eligible(_capacity_error(CWSANDBOX_RUNNER_OVERLOADED))

    def test_runner_unavailable_is_eligible(self) -> None:
        error = _MockRpcErrorWithDetails(
            grpc.StatusCode.UNAVAILABLE,
            "runner down",
            reason=CWSANDBOX_RUNNER_UNAVAILABLE,
        )
        assert isinstance(_translate_rpc_error(error), SandboxUnavailableError)
        assert _is_spillover_eligible(error)

    def test_resource_ceiling_is_eligible(self) -> None:
        error = _ceiling_error()
        exc = _translate_rpc_error(error)
        assert isinstance(exc, SandboxValidationError)
        assert exc.reason == CWSANDBOX_RESOURCE_CEILING_EXCEEDED
        assert exc.field_violations[0].field == "spec.containers[0].resources.memory"
        assert _is_spillover_eligible(error)

    def test_other_invalid_argument_is_not_eligible(self) -> None:
        error = _MockRpcErrorWithDetails(
            grpc.StatusCode.INVALID_ARGUMENT,
            "bad field",
            reason=CWSANDBOX_INVALID_REQUEST,
        )
        assert not _is_spillover_eligible(error)


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
    @pytest.mark.parametrize(
        ("code", "domain"),
        [
            (grpc.StatusCode.INVALID_ARGUMENT, "proxy.example.com"),
            (grpc.StatusCode.INVALID_ARGUMENT, ""),
            (grpc.StatusCode.UNAUTHENTICATED, "cwsandbox.com"),
            (grpc.StatusCode.PERMISSION_DENIED, "cwsandbox.com"),
            (grpc.StatusCode.DEADLINE_EXCEEDED, "cwsandbox.com"),
            (grpc.StatusCode.UNAVAILABLE, "cwsandbox.com"),
            (grpc.StatusCode.INTERNAL, "cwsandbox.com"),
        ],
    )
    def test_ceiling_reason_requires_matching_domain_and_status(
        self, code: grpc.StatusCode, domain: str
    ) -> None:
        error = _MockRpcErrorWithDetails(
            code,
            reason=CWSANDBOX_RESOURCE_CEILING_EXCEEDED,
            domain=domain,
        )
        sandbox, stub, err = _run_with_create_side_effect(
            [error, _create_sandbox_response()],
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
        )
        assert sandbox is None
        assert isinstance(err, CWSandboxError)
        assert err.reason == CWSANDBOX_RESOURCE_CEILING_EXCEEDED
        assert stub.CreateSandbox.call_count == 1

    @pytest.mark.parametrize("error", [_capacity_error(), _ceiling_error()])
    def test_strict_does_not_retry(self, error: grpc.RpcError) -> None:
        sandbox, stub, err = _run_with_create_side_effect(
            [error],
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.STRICT,
            runner_ids=["runner-a"],
        )
        assert sandbox is None
        assert isinstance(err, SandboxError)
        assert err.__cause__ is error
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

    @pytest.mark.parametrize(
        ("primary", "spillover", "alternate", "runner_ids"),
        [
            (
                PlacementMode.CKS,
                PlacementSpillover.CKS_THEN_SERVERLESS,
                PlacementMode.SERVERLESS,
                ["runner-a"],
            ),
            (
                PlacementMode.SERVERLESS,
                PlacementSpillover.SERVERLESS_THEN_CKS,
                PlacementMode.CKS,
                None,
            ),
        ],
    )
    def test_retries_on_resource_ceiling_preserving_spec(
        self,
        primary: PlacementMode,
        spillover: PlacementSpillover,
        alternate: PlacementMode,
        runner_ids: list[str] | None,
    ) -> None:
        ok = _create_sandbox_response("spilled-ceiling")
        sandbox, stub, err = _run_with_create_side_effect(
            [_ceiling_error(), ok],
            placement_mode=primary,
            placement_spillover=spillover,
            runner_ids=runner_ids,
            resources={"cpu": "2", "memory": "4Gi"},
        )
        assert err is None
        assert sandbox is not None
        assert stub.CreateSandbox.call_count == 2
        req1 = stub.CreateSandbox.call_args_list[0].args[0]
        req2 = stub.CreateSandbox.call_args_list[1].args[0]
        assert req1.sandbox.spec.mode == sandbox_pb2.SandboxMode.Value(
            f"SANDBOX_MODE_{primary.name}"
        )
        assert list(req1.sandbox.spec.runner_ids) == (runner_ids or [])
        assert req2.sandbox.spec.mode == sandbox_pb2.SandboxMode.Value(
            f"SANDBOX_MODE_{alternate.name}"
        )
        assert list(req2.sandbox.spec.runner_ids) == []
        assert req1.request_id != req2.request_id
        expected_spec = sandbox_pb2.SandboxSpec()
        expected_spec.CopyFrom(req1.sandbox.spec)
        expected_spec.mode = req2.sandbox.spec.mode
        expected_spec.ClearField("runner_ids")
        assert req2.sandbox.spec == expected_spec
        assert sandbox._placement_mode == alternate
        assert sandbox._runner_ids is None

    def test_resource_ceiling_on_both_modes_chains_cause(self) -> None:
        sandbox, stub, err = _run_with_create_side_effect(
            [_ceiling_error("cks ceiling"), _ceiling_error("serverless ceiling")],
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
        )
        assert sandbox is None
        assert isinstance(err, SandboxValidationError)
        assert err.reason == CWSANDBOX_RESOURCE_CEILING_EXCEEDED
        assert isinstance(err.__cause__, SandboxValidationError)
        assert err.__cause__.reason == CWSANDBOX_RESOURCE_CEILING_EXCEEDED
        assert err.field_violations[0].description == "serverless ceiling"
        assert err.__cause__.field_violations[0].description == "cks ceiling"
        assert any(
            CWSANDBOX_RESOURCE_CEILING_EXCEEDED in note for note in getattr(err, "__notes__", [])
        )
        assert stub.CreateSandbox.call_count == 2

    def test_other_invalid_argument_does_not_retry(self) -> None:
        bad = _MockRpcErrorWithDetails(
            grpc.StatusCode.INVALID_ARGUMENT,
            "bad field",
            reason=CWSANDBOX_INVALID_REQUEST,
        )
        sandbox, stub, err = _run_with_create_side_effect(
            [bad],
            placement_mode=PlacementMode.CKS,
            placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,
        )
        assert sandbox is None
        assert isinstance(err, CWSandboxError)
        assert err.reason == CWSANDBOX_INVALID_REQUEST
        assert stub.CreateSandbox.call_count == 1

    @pytest.mark.parametrize(
        ("first", "second", "exception_type"),
        [
            (
                _capacity_error(),
                _capacity_error(CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED),
                SandboxResourceExhaustedError,
            ),
            (_ceiling_error(), _ceiling_error(), SandboxValidationError),
        ],
    )
    def test_restore_on_failed_spill_second_start_retries_primary(
        self, first: grpc.RpcError, second: grpc.RpcError, exception_type: type[SandboxError]
    ) -> None:
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
            with pytest.raises(exception_type):
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

    @pytest.mark.parametrize(
        ("code", "reason", "domain"),
        [
            (grpc.StatusCode.DEADLINE_EXCEEDED, None, "cwsandbox.com"),
            (
                grpc.StatusCode.DEADLINE_EXCEEDED,
                CWSANDBOX_RESOURCE_CEILING_EXCEEDED,
                "cwsandbox.com",
            ),
            (
                grpc.StatusCode.UNAVAILABLE,
                CWSANDBOX_RESOURCE_CEILING_EXCEEDED,
                "cwsandbox.com",
            ),
            (grpc.StatusCode.INTERNAL, CWSANDBOX_RESOURCE_CEILING_EXCEEDED, "cwsandbox.com"),
            (grpc.StatusCode.INTERNAL, "CWSANDBOX_INTERNAL_ERROR", "cwsandbox.com"),
            (
                grpc.StatusCode.UNAVAILABLE,
                CWSANDBOX_RUNNER_UNAVAILABLE,
                "proxy.example.com",
            ),
        ],
    )
    def test_ambiguous_spill_second_keeps_spilled_request_id(
        self, code: grpc.StatusCode, reason: str | None, domain: str
    ) -> None:
        first = _ceiling_error()
        second = _MockRpcErrorWithDetails(
            code,
            "maybe committed",
            reason=reason,
            domain=domain,
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
            with pytest.raises(SandboxError):
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

    @pytest.mark.parametrize(
        ("code", "reason", "rejected"),
        [
            (grpc.StatusCode.RESOURCE_EXHAUSTED, CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED, True),
            (grpc.StatusCode.INVALID_ARGUMENT, CWSANDBOX_INVALID_REQUEST, True),
            (grpc.StatusCode.INVALID_ARGUMENT, CWSANDBOX_RESOURCE_CEILING_EXCEEDED, True),
            (grpc.StatusCode.INVALID_ARGUMENT, None, True),
            (grpc.StatusCode.FAILED_PRECONDITION, CWSANDBOX_SERVERLESS_NOT_ALLOWED, True),
            (grpc.StatusCode.UNAUTHENTICATED, None, True),
            (grpc.StatusCode.PERMISSION_DENIED, None, True),
            (grpc.StatusCode.NOT_FOUND, None, True),
            (grpc.StatusCode.UNAVAILABLE, CWSANDBOX_RUNNER_UNAVAILABLE, True),
            (grpc.StatusCode.DEADLINE_EXCEEDED, None, False),
            (grpc.StatusCode.UNAVAILABLE, None, False),
            (grpc.StatusCode.UNAVAILABLE, CWSANDBOX_BACKEND_UNAVAILABLE, False),
            (grpc.StatusCode.RESOURCE_EXHAUSTED, None, False),
            (grpc.StatusCode.RESOURCE_EXHAUSTED, "UNKNOWN_QUOTA", False),
            (grpc.StatusCode.INTERNAL, None, False),
            (grpc.StatusCode.INTERNAL, "CWSANDBOX_INTERNAL_ERROR", False),
            (grpc.StatusCode.UNKNOWN, CWSANDBOX_RESOURCE_CEILING_EXCEEDED, False),
            (grpc.StatusCode.CANCELLED, CWSANDBOX_RESOURCE_CEILING_EXCEEDED, False),
        ],
    )
    def test_create_attempt_reject_classification(
        self, code: grpc.StatusCode, reason: str | None, rejected: bool
    ) -> None:
        error = _MockRpcErrorWithDetails(code, reason=reason)
        assert _create_attempt_definitely_rejected(error) is rejected
