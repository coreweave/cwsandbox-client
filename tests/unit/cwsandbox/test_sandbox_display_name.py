# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for sandbox display_name: create, read-back, lookup, and errors."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.rpc import error_details_pb2, status_pb2

from cwsandbox import (
    Sandbox,
    SandboxDefaults,
    SandboxError,
    SandboxNameInUseError,
    SandboxStatus,
    Session,
)
from cwsandbox._error_info import CWSANDBOX_SANDBOX_DISPLAY_NAME_IN_USE
from cwsandbox._proto import sandbox_pb2
from cwsandbox._sandbox import _Terminal, _translate_rpc_error

_UUID = "6f1c1f9e-3b0a-4c55-9d1e-0a6c2b7e9f10"


def _response(
    sandbox_id: str = _UUID,
    *,
    display_name: str = "",
    state: int = sandbox_pb2.STATE_PENDING,
) -> sandbox_pb2.Sandbox:
    return sandbox_pb2.Sandbox(
        sandbox_id=sandbox_id,
        display_name=display_name,
        status=sandbox_pb2.SandboxStatus(state=state),
    )


def _rpc_error(reason: str | None, code: grpc.StatusCode) -> grpc.RpcError:
    status = status_pb2.Status(message="boom")
    if reason is not None:
        info = error_details_pb2.ErrorInfo(reason=reason, domain="cwsandbox.com")
        status.details.add().Pack(info)
    status_bytes = status.SerializeToString()

    class _Err(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return code

        def details(self) -> str:
            return "display_name is already in use by an active sandbox"

        def trailing_metadata(self) -> tuple[tuple[str, bytes], ...]:
            return (("grpc-status-details-bin", status_bytes),)

    return _Err()


def _stub(response: Any = None, *, error: grpc.RpcError | None = None) -> MagicMock:
    stub = MagicMock()
    for rpc in ("CreateSandbox", "CreateSandboxFromTemplate", "CreateSandboxFromFile"):
        mock = AsyncMock(side_effect=error) if error else AsyncMock(return_value=response)
        setattr(stub, rpc, mock)
    return stub


def _with_stub(stub: MagicMock) -> Any:
    async def ensure_client(sandbox: Sandbox) -> None:
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = stub

    return patch.object(Sandbox, "_ensure_client", ensure_client)


def _finish(sandbox: Sandbox) -> None:
    sandbox._state = _Terminal(sandbox_id=_UUID, status=SandboxStatus.COMPLETED)


class TestDisplayNameOnCreate:
    def test_run_sends_name_on_sandbox(self) -> None:
        stub = _stub(_response(display_name="worker-1"))
        with _with_stub(stub):
            sandbox = Sandbox.run(display_name="worker-1")
        request = stub.CreateSandbox.call_args.args[0]
        assert request.sandbox.display_name == "worker-1"
        assert sandbox.display_name == "worker-1"
        _finish(sandbox)

    def test_run_without_name_sends_empty(self) -> None:
        stub = _stub(_response())
        with _with_stub(stub):
            sandbox = Sandbox.run()
        assert stub.CreateSandbox.call_args.args[0].sandbox.display_name == ""
        assert sandbox.display_name is None
        _finish(sandbox)

    def test_run_from_template_sends_top_level_name(self) -> None:
        stub = _stub(_response(display_name="tmpl-1"))
        with _with_stub(stub):
            sandbox = Sandbox.run_from_template("tmpl-id", display_name="tmpl-1")
        request = stub.CreateSandboxFromTemplate.call_args.args[0]
        assert request.display_name == "tmpl-1"
        assert sandbox.display_name == "tmpl-1"
        _finish(sandbox)

    def test_run_from_file_sends_top_level_name(self) -> None:
        stub = _stub(_response(display_name="compose-1"))
        with _with_stub(stub):
            sandbox = Sandbox.run_from_file(
                b"services:\n  main:\n    image: python:3.11\n",
                primary_service="main",
                display_name="compose-1",
            )
        request = stub.CreateSandboxFromFile.call_args.args[0]
        assert request.display_name == "compose-1"
        assert sandbox.display_name == "compose-1"
        _finish(sandbox)

    def test_session_sandbox_sends_name(self) -> None:
        stub = _stub(_response(display_name="sess-1"))
        session = Session(SandboxDefaults(tags=("t",)))
        sandbox = session.sandbox(display_name="sess-1")
        assert sandbox.display_name == "sess-1"
        with _with_stub(stub):
            sandbox.start().result()
        assert stub.CreateSandbox.call_args.args[0].sandbox.display_name == "sess-1"
        _finish(sandbox)

    def test_requested_name_visible_before_start(self) -> None:
        assert Sandbox(display_name="pending").display_name == "pending"

    def test_response_overrides_requested_name(self) -> None:
        stub = _stub(_response(display_name=""))
        with _with_stub(stub):
            sandbox = Sandbox.run(display_name="asked")
        assert sandbox.display_name is None
        _finish(sandbox)


class TestDisplayNameValidation:
    @pytest.mark.parametrize(
        "name",
        [
            "a",
            "worker 1",
            "Ünïcode-名前",
            "x" * 128,
            "é" * 64,
            "6f1c1f9e-3b0a-4c55-9d1e-0a6c2b7e9f1",
            "6f1c1f9e3b0a4c559d1e0a6c2b7e9f1g",
        ],
    )
    def test_accepts(self, name: str) -> None:
        assert Sandbox(display_name=name).display_name == name

    @pytest.mark.parametrize("name", [None, ""])
    def test_empty_means_unset(self, name: str | None) -> None:
        assert Sandbox(display_name=name).display_name is None

    @pytest.mark.parametrize(
        "name",
        [
            "x" * 129,
            "é" * 65,
            " lead",
            "trail ",
            " nbsp",
            "trail　",
            "a/b",
            "a\\b",
            "a:b",
            "tab\tinside",
            "nul\x00",
            "del\x7f",
            "c1\x85",
            "lone\ud800",
            _UUID,
            _UUID.upper(),
            _UUID.replace("-", ""),
            "{" + _UUID + "}",
            "(" + _UUID + ")",
            "x" + _UUID + "y",
            "urn:uuid:" + _UUID,
        ],
    )
    def test_rejects(self, name: str) -> None:
        with pytest.raises(ValueError, match="display_name"):
            Sandbox(display_name=name)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError, match="display_name"):
            Sandbox(display_name=123)  # type: ignore[arg-type]

    def test_run_rejects_before_rpc(self) -> None:
        stub = _stub(_response())
        with _with_stub(stub), pytest.raises(ValueError, match="display_name"):
            Sandbox.run(display_name="a/b")
        stub.CreateSandbox.assert_not_called()


class TestDisplayNameDefaults:
    def test_sandbox_defaults_has_no_field(self) -> None:
        with pytest.raises(TypeError):
            SandboxDefaults(display_name="x")  # type: ignore[call-arg]

    def test_from_dict_rejects(self) -> None:
        with pytest.raises(TypeError, match="display_name"):
            SandboxDefaults.from_dict({"display_name": "x"})

    def test_session_mapping_defaults_rejects(self) -> None:
        with pytest.raises(TypeError, match="display_name"):
            Session({"display_name": "x"})


class TestNameInUseError:
    def test_reason_maps_to_typed_error(self) -> None:
        err = _translate_rpc_error(
            _rpc_error(CWSANDBOX_SANDBOX_DISPLAY_NAME_IN_USE, grpc.StatusCode.ALREADY_EXISTS),
            operation="Create sandbox",
        )
        assert isinstance(err, SandboxNameInUseError)
        assert isinstance(err, SandboxError)
        assert err.reason == CWSANDBOX_SANDBOX_DISPLAY_NAME_IN_USE

    def test_bare_already_exists_is_not_name_error(self) -> None:
        err = _translate_rpc_error(
            _rpc_error(None, grpc.StatusCode.ALREADY_EXISTS), operation="Create sandbox"
        )
        assert not isinstance(err, SandboxNameInUseError)

    @pytest.mark.parametrize(
        ("factory", "rpc"),
        [
            (lambda: Sandbox.run(display_name="taken"), "CreateSandbox"),
            (
                lambda: Sandbox.run_from_template("tmpl-id", display_name="taken"),
                "CreateSandboxFromTemplate",
            ),
            (
                lambda: Sandbox.run_from_file(
                    b"services: {}\n", primary_service="main", display_name="taken"
                ),
                "CreateSandboxFromFile",
            ),
        ],
    )
    def test_create_raises_name_in_use(self, factory: Any, rpc: str) -> None:
        stub = _stub(
            error=_rpc_error(CWSANDBOX_SANDBOX_DISPLAY_NAME_IN_USE, grpc.StatusCode.ALREADY_EXISTS)
        )
        with _with_stub(stub), pytest.raises(SandboxNameInUseError):
            factory()
        getattr(stub, rpc).assert_called_once()


class TestFromIdByName:
    @pytest.mark.asyncio
    async def test_handle_uses_resolved_uuid(self, mock_api_key: str) -> None:
        get_stub = MagicMock()
        get_stub.GetSandbox = AsyncMock(
            return_value=_response(display_name="worker-1", state=sandbox_pb2.STATE_RUNNING)
        )
        channel = MagicMock()
        channel.close = AsyncMock()
        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=channel),
            patch("cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub", return_value=get_stub),
        ):
            sandbox = await Sandbox.from_id("worker-1")

        assert get_stub.GetSandbox.call_args.args[0].sandbox_id == "worker-1"
        assert sandbox.sandbox_id == _UUID
        assert sandbox.display_name == "worker-1"

        later_stub = MagicMock()
        later_stub.GetSandbox = AsyncMock(
            return_value=_response(display_name="worker-1", state=sandbox_pb2.STATE_RUNNING)
        )
        with _with_stub(later_stub):
            await sandbox._get_status_async()
        assert later_stub.GetSandbox.call_args.args[0].sandbox_id == _UUID

    @pytest.mark.asyncio
    async def test_list_reads_back_name(self, mock_api_key: str) -> None:
        stub = MagicMock()
        stub.ListSandboxes = AsyncMock(
            return_value=sandbox_pb2.ListSandboxesResponse(
                sandboxes=[_response(display_name="listed", state=sandbox_pb2.STATE_RUNNING)]
            )
        )
        channel = MagicMock()
        channel.close = AsyncMock()
        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=channel),
            patch("cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub", return_value=stub),
        ):
            sandboxes = await Sandbox.list()
        assert [sb.display_name for sb in sandboxes] == ["listed"]
