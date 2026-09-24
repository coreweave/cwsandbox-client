# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Retry of delete, stop, and gateway read_file on server-hinted UNAVAILABLE."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.protobuf import duration_pb2
from google.rpc import error_details_pb2, status_pb2

import cwsandbox._sandbox as sandbox_module
from cwsandbox import Sandbox, SandboxDefaults, SandboxStatus
from cwsandbox._proto import sandbox_pb2
from cwsandbox._sandbox import (
    _retry_transient_unavailable,
    _Starting,
)
from cwsandbox._types import DataPlaneMode
from cwsandbox.exceptions import (
    SandboxError,
    SandboxNotFoundError,
    SandboxUnavailableError,
)

RUNNER_UNAVAILABLE = "CWSANDBOX_RUNNER_UNAVAILABLE"


class _RpcError(grpc.RpcError):
    def __init__(
        self,
        code: grpc.StatusCode,
        *,
        reason: str | None = None,
        retry_delay: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._code = code
        self._trailing: list[tuple[str, bytes]] = []
        if reason is None and retry_delay is None:
            return
        status = status_pb2.Status(code=code.value[0], message="boom")
        if reason is not None:
            status.details.add().Pack(
                error_details_pb2.ErrorInfo(reason=reason, domain="cwsandbox.com")
            )
        if retry_delay is not None:
            seconds, nanos = retry_delay
            status.details.add().Pack(
                error_details_pb2.RetryInfo(
                    retry_delay=duration_pb2.Duration(seconds=seconds, nanos=nanos)
                )
            )
        self._trailing.append(("grpc-status-details-bin", status.SerializeToString()))

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return "boom"

    def trailing_metadata(self) -> list[tuple[str, bytes]]:  # type: ignore[override]
        return self._trailing


class _ModuleProxy:
    """Stand-in for a stdlib module inside ``cwsandbox._sandbox`` only.

    Overrides a few functions and delegates the rest, so patching the SDK's
    ``asyncio`` / ``time`` / ``random`` binding never touches the shared
    stdlib module that other tasks and the SDK loop thread use.
    """

    def __init__(self, module: Any, **overrides: Any) -> None:
        self._module = module
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._module, name)


def _patch_sleep(fn: Any) -> Any:
    return patch.object(sandbox_module, "asyncio", _ModuleProxy(asyncio, sleep=fn))


def _patch_monotonic(fn: Any) -> Any:
    return patch.object(sandbox_module, "time", _ModuleProxy(time, monotonic=fn))


def _patch_uniform(fn: Any) -> Any:
    return patch.object(sandbox_module, "random", _ModuleProxy(random, uniform=fn))


# 1 ms hint: exercises the real sleep without slowing the suite.
def _hinted(reason: str = RUNNER_UNAVAILABLE) -> _RpcError:
    return _RpcError(grpc.StatusCode.UNAVAILABLE, reason=reason, retry_delay=(0, 1_000_000))


def _not_found() -> _RpcError:
    return _RpcError(grpc.StatusCode.NOT_FOUND, reason="CWSANDBOX_SANDBOX_NOT_FOUND")


class _Calls:
    """Attempt callable that replays a script of errors/results and records timeouts."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.timeouts: list[float] = []

    async def __call__(self, rpc_timeout: float) -> object:
        self.timeouts.append(rpc_timeout)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def _run(calls: _Calls, **kwargs: object) -> object:
    opts: dict[str, object] = {"timeout": 30.0, "operation": "op", "enabled": True}
    opts.update(kwargs)
    return await _retry_transient_unavailable(calls, **opts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helper policy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_retries_hinted_unavailable_until_success(self) -> None:
        calls = _Calls(_hinted(), _hinted(), "ok")
        assert await _run(calls) == "ok"
        assert len(calls.timeouts) == 3

    @pytest.mark.asyncio
    async def test_bare_unavailable_is_not_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE)
        calls = _Calls(err, "ok")
        with pytest.raises(grpc.RpcError) as exc_info:
            await _run(calls)
        assert exc_info.value is err
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_unavailable_reason_on_other_status_is_not_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.INTERNAL, reason=RUNNER_UNAVAILABLE, retry_delay=(0, 1))
        calls = _Calls(err, "ok")
        with pytest.raises(grpc.RpcError):
            await _run(calls)
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_unavailable_reason_without_retry_info_is_not_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, reason=RUNNER_UNAVAILABLE)
        calls = _Calls(err, "ok")
        with pytest.raises(grpc.RpcError):
            await _run(calls)
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_negative_delay_is_not_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(-1, 0))
        calls = _Calls(err, "ok")
        with pytest.raises(grpc.RpcError):
            await _run(calls)
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_zero_delay_is_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(0, 0))
        calls = _Calls(err, "ok")
        assert await _run(calls) == "ok"
        assert len(calls.timeouts) == 2

    @pytest.mark.asyncio
    async def test_attempts_are_capped_and_last_error_surfaces(self) -> None:
        errors = [_hinted(), _hinted(), _hinted(), _hinted()]
        calls = _Calls(*errors)
        with pytest.raises(grpc.RpcError) as exc_info:
            await _run(calls)
        assert len(calls.timeouts) == 3
        assert exc_info.value is errors[2]

    @pytest.mark.asyncio
    async def test_max_attempts_override(self) -> None:
        calls = _Calls(_hinted(), _hinted(), "ok")
        with pytest.raises(grpc.RpcError):
            await _run(calls, max_attempts=2)
        assert len(calls.timeouts) == 2

    @pytest.mark.asyncio
    async def test_disabled_makes_one_attempt(self) -> None:
        calls = _Calls(_hinted(), "ok")
        with pytest.raises(grpc.RpcError):
            await _run(calls, enabled=False)
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_delay_longer_than_budget_is_not_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(5, 0))
        calls = _Calls(err, "ok")
        with (
            _patch_sleep(sleep := AsyncMock()),
            pytest.raises(grpc.RpcError),
        ):
            await _run(calls, timeout=1.0)
        assert len(calls.timeouts) == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_attempt_gets_full_timeout_later_ones_the_remainder(self) -> None:
        calls = _Calls(_hinted(), "ok")
        await _run(calls, timeout=30.0)
        assert calls.timeouts[0] == 30.0
        assert 0 < calls.timeouts[1] < 30.0

    @pytest.mark.asyncio
    async def test_budget_exhausted_during_sleep_stops(self) -> None:
        now = [100.0]
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(1, 0))

        async def _fake_sleep(seconds: float) -> None:
            now[0] += 16.0  # overslept: 4 s left, under the 5 s floor

        calls = _Calls(err, "ok")
        with (
            _patch_monotonic(lambda: now[0]),
            _patch_sleep(_fake_sleep),
            pytest.raises(grpc.RpcError),
        ):
            await _run(calls, timeout=20.0)
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_hint_above_cap_is_raised_not_slept(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(11, 0))
        calls = _Calls(err, "ok")
        with (
            _patch_sleep(sleep := AsyncMock()),
            pytest.raises(grpc.RpcError) as exc_info,
        ):
            await _run(calls, timeout=300.0)
        assert exc_info.value is err
        assert len(calls.timeouts) == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_hint_at_cap_is_retried(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(10, 0))
        calls = _Calls(err, "ok")
        with _patch_sleep(AsyncMock()), _patch_uniform(lambda a, b: a):
            assert await _run(calls, timeout=300.0) == "ok"
        assert len(calls.timeouts) == 2

    @pytest.mark.parametrize(
        ("timeout", "retried"),
        [
            pytest.param(9.9, False, id="delay-plus-floor-exceeds-budget"),
            pytest.param(10.1, True, id="delay-plus-floor-fits-budget"),
        ],
    )
    @pytest.mark.asyncio
    async def test_retry_needs_min_time_left_for_next_attempt(
        self, timeout: float, retried: bool
    ) -> None:
        # 5 s hint + 5 s floor for the next attempt.
        now = [100.0]
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(5, 0))

        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            now[0] += seconds

        calls = _Calls(err, "ok")
        with (
            _patch_monotonic(lambda: now[0]),
            _patch_sleep(_fake_sleep),
            _patch_uniform(lambda a, b: a),
        ):
            if retried:
                assert await _run(calls, timeout=timeout) == "ok"
            else:
                with pytest.raises(grpc.RpcError):
                    await _run(calls, timeout=timeout)
        assert len(calls.timeouts) == (2 if retried else 1)
        # No pointless backoff when the next attempt could not fit anyway.
        assert slept == ([5.0] if retried else [])

    @pytest.mark.asyncio
    async def test_jitter_never_sleeps_below_hint(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(0, 1_000_000))
        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        uniform_args: list[tuple[float, float]] = []

        def _uniform(a: float, b: float) -> float:
            uniform_args.append((a, b))
            return a

        with (
            _patch_sleep(_fake_sleep),
            _patch_uniform(_uniform),
        ):
            await _run(_Calls(err, "ok"))
        assert uniform_args == [(1.0, 1.2)]
        assert slept == [pytest.approx(0.001)]

    @pytest.mark.asyncio
    async def test_still_valid_false_after_sleep_stops(self) -> None:
        err = _hinted()
        calls = _Calls(err, "ok")
        with pytest.raises(grpc.RpcError) as exc_info:
            await _run(calls, still_valid=lambda: False)
        assert exc_info.value is err
        assert len(calls.timeouts) == 1

    @pytest.mark.asyncio
    async def test_cancel_during_backoff_propagates_without_another_attempt(self) -> None:
        err = _RpcError(grpc.StatusCode.UNAVAILABLE, retry_delay=(9, 0))
        calls = _Calls(err, "ok")
        task = asyncio.ensure_future(_run(calls, timeout=300.0))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls.timeouts) == 1


# ---------------------------------------------------------------------------
# Sandbox.delete (classmethod)
# ---------------------------------------------------------------------------


def _patch_delete_stub(stub: MagicMock) -> list[object]:
    channel = MagicMock()
    channel.close = AsyncMock()
    return [
        patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
        patch("cwsandbox._sandbox.create_channel", return_value=channel),
        patch("cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub", return_value=stub),
    ]


async def _delete(stub: MagicMock, **kwargs: object) -> None:
    p1, p2, p3 = _patch_delete_stub(stub)
    with p1, p2, p3:  # type: ignore[attr-defined]
        await Sandbox.delete("sb-1", **kwargs)  # type: ignore[arg-type]


class TestDeleteClassMethod:
    @pytest.mark.asyncio
    async def test_hinted_runner_unavailable_is_retried(self, mock_api_key: str) -> None:
        stub = MagicMock()
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), sandbox_pb2.DeleteSandboxResponse()])
        await _delete(stub)
        assert stub.DeleteSandbox.await_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_surface_translated_last_error(self, mock_api_key: str) -> None:
        last = _hinted("CWSANDBOX_SANDBOX_ROUTE_UNAVAILABLE")
        stub = MagicMock()
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), _hinted(), last])
        with pytest.raises(SandboxUnavailableError) as exc_info:
            await _delete(stub)
        assert stub.DeleteSandbox.await_count == 3
        assert exc_info.value.reason == "CWSANDBOX_SANDBOX_ROUTE_UNAVAILABLE"
        assert exc_info.value.retry_delay is not None
        assert exc_info.value.__cause__ is last

    @pytest.mark.asyncio
    async def test_not_found_on_retry_counts_as_deleted(self, mock_api_key: str) -> None:
        stub = MagicMock()
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), _not_found()])
        await _delete(stub)
        assert stub.DeleteSandbox.await_count == 2

    @pytest.mark.asyncio
    async def test_not_found_on_first_attempt_still_raises(self, mock_api_key: str) -> None:
        stub = MagicMock()
        stub.DeleteSandbox = AsyncMock(side_effect=[_not_found()])
        with pytest.raises(SandboxNotFoundError):
            await _delete(stub)


# ---------------------------------------------------------------------------
# Sandbox.stop (instance)
# ---------------------------------------------------------------------------


def _stoppable(defaults: SandboxDefaults | None = None) -> tuple[Sandbox, MagicMock]:
    sandbox = Sandbox(command="sleep", args=["infinity"], defaults=defaults)
    sandbox._sandbox_id = "sb-1"
    sandbox._state = _Starting(sandbox_id="sb-1")
    sandbox._channel = MagicMock()
    sandbox._channel.close = AsyncMock()
    sandbox._stub = MagicMock()
    return sandbox, sandbox._stub


class TestStop:
    def test_plain_stop_retries_hinted_unavailable(self) -> None:
        sandbox, stub = _stoppable()
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), sandbox_pb2.DeleteSandboxResponse()])
        with patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock):
            sandbox.stop().result()
        assert stub.DeleteSandbox.await_count == 2

    def test_snapshot_on_stop_is_not_retried(self) -> None:
        sandbox, stub = _stoppable()
        sandbox._scratch_volume_names = ("workspace",)
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), sandbox_pb2.DeleteSandboxResponse()])
        with (
            patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock),
            pytest.raises(SandboxError),
        ):
            sandbox.stop(snapshot_on_stop=True).result()
        assert stub.DeleteSandbox.await_count == 1

    def test_not_found_on_retry_counts_as_stopped(self) -> None:
        sandbox, stub = _stoppable()
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), _not_found()])
        with patch.object(
            sandbox, "_await_terminal_after_stop", new_callable=AsyncMock
        ) as await_terminal:
            sandbox.stop().result()
        assert stub.DeleteSandbox.await_count == 2
        assert sandbox.status == SandboxStatus.TERMINATED
        await_terminal.assert_not_awaited()

    def test_not_found_on_first_attempt_still_raises(self) -> None:
        sandbox, stub = _stoppable()
        stub.DeleteSandbox = AsyncMock(side_effect=[_not_found()])
        with (
            patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock),
            pytest.raises(SandboxNotFoundError),
        ):
            sandbox.stop().result()

    @pytest.mark.asyncio
    async def test_channel_closed_during_backoff_stops_retrying(self) -> None:
        sandbox, stub = _stoppable()
        stub.DeleteSandbox = AsyncMock(side_effect=[_hinted(), sandbox_pb2.DeleteSandboxResponse()])
        real_sleep = asyncio.sleep

        async def _sleep_then_close(seconds: float) -> None:
            # A cancelled stop() joiner tears down the channel mid-backoff and
            # a later _ensure_client() opens a new one: the old stub is stale.
            sandbox._stub = MagicMock()
            await real_sleep(0)

        with (
            patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock),
            _patch_sleep(_sleep_then_close),
            pytest.raises(SandboxUnavailableError),
        ):
            await sandbox._do_stop(
                snapshot_on_stop=False,
                graceful_shutdown_seconds=10.0,
                missing_ok=False,
                wait_for_ready=True,
                request_id=None,
            )
        assert stub.DeleteSandbox.await_count == 1


# ---------------------------------------------------------------------------
# read_file (unary)
# ---------------------------------------------------------------------------


def _read_patches(sandbox: Sandbox) -> list[object]:
    return [
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
    ]


def _running(mode: DataPlaneMode, defaults: SandboxDefaults | None = None) -> Sandbox:
    sandbox = Sandbox(data_plane_mode=mode, defaults=defaults)
    sandbox._sandbox_id = "sandbox-1"
    sandbox._stub = MagicMock()
    return sandbox


async def _read(sandbox: Sandbox) -> bytes:
    p1, p2, p3 = _read_patches(sandbox)
    with p1, p2, p3:  # type: ignore[attr-defined]
        return await sandbox._read_file_unary_async("/tmp/f", 30)


class TestReadFile:
    @pytest.mark.asyncio
    async def test_gateway_read_retries_hinted_unavailable(self) -> None:
        sandbox = _running(DataPlaneMode.GATEWAY)
        sandbox._stub.ReadFile = AsyncMock(
            side_effect=[_hinted(), sandbox_pb2.ReadFileResponse(content=b"ok")]
        )
        assert await _read(sandbox) == b"ok"
        assert sandbox._stub.ReadFile.await_count == 2

    @pytest.mark.asyncio
    async def test_direct_read_adds_no_hinted_retry(self) -> None:
        sandbox = _running(DataPlaneMode.DIRECT)
        direct_stub = MagicMock()
        direct_stub.ReadFile = AsyncMock(
            side_effect=[_hinted(), sandbox_pb2.ReadFileResponse(content=b"ok")]
        )
        lease = MagicMock(stub=direct_stub)
        lease.release = AsyncMock()
        lease.discard = AsyncMock()
        sandbox._direct_data_plane.acquire = AsyncMock(return_value=lease)
        with (
            _patch_sleep(sleep := AsyncMock()),
            pytest.raises(SandboxUnavailableError),
        ):
            await _read(sandbox)
        assert direct_stub.ReadFile.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_direct_retirement_then_gateway_shares_three_call_cap(self) -> None:
        sandbox = _running(DataPlaneMode.AUTO)
        direct_stub = MagicMock()
        direct_stub.ReadFile = AsyncMock(
            side_effect=_RpcError(
                grpc.StatusCode.UNAVAILABLE, reason="CWSANDBOX_RUNNER_SHARD_RETIRING"
            )
        )
        lease = MagicMock(stub=direct_stub)
        lease.release = AsyncMock()
        lease.discard = AsyncMock()
        # First pass goes direct and hits retirement; second pass falls back
        # to the gateway, which keeps failing with a hinted UNAVAILABLE.
        sandbox._direct_data_plane.acquire = AsyncMock(
            side_effect=[lease, sandbox_module.DirectDataPlaneUnavailable("gone")]
        )
        sandbox._stub.ReadFile = AsyncMock(side_effect=[_hinted(), _hinted(), _hinted()])
        with pytest.raises(SandboxUnavailableError):
            await _read(sandbox)
        assert direct_stub.ReadFile.await_count == 1
        assert sandbox._stub.ReadFile.await_count == 2

    @staticmethod
    def _retiring_then_gateway(sandbox: Sandbox, retiring: _RpcError, cost: float) -> list[float]:
        """Direct pass fails with shard retirement after ``cost`` seconds, then gateway."""
        now = [100.0]

        async def _slow_direct(*args: Any, **kwargs: Any) -> Any:
            now[0] += cost
            raise retiring

        direct_stub = MagicMock()
        direct_stub.ReadFile = AsyncMock(side_effect=_slow_direct)
        lease = MagicMock(stub=direct_stub)
        lease.release = AsyncMock()
        lease.discard = AsyncMock()
        sandbox._direct_data_plane.acquire = AsyncMock(
            side_effect=[lease, sandbox_module.DirectDataPlaneUnavailable("gone")]
        )
        sandbox._stub.ReadFile = AsyncMock(return_value=sandbox_pb2.ReadFileResponse(content=b"ok"))
        return now

    @staticmethod
    def _retiring() -> _RpcError:
        return _RpcError(grpc.StatusCode.UNAVAILABLE, reason="CWSANDBOX_RUNNER_SHARD_RETIRING")

    @pytest.mark.asyncio
    async def test_retirement_fallback_keeps_full_timeout(self) -> None:
        # The gateway fallback after a slow direct shard-retirement pass gets
        # the full per-call timeout, as before this retry existed.
        sandbox = _running(DataPlaneMode.AUTO)
        now = self._retiring_then_gateway(sandbox, self._retiring(), cost=60.0)
        with _patch_monotonic(lambda: now[0]):
            assert await _read(sandbox) == b"ok"
        assert sandbox._stub.ReadFile.call_args.kwargs["timeout"] == 30

    @pytest.mark.asyncio
    async def test_zero_timeout_gateway_read_still_calls_rpc(self) -> None:
        sandbox = _running(DataPlaneMode.GATEWAY)
        sandbox._stub.ReadFile = AsyncMock(return_value=sandbox_pb2.ReadFileResponse(content=b"ok"))
        p1, p2, p3 = _read_patches(sandbox)
        with p1, p2, p3:  # type: ignore[attr-defined]
            assert await sandbox._read_file_unary_async("/tmp/f", 0) == b"ok"
        assert sandbox._stub.ReadFile.call_args.kwargs["timeout"] == 0

    @pytest.mark.asyncio
    async def test_channel_replaced_during_backoff_stops_retrying(self) -> None:
        sandbox = _running(DataPlaneMode.GATEWAY)
        gateway_stub = sandbox._stub
        gateway_stub.ReadFile = AsyncMock(
            side_effect=[_hinted(), sandbox_pb2.ReadFileResponse(content=b"ok")]
        )
        real_sleep = asyncio.sleep

        async def _sleep_then_close(seconds: float) -> None:
            sandbox._stub = MagicMock()  # channel closed and reopened meanwhile
            await real_sleep(0)

        with (
            _patch_sleep(_sleep_then_close),
            pytest.raises(SandboxUnavailableError),
        ):
            await _read(sandbox)
        assert gateway_stub.ReadFile.await_count == 1
