# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for polling deduplication in _wait_until_running_async."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from cwsandbox import Sandbox
from cwsandbox._proto import gateway_pb2
from cwsandbox._sandbox import (
    SandboxStatus,
    _classify_poll_error,
    _NotStarted,
    _Running,
    _Starting,
    _Stopping,
    _Terminal,
)
from cwsandbox.exceptions import (
    CWSandboxAuthenticationError,
    CWSandboxError,
    SandboxCommandTimeoutError,
    SandboxError,
    SandboxFailedError,
    SandboxNotFoundError,
    SandboxNotRunningError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxTerminatedError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eliminate poll-interval sleeps so tests run on mock timing alone."""
    monkeypatch.setattr("cwsandbox._sandbox.DEFAULT_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr("cwsandbox._sandbox.DEFAULT_MAX_POLL_INTERVAL_SECONDS", 0.0)


def _install_recording_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> list[float]:
    """Replace ``cwsandbox._sandbox.asyncio.sleep`` with a yielding recorder.

    Unlike ``AsyncMock()``, this wrapper still yields to the event loop via a
    no-op future awaited with the original ``asyncio.sleep`` so cancellation
    scheduling remains correct. Returns a list that each invocation appends to
    for post-hoc inspection.

    Patching ``cwsandbox._sandbox.asyncio.sleep`` actually monkeypatches the
    shared ``asyncio`` module's ``sleep`` attribute, so we must capture the
    original before installing the recorder to avoid recursing into the
    recorder from the yield.
    """
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def _recording_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        await real_sleep(0)  # yield but don't actually wait

    monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", _recording_sleep)
    return sleep_calls


def _make_sandbox(
    sandbox_id: str = "test-sandbox",
    status: SandboxStatus | None = None,
    timeout: float = 300.0,
) -> Sandbox:
    """Create a Sandbox wired up for async polling tests.

    Sets _sandbox_id, _channel, _stub, and _auth_metadata so that
    _ensure_started_async and _ensure_client are no-ops.
    """
    sandbox = Sandbox(command="sleep", args=["infinity"])
    sandbox._sandbox_id = sandbox_id
    sandbox._channel = MagicMock()
    sandbox._stub = MagicMock()
    sandbox._auth_metadata = ()
    sandbox._request_timeout_seconds = timeout
    if status is not None:
        if status in (SandboxStatus.RUNNING, SandboxStatus.PAUSED):
            sandbox._state = _Running(sandbox_id=sandbox_id, status=status)
        elif status in (SandboxStatus.COMPLETED, SandboxStatus.FAILED, SandboxStatus.TERMINATED):
            sandbox._state = _Terminal(sandbox_id=sandbox_id, status=status)
    else:
        sandbox._state = _Starting(sandbox_id=sandbox_id)
    return sandbox


def _get_response(status: int, **kwargs: object) -> MagicMock:
    """Build a mock GetSandboxResponse with the given proto status."""
    resp = MagicMock()
    resp.sandbox_status = status
    resp.runner_id = kwargs.get("runner_id", "")
    resp.runner_group_id = kwargs.get("runner_group_id", "")
    resp.profile_id = kwargs.get("profile_id", "")
    resp.started_at_time = kwargs.get("started_at_time", None)
    resp.returncode = kwargs.get("returncode", 0)
    return resp


# ---------------------------------------------------------------------------
# _wait_until_running_async tests
# ---------------------------------------------------------------------------


class TestRunningDedup:
    """Verify _wait_until_running_async shares a single polling task."""

    @pytest.mark.asyncio
    async def test_concurrent_waiters_share_single_poll(self) -> None:
        """N concurrent waiters produce only one set of Get calls, not N sets."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _get_response(gateway_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        n_waiters = 5
        tasks = [asyncio.create_task(sandbox._wait_until_running_async()) for _ in range(n_waiters)]
        await asyncio.gather(*tasks)

        # The poll loop ran 3 times (2 PENDING + 1 RUNNING).
        # Without dedup it would be N*3 = 15.
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fast_path_skips_api_call(self) -> None:
        """When status is already RUNNING, no API calls are made."""
        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        await sandbox._wait_until_running_async()
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_completed_returns(self) -> None:
        """COMPLETED terminal state returns without error or API call."""
        sandbox = _make_sandbox(status=SandboxStatus.COMPLETED)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        await sandbox._wait_until_running_async()
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_failed_raises(self) -> None:
        """FAILED terminal state raises SandboxFailedError."""
        sandbox = _make_sandbox(status=SandboxStatus.FAILED)

        with pytest.raises(SandboxFailedError):
            await sandbox._wait_until_running_async()

    @pytest.mark.asyncio
    async def test_fast_path_terminated_raises(self) -> None:
        """TERMINATED terminal state raises SandboxTerminatedError."""
        sandbox = _make_sandbox(status=SandboxStatus.TERMINATED)

        with pytest.raises(SandboxTerminatedError):
            await sandbox._wait_until_running_async()

    @pytest.mark.asyncio
    async def test_fast_path_paused_returns(self) -> None:
        """PAUSED state is treated as running - returns without API call."""
        sandbox = _make_sandbox()
        sandbox._state = _Running(sandbox_id="test-sandbox", status=SandboxStatus.PAUSED)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        await sandbox._wait_until_running_async()
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_through_to_paused(self) -> None:
        """Sandbox starting in _Starting transitions to _Running with PAUSED status."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _get_response(gateway_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_PAUSED)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        assert isinstance(sandbox._state, _Running)
        assert sandbox._state.status == SandboxStatus.PAUSED
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_fast_path_cancelled_raises(self) -> None:
        """Cancelled _NotStarted raises SandboxNotRunningError."""
        sandbox = _make_sandbox()
        sandbox._state = _NotStarted(cancelled=True)

        with pytest.raises(SandboxNotRunningError, match="cancelled before starting"):
            await sandbox._wait_until_running_async()

    @pytest.mark.asyncio
    async def test_cancellation_isolation(self) -> None:
        """Cancelling one waiter does not cancel the shared polling task."""
        poll_round = 0
        advance = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal poll_round
            poll_round += 1
            if poll_round == 1:
                advance.set()
                return _get_response(gateway_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        waiter_a = asyncio.create_task(sandbox._wait_until_running_async())
        waiter_b = asyncio.create_task(sandbox._wait_until_running_async())

        # Wait until the poll loop is underway, then cancel waiter_a
        await advance.wait()
        waiter_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_a

        # waiter_b should still complete
        await waiter_b
        assert sandbox.status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_timeout_isolation(self) -> None:
        """A short-timeout waiter fails without killing a long-timeout waiter."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                await asyncio.sleep(0.05)
                return _get_response(gateway_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox(timeout=60.0)
        sandbox._stub.Get = mock_get

        short_waiter = asyncio.create_task(sandbox._wait_until_running_async(timeout=0.01))
        long_waiter = asyncio.create_task(sandbox._wait_until_running_async(timeout=60.0))

        # Short waiter should time out
        with pytest.raises(SandboxTimeoutError):
            await short_waiter

        # Long waiter should succeed
        await long_waiter
        assert sandbox.status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_while_waiting(self) -> None:
        """Calling stop() while a waiter is blocked raises SandboxNotRunningError."""
        poll_started = asyncio.Event()
        stop_called = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            if not stop_called.is_set():
                poll_started.set()
                # Block long enough for stop() to fire
                await asyncio.sleep(10)
                return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)
            # After stop, return terminal immediately
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get
        stop_response = MagicMock()
        stop_response.success = True
        sandbox._stub.Stop = AsyncMock(return_value=stop_response)
        sandbox._channel.close = AsyncMock()

        waiter = asyncio.create_task(sandbox._wait_until_running_async())
        await poll_started.wait()

        # Stop the sandbox (cancels the running task, polls to terminal)
        stop_called.set()
        await sandbox._stop_async()

        with pytest.raises(SandboxNotRunningError):
            await waiter

    @pytest.mark.asyncio
    async def test_task_slot_cleared_after_fatal_error(self) -> None:
        """After a fatal poll failure, _running_task is cleared so a fresh waiter retries.

        A fatal status code (PERMISSION_DENIED) surfaces through the retry
        classifier without retry; the task slot must still be cleared so the
        next waiter starts a fresh poll instead of joining the dead task.
        """
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.PERMISSION_DENIED)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        # First waiter gets the translated fatal error
        with pytest.raises(CWSandboxAuthenticationError):
            await sandbox._wait_until_running_async()

        # _on_poll_task_done should have cleared _running_task
        assert sandbox._running_task is None

        # Second waiter retries and succeeds (fresh task)
        await sandbox._wait_until_running_async()
        assert sandbox.status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_running_waiters_share_retry_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two startup waiters share _running_task across a retry sequence."""
        _install_recording_sleep(monkeypatch)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        waiter_a = asyncio.create_task(sandbox._wait_until_running_async())
        waiter_b = asyncio.create_task(sandbox._wait_until_running_async())
        await asyncio.gather(waiter_a, waiter_b)

        # Shared task - Get count equals the sequence length (UNAVAILABLE then
        # RUNNING), not 2x.
        assert attempt == 2
        assert sandbox.status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_running_retry_through_to_terminating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup retry lands on TERMINATING: sandbox enters _Stopping, not _Running."""
        _install_recording_sleep(monkeypatch)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_TERMINATING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        # TERMINATING is a stable status from _poll_until_stable's perspective,
        # so _wait_until_running_async returns (the sandbox is past the startup
        # transient), and the lifecycle state reflects _Stopping, not _Running.
        await sandbox._wait_until_running_async()

        assert attempt == 2
        assert isinstance(sandbox._state, _Stopping)


# ---------------------------------------------------------------------------
# _wait_until_complete_async tests
# ---------------------------------------------------------------------------


class TestCompleteDedup:
    """Verify _wait_until_complete_async shares a single polling task."""

    @pytest.mark.asyncio
    async def test_concurrent_waiters_share_single_poll(self) -> None:
        """N concurrent waiters produce only one set of Get calls, not N sets."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        n_waiters = 5
        tasks = [
            asyncio.create_task(sandbox._wait_until_complete_async()) for _ in range(n_waiters)
        ]
        await asyncio.gather(*tasks)

        # Poll loop: 2 RUNNING + 1 COMPLETED = 3 calls. Without dedup: 15.
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_per_waiter_raise_on_termination(self) -> None:
        """One waiter raises on TERMINATED, the other does not."""

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            return _get_response(gateway_pb2.SANDBOX_STATUS_TERMINATED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        raises_waiter = asyncio.create_task(
            sandbox._wait_until_complete_async(raise_on_termination=True)
        )
        quiet_waiter = asyncio.create_task(
            sandbox._wait_until_complete_async(raise_on_termination=False)
        )

        with pytest.raises(SandboxTerminatedError):
            await raises_waiter

        # Should not raise
        await quiet_waiter

    @pytest.mark.asyncio
    async def test_fast_path_when_terminal(self) -> None:
        """When state is already terminal, no API calls are made."""
        sandbox = _make_sandbox(status=SandboxStatus.COMPLETED)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        await sandbox._wait_until_complete_async()
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_failed_raises(self) -> None:
        """FAILED terminal state raises SandboxFailedError without API call."""
        sandbox = _make_sandbox(status=SandboxStatus.FAILED)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        with pytest.raises(SandboxFailedError):
            await sandbox._wait_until_complete_async()
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_terminated_raises(self) -> None:
        """TERMINATED terminal state raises SandboxTerminatedError by default."""
        sandbox = _make_sandbox(status=SandboxStatus.TERMINATED)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        with pytest.raises(SandboxTerminatedError):
            await sandbox._wait_until_complete_async(raise_on_termination=True)
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_terminated_no_raise(self) -> None:
        """TERMINATED with raise_on_termination=False returns without error or API call."""
        sandbox = _make_sandbox(status=SandboxStatus.TERMINATED)
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        await sandbox._wait_until_complete_async(raise_on_termination=False)
        sandbox._stub.Get.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_isolation(self) -> None:
        """Cancelling one waiter does not cancel the shared polling task."""
        poll_round = 0
        advance = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal poll_round
            poll_round += 1
            if poll_round == 1:
                advance.set()
                return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        waiter_a = asyncio.create_task(sandbox._wait_until_complete_async())
        waiter_b = asyncio.create_task(sandbox._wait_until_complete_async())

        # Wait until the poll loop is underway, then cancel waiter_a
        await advance.wait()
        waiter_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_a

        # waiter_b should still complete
        await waiter_b
        assert sandbox.status == SandboxStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_timeout_isolation(self) -> None:
        """A short-timeout waiter fails without killing a long-timeout waiter."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                await asyncio.sleep(0.05)
                return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING, timeout=60.0)
        sandbox._stub.Get = mock_get

        short_waiter = asyncio.create_task(sandbox._wait_until_complete_async(timeout=0.01))
        long_waiter = asyncio.create_task(sandbox._wait_until_complete_async(timeout=60.0))

        # Short waiter should time out
        with pytest.raises(SandboxTimeoutError):
            await short_waiter

        # Long waiter should succeed
        await long_waiter
        assert sandbox.status == SandboxStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stop_while_waiting(self) -> None:
        """Calling stop() during wait raises SandboxNotRunningError."""
        poll_started = asyncio.Event()
        stop_called = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            if not stop_called.is_set():
                poll_started.set()
                # Wait for stop to be called, not a fixed sleep
                try:
                    await asyncio.wait_for(stop_called.wait(), timeout=5.0)
                except TimeoutError:
                    pass
                return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)
            # After stop, return terminal immediately
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get
        stop_response = MagicMock()
        stop_response.success = True
        sandbox._stub.Stop = AsyncMock(return_value=stop_response)
        sandbox._channel.close = AsyncMock()

        waiter = asyncio.create_task(sandbox._wait_until_complete_async())
        await poll_started.wait()

        # Stop the sandbox - sets _Stopping and polls to terminal
        stop_called.set()
        await sandbox._stop_async()

        # Waiter sees either CancelledError (-> SandboxNotRunningError via
        # _stop_owned) or a terminal state (-> SandboxTerminatedError)
        with pytest.raises((SandboxNotRunningError, SandboxTerminatedError)):
            await waiter

    @pytest.mark.asyncio
    async def test_task_slot_cleared_after_fatal_error(self) -> None:
        """After a fatal poll failure, _complete_task is cleared so a fresh waiter retries.

        A fatal status code (PERMISSION_DENIED) surfaces through the retry
        classifier without retry; the task slot must still be cleared so the
        next waiter starts a fresh poll instead of joining the dead task.
        """
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.PERMISSION_DENIED)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        with pytest.raises(CWSandboxAuthenticationError):
            await sandbox._wait_until_complete_async()

        assert sandbox._complete_task is None

        await sandbox._wait_until_complete_async()
        assert sandbox.status == SandboxStatus.COMPLETED


class TestPollRpcTimeout:
    """Verify poll Get RPCs use poll_rpc_timeout_seconds, not request_timeout_seconds."""

    @pytest.mark.asyncio
    async def test_poll_get_uses_poll_rpc_timeout(self) -> None:
        """The poll Get call passes _poll_rpc_timeout_seconds as its timeout."""
        captured_timeouts: list[float] = []

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            captured_timeouts.append(timeout)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox(timeout=300.0)
        sandbox._poll_rpc_timeout_seconds = 5.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        assert captured_timeouts == [5.0]
        # request_timeout_seconds is preserved for non-poll RPCs
        assert sandbox._request_timeout_seconds == 300.0


# ---------------------------------------------------------------------------
# _classify_poll_error tests
# ---------------------------------------------------------------------------


class _FakeRpcError(grpc.RpcError):
    """Minimal real grpc.RpcError subclass usable in raise/__cause__ contexts.

    MagicMock(spec=AioRpcError) is not a real BaseException, so it cannot be
    set as __cause__ or raised. A real subclass is required both for
    classifier tests that chain it via __cause__ and for retry tests that
    surface it through _translate_rpc_error.

    Implements just enough of the grpc.RpcError / AioRpcError surface that
    the SDK's translator touches: ``code()``, ``details()``, and
    ``trailing_metadata()``.
    """

    def __init__(self, code: grpc.StatusCode, details: str = "") -> None:
        super().__init__()
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def trailing_metadata(self) -> tuple[tuple[str, str], ...]:
        return ()

    def initial_metadata(self) -> tuple[tuple[str, str], ...]:
        return ()


def _rpc_error(code: grpc.StatusCode) -> _FakeRpcError:
    """Build a grpc.RpcError exposing the given status code."""
    return _FakeRpcError(code)


def _translated_from(code: grpc.StatusCode, cls: type[CWSandboxError]) -> CWSandboxError:
    """Simulate a translated exception chained from a grpc error with the given code."""
    cause = _rpc_error(code)
    exc = cls("translated")
    exc.__cause__ = cause
    return exc


class TestClassifyPollError:
    """Verify the classifier dispatches on exception class.

    The classifier is class-based (not cause-based): it considers the
    translated exception class only. Retryable set is
    ``(SandboxUnavailableError, SandboxRequestTimeoutError,
    SandboxResourceExhaustedError)``. SandboxNotFoundError is always fatal
    (short-circuit). Everything else is fatal.
    """

    def test_unavailable_error_is_retryable(self) -> None:
        """SandboxUnavailableError (from UNAVAILABLE / UNAVAILABLE_REASONS) retries."""
        exc = _translated_from(grpc.StatusCode.UNAVAILABLE, SandboxUnavailableError)
        assert _classify_poll_error(exc) == "retryable"

    def test_request_timeout_error_is_retryable(self) -> None:
        """SandboxRequestTimeoutError (from DEADLINE_EXCEEDED) retries."""
        exc = _translated_from(grpc.StatusCode.DEADLINE_EXCEEDED, SandboxRequestTimeoutError)
        assert _classify_poll_error(exc) == "retryable"

    def test_resource_exhausted_error_is_retryable(self) -> None:
        """SandboxResourceExhaustedError (from RESOURCE_EXHAUSTED) retries."""
        exc = _translated_from(grpc.StatusCode.RESOURCE_EXHAUSTED, SandboxResourceExhaustedError)
        assert _classify_poll_error(exc) == "retryable"

    def test_raw_not_running_is_fatal(self) -> None:
        """Raw SandboxNotRunningError (local-stop / CANCELLED) is fatal, not retryable.

        After R1, only SandboxUnavailableError subclass is retryable. The base
        SandboxNotRunningError covers CANCELLED translation and local-stop
        paths where retry would be wrong.
        """
        exc = SandboxNotRunningError("local stop")
        assert _classify_poll_error(exc) == "fatal"

    def test_not_running_from_cancelled_is_fatal(self) -> None:
        """SandboxNotRunningError translated from CANCELLED classifies fatal.

        _translate_rpc_error maps CANCELLED -> raw SandboxNotRunningError (not
        SandboxUnavailableError), so the class-based classifier correctly
        treats it as fatal: cancellation should not be retried.
        """
        exc = _translated_from(grpc.StatusCode.CANCELLED, SandboxNotRunningError)
        assert _classify_poll_error(exc) == "fatal"

    def test_raw_timeout_error_is_fatal(self) -> None:
        """Raw SandboxTimeoutError (not its retryable subclass) is fatal."""
        exc = SandboxTimeoutError("timed out")
        assert _classify_poll_error(exc) == "fatal"

    def test_command_timeout_is_fatal(self) -> None:
        """SandboxCommandTimeoutError is fatal - the command itself exceeded its budget."""
        exc = SandboxCommandTimeoutError("command timed out")
        assert _classify_poll_error(exc) == "fatal"

    def test_internal_is_fatal(self) -> None:
        """INTERNAL translated to SandboxError is fatal."""
        exc = _translated_from(grpc.StatusCode.INTERNAL, SandboxError)
        assert _classify_poll_error(exc) == "fatal"

    def test_not_found_is_fatal(self) -> None:
        """Class-first check on SandboxNotFoundError returns fatal so retry is skipped."""
        exc = _translated_from(grpc.StatusCode.NOT_FOUND, SandboxNotFoundError)
        assert _classify_poll_error(exc) == "fatal"

    def test_not_found_by_code_only_is_fatal(self) -> None:
        """A non-SandboxNotFoundError CWSandboxError chained from NOT_FOUND is fatal."""
        exc = _translated_from(grpc.StatusCode.NOT_FOUND, SandboxError)
        assert _classify_poll_error(exc) == "fatal"

    def test_reason_mapped_not_found_is_fatal_over_internal(self) -> None:
        """SandboxNotFoundError chained from INTERNAL (reason-mapped) is fatal.

        Older backends can emit ``INTERNAL`` with ``CWSANDBOX_SANDBOX_NOT_FOUND``
        and the translator maps that to ``SandboxNotFoundError``. The class-
        first gate keeps NOT_FOUND fatal regardless of the underlying gRPC
        code.
        """
        exc = _translated_from(grpc.StatusCode.INTERNAL, SandboxNotFoundError)
        assert _classify_poll_error(exc) == "fatal"

    def test_permission_denied_is_fatal(self) -> None:
        exc = _translated_from(grpc.StatusCode.PERMISSION_DENIED, SandboxError)
        assert _classify_poll_error(exc) == "fatal"

    def test_unauthenticated_is_fatal(self) -> None:
        exc = _translated_from(grpc.StatusCode.UNAUTHENTICATED, SandboxError)
        assert _classify_poll_error(exc) == "fatal"

    def test_invalid_argument_is_fatal(self) -> None:
        exc = _translated_from(grpc.StatusCode.INVALID_ARGUMENT, SandboxError)
        assert _classify_poll_error(exc) == "fatal"

    def test_local_stop_without_cause_is_fatal(self) -> None:
        """SandboxNotRunningError raised locally (no __cause__) is not retryable."""
        exc = SandboxNotRunningError("Sandbox has been stopped")
        # No __cause__ set - this is the local-stop path
        assert _classify_poll_error(exc) == "fatal"

    def test_non_grpc_cause_is_fatal(self) -> None:
        """A CWSandboxError with a non-grpc cause is not retryable."""
        exc = SandboxError("wrapped")
        exc.__cause__ = RuntimeError("something else")
        assert _classify_poll_error(exc) == "fatal"


# ---------------------------------------------------------------------------
# _poll_with_retry tests
# ---------------------------------------------------------------------------


class TestPollWithRetry:
    """Verify retry wrapper honors the budget and classification correctly."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[UNAVAILABLE, UNAVAILABLE, RUNNING] reaches RUNNING and returns."""
        _install_recording_sleep(monkeypatch)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt <= 2:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        assert attempt == 3
        assert sandbox.status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_budget_exhaustion_reraises_last_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated UNAVAILABLE with budget=0 re-raises immediately."""
        _install_recording_sleep(monkeypatch)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            raise _rpc_error(grpc.StatusCode.UNAVAILABLE)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 0.0
        sandbox._stub.Get = mock_get

        with pytest.raises(SandboxNotRunningError):
            await sandbox._wait_until_running_async()

        # Budget=0 disables retry, so exactly one attempt occurred.
        assert attempt == 1

    @pytest.mark.asyncio
    async def test_not_found_reraised_for_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NOT_FOUND is classified separately and re-raised without retry."""
        _install_recording_sleep(monkeypatch)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            raise _rpc_error(grpc.StatusCode.NOT_FOUND)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        with pytest.raises(SandboxNotFoundError):
            await sandbox._wait_until_running_async()

        assert attempt == 1  # No retry for NOT_FOUND

    @pytest.mark.asyncio
    async def test_fatal_reraised_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PERMISSION_DENIED is fatal and raises immediately."""
        _install_recording_sleep(monkeypatch)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            raise _rpc_error(grpc.StatusCode.PERMISSION_DENIED)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        # PERMISSION_DENIED is translated to CWSandboxAuthenticationError. The
        # retry classifier treats it as fatal and re-raises without retry.
        with pytest.raises(CWSandboxAuthenticationError):
            await sandbox._wait_until_running_async()

        assert attempt == 1

    @pytest.mark.asyncio
    async def test_retry_budget_bounded_by_wall_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retries stop once wall-clock exceeds the deadline.

        Uses a scripted clock (deque.popleft) that only responds to calls
        from the ``cwsandbox._sandbox`` module and falls back to the real
        clock otherwise, so asyncio's internal ``time.monotonic`` lookups
        do not drain the script. If the retry loop's internal shape changes
        in a way that reorders the scripted calls, the test fails with a
        clear mismatch rather than an opaque assertion.
        """
        import sys
        import time as real_time
        from collections import deque

        # Capture the real monotonic BEFORE monkeypatching - otherwise the
        # patched version replaces time.monotonic globally and the fallback
        # branch recurses.
        real_monotonic = real_time.monotonic

        _install_recording_sleep(monkeypatch)

        # Sequence consumed only by cwsandbox._sandbox.time.monotonic lookups.
        # The retry-deadline timer starts on the first retryable failure, not
        # at function entry, so the first RPC timeout clamp is computed from
        # the static budget (no clock read).
        #   0.0 -> deadline setup on first failure (retry_deadline = 0 + 1 = 1.0)
        #   0.5 -> post-exception gate on first retry (0.5 < 1.0, proceed)
        #   2.0 -> post-sleep gate after first retry (2.0 >= 1.0, raise last_exc)
        clock_values = deque([0.0, 0.5, 2.0])

        def scripted_clock() -> float:
            # Only scripted when the caller is cwsandbox._sandbox; otherwise
            # fall through to the real clock so asyncio's scheduler is not
            # affected.
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller == "cwsandbox._sandbox":
                try:
                    return clock_values.popleft()
                except IndexError:
                    return 2.0
            return real_monotonic()

        monkeypatch.setattr("cwsandbox._sandbox.time.monotonic", scripted_clock)
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            raise _rpc_error(grpc.StatusCode.UNAVAILABLE)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 1.0
        sandbox._stub.Get = mock_get

        with pytest.raises(SandboxNotRunningError):
            await sandbox._wait_until_running_async()

        # Scripted clock: first attempt fires, post-sleep gate stops the
        # second. Exactly 1 Get call.
        assert attempt == 1

    @pytest.mark.asyncio
    async def test_retry_state_does_not_leak_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Budget exhaustion in one poll does not poison subsequent calls."""
        _install_recording_sleep(monkeypatch)

        call_log: list[str] = []

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            if not call_log:
                # First invocation: fail to exhaust the budget
                call_log.append("fail")
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            # Later invocations succeed
            call_log.append("ok")
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 0.0
        sandbox._stub.Get = mock_get

        with pytest.raises(SandboxNotRunningError):
            await sandbox._wait_until_running_async()

        # Reset state for a fresh call
        sandbox._state = _Starting(sandbox_id="test-sandbox")
        sandbox._running_task = None
        # Second call should start with a fresh budget and succeed
        await sandbox._wait_until_running_async()
        assert sandbox.status == SandboxStatus.RUNNING


class TestPollRetryJitterBounds:
    """Verify decorrelated-jitter bounds fed to random.uniform on each retry.

    The computed-backoff path picks ``sleep_for`` from
    ``random.uniform(base, ceiling)`` where
    ``ceiling = min(cap, prev_sleep * backoff_factor, remaining)``.
    This test asserts the exact (low, high) args on consecutive retries
    rather than just the resulting sleep durations.
    """

    @pytest.mark.asyncio
    async def test_jitter_bounds_expand_across_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """random.uniform receives (base, expected_ceiling) on each retry."""
        from cwsandbox._defaults import (
            DEFAULT_MAX_POLL_INTERVAL_SECONDS,
            DEFAULT_POLL_BACKOFF_FACTOR,
            DEFAULT_POLL_INTERVAL_SECONDS,
        )

        # Restore real interval constants since the autouse _fast_polling
        # fixture zeros them out. Override _sandbox module-scope constants
        # that _poll_with_retry reads.
        monkeypatch.setattr(
            "cwsandbox._sandbox.DEFAULT_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS
        )
        monkeypatch.setattr(
            "cwsandbox._sandbox.DEFAULT_MAX_POLL_INTERVAL_SECONDS",
            DEFAULT_MAX_POLL_INTERVAL_SECONDS,
        )

        # No real sleeping - AsyncMock completes instantly.
        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", AsyncMock())

        # Deterministic midpoint selection. Also record args for assertion.
        uniform_args: list[tuple[float, float]] = []

        def fake_uniform(low: float, high: float) -> float:
            uniform_args.append((low, high))
            return (low + high) / 2.0

        monkeypatch.setattr("cwsandbox._sandbox.random.uniform", fake_uniform)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt <= 3:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        # Large budget so `remaining` is not the binding constraint.
        sandbox._poll_retry_budget_seconds = 300.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # Three UNAVAILABLE -> three retry sleeps -> three uniform() calls.
        assert len(uniform_args) == 3
        base = DEFAULT_POLL_INTERVAL_SECONDS
        cap = DEFAULT_MAX_POLL_INTERVAL_SECONDS

        # Retry 1: prev_sleep starts at base=0.2, ceiling=min(cap, 0.2*1.5, remaining)
        # = min(2.0, 0.3, ~300) = 0.3.
        low1, high1 = uniform_args[0]
        assert low1 == base
        assert high1 == pytest.approx(min(cap, base * DEFAULT_POLL_BACKOFF_FACTOR), rel=1e-6)
        first_sleep = (low1 + high1) / 2.0

        # Retry 2: prev_sleep is first_sleep; ceiling grows by factor.
        low2, high2 = uniform_args[1]
        assert low2 == base
        expected_high2 = min(cap, first_sleep * DEFAULT_POLL_BACKOFF_FACTOR)
        # `max(base, ...)` in the formula keeps ceiling >= base.
        assert high2 == pytest.approx(max(base, expected_high2), rel=1e-6)

        # Retry 3: further growth, capped at DEFAULT_MAX_POLL_INTERVAL_SECONDS.
        low3, high3 = uniform_args[2]
        second_sleep = (low2 + high2) / 2.0
        expected_high3 = min(cap, second_sleep * DEFAULT_POLL_BACKOFF_FACTOR)
        assert low3 == base
        assert high3 == pytest.approx(max(base, expected_high3), rel=1e-6)


class TestPollRetryRpcTimeoutClamp:
    """Verify per-RPC timeout is clamped by the remaining retry budget.

    The next attempt's rpc timeout is set to
    ``min(poll_rpc_timeout_seconds, max(0.1, remaining_budget))`` so a
    wedged Get cannot run past the budget ceiling. When budget has plenty
    of headroom the configured rpc_timeout binds; when the remaining
    budget is smaller, remaining binds; at the extreme end, the 0.1s
    floor wins.
    """

    def _make_scoped_scripted_clock(self, monkeypatch: pytest.MonkeyPatch, values: list[float]):
        """Install a scripted clock that only drives cwsandbox._sandbox calls.

        asyncio's internal scheduler also calls time.monotonic(), so the
        scripted values are filtered by caller module. Other callers see the
        real wall clock. We capture the real monotonic BEFORE monkeypatching
        so the fallback does not recurse through the patched binding.
        """
        import sys
        import time as real_time
        from collections import deque

        real_monotonic = real_time.monotonic
        clock_values = deque(values)
        fallback = values[-1] if values else 0.0

        def scripted_clock() -> float:
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller == "cwsandbox._sandbox":
                try:
                    return clock_values.popleft()
                except IndexError:
                    return fallback
            return real_monotonic()

        monkeypatch.setattr("cwsandbox._sandbox.time.monotonic", scripted_clock)

    @pytest.mark.asyncio
    async def test_second_call_uses_rpc_timeout_when_budget_ample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """budget=30s, rpc_timeout=15s: second call timeout equals 15s (rpc_timeout binds)."""
        _install_recording_sleep(monkeypatch)

        # Sequence consumed by cwsandbox._sandbox.time.monotonic lookups.
        # The retry-deadline timer starts on first retryable failure, not at
        # function entry, so the pre-loop clamp does not read the clock:
        #   0.0 -> deadline setup on first failure (retry_deadline = 0 + 30 = 30)
        #   0.0 -> post-exception gate after first attempt (0 < 30, proceed)
        #   0.0 -> post-sleep gate (0 < 30, proceed; post_sleep_remaining = 30)
        # First call clamp: min(15.0, 30.0) = 15.0 (from static budget).
        # Second call clamp: min(15.0, max(0.1, 30.0)) = 15.0.
        self._make_scoped_scripted_clock(monkeypatch, [0.0, 0.0, 0.0])

        captured_timeouts: list[float | None] = []
        attempt = 0

        async def mock_get(
            request: object, timeout: float | None = None, metadata: object = ()
        ) -> MagicMock:
            nonlocal attempt
            captured_timeouts.append(timeout)
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 30.0
        sandbox._poll_rpc_timeout_seconds = 15.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # First call timeout is the pre-loop clamp; second is post-sleep clamp.
        # With ample budget, both clamp to rpc_timeout_seconds=15.0.
        assert len(captured_timeouts) == 2
        assert captured_timeouts[1] == pytest.approx(15.0, rel=1e-6)

    @pytest.mark.asyncio
    async def test_second_call_uses_remaining_when_budget_smaller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """budget=5s, rpc_timeout=15s: post-sleep remaining binds for second call."""
        _install_recording_sleep(monkeypatch)

        # Sequence. Deadline timer starts on first retryable failure, not at
        # function entry; the pre-loop clamp does not read the clock.
        #   0.0 -> deadline setup on first failure (retry_deadline = 0 + 5 = 5)
        #   0.0 -> post-exception gate (0 < 5, proceed)
        #   2.0 -> post-sleep gate (2 < 5, post_sleep_remaining = 3)
        # First call clamp: min(15.0, 5.0) = 5.0 (from static budget).
        # Second call clamp: min(15.0, max(0.1, 3.0)) = 3.0 (remaining binds).
        self._make_scoped_scripted_clock(monkeypatch, [0.0, 0.0, 2.0])

        captured_timeouts: list[float | None] = []
        attempt = 0

        async def mock_get(
            request: object, timeout: float | None = None, metadata: object = ()
        ) -> MagicMock:
            nonlocal attempt
            captured_timeouts.append(timeout)
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 5.0
        sandbox._poll_rpc_timeout_seconds = 15.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # Second call: min(15.0, max(0.1, 3.0)) = 3.0 (remaining binds).
        assert len(captured_timeouts) == 2
        assert captured_timeouts[1] == pytest.approx(3.0, rel=1e-3)

    @pytest.mark.asyncio
    async def test_second_call_floors_at_min_when_budget_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """budget=5s, remaining=~0s after sleep: second call timeout floors at 0.1s."""
        _install_recording_sleep(monkeypatch)

        # Sequence. Deadline timer starts on first retryable failure, not at
        # function entry; the pre-loop clamp does not read the clock.
        #   0.0 -> deadline setup on first failure (retry_deadline = 0 + 5 = 5)
        #   4.9 -> post-exception gate (4.9 < 5, proceed to sleep)
        #   4.95 -> post-sleep gate (4.95 < 5, post_sleep_remaining = 0.05)
        # Second call clamp: min(15.0, max(0.1, 0.05)) = 0.1 (floor wins).
        self._make_scoped_scripted_clock(monkeypatch, [0.0, 4.9, 4.95])

        captured_timeouts: list[float | None] = []
        attempt = 0

        async def mock_get(
            request: object, timeout: float | None = None, metadata: object = ()
        ) -> MagicMock:
            nonlocal attempt
            captured_timeouts.append(timeout)
            attempt += 1
            if attempt == 1:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 5.0
        sandbox._poll_rpc_timeout_seconds = 15.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # Second call timeout: min(15.0, max(0.1, 0.05)) = 0.1 (floor wins).
        assert len(captured_timeouts) == 2
        assert captured_timeouts[1] == pytest.approx(0.1, rel=1e-3)


# ---------------------------------------------------------------------------
# _synthesize_terminal_state and post-stop NOT_FOUND handling
# ---------------------------------------------------------------------------


class TestRetryDelayHonored:
    """Verify _poll_with_retry uses exc.retry_delay when set, capped correctly."""

    @pytest.mark.asyncio
    async def test_hinted_delay_used_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 5s hint is used instead of computed backoff on the first retry."""
        sleep_calls: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", mock_sleep)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                # Raise a translated exception carrying retry_delay directly so
                # we test the _poll_with_retry read path, not the translator.
                exc = SandboxUnavailableError("unavailable", retry_delay=timedelta(seconds=5))
                exc.__cause__ = _rpc_error(grpc.StatusCode.UNAVAILABLE)
                raise exc
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 60.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # The retry sleep should honor the 5s hint exactly (remaining budget
        # is ~60s and the 10s cap does not kick in).
        assert sleep_calls == [5.0]

    @pytest.mark.asyncio
    async def test_hinted_delay_capped_at_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 60s hint is capped at MAX_POLL_RETRY_HINTED_DELAY_SECONDS (10s)."""
        sleep_calls: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", mock_sleep)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                exc = SandboxUnavailableError("unavailable", retry_delay=timedelta(seconds=60))
                exc.__cause__ = _rpc_error(grpc.StatusCode.UNAVAILABLE)
                raise exc
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 300.0  # plenty of room
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        assert sleep_calls == [10.0]

    @pytest.mark.asyncio
    async def test_no_hint_uses_computed_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without retry_delay, backoff starts at DEFAULT_POLL_INTERVAL_SECONDS."""
        sleep_calls: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", mock_sleep)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                # No retry_delay on the exception.
                exc = SandboxUnavailableError("unavailable")
                exc.__cause__ = _rpc_error(grpc.StatusCode.UNAVAILABLE)
                raise exc
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 60.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # Autouse fixture patches DEFAULT_POLL_INTERVAL_SECONDS to 0.0, so
        # computed backoff sleeps for 0s. The key check is that we did not
        # use a nonzero hint.
        assert sleep_calls == [0.0]

    @pytest.mark.asyncio
    async def test_hinted_delay_bounded_by_remaining_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 5s hint is truncated to remaining budget if the budget is smaller."""
        sleep_calls: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", mock_sleep)

        # Time progresses: 0.0 for deadline set, 0.0 when computing sleep.
        # With budget=2.0, remaining at computation is 2.0, hint is 5.0,
        # cap is 10.0 - should pick min(5, 2, 10) = 2.
        current = [0.0]

        def fake_monotonic() -> float:
            return current[0]

        monkeypatch.setattr("cwsandbox._sandbox.time.monotonic", fake_monotonic)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                exc = SandboxUnavailableError("unavailable", retry_delay=timedelta(seconds=5))
                exc.__cause__ = _rpc_error(grpc.StatusCode.UNAVAILABLE)
                raise exc
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 2.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        assert sleep_calls == [2.0]

    @pytest.mark.asyncio
    async def test_zero_hint_falls_back_to_computed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A retry_delay of 0 is treated as absent - fall back to computed backoff."""
        sleep_calls: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", mock_sleep)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                exc = SandboxUnavailableError("unavailable", retry_delay=timedelta(seconds=0))
                exc.__cause__ = _rpc_error(grpc.StatusCode.UNAVAILABLE)
                raise exc
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 60.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # Autouse fixture patches poll interval to 0.0, so computed backoff is 0.
        assert sleep_calls == [0.0]

    @pytest.mark.asyncio
    async def test_negative_hinted_delay_falls_through_to_computed_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A negative retry_delay falls through to the jittered computed backoff.

        The hinted-delay branch gates on ``hinted_delay > 0``. A negative
        timedelta must not drive an ``asyncio.sleep(-5)`` call; it must fall
        through to the computed path and eventually succeed.
        """
        sleep_calls = _install_recording_sleep(monkeypatch)

        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                exc = SandboxUnavailableError("unavailable", retry_delay=timedelta(seconds=-5))
                exc.__cause__ = _rpc_error(grpc.StatusCode.UNAVAILABLE)
                raise exc
            return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._poll_retry_budget_seconds = 60.0
        sandbox._stub.Get = mock_get

        await sandbox._wait_until_running_async()

        # No negative sleep was ever requested.
        assert all(s >= 0 for s in sleep_calls), sleep_calls
        # Exactly one retry sleep (from the computed path).
        assert len(sleep_calls) == 1
        # Autouse fixture patches poll interval to 0.0, so computed jittered
        # backoff is also 0.0.  The key check is non-negativity; the retry
        # then succeeds on the second attempt.
        assert sandbox.status == SandboxStatus.RUNNING
        assert attempt == 2


# ---------------------------------------------------------------------------
# Cross-cutting concurrency tests: retry + NOT_FOUND + shared task semantics
# ---------------------------------------------------------------------------


class TestConcurrencyAcrossRetryAndNotFound:
    """Exercise retry and NOT_FOUND handling under concurrent waiters."""

    @pytest.mark.asyncio
    async def test_two_waiters_share_retry_sequence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two waiters on _complete_task see COMPLETED through mixed retry/success."""
        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", AsyncMock())
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # Sequence: UNAVAILABLE, RUNNING, UNAVAILABLE, COMPLETED
            if call_count == 1:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            if call_count == 2:
                return _get_response(gateway_pb2.SANDBOX_STATUS_RUNNING)
            if call_count == 3:
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        waiter_a = asyncio.create_task(
            sandbox._wait_until_complete_async(raise_on_termination=False)
        )
        waiter_b = asyncio.create_task(
            sandbox._wait_until_complete_async(raise_on_termination=False)
        )
        await asyncio.gather(waiter_a, waiter_b)

        # Shared task - Get count equals the sequence length, not 2x.
        assert call_count == 4
        assert isinstance(sandbox._state, _Terminal)
        assert sandbox._state.status == SandboxStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_two_waiters_see_same_fatal_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent waiters on a failing _complete_task see the same exception.

        Also asserts dedup: Get is called exactly once (fatal, no retry) and
        both waiter tasks share the same _complete_task instance.
        """
        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", AsyncMock())
        call_count = 0
        get_barrier = asyncio.Event()
        second_waiter_joined = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            get_barrier.set()
            await second_waiter_joined.wait()
            raise _rpc_error(grpc.StatusCode.PERMISSION_DENIED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        waiter_a = asyncio.create_task(sandbox._wait_until_complete_async())
        # Wait for first waiter to establish shared _complete_task.
        await get_barrier.wait()
        shared_task_a = sandbox._complete_task
        assert shared_task_a is not None

        waiter_b = asyncio.create_task(sandbox._wait_until_complete_async())
        # Yield so waiter_b enters and joins the shared task rather than
        # creating a new one.
        await asyncio.sleep(0)
        shared_task_b = sandbox._complete_task
        assert shared_task_b is shared_task_a  # same shared task

        # Release the gate; both waiters resolve via the shared task.
        second_waiter_joined.set()
        results = await asyncio.gather(waiter_a, waiter_b, return_exceptions=True)

        # Single Get call: PERMISSION_DENIED classifies fatal, no retry.
        # Shared task means both waiters join it, not 2x Get.
        assert call_count == 1
        # Both waiters see the same exception instance (shared shielded task).
        assert all(isinstance(r, CWSandboxAuthenticationError) for r in results)
        # Same message = same translation from the same underlying gRPC error.
        assert str(results[0]) == str(results[1])

    @pytest.mark.asyncio
    async def test_retrying_task_survives_short_waiter_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short-timeout waiter fails while a long-timeout waiter rides the retry to COMPLETED.

        Validates the asyncio.shield() contract: cancelling one waiter (because
        its timeout fired while the shared retry loop was mid-retry) must not
        kill the shared task. The remaining waiter must still observe COMPLETED.
        """
        # Capture real asyncio.sleep before patching so mock_get can block.
        real_sleep = asyncio.sleep
        monkeypatch.setattr("cwsandbox._sandbox.asyncio.sleep", AsyncMock())
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # Short waiter's timeout fires during the retry sequence; long
            # waiter must still see the final COMPLETED.
            if call_count <= 3:
                await real_sleep(0.05)
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        short_waiter = asyncio.create_task(
            sandbox._wait_until_complete_async(timeout=0.01, raise_on_termination=False)
        )
        long_waiter = asyncio.create_task(
            sandbox._wait_until_complete_async(timeout=60.0, raise_on_termination=False)
        )

        # Short waiter should time out without killing the shared task.
        with pytest.raises(SandboxTimeoutError):
            await short_waiter

        # Long waiter sees COMPLETED after the retry sequence finishes.
        await long_waiter
        assert isinstance(sandbox._state, _Terminal)
        assert sandbox._state.status == SandboxStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancelled_during_retry_sleep_does_not_kill_shared_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling waiter_a mid-retry-sleep does not stop waiter_b from seeing COMPLETED.

        The retry loop uses a real asyncio.sleep for the backoff interval; while
        that sleep is active, cancelling one waiter must leave the shared
        _complete_task alive (asyncio.shield) so the other waiter still sees
        the eventual COMPLETED result.
        """
        # Deliberately NOT patching asyncio.sleep - use a real small interval.
        monkeypatch.setattr("cwsandbox._sandbox.DEFAULT_POLL_INTERVAL_SECONDS", 0.05)
        monkeypatch.setattr("cwsandbox._sandbox.DEFAULT_MAX_POLL_INTERVAL_SECONDS", 0.1)

        call_count = 0
        first_call_started = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_call_started.set()
                raise _rpc_error(grpc.StatusCode.UNAVAILABLE)
            return _get_response(gateway_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._poll_retry_budget_seconds = 5.0
        sandbox._stub.Get = mock_get

        waiter_a = asyncio.create_task(
            sandbox._wait_until_complete_async(raise_on_termination=False)
        )
        waiter_b = asyncio.create_task(
            sandbox._wait_until_complete_async(raise_on_termination=False)
        )

        # Wait for the first Get call to fire, then cancel waiter_a
        # (waiter_a is suspended inside asyncio.wait_for -> asyncio.shield).
        await first_call_started.wait()
        waiter_a.cancel()

        with pytest.raises(asyncio.CancelledError):
            await waiter_a

        # waiter_b should still see the COMPLETED result once the retry sleep
        # finishes and the second Get is made.
        await waiter_b

        assert call_count == 2  # first UNAVAILABLE + second COMPLETED
        assert isinstance(sandbox._state, _Terminal)
        assert sandbox._state.status == SandboxStatus.COMPLETED
