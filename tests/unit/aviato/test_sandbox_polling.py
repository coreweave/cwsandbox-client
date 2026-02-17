# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for polling deduplication in _wait_until_running_async."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreweave.aviato.v1beta1 import atc_pb2

from aviato import Sandbox
from aviato._sandbox import SandboxStatus
from aviato.exceptions import (
    SandboxNotRunningError,
    SandboxTerminatedError,
    SandboxTimeoutError,
)


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eliminate poll-interval sleeps so tests run on mock timing alone."""
    monkeypatch.setattr("aviato._sandbox.DEFAULT_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr("aviato._sandbox.DEFAULT_MAX_POLL_INTERVAL_SECONDS", 0.0)


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
        sandbox._status = status
    return sandbox


def _get_response(status: int, **kwargs: object) -> MagicMock:
    """Build a mock GetSandboxResponse with the given proto status."""
    resp = MagicMock()
    resp.sandbox_status = status
    resp.tower_id = kwargs.get("tower_id", "")
    resp.tower_group_id = kwargs.get("tower_group_id", "")
    resp.runway_id = kwargs.get("runway_id", "")
    resp.started_at_time = kwargs.get("started_at_time", None)
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
                return _get_response(atc_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)

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
    async def test_cancellation_isolation(self) -> None:
        """Cancelling one waiter does not cancel the shared polling task."""
        poll_round = 0
        advance = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal poll_round
            poll_round += 1
            if poll_round == 1:
                advance.set()
                return _get_response(atc_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)

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
        assert sandbox._status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_timeout_isolation(self) -> None:
        """A short-timeout waiter fails without killing a long-timeout waiter."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                await asyncio.sleep(0.05)
                return _get_response(atc_pb2.SANDBOX_STATUS_PENDING)
            return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox(timeout=60.0)
        sandbox._stub.Get = mock_get

        short_waiter = asyncio.create_task(sandbox._wait_until_running_async(timeout=0.01))
        long_waiter = asyncio.create_task(sandbox._wait_until_running_async(timeout=60.0))

        # Short waiter should time out
        with pytest.raises(SandboxTimeoutError):
            await short_waiter

        # Long waiter should succeed
        await long_waiter
        assert sandbox._status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_while_waiting(self) -> None:
        """Calling stop() while a waiter is blocked raises SandboxNotRunningError."""
        poll_started = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            poll_started.set()
            # Block long enough for stop() to fire
            await asyncio.sleep(10)
            return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get
        stop_response = MagicMock()
        stop_response.success = True
        sandbox._stub.Stop = AsyncMock(return_value=stop_response)
        sandbox._channel.close = AsyncMock()

        waiter = asyncio.create_task(sandbox._wait_until_running_async())
        await poll_started.wait()

        # Stop the sandbox (cancels the running task)
        await sandbox._stop_async()

        with pytest.raises(SandboxNotRunningError):
            await waiter

    @pytest.mark.asyncio
    async def test_task_failure_recovery(self) -> None:
        """After a failed poll, _running_task is cleared so the next waiter can retry."""
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise Exception("transient network error")
            return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)

        sandbox = _make_sandbox()
        sandbox._stub.Get = mock_get

        # First waiter gets the error (wrapped by _translate_rpc_error or propagated)
        with pytest.raises(Exception, match="transient network error"):
            await sandbox._wait_until_running_async()

        # _on_poll_task_done should have cleared _running_task
        assert sandbox._running_task is None

        # Second waiter retries and succeeds
        await sandbox._wait_until_running_async()
        assert sandbox._status == SandboxStatus.RUNNING


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
                return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)
            return _get_response(atc_pb2.SANDBOX_STATUS_COMPLETED)

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
            return _get_response(atc_pb2.SANDBOX_STATUS_TERMINATED)

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
    async def test_fast_path_when_returncode_set(self) -> None:
        """When _returncode is already set, no API calls are made."""
        sandbox = _make_sandbox(status=SandboxStatus.COMPLETED)
        sandbox._returncode = 0
        sandbox._stub.Get = AsyncMock(side_effect=AssertionError("should not be called"))

        await sandbox._wait_until_complete_async()
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
                return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)
            return _get_response(atc_pb2.SANDBOX_STATUS_COMPLETED)

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
        assert sandbox._status == SandboxStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_timeout_isolation(self) -> None:
        """A short-timeout waiter fails without killing a long-timeout waiter."""
        call_count = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                await asyncio.sleep(0.05)
                return _get_response(atc_pb2.SANDBOX_STATUS_RUNNING)
            return _get_response(atc_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING, timeout=60.0)
        sandbox._stub.Get = mock_get

        short_waiter = asyncio.create_task(sandbox._wait_until_complete_async(timeout=0.01))
        long_waiter = asyncio.create_task(sandbox._wait_until_complete_async(timeout=60.0))

        # Short waiter should time out
        with pytest.raises(SandboxTimeoutError):
            await short_waiter

        # Long waiter should succeed
        await long_waiter
        assert sandbox._status == SandboxStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stop_while_waiting(self) -> None:
        """Calling stop() during wait raises SandboxNotRunningError."""
        poll_started = asyncio.Event()

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            poll_started.set()
            # Block long enough for stop() to fire
            await asyncio.sleep(10)
            return _get_response(atc_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get
        stop_response = MagicMock()
        stop_response.success = True
        sandbox._stub.Stop = AsyncMock(return_value=stop_response)
        sandbox._channel.close = AsyncMock()

        waiter = asyncio.create_task(sandbox._wait_until_complete_async())
        await poll_started.wait()

        # Stop the sandbox (cancels the complete task)
        await sandbox._stop_async()

        with pytest.raises(SandboxNotRunningError):
            await waiter

    @pytest.mark.asyncio
    async def test_task_failure_recovery(self) -> None:
        """After a failed poll, _complete_task is cleared so the next waiter can retry."""
        attempt = 0

        async def mock_get(request: object, timeout: float = 0, metadata: object = ()) -> MagicMock:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise Exception("transient network error")
            return _get_response(atc_pb2.SANDBOX_STATUS_COMPLETED)

        sandbox = _make_sandbox(status=SandboxStatus.RUNNING)
        sandbox._stub.Get = mock_get

        with pytest.raises(Exception, match="transient network error"):
            await sandbox._wait_until_complete_async()

        assert sandbox._complete_task is None

        await sandbox._wait_until_complete_async()
        assert sandbox._status == SandboxStatus.COMPLETED
