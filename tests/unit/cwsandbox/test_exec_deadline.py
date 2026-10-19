# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Execution budgets are enforced by the StreamExec transport deadline."""

import asyncio
import time
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from cwsandbox import Sandbox
from cwsandbox._proto import sandbox_pb2
from cwsandbox.exceptions import SandboxTimeoutError


class DeadlineCall:
    """A transport double that honors the supplied deadline."""

    def __init__(self, timeout, *, duration=0.1, output=False, unavailable=False):
        self.deadline = time.monotonic() + timeout if timeout is not None else None
        self.duration = duration
        self.output = output
        self.unavailable = unavailable
        self.cancel = MagicMock()

    async def responses(self):
        if self.output:
            yield sandbox_pb2.ExecStreamResponse(
                output=sandbox_pb2.ExecStreamOutput(
                    stream=sandbox_pb2.ExecStreamOutput.STREAM_STDOUT, data=b"started\n"
                )
            )
        remaining = max(0, self.deadline - time.monotonic()) if self.deadline else None
        try:
            await asyncio.wait_for(asyncio.sleep(self.duration), timeout=remaining)
        except TimeoutError:
            raise grpc.aio.AioRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED, (), (), "deadline exceeded"
            ) from None
        if self.unavailable:
            raise grpc.aio.AioRpcError(grpc.StatusCode.UNAVAILABLE, (), (), "try again")
        yield sandbox_pb2.ExecStreamResponse(exit=sandbox_pb2.ExecStreamExit(exit_code=0))

    def __aiter__(self):
        return self.responses()


def sandbox_with_transport(*, duration=0.1, output=False, unavailable=False):
    sandbox = Sandbox()
    sandbox._sandbox_id = "test-sandbox"
    calls = []

    def stream(**kwargs):
        call = DeadlineCall(
            kwargs["timeout"], duration=duration, output=output, unavailable=unavailable
        )
        calls.append(call)
        return call

    prepared = MagicMock(metadata=(), is_direct=unavailable)
    prepared.stub.StreamExec.side_effect = stream
    prepared.release = AsyncMock()
    prepared.discard = AsyncMock()
    sandbox._prepare_streaming_call = AsyncMock(return_value=prepared)
    return sandbox, prepared, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("output", [False, True])
async def test_exec_deadline_expires_before_command_finishes(binary, output):
    sandbox, prepared, calls = sandbox_with_transport(output=output)
    stdout, stderr = asyncio.Queue(), asyncio.Queue()
    with pytest.raises(SandboxTimeoutError):
        if binary:
            await sandbox._exec_streaming_binary_async(
                ["sleep", "1"], timeout_seconds=0.02, operation="Execute command"
            )
        else:
            await sandbox._exec_streaming_async(
                ["sleep", "1"], stdout, stderr, timeout_seconds=0.02
            )
    assert 0 < prepared.stub.StreamExec.call_args.kwargs["timeout"] <= 0.02
    calls[0].cancel.assert_called_once()
    if not binary:
        if output:
            assert stdout.get_nowait() == "started\n"
        assert stdout.get_nowait() is None
        assert stderr.get_nowait() is None


@pytest.mark.asyncio
async def test_initial_readiness_wait_does_not_consume_exec_budget():
    sandbox, prepared, _ = sandbox_with_transport(duration=0)

    async def prepare():
        await asyncio.sleep(0.03)
        return prepared

    sandbox._prepare_streaming_call.side_effect = prepare
    result = await sandbox._exec_streaming_async(
        ["true"], asyncio.Queue(), asyncio.Queue(), timeout_seconds=0.02
    )
    assert result.returncode == 0
    assert 0.01 < prepared.stub.StreamExec.call_args.kwargs["timeout"] <= 0.02


@pytest.mark.asyncio
async def test_reconnect_cannot_reset_execution_budget():
    sandbox, prepared, calls = sandbox_with_transport(duration=0.01, unavailable=True)
    attempts = 0

    async def prepare():
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            await asyncio.sleep(0.1)
        return prepared

    sandbox._prepare_streaming_call.side_effect = prepare
    stdout, stderr = asyncio.Queue(), asyncio.Queue()
    with pytest.raises(SandboxTimeoutError, match="reconnecting"):
        await sandbox._exec_streaming_async(["true"], stdout, stderr, timeout_seconds=0.04)
    assert attempts == 2
    assert len(calls) == 1
    assert stdout.get_nowait() is None
    assert stderr.get_nowait() is None


@pytest.mark.asyncio
async def test_retry_uses_remaining_execution_budget():
    sandbox, prepared, calls = sandbox_with_transport(duration=0.01, unavailable=True)
    attempts = 0

    async def prepare():
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            await asyncio.sleep(0.01)
            prepared.stub.StreamExec.side_effect = lambda **kw: DeadlineCall(
                kw["timeout"], duration=0
            )
        return prepared

    sandbox._prepare_streaming_call.side_effect = prepare
    result = await sandbox._exec_streaming_async(
        ["true"], asyncio.Queue(), asyncio.Queue(), timeout_seconds=1
    )
    assert result.returncode == 0
    assert len(calls) == 1
    assert prepared.stub.StreamExec.call_count == 2
    first, second = prepared.stub.StreamExec.call_args_list
    assert 0 < second.kwargs["timeout"] < first.kwargs["timeout"] <= 1


@pytest.mark.asyncio
async def test_cancelled_exec_cancels_transport_and_closes_streams():
    sandbox, _, calls = sandbox_with_transport(duration=10)
    stdout, stderr = asyncio.Queue(), asyncio.Queue()
    task = asyncio.create_task(
        sandbox._exec_streaming_async(["sleep", "10"], stdout, stderr, timeout_seconds=20)
    )
    while not calls:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    calls[0].cancel.assert_called_once()
    assert stdout.get_nowait() is None
    assert stderr.get_nowait() is None


@pytest.mark.asyncio
async def test_binary_preparation_is_bounded_by_deadline():
    sandbox, prepared, _ = sandbox_with_transport()
    cancelled = asyncio.Event()

    async def prepare():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    sandbox._prepare_streaming_call.side_effect = prepare
    with pytest.raises(SandboxTimeoutError):
        await asyncio.wait_for(
            sandbox._exec_streaming_binary_async(
                ["true"], timeout_seconds=0.02, operation="Read file"
            ),
            timeout=1,
        )
    assert cancelled.is_set()
    prepared.stub.StreamExec.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_reconnect", [False, True])
async def test_binary_retry_preserves_deadline(slow_reconnect):
    sandbox, prepared, calls = sandbox_with_transport(duration=0.01, unavailable=True)
    attempts = 0

    async def prepare():
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            await asyncio.sleep(1 if slow_reconnect else 0.01)
            prepared.stub.StreamExec.side_effect = lambda **kw: DeadlineCall(
                kw["timeout"], duration=0
            )
        return prepared

    sandbox._prepare_streaming_call.side_effect = prepare
    with patch("cwsandbox._sandbox._is_runner_shard_retiring_error", return_value=True):
        operation = sandbox._exec_streaming_binary_async(
            ["true"], timeout_seconds=0.2, operation="Read file"
        )
        if slow_reconnect:
            with pytest.raises(SandboxTimeoutError, match="connecting"):
                await asyncio.wait_for(operation, timeout=2)
        else:
            assert (await operation)[0] == 0
            first, second = prepared.stub.StreamExec.call_args_list
            assert 0 < second.kwargs["timeout"] < first.kwargs["timeout"] <= 0.2
    assert attempts == 2
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_exec", [False, True])
async def test_exec_cleans_up_pending_stdin_read(cancel_exec):
    sandbox, prepared, _ = sandbox_with_transport()
    stdin = asyncio.Queue()
    baseline = asyncio.all_tasks()
    sender = None

    class StdinCall(DeadlineCall):
        async def responses(self):
            yield sandbox_pb2.ExecStreamResponse(ready=sandbox_pb2.ExecStreamReady())
            async for response in super().responses():
                yield response

    async def consume_requests(iterator):
        async for _ in iterator:
            pass

    def stream(**kwargs):
        nonlocal sender
        sender = asyncio.create_task(consume_requests(kwargs["request_iterator"]))
        call = StdinCall(kwargs["timeout"], duration=10)
        call.cancel.side_effect = sender.cancel
        return call

    prepared.stub.StreamExec.side_effect = stream
    task = asyncio.create_task(
        sandbox._exec_streaming_async(
            ["cat"],
            asyncio.Queue(),
            asyncio.Queue(),
            stdin_queue=stdin,
            timeout_seconds=1 if cancel_exec else 0.05,
        )
    )
    try:
        if cancel_exec:

            async def wait_for_stdin():
                while not stdin._getters:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_stdin(), timeout=1)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel_exec else SandboxTimeoutError):
            await asyncio.wait_for(task, timeout=2)
        assert sender is not None
        with suppress(asyncio.CancelledError):
            await sender
        assert not stdin._getters
        assert not (asyncio.all_tasks() - baseline)
    finally:
        remaining = asyncio.all_tasks() - baseline
        for pending in remaining:
            pending.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
