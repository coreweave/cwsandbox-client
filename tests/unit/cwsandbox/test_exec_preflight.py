# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Exec streams must finish even when the command never starts."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cwsandbox import Sandbox, SandboxStatus
from cwsandbox._sandbox import _Terminal
from cwsandbox._types import StreamWriter
from cwsandbox.exceptions import SandboxExecutionError, SandboxNotRunningError
from tests.unit.cwsandbox.test_exec_deadline import sandbox_with_transport


@pytest.mark.asyncio
async def test_stopped_sandbox_streams_finish_before_result_is_consumed():
    sandbox = Sandbox()
    sandbox._sandbox_id = "stopped-sandbox"
    sandbox._state = _Terminal(sandbox_id="stopped-sandbox", status=SandboxStatus.COMPLETED)
    sandbox._ensure_started_async = AsyncMock()
    process = sandbox.exec(["echo", "hello"], stdin=True)

    async def drain(reader):
        return [chunk async for chunk in reader]

    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.gather(drain(process.stdout), drain(process.stderr)), timeout=1
        )
        assert stdout == stderr == []
        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            await process
        assert process.stdin is not None
        with pytest.raises(SandboxExecutionError, match="stream has failed"):
            process.stdin.write(b"late input")
    finally:
        process.stdout.close()
        process.stderr.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["prepare", "transport", "retry_cleanup"])
async def test_exec_failure_closes_streams_and_stdin(failure_stage):
    sandbox, prepared, _ = sandbox_with_transport(
        duration=0, unavailable=failure_stage == "retry_cleanup"
    )
    failure = RuntimeError("connection setup failed")
    if failure_stage == "prepare":
        sandbox._prepare_streaming_call.side_effect = failure
    elif failure_stage == "transport":
        prepared.stub.StreamExec.side_effect = failure
    else:
        prepared.discard.side_effect = failure

    stdout, stderr, stdin = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
    writer = StreamWriter(stdin, MagicMock())
    with pytest.raises(RuntimeError, match="connection setup failed"):
        await sandbox._exec_streaming_async(
            ["cat"], stdout, stderr, stdin_queue=stdin, stdin_writer=writer
        )

    assert stdout.get_nowait() is None
    assert stderr.get_nowait() is None
    assert stdout.empty() and stderr.empty()
    with pytest.raises(SandboxExecutionError, match="stream has failed"):
        writer.write(b"late input")


@pytest.mark.asyncio
async def test_cancellation_during_preparation_closes_streams():
    sandbox, _, _ = sandbox_with_transport()
    preparing = asyncio.Event()

    async def prepare():
        preparing.set()
        await asyncio.Event().wait()

    sandbox._prepare_streaming_call.side_effect = prepare
    stdout, stderr = asyncio.Queue(), asyncio.Queue()
    task = asyncio.create_task(sandbox._exec_streaming_async(["true"], stdout, stderr))
    await asyncio.wait_for(preparing.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stdout.get_nowait() is None
    assert stderr.get_nowait() is None
