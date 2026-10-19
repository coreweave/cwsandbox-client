# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Stream consumers and producers may run on different event loops."""

import asyncio
from collections.abc import Iterator
from contextlib import suppress

import pytest

from cwsandbox._loop_manager import _LoopManager
from cwsandbox._types import StreamReader


@pytest.fixture
def manager() -> Iterator[_LoopManager]:
    _LoopManager._reset_for_testing()
    manager = _LoopManager.get()
    loop = manager.get_loop()
    loop.set_debug(True)
    yield manager
    _LoopManager._reset_for_testing()
    loop.close()


async def bound_queue() -> asyncio.Queue:
    """Bind an empty queue to the producer loop before returning it."""
    queue = asyncio.Queue(maxsize=1)
    waiter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    waiter.cancel()
    with suppress(asyncio.CancelledError):
        await waiter
    return queue


async def wait_for_getter(queue: asyncio.Queue) -> None:
    while not queue._getters:
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["line\n", b"bytes", None, ValueError("producer failed")])
async def test_async_reader_receives_from_producer_loop(manager, value):
    queue = await asyncio.wrap_future(manager.run_async(bound_queue()))
    reader = StreamReader(queue, manager)

    async def produce():
        await asyncio.sleep(0.01)
        await queue.put(value)

    producer = manager.run_async(produce())
    try:
        if value is None:
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(reader), timeout=2)
        elif isinstance(value, Exception):
            with pytest.raises(ValueError, match="producer failed"):
                await asyncio.wait_for(anext(reader), timeout=2)
        else:
            assert await asyncio.wait_for(anext(reader), timeout=2) == value
    finally:
        await asyncio.wrap_future(producer)


@pytest.mark.asyncio
async def test_close_wakes_pending_async_read(manager):
    queue = await asyncio.wrap_future(manager.run_async(bound_queue()))
    reader = StreamReader(queue, manager)
    consumer = asyncio.create_task(anext(reader))
    try:
        await asyncio.wait_for(
            asyncio.wrap_future(manager.run_async(wait_for_getter(queue))), timeout=2
        )
        reader.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(consumer, timeout=2)
    finally:
        consumer.cancel()
        with suppress(asyncio.CancelledError, StopAsyncIteration, RuntimeError):
            await consumer


@pytest.mark.asyncio
async def test_cancelled_read_does_not_consume_next_item(manager):
    queue = await asyncio.wrap_future(manager.run_async(bound_queue()))
    reader = StreamReader(queue, manager)
    consumer = asyncio.create_task(anext(reader))
    await asyncio.wait_for(
        asyncio.wrap_future(manager.run_async(wait_for_getter(queue))), timeout=2
    )
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.wrap_future(manager.run_async(queue.put("next")))
    assert await asyncio.wait_for(anext(reader), timeout=2) == "next"


def test_close_wakes_pending_sync_read(manager):
    from concurrent.futures import ThreadPoolExecutor

    queue = manager.run_sync(bound_queue())
    reader = StreamReader(queue, manager)
    with ThreadPoolExecutor(max_workers=1) as pool:
        consumer = pool.submit(next, reader)
        manager.run_async(wait_for_getter(queue)).result(timeout=2)
        reader.close()
        with pytest.raises(StopIteration):
            consumer.result(timeout=2)


@pytest.mark.asyncio
async def test_close_with_full_queue_ends_iteration(manager):
    queue = await asyncio.wrap_future(manager.run_async(bound_queue()))
    await asyncio.wrap_future(manager.run_async(queue.put("buffered")))
    reader = StreamReader(queue, manager)
    reader.close()
    await asyncio.wrap_future(manager.run_async(asyncio.sleep(0)))
    with pytest.raises(StopAsyncIteration):
        await anext(reader)


@pytest.mark.asyncio
async def test_reader_also_works_on_producer_loop(manager):
    async def consume():
        queue = asyncio.Queue()
        await queue.put("same loop")
        return await anext(StreamReader(queue, manager))

    assert await asyncio.wrap_future(manager.run_async(consume())) == "same loop"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["last line", None, ValueError("producer failed")])
@pytest.mark.parametrize("resume_sync", [False, True])
async def test_cancel_after_dequeue_preserves_item(manager, value, resume_sync):
    queue = await asyncio.wrap_future(manager.run_async(bound_queue()))
    reader = StreamReader(queue, manager)
    consumer = asyncio.create_task(anext(reader))
    await asyncio.wait_for(
        asyncio.wrap_future(manager.run_async(wait_for_getter(queue))), timeout=2
    )
    # Let the owner loop dequeue while this loop cannot deliver the result.
    manager.run_async(queue.put(value)).result(timeout=2)
    manager.run_async(asyncio.sleep(0)).result(timeout=2)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    # A distinct following item makes loss fail without a blocked reader.
    await asyncio.wrap_future(manager.run_async(queue.put("unexpected" if value is None else None)))
    if resume_sync:
        if value is None:
            with pytest.raises(StopIteration):
                next(reader)
        elif isinstance(value, Exception):
            with pytest.raises(ValueError, match="producer failed"):
                next(reader)
        else:
            assert next(reader) == value
    elif value is None:
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(reader), timeout=2)
    elif isinstance(value, Exception):
        with pytest.raises(ValueError, match="producer failed"):
            await asyncio.wait_for(anext(reader), timeout=2)
    else:
        assert await asyncio.wait_for(anext(reader), timeout=2) == value
