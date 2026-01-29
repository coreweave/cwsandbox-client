"""A Python client library for Aviato sandboxes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar, overload

# Import cleanup module to install atexit and signal handlers
from aviato import _cleanup as _cleanup  # noqa: F401

# Patch ConnectRPC to ignore unknown protobuf fields (must be before other imports)
from aviato import _compat as _compat  # noqa: F401
from aviato._defaults import SandboxDefaults
from aviato._loop_manager import _LoopManager
from aviato._sandbox import Sandbox, SandboxStatus
from aviato._session import Session
from aviato._types import (
    OperationRef,
    Process,
    ProcessResult,
    Serialization,
    StreamReader,
)
from aviato.exceptions import (
    AsyncFunctionError,
    AviatoAuthenticationError,
    AviatoError,
    FunctionError,
    FunctionSerializationError,
    SandboxError,
    SandboxExecutionError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotFoundError,
    SandboxNotRunningError,
    SandboxTerminatedError,
    SandboxTimeoutError,
    WandbAuthError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")

# Type alias for things that can be waited on
Waitable = Sandbox | OperationRef[Any] | Process


@overload
def results(ref: OperationRef[T], /) -> T: ...


@overload
def results(refs: Sequence[OperationRef[T]], /) -> list[T]: ...


def results(refs: OperationRef[T] | Sequence[OperationRef[T]], /) -> T | list[T]:
    """Block for one or more OperationRefs and return results.

    This is a convenience function for retrieving results from OperationRefs.
    For a single ref, returns the result directly. For a sequence of refs,
    returns a list of results in the same order.

    Args:
        refs: A single OperationRef or a sequence of OperationRefs.

    Returns:
        The result(s) from the operation(s).

    Raises:
        Exception: Any exception raised by the underlying operation(s).

    Examples:
        Single ref:
        ```python
        data = aviato.results(sandbox.read_file("/path"))
        ```

        Multiple refs:
        ```python
        all_results = aviato.results([sb.read_file(f) for f in files])
        ```
    """
    if isinstance(refs, OperationRef):
        return refs.result()
    return [ref.result() for ref in refs]


def wait(
    waitables: Sequence[Waitable],
    num_returns: int | None = None,
    timeout: float | None = None,
) -> tuple[list[Waitable], list[Waitable]]:
    """Wait for waitables to complete, return (done, pending).

    Each waitable type has natural "wait for" behavior:
    - Sandbox: waits until RUNNING status
    - OperationRef: waits until operation completes
    - Process: waits until process completes

    Args:
        waitables: Sequence of Sandbox, OperationRef, or Process objects.
        num_returns: If specified, return after this many complete.
            If None, wait for all to complete.
        timeout: Maximum seconds to wait. If None, wait forever.

    Returns:
        Tuple of (done, pending) lists containing the original waitable objects.

    Raises:
        ValueError: If num_returns is less than 1.

    Examples:
        Wait for all sandboxes to be running:
        ```python
        sandboxes = [Sandbox.run(...) for _ in range(5)]
        done, pending = aviato.wait(sandboxes)
        ```

        Wait for first 2 operations to complete:
        ```python
        refs = [sb.read_file(f) for f in files]
        done, pending = aviato.wait(refs, num_returns=2)
        ```

        Wait with timeout:
        ```python
        done, pending = aviato.wait(procs, timeout=30.0)
        ```
    """
    if num_returns is not None and num_returns < 1:
        raise ValueError(f"num_returns must be at least 1, got {num_returns}")

    if not waitables:
        return [], []

    loop_manager = _LoopManager.get()
    return loop_manager.run_sync(_wait_async(waitables, num_returns, timeout))


async def _wait_async(
    waitables: Sequence[Waitable],
    num_returns: int | None,
    timeout: float | None,
) -> tuple[list[Waitable], list[Waitable]]:
    """Internal async implementation of wait()."""
    loop = asyncio.get_running_loop()

    # Calculate deadline for timeout tracking across multiple wait rounds
    deadline: float | None = None
    if timeout is not None:
        deadline = loop.time() + timeout

    def _remaining_timeout() -> float | None:
        """Compute remaining time until deadline, or None if no timeout."""
        if deadline is None:
            return None
        remaining = deadline - loop.time()
        return max(0.0, remaining)

    async def _wrap_future(future: Any) -> Any:
        """Wrap a concurrent.futures.Future as a coroutine."""
        return await asyncio.wrap_future(future)

    def _to_awaitable(w: Waitable) -> asyncio.Task[Any]:
        if isinstance(w, Sandbox):
            return asyncio.create_task(w._wait_until_running_async())
        if isinstance(w, Process):
            return asyncio.create_task(_wrap_future(w._future))
        # OperationRef
        return asyncio.create_task(_wrap_future(w._future))

    # Create tasks mapping back to original objects
    tasks: dict[asyncio.Task[Any], Waitable] = {}
    for w in waitables:
        task = _to_awaitable(w)
        tasks[task] = w

    # Determine return_when strategy
    if num_returns is not None and num_returns < len(waitables):
        return_when = asyncio.FIRST_COMPLETED
    else:
        return_when = asyncio.ALL_COMPLETED

    # Wait for tasks
    done_tasks, pending_tasks = await asyncio.wait(
        tasks.keys(),
        timeout=_remaining_timeout(),
        return_when=return_when,
    )

    # If num_returns specified with FIRST_COMPLETED, may need multiple rounds
    if num_returns is not None and return_when == asyncio.FIRST_COMPLETED:
        while len(done_tasks) < num_returns and pending_tasks:
            remaining = _remaining_timeout()
            # If timeout expired, stop waiting
            if remaining is not None and remaining <= 0:
                break

            more_done, pending_tasks = await asyncio.wait(
                pending_tasks,
                timeout=0,  # Non-blocking check for already-done tasks
                return_when=asyncio.FIRST_COMPLETED,
            )
            done_tasks = done_tasks | more_done
            if not more_done:
                # No more immediately done, need to actually wait
                remaining = _remaining_timeout()
                if remaining is not None and remaining <= 0:
                    break
                if len(done_tasks) < num_returns and pending_tasks:
                    more_done, pending_tasks = await asyncio.wait(
                        pending_tasks,
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    done_tasks = done_tasks | more_done

    # Map tasks back to original waitables
    done = [tasks[t] for t in done_tasks]
    pending = [tasks[t] for t in pending_tasks]

    # Trim to num_returns if we got more than requested
    if num_returns is not None and len(done) > num_returns:
        extra = done[num_returns:]
        done = done[:num_returns]
        pending = extra + pending

    return done, pending


__all__ = [
    "AsyncFunctionError",
    "AviatoAuthenticationError",
    "AviatoError",
    "FunctionError",
    "FunctionSerializationError",
    "OperationRef",
    "Process",
    "ProcessResult",
    "Sandbox",
    "SandboxDefaults",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxFailedError",
    "SandboxFileError",
    "SandboxNotFoundError",
    "SandboxNotRunningError",
    "SandboxStatus",
    "SandboxTerminatedError",
    "SandboxTimeoutError",
    "Serialization",
    "Session",
    "StreamReader",
    "Waitable",
    "WandbAuthError",
    "results",
    "wait",
]
