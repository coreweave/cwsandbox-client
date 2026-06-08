# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Exception hierarchy for CWSandbox operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from cwsandbox._types import ProcessResult


class CWSandboxError(Exception):
    """Base exception for all CWSandbox operations.

    Attributes:
        reason: Structured error reason parsed from ``google.rpc.ErrorInfo``
            (e.g. ``"CWSANDBOX_FILE_NOT_FOUND"``), or ``None`` when the error
            did not carry AIP-193 details.
        metadata: Machine-readable metadata from ``ErrorInfo.metadata``. Empty
            dict when the error carried no metadata.
        retry_delay: Suggested client retry delay parsed from
            ``google.rpc.RetryInfo``, or ``None`` when absent.

    Note:
        ``reason``, ``metadata``, and ``retry_delay`` are raw server-supplied
        values and populate even when the ErrorInfo domain is not trusted.
        Callers that branch on ``reason`` should also verify the exception
        class, which IS domain-gated.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.metadata: dict[str, str] = dict(metadata) if metadata else {}
        self.retry_delay = retry_delay


class CWSandboxAuthenticationError(CWSandboxError):
    """Raised when authentication fails."""


class SandboxError(CWSandboxError):
    """Base exception for sandbox operations."""


class SandboxNotRunningError(SandboxError):
    """Raised when an operation requires a running sandbox."""


class SandboxTimeoutError(SandboxError):
    """Raised when a sandbox operation times out."""


class SandboxTerminatedError(SandboxError):
    """Raised when a sandbox was terminated externally."""


class SandboxFailedError(SandboxError):
    """Raised when a sandbox fails to start or encounters a fatal error."""


class SandboxNotFoundError(SandboxError):
    """Raised when a sandbox is not found (e.g., already deleted).

    Attributes:
        sandbox_id: The ID of the sandbox that was not found, or None.
    """

    def __init__(
        self,
        message: str,
        *,
        sandbox_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message, reason=reason, metadata=metadata, retry_delay=retry_delay)
        self.sandbox_id = sandbox_id


class SandboxExecutionError(SandboxError):
    """Raised when command execution fails inside a sandbox.

    Attributes:
        exec_result: The ``ProcessResult`` from the failed execution, or None.
        exception_type: Python exception class name from sandbox stderr, or None.
        exception_message: Exception message from sandbox stderr, or None.
    """

    def __init__(
        self,
        message: str,
        *,
        exec_result: ProcessResult | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message, reason=reason, metadata=metadata, retry_delay=retry_delay)
        self.exec_result = exec_result
        self.exception_type = exception_type
        self.exception_message = exception_message


class SandboxStreamBackpressureError(SandboxExecutionError):
    """Raised when an output stream ended early because it was not being read
    fast enough to keep up with the command's output.

    When a command produces output faster than your code reads it, the stream
    is ended with this explicit error rather than silently dropping output and
    still reporting success. Some output has therefore likely been lost.

    Subclasses ``SandboxExecutionError`` so existing ``except
    SandboxExecutionError`` handlers still catch it; catch this type
    specifically to handle a too-slow reader distinctly from a command failure.

    Attributes:
        stream_code: The terminal ``ExecStreamError.code`` that triggered this
            exception (``"STREAM_BACKPRESSURE"``). This is a streaming-channel
            code, NOT an AIP-193 ErrorInfo ``reason`` — the two namespaces are
            kept distinct, so ``.reason`` is ``None`` here and callers should
            branch on the exception class (or ``stream_code``) rather than
            ``.reason``.

    How to avoid it:

    - Read the stream as output arrives — iterate the reader / drain stdout in
      a tight loop and move slow work (disk writes, network calls) off the read
      loop. The common cause is doing per-chunk work inline; drain into a fast
      local sink (e.g. a file) first and process afterward. See
      ``examples/large_file_streaming.py``.
    - For very large files, use ``read_file_streaming`` /
      ``write_file_streaming`` (chunked) instead of reading everything at once.
    - If the *destination* itself cannot keep up no matter how tight the loop
      (a rate-limited API, a slow disk, a human watching a terminal), no amount
      of loop-tightening helps — split the work into smaller transfers, or move
      very large payloads out of the streaming path entirely.
    - This is not a transient error — retrying the same consumer pattern will
      hit it again. Fix the read pace (or chunk the work) first, then retry.
    """

    def __init__(
        self,
        message: str,
        *,
        stream_code: str | None = None,
        exec_result: ProcessResult | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        # `reason` is intentionally not accepted/forwarded: STREAM_BACKPRESSURE
        # is an ExecStreamError stream code, not an AIP-193 ErrorInfo reason, so
        # it does not belong in the `.reason` namespace. It is exposed via
        # `.stream_code` instead.
        super().__init__(
            message,
            exec_result=exec_result,
            exception_type=exception_type,
            exception_message=exception_message,
            metadata=metadata,
            retry_delay=retry_delay,
        )
        self.stream_code = stream_code


class SandboxStreamTruncatedError(SandboxExecutionError):
    """Raised when a command's output was truncated in transit, even though the
    command ran to completion.

    The command exited normally, but not all of its output reached you: some
    was lost on the way back. The stream is ended with this explicit error
    rather than silently returning partial output alongside a success exit
    code, so you can tell complete output apart from a quiet truncation.

    Subclasses ``SandboxExecutionError`` so existing ``except
    SandboxExecutionError`` handlers still catch it; catch this type
    specifically to handle truncated output distinctly from a command failure.

    Attributes:
        stream_code: The terminal ``ExecStreamError.code`` that triggered this
            exception (``"STREAM_TRUNCATED"``). This is a streaming-channel
            code, NOT an AIP-193 ErrorInfo ``reason`` — the two namespaces are
            kept distinct, so ``.reason`` is ``None`` here and callers should
            branch on the exception class (or ``stream_code``) rather than
            ``.reason``.

    What to do:

    - For large output, write it to a file in the sandbox and retrieve the file
      (``read_file_streaming``) instead of streaming it back over stdout. This
      avoids the streaming path that truncated and is the recommended approach
      for anything beyond a few megabytes.
    - Re-running the command will stream over the same path and may truncate
      again; it can also have side effects. Re-run only if the command is
      idempotent (a pure read such as ``cat`` / ``ls``), and prefer the
      file-based approach above for large output regardless.
    - Truncation is more likely the larger and faster the output is, and the
      slower or more distant the client; there is no exact size below which it
      cannot happen.
    """

    def __init__(
        self,
        message: str,
        *,
        stream_code: str | None = None,
        exec_result: ProcessResult | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        # `reason` is intentionally not accepted/forwarded: STREAM_TRUNCATED is
        # an ExecStreamError stream code, not an AIP-193 ErrorInfo reason, so it
        # does not belong in the `.reason` namespace. It is exposed via
        # `.stream_code` instead.
        super().__init__(
            message,
            exec_result=exec_result,
            exception_type=exception_type,
            exception_message=exception_message,
            metadata=metadata,
            retry_delay=retry_delay,
        )
        self.stream_code = stream_code


class SandboxFileError(SandboxError):
    """Raised when a file operation fails in the sandbox.

    This is a sandbox infrastructure error, not a user code error.
    Inherits from SandboxError since it's a sandbox operation failure.

    Attributes:
        filepath: The path of the file that caused the error, or None.
    """

    def __init__(
        self,
        message: str,
        *,
        filepath: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message, reason=reason, metadata=metadata, retry_delay=retry_delay)
        self.filepath = filepath


class SandboxUnavailableError(SandboxNotRunningError):
    """Raised when the sandbox service is transiently unavailable.

    Emitted for gRPC ``UNAVAILABLE`` and AIP-193 UNAVAILABLE_REASONS.
    Poll loop treats this as retryable. This retry contract is
    poll-specific; callers of non-poll operations must decide their
    own retry policy.
    """


class SandboxRequestTimeoutError(SandboxTimeoutError):
    """Raised when a gRPC request exceeded its deadline.

    Emitted for gRPC ``DEADLINE_EXCEEDED``. Poll loop treats this as
    retryable.
    """


class SandboxCommandTimeoutError(SandboxTimeoutError):
    """Raised when the user's command inside the sandbox timed out.

    Emitted when the backend signals ``CWSANDBOX_COMMAND_TIMEOUT`` via
    AIP-193 reason. This is NOT retryable - the command itself exceeded
    its timeout budget.
    """


class SandboxResourceExhaustedError(SandboxError):
    """Raised when the sandbox service is under resource pressure.

    Emitted for gRPC ``RESOURCE_EXHAUSTED``. Poll loop treats this as
    retryable, but callers should back off - the server is overloaded.
    """


class SandboxTerminalStateUnavailableError(SandboxError):
    """Raised when the backend does not report a terminal state after stop.

    After a successful ``stop()``, the backend should return the persisted
    terminal state (COMPLETED or FAILED) on subsequent ``Get`` calls. A
    narrow race between the backend's terminal-state write and the client's
    next poll, or backend-rollout skew, can surface as NOT_FOUND. The SDK
    retries for a brief budget; if NOT_FOUND persists past that budget,
    this exception is raised.

    Callers seeing this should treat the outcome as ambiguous - the stop
    succeeded, but whether the user's workload completed or failed is not
    observable from the client.
    """


class SandboxSnapshotError(SandboxError):
    """Base exception for file-system snapshot (FSS) operation failures.

    Raised by ``snapshot()``, snapshot restore on ``run()``,
    ``stop(snapshot_on_stop=True)``, and the snapshot management methods.
    Also used directly for terminal internal failures (restore/create
    failed, bucket unavailable, auth/transport failures).

    Attributes:
        file_system_snapshot_id: The file-system snapshot ID involved, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        file_system_snapshot_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message, reason=reason, metadata=metadata, retry_delay=retry_delay)
        self.file_system_snapshot_id = file_system_snapshot_id


class SnapshotNotFoundError(SandboxSnapshotError):
    """Raised when a snapshot ID is unknown, deleted, or owned by another org.

    Emitted for ``CWSANDBOX_FSS_NOT_FOUND``.
    """


class SnapshotNotReadyError(SandboxSnapshotError):
    """Raised when a snapshot is not in the READY state for the operation.

    Emitted for ``CWSANDBOX_FSS_NOT_READY`` (e.g. restoring from a snapshot
    that is still CREATING or has FAILED).
    """


class SnapshotNotSupportedError(SandboxSnapshotError):
    """Raised when file-system snapshots are not enabled for the organization.

    Emitted for ``CWSANDBOX_FSS_NOT_SUPPORTED``. FSS is gated by a per-org
    allowlist on the backend; an org not on the list cannot create, restore,
    or manage snapshots.
    """


class SnapshotSizeExceededError(SandboxSnapshotError):
    """Raised when the requested mount size exceeds the configured maximum.

    Emitted for ``CWSANDBOX_FSS_SIZE_EXCEEDED``.
    """


class SnapshotQuotaExceededError(SandboxSnapshotError):
    """Raised when the organization's snapshot quota is exhausted.

    Emitted for ``CWSANDBOX_FSS_QUOTA_EXCEEDED``.
    """


class SnapshotBucketMismatchError(SandboxSnapshotError):
    """Raised when a snapshot's bucket differs from the org's current bucket.

    Emitted for ``CWSANDBOX_FSS_BUCKET_MISMATCH``. Reversible by reverting the
    org's snapshot bucket configuration to the one the snapshot was written to.
    """


class SnapshotWaitTimeoutError(SandboxTimeoutError):
    """Raised when ``wait_for_ready`` exceeded its budget before the snapshot was READY.

    Emitted for ``CWSANDBOX_FSS_WAIT_TIMEOUT``. The snapshot may still complete
    server-side; poll with ``Sandbox.get_snapshot()`` to check.
    """


class SnapshotBackendThrottledError(SandboxUnavailableError):
    """Raised when the snapshot backend is transiently throttled or at capacity.

    Emitted for ``CWSANDBOX_FSS_BACKEND_THROTTLED`` and
    ``CWSANDBOX_FSS_INFLIGHT_LIMIT`` (gRPC ``UNAVAILABLE``). Inherits the
    retryable contract of ``SandboxUnavailableError``; callers should back off.
    """


class DiscoveryError(CWSandboxError):
    """Base exception for discovery operations (runners, profiles).

    Covers domain errors (not-found) and transport errors (timeout,
    unavailable). Authentication errors use
    ``CWSandboxAuthenticationError`` instead.
    """


class RunnerNotFoundError(DiscoveryError):
    """Raised when a runner ID is not found.

    Attributes:
        runner_id: The ID of the runner that was not found.
    """

    def __init__(
        self,
        message: str,
        *,
        runner_id: str,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message, reason=reason, metadata=metadata, retry_delay=retry_delay)
        self.runner_id = runner_id


class ProfileNotFoundError(DiscoveryError):
    """Raised when a profile is not found.

    Attributes:
        profile_name: The name of the profile that was not found.
        runner_id: The runner ID if specified in the request, or None.
    """

    def __init__(
        self,
        message: str,
        *,
        profile_name: str,
        runner_id: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, str] | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        super().__init__(message, reason=reason, metadata=metadata, retry_delay=retry_delay)
        self.profile_name = profile_name
        self.runner_id = runner_id


class FunctionError(CWSandboxError):
    """Base exception for function execution operations."""


class AsyncFunctionError(FunctionError):
    """Raised when an async function is passed to @session.function().

    Async functions are not supported because the sandbox executes Python
    synchronously. The decorated function must be a regular (sync) function.
    """
