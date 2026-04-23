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


class FunctionSerializationError(FunctionError):
    """Raised when arguments, referenced globals, or closures cannot be serialized."""
