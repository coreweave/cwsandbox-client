"""Exception hierarchy for Aviato operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aviato._types import ProcessResult


class AviatoError(Exception):
    """Base exception for all Aviato operations."""


class AviatoAuthenticationError(AviatoError):
    """Raised when authentication fails."""


class SandboxError(AviatoError):
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
    """Raised when a sandbox is not found (e.g., already deleted)."""

    def __init__(self, message: str, *, sandbox_id: str | None = None) -> None:
        super().__init__(message)
        self.sandbox_id = sandbox_id


class SandboxExecutionError(SandboxError):
    """Raised when command execution fails inside a sandbox.

    Access execution details via exec_result
    """

    def __init__(
        self,
        message: str,
        *,
        exec_result: ProcessResult | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.exec_result = exec_result
        self.exception_type = exception_type
        self.exception_message = exception_message


class SandboxFileError(SandboxError):
    """Raised when a file operation fails in the sandbox.

    This is a sandbox infrastructure error, not a user code error.
    Inherits from SandboxError since it's a sandbox operation failure.
    """

    def __init__(self, message: str, *, filepath: str | None = None) -> None:
        super().__init__(message)
        self.filepath = filepath


class FunctionError(AviatoError):
    """Base exception for function execution operations."""


class AsyncFunctionError(FunctionError):
    """Raised when an async function is passed to @session.function().

    Async functions are not supported because the sandbox executes Python
    synchronously. The decorated function must be a regular (sync) function.
    """


class FunctionSerializationError(FunctionError):
    """Raised when arguments, referenced globals, or closures cannot be serialized."""


class WandbAuthError(AviatoAuthenticationError):
    """Raised when W&B authentication is misconfigured."""
