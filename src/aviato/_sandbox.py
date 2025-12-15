from __future__ import annotations

import asyncio
import base64
import logging
import os
import shlex
import time
import warnings
from typing import TYPE_CHECKING, Any

import httpx
from coreweave.aviato.v1beta1 import atc_connect, atc_pb2

from aviato._defaults import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    DEFAULT_MAX_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_BACKOFF_FACTOR,
    DEFAULT_POLL_INTERVAL_SECONDS,
    SandboxDefaults,
)
from aviato._types import ExecResult
from aviato.exceptions import (
    SandboxExecutionError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotRunningError,
    SandboxTerminatedError,
    SandboxTimeoutError,
)

if TYPE_CHECKING:
    from aviato._session import Session

logger = logging.getLogger(__name__)


class Sandbox:
    """Aviato sandbox client.

    Supports two construction patterns:

    1. Constructor:

        async with Sandbox(
            command="sleep",
            args=["infinity"],
            container_image="python:3.11",
        ) as sandbox:
            result = await sandbox.exec(["echo", "hello"])

    2. Factory method:

        sandbox = await Sandbox.create("echo", "hello", "world")
        sandbox = await Sandbox.create("python", "-c", "print('hi')")

    """

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        defaults: SandboxDefaults | None = None,
        container_image: str | None = None,
        tags: list[str] | None = None,
        base_url: str | None = None,
        request_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        _session: Session | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a sandbox (does not start it).

        Args:
            command: The command to run in the sandbox
            args: Optional arguments for the command
            defaults: Optional SandboxDefaults to apply
            container_image: Container image to use (default: python:3.11)
            tags: Optional tags for the sandbox
            base_url: Aviato API URL (default: AVIATO_BASE_URL env or localhost)
            request_timeout_seconds: Timeout for API requests (client-side, default: 300s)
            max_lifetime_seconds: Max sandbox lifetime (server-side). If not set,
                the backend controls the default.
            runway_ids: Optional list of runway IDs
            tower_ids: Optional list of tower IDs
            **kwargs: Additional configuration options
        """
        self._defaults = defaults or SandboxDefaults()
        self._session = _session

        self._command = command
        self._args = args

        # Apply defaults with explicit values taking precedence
        self._container_image = container_image or self._defaults.container_image
        self._base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or self._defaults.base_url
        ).rstrip("/")
        self._request_timeout_seconds = (
            request_timeout_seconds or self._defaults.request_timeout_seconds
        )
        self._max_lifetime_seconds = (
            max_lifetime_seconds
            if max_lifetime_seconds is not None
            else self._defaults.max_lifetime_seconds
        )

        self._tags = self._defaults.merge_tags(tags)

        self._runway_ids = runway_ids or (
            list(self._defaults.runway_ids) if self._defaults.runway_ids else None
        )
        self._tower_ids = tower_ids or (
            list(self._defaults.tower_ids) if self._defaults.tower_ids else None
        )

        self._start_kwargs = kwargs
        self._client: atc_connect.ATCServiceClient | None = None
        self._sandbox_id: str | None = None
        self._returncode: int | None = None
        self._tower_id: str | None = None
        self._runway_id: str | None = None
        self._stopped = False

    @classmethod
    async def create(
        cls,
        *args: str,
        container_image: str | None = None,
        defaults: SandboxDefaults | None = None,
        request_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        **kwargs: Any,
    ) -> Sandbox:
        """Factory method for quick sandbox creation with auto-start.

        The first positional arg is the command, remaining args are its arguments.
        The sandbox is started before returning.

        Args:
            *args: Command and arguments (e.g., "echo", "hello", "world")
            container_image: Container image to use
            defaults: Optional SandboxDefaults to apply
            request_timeout_seconds: Timeout for API requests (client-side)
            max_lifetime_seconds: Max sandbox lifetime (server-side)
            **kwargs: Additional sandbox configuration

        Returns:
            A started Sandbox instance

        Raises:
            ValueError: If no positional arguments provided
            SandboxFailedError: If sandbox fails to start
            SandboxTimeoutError: If start times out

        Example:
            # Quick one-liner
            sandbox = await Sandbox.create("echo", "hello")

            # With options
            sandbox = await Sandbox.create(
                "python", "train.py",
                container_image="pytorch/pytorch:latest",
            )

            # Remember to stop when done
            await sandbox.stop()

            # For automatic cleanup, use the constructor with context manager instead:
            # async with Sandbox(command="python", args=["-c", "print(1)"]) as sb:
            #     result = await sb.exec(["python", "--version"])
        """
        if not args:
            raise ValueError("At least one positional argument (command) is required")

        command = args[0]
        cmd_args = list(args[1:]) if len(args) > 1 else None

        sandbox = cls(
            command=command,
            args=cmd_args,
            container_image=container_image or DEFAULT_CONTAINER_IMAGE,
            defaults=defaults,
            request_timeout_seconds=request_timeout_seconds,
            max_lifetime_seconds=max_lifetime_seconds,
            **kwargs,
        )
        logger.debug("Creating sandbox with command: %s", command)
        await sandbox.start()
        return sandbox

    @classmethod
    def session(
        cls,
        defaults: SandboxDefaults | None = None,
    ) -> Session:
        """Create a session for managing multiple sandboxes.

        Sessions provide:
        - Shared configuration via defaults
        - Automatic cleanup of orphaned sandboxes
        - Function execution via @session.function() decorator

        Args:
            defaults: Optional defaults to apply to sandboxes created via session

        Returns:
            A Session instance

        Example:
            session = Sandbox.session(defaults)
            sb = session.create(command="sleep", args=["infinity"])

            @session.function()
            def compute(x, y):
                return x + y

            await session.close()
        """
        from aviato._session import Session

        return Session(defaults)

    @property
    def sandbox_id(self) -> str | None:
        """The unique sandbox ID, or None if not yet started."""
        return self._sandbox_id

    @property
    def returncode(self) -> int | None:
        """Exit code if sandbox has completed, None if still running.

        Use wait() to block until the sandbox completes.
        """
        return self._returncode

    @property
    def tower_id(self) -> str | None:
        """Tower where sandbox is running, or None if not started."""
        return self._tower_id

    @property
    def runway_id(self) -> str | None:
        """Runway where sandbox is running, or None if not started."""
        return self._runway_id

    def __repr__(self) -> str:
        if self._sandbox_id is None:
            status = "not_started"
        elif self._stopped:
            status = "stopped"
        elif self._returncode is not None:
            status = f"completed(rc={self._returncode})"
        else:
            status = "running"
        return f"<Sandbox id={self._sandbox_id} status={status}>"

    async def get_status(self) -> str:
        """Get the current status of the sandbox from the backend.

        Returns:
            Status string: "running", "completed", "failed", "terminated",
            "creating", "pending", "paused", or "unspecified"

        Raises:
            SandboxNotRunningError: If sandbox has not been started

        Example:
            sandbox = await Sandbox.create("sleep", "10")
            status = await sandbox.get_status()
            print(f"Sandbox is {status}")  # "running"

            await sandbox.wait()
            status = await sandbox.get_status()
            print(f"Sandbox is {status}")  # "completed"
        """
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("Sandbox has not been started")

        await self._ensure_client()
        assert self._client is not None

        request = atc_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
        response = await self._client.get(request)

        # "SANDBOX_STATUS_RUNNING" -> "running"
        name = atc_pb2.SandboxStatus.Name(response.sandbox_status)
        return name.replace("SANDBOX_STATUS_", "").lower()

    # Context manager

    async def __aenter__(self) -> Sandbox:
        """Start the sandbox."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop the sandbox and clean up.

        # TODO: Add proper cancellation support - if CancelledError is raised,
        # we should still attempt cleanup but not suppress the cancellation.
        """
        if self._sandbox_id is not None:
            await self.stop()

    def __del__(self) -> None:
        """Warn if sandbox was not properly stopped."""
        if self._sandbox_id is not None and self._returncode is None:
            warnings.warn(
                f"Sandbox {self._sandbox_id} was not stopped. "
                "Use 'await sandbox.stop()' or the context manager pattern.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_client(self) -> None:
        """Ensure the Connect RPC client is initialized."""
        if self._client is not None:
            return

        token = os.environ.get("AVIATO_API_KEY", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        session = httpx.AsyncClient(
            timeout=httpx.Timeout(self._request_timeout_seconds),
            headers=headers,
        )
        self._client = atc_connect.ATCServiceClient(
            address=self._base_url,
            session=session,
            proto_json=True,
        )
        logger.debug("Initialized client for %s", self._base_url)

    async def _close_client(self) -> None:
        """Close the HTTP client if open."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.debug("Closed client")

    async def _poll_until_stable(
        self,
        timeout_seconds: float,
        timeout_message: str,
    ) -> atc_pb2.GetSandboxResponse:
        """Poll sandbox status until a stable state is reached.

        Returns the response when sandbox reaches a stable state (RUNNING,
        COMPLETED, FAILED, TERMINATED, or UNSPECIFIED). Transient states
        like CREATING, PENDING, and PAUSED are polled through. Raises on timeout.

        Args:
            timeout_seconds: Maximum time to wait
            timeout_message: Message for SandboxTimeoutError if timeout occurs

        Returns:
            The GetSandboxResponse with a stable status
        """
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox ID available")

        await self._ensure_client()
        assert self._client is not None

        start_time = time.monotonic()
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_seconds:
                raise SandboxTimeoutError(timeout_message)

            request = atc_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
            response = await self._client.get(request)

            logger.debug(
                "Sandbox %s status: %s",
                self._sandbox_id,
                response.sandbox_status,
            )

            # Stable states - return for caller to handle
            if response.sandbox_status in (
                atc_pb2.SANDBOX_STATUS_RUNNING,
                atc_pb2.SANDBOX_STATUS_COMPLETED,
                atc_pb2.SANDBOX_STATUS_FAILED,
                atc_pb2.SANDBOX_STATUS_TERMINATED,
                atc_pb2.SANDBOX_STATUS_UNSPECIFIED,
            ):
                return response

            # Transient states - keep polling
            await asyncio.sleep(poll_interval)
            poll_interval = min(
                poll_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                DEFAULT_MAX_POLL_INTERVAL_SECONDS,
            )

    async def _wait_for_ready(self, timeout_seconds: float) -> None:
        """Wait for the sandbox to reach RUNNING status."""
        response = await self._poll_until_stable(
            timeout_seconds,
            f"Sandbox {self._sandbox_id} did not become ready within {timeout_seconds}s",
        )

        match response.sandbox_status:
            case atc_pb2.SANDBOX_STATUS_RUNNING:
                self._tower_id = response.tower_id or None
                self._runway_id = response.runway_id or None
                logger.info("Sandbox %s is running", self._sandbox_id)
            case atc_pb2.SANDBOX_STATUS_FAILED:
                raise SandboxFailedError(f"Sandbox {self._sandbox_id} failed to start")
            case atc_pb2.SANDBOX_STATUS_TERMINATED:
                raise SandboxTerminatedError(f"Sandbox {self._sandbox_id} was terminated")
            case atc_pb2.SANDBOX_STATUS_COMPLETED | atc_pb2.SANDBOX_STATUS_UNSPECIFIED:
                # COMPLETED: finished successfully
                # UNSPECIFIED: sandbox already cleaned up (fast command)
                # TODO: Adjust UNSPECIFIED behavior once backend has proper accounting
                self._tower_id = response.tower_id or None
                self._runway_id = response.runway_id or None
                self._returncode = 0
                logger.info("Sandbox %s completed during startup", self._sandbox_id)

    # Lifecycle methods

    async def start(self, *, timeout_seconds: float | None = None) -> str:
        """Start the sandbox and wait for it to be ready.

        Returns:
            The sandbox ID

        Raises:
            SandboxFailedError: If sandbox fails to start
            SandboxTimeoutError: If start operation times out

        Example:
            sandbox = Sandbox(command="sleep", args=["infinity"], ...)
            sandbox_id = await sandbox.start()
            print(f"Started sandbox: {sandbox_id}")
            print(f"Running on tower: {sandbox.tower_id}")
        """
        timeout = timeout_seconds or self._request_timeout_seconds

        await self._ensure_client()
        assert self._client is not None

        request_kwargs: dict[str, Any] = {
            "command": self._command,
            "args": self._args or [],
            "container_image": self._container_image,
            "tags": self._tags or [],
        }

        if self._max_lifetime_seconds is not None:
            request_kwargs["max_lifetime_seconds"] = int(self._max_lifetime_seconds)
        if self._runway_ids:
            request_kwargs["runway_ids"] = self._runway_ids
        if self._tower_ids:
            request_kwargs["tower_ids"] = self._tower_ids

        request_kwargs.update(self._start_kwargs)

        logger.debug("Starting sandbox with image %s", self._container_image)

        request = atc_pb2.StartSandboxRequest(**request_kwargs)
        response = await self._client.start(request)
        sandbox_id = str(response.sandbox_id)
        self._sandbox_id = sandbox_id

        logger.debug("Sandbox %s created, waiting for ready", sandbox_id)
        await self._wait_for_ready(timeout)
        return sandbox_id

    async def wait(
        self,
        *,
        raise_on_termination: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """Wait for the sandbox to complete.

        Blocks until the sandbox process exits. After this returns successfully,
        returncode will be available.

        Args:
            raise_on_termination: If True (default), raises SandboxTerminatedError
                if sandbox was terminated externally. Set to False to handle
                termination gracefully without exceptions.
            timeout_seconds: Optional timeout for the wait operation.

        Raises:
            SandboxTimeoutError: If the sandbox or wait operation times out
            SandboxTerminatedError: If sandbox was terminated (and raise_on_termination=True)
            SandboxFailedError: If sandbox failed

        Example:
            # Normal usage - let exceptions propagate
            sandbox = await Sandbox.create("python", "-c", "print('done')")
            await sandbox.wait()
            print(f"Exit code: {sandbox.returncode}")

            # Graceful handling of termination
            await sandbox.wait(raise_on_termination=False)
            if sandbox.returncode is None:
                print("Sandbox was terminated")
        """
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        if self._returncode is not None:
            return

        timeout = timeout_seconds or self._request_timeout_seconds
        start_time = time.monotonic()
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS

        while True:
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                raise SandboxTimeoutError(f"Timed out waiting for sandbox {self._sandbox_id}")

            response = await self._poll_until_stable(
                remaining,
                f"Timed out waiting for sandbox {self._sandbox_id}",
            )

            match response.sandbox_status:
                case atc_pb2.SANDBOX_STATUS_COMPLETED:
                    self._returncode = 0
                    logger.info("Sandbox %s completed", self._sandbox_id)
                    return
                case atc_pb2.SANDBOX_STATUS_FAILED:
                    raise SandboxFailedError(f"Sandbox {self._sandbox_id} failed")
                case atc_pb2.SANDBOX_STATUS_TERMINATED:
                    self._returncode = None
                    logger.info("Sandbox %s was terminated", self._sandbox_id)
                    if raise_on_termination:
                        raise SandboxTerminatedError(f"Sandbox {self._sandbox_id} was terminated")
                    return
                case atc_pb2.SANDBOX_STATUS_UNSPECIFIED:
                    # Sandbox no longer tracked by backend
                    # TODO: Adjust this behavior once the backend has proper accounting
                    self._returncode = 0
                    logger.info(
                        "Sandbox %s status unspecified (assuming completed)",
                        self._sandbox_id,
                    )
                    return
                case atc_pb2.SANDBOX_STATUS_RUNNING:
                    # Still running - wait before polling again
                    await asyncio.sleep(poll_interval)
                    poll_interval = min(
                        poll_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                        DEFAULT_MAX_POLL_INTERVAL_SECONDS,
                    )

    async def stop(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    ) -> bool:
        """Stop the sandbox and close the client.

        The sandbox is deregistered from its session regardless of whether
        the stop was successful, since the sandbox is no longer usable.

        Returns:
            True if the backend confirmed the stop (or sandbox was already
            stopped), False if the stop failed.
        """
        if self._stopped:
            logger.debug("stop() called on already-stopped sandbox %s", self._sandbox_id)
            return True

        if self._sandbox_id is None:
            logger.debug("stop() called on sandbox that was never started")
            return True

        self._stopped = True

        await self._ensure_client()
        assert self._client is not None

        logger.debug("Stopping sandbox %s", self._sandbox_id)

        request = atc_pb2.StopSandboxRequest(
            sandbox_id=self._sandbox_id,
            graceful_shutdown_seconds=int(graceful_shutdown_seconds),
            snapshot_on_stop=snapshot_on_stop,
        )

        try:
            response = await self._client.stop(request)
            success = bool(response.success)

            if success:
                logger.info("Sandbox %s stopped successfully", self._sandbox_id)
            else:
                error_msg = response.error_message or "unknown error"
                logger.warning(
                    "Sandbox %s stop returned failure: %s",
                    self._sandbox_id,
                    error_msg,
                )

            # Deregister from session regardless of success - the sandbox
            # is no longer usable either way
            if self._session is not None:
                self._session._deregister_sandbox(self)

            return success
        finally:
            await self._close_client()

    async def exec(
        self,
        command: list[str],
        *,
        check: bool = False,
        timeout_seconds: float | None = None,
    ) -> ExecResult:
        """Execute a command in the running sandbox.

        Args:
            command: Command and arguments to execute
            check: If True, raise SandboxExecutionError on non-zero returncode
            timeout_seconds: Timeout for the command execution

        Returns:
            ExecResult with stdout, stderr, and returncode

        Raises:
            SandboxNotRunningError: If sandbox is not running
            SandboxExecutionError: If check=True and command returns non-zero
            asyncio.TimeoutError: If command exceeds timeout_seconds
        """
        timeout = timeout_seconds or self._request_timeout_seconds

        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        await self._ensure_client()
        assert self._client is not None

        if not command:
            raise ValueError("Command cannot be empty")

        logger.debug("Executing command in sandbox %s: %s", self._sandbox_id, shlex.join(command))

        request = atc_pb2.ExecSandboxRequest(
            sandbox_id=self._sandbox_id,
            command=command,
            max_timeout_seconds=int(timeout),
        )

        response = await asyncio.wait_for(
            self._client.exec(request),
            timeout=timeout,
        )
        logger.debug("Command completed with exit code %d", response.result.exit_code)

        stdout_raw = response.result.stdout
        stderr_raw = response.result.stderr

        # TODO: Update handling stdout/stderr datatypes when it changes in aviato-core
        result = ExecResult(
            stdout_bytes=base64.b64decode(stdout_raw) if stdout_raw else b"",
            stderr_bytes=stderr_raw.encode() if stderr_raw else b"",
            returncode=response.result.exit_code,
            command=command,
        )

        if check and result.returncode != 0:
            raise SandboxExecutionError(
                f"Command {shlex.join(command)} failed with exit code {result.returncode}",
                exec_result=result,
            )

        return result

    async def read_file(
        self,
        filepath: str,
        *,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Read a file from the sandbox filesystem.

        Raises:
            SandboxNotRunningError: If sandbox is not running
            asyncio.TimeoutError: If operation exceeds timeout_seconds
        """
        timeout = timeout_seconds or self._request_timeout_seconds

        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        await self._ensure_client()
        assert self._client is not None

        logger.debug("Reading file from sandbox %s: %s", self._sandbox_id, filepath)

        request = atc_pb2.RetrieveFileSandboxRequest(
            sandbox_id=self._sandbox_id,
            filepath=filepath,
            max_timeout_seconds=int(timeout),
        )

        response = await asyncio.wait_for(
            self._client.retrieve_file(request),
            timeout=timeout,
        )

        if not response.success:
            logger.warning("Failed to read file %s from sandbox %s", filepath, self._sandbox_id)
            raise SandboxFileError(
                f"Failed to read file '{filepath}': {response.error_message}",
                filepath=filepath,
            )

        return bytes(response.file_contents)

    async def write_file(
        self,
        filepath: str,
        contents: bytes,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Write a file to the sandbox filesystem.

        Raises:
            SandboxNotRunningError: If sandbox is not running
            SandboxFileError: If the file could not be written
            asyncio.TimeoutError: If operation exceeds timeout_seconds
        """
        timeout = timeout_seconds or self._request_timeout_seconds

        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        await self._ensure_client()
        assert self._client is not None

        logger.debug(
            "Writing file to sandbox %s: %s (%d bytes)",
            self._sandbox_id,
            filepath,
            len(contents),
        )

        request = atc_pb2.AddFileSandboxRequest(
            sandbox_id=self._sandbox_id,
            filepath=filepath,
            file_contents=contents,
            max_timeout_seconds=int(timeout),
        )

        response = await asyncio.wait_for(
            self._client.add_file(request),
            timeout=timeout,
        )

        if not response.success:
            logger.warning("Failed to write file %s to sandbox %s", filepath, self._sandbox_id)
            raise SandboxFileError(
                f"Failed to write file '{filepath}'",
                filepath=filepath,
            )
