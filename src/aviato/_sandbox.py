from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import sys
import time
import warnings
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from coreweave.aviato.v1beta1 import atc_connect, atc_pb2, streaming_connect, streaming_pb2
from google.protobuf import timestamp_pb2

from aviato._auth import resolve_auth
from aviato._defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
    DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    DEFAULT_MAX_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_BACKOFF_FACTOR,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    SandboxDefaults,
)
from aviato._types import ExecResult
from aviato.exceptions import (
    SandboxError,
    SandboxExecutionError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotFoundError,
    SandboxNotRunningError,
    SandboxTerminatedError,
    SandboxTimeoutError,
)

if TYPE_CHECKING:
    from aviato._session import Session

logger = logging.getLogger(__name__)


def _default_stdout_callback(data: bytes) -> None:
    """Default callback that writes to stdout."""
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _default_stderr_callback(data: bytes) -> None:
    """Default callback that writes to stderr."""
    sys.stderr.buffer.write(data)
    sys.stderr.buffer.flush()


class SandboxStatus(StrEnum):
    """Sandbox status values."""

    RUNNING = "running"
    CREATING = "creating"
    PENDING = "pending"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    UNSPECIFIED = "unspecified"

    @classmethod
    def from_proto(cls, proto_status: int) -> SandboxStatus:
        """Convert protobuf status enum to SandboxStatus"""
        proto_name = atc_pb2.SandboxStatus.Name(proto_status)
        enum_name = proto_name.replace("SANDBOX_STATUS_", "")
        return cls[enum_name]

    def to_proto(self) -> int:
        """Convert SandboxStatus to protobuf enum"""
        proto_name = f"SANDBOX_STATUS_{self.name}"
        return atc_pb2.SandboxStatus.Value(proto_name)


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
        command: str | None = None,
        args: list[str] | None = None,
        defaults: SandboxDefaults | None = None,
        container_image: str | None = None,
        tags: list[str] | None = None,
        base_url: str | None = None,
        request_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
        _session: Session | None = None,
    ) -> None:
        """Initialize a sandbox (does not start it).

        Args:
            command: Optional command to run in the sandbox
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
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            service: Service configuration for network access
            max_timeout_seconds: Maximum timeout for sandbox operations
        """

        self._defaults = defaults or SandboxDefaults()
        self._session = _session

        self._command = command or self._defaults.command
        self._args = args if args is not None else list(self._defaults.args)

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

        self._start_kwargs: dict[str, Any] = {}
        if resources is not None:
            self._start_kwargs["resources"] = resources
        if mounted_files is not None:
            self._start_kwargs["mounted_files"] = mounted_files
        if s3_mount is not None:
            self._start_kwargs["s3_mount"] = s3_mount
        if ports is not None:
            self._start_kwargs["ports"] = ports
        if service is not None:
            self._start_kwargs["service"] = service
        if max_timeout_seconds is not None:
            self._start_kwargs["max_timeout_seconds"] = max_timeout_seconds
        self._client: atc_connect.ATCServiceClient | None = None
        self._streaming_client: streaming_connect.ATCStreamingServiceClient | None = None
        self._sandbox_id: str | None = None
        self._returncode: int | None = None
        self._tower_id: str | None = None
        self._runway_id: str | None = None
        # TODO: Remove _stopped once backend adds proper accounting for stopped/terminated
        # sandboxes. Currently backend stops tracking sandbox IDs after stop(), so we need
        # this client-side flag as a workaround. Once backend properly tracks stopped
        # sandboxes, we can remove this and query backend status directly.
        self._stopped = False
        # Lock for thread-safe streaming client initialization
        self._streaming_client_lock: asyncio.Lock | None = None

        self._status: SandboxStatus | None = None
        self._status_updated_at: datetime | None = None
        self._started_at: datetime | None = None
        self._tower_group_id: str | None = None

    @classmethod
    async def create(
        cls,
        *args: str,
        container_image: str | None = None,
        defaults: SandboxDefaults | None = None,
        request_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        tags: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
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
            tags: Optional tags for the sandbox
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            service: Service configuration for network access
            max_timeout_seconds: Maximum timeout for sandbox operations

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
                resources={"cpu": "2", "memory": "4Gi"},
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
            container_image=container_image,
            defaults=defaults,
            request_timeout_seconds=request_timeout_seconds,
            max_lifetime_seconds=max_lifetime_seconds,
            tags=tags,
            resources=resources,
            mounted_files=mounted_files,
            s3_mount=s3_mount,
            ports=ports,
            service=service,
            max_timeout_seconds=max_timeout_seconds,
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

    @classmethod
    def _from_sandbox_info(
        cls,
        sandbox_id: str,
        sandbox_status: int,
        started_at_time: timestamp_pb2.Timestamp | None,
        tower_id: str,
        tower_group_id: str,
        runway_id: str,
        base_url: str,
        timeout_seconds: float,
    ) -> Sandbox:
        """Create a Sandbox instance from sandbox info fields"""
        sandbox = cls.__new__(cls)
        # Initialize state for a discovered sandbox
        sandbox._sandbox_id = sandbox_id
        sandbox._status = SandboxStatus.from_proto(sandbox_status)
        sandbox._status_updated_at = datetime.now(UTC)
        sandbox._started_at = started_at_time.ToDatetime() if started_at_time else None
        sandbox._tower_id = tower_id or None
        sandbox._tower_group_id = tower_group_id or None
        sandbox._runway_id = runway_id or None
        sandbox._base_url = base_url
        sandbox._request_timeout_seconds = timeout_seconds
        # These are not applicable for discovered sandboxes
        sandbox._command = None
        sandbox._args = None
        sandbox._container_image = None
        sandbox._tags = None
        sandbox._max_lifetime_seconds = None
        sandbox._runway_ids = None
        sandbox._tower_ids = None
        sandbox._client = None
        sandbox._streaming_client = None
        sandbox._streaming_client_lock = None
        sandbox._stopped = False
        sandbox._returncode = None
        sandbox._session = None
        sandbox._defaults = SandboxDefaults()
        sandbox._start_kwargs = {}
        return sandbox

    @classmethod
    async def list(
        cls,
        *,
        tags: list[str] | None = None,
        status: str | None = None,
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> list[Sandbox]:
        """List existing sandboxes with optional filters.

        Returns Sandbox instances that can be used for operations like
        exec(), stop(), get_status(), read_file(), write_file(), etc.

        Args:
            tags: Filter by tags (sandboxes must have ALL specified tags)
            status: Filter by status ("running", "completed", "failed", etc.)
            runway_ids: Filter by runway IDs
            tower_ids: Filter by tower IDs
            base_url: Override API URL (default: AVIATO_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)

        Returns:
            List of Sandbox instances for matching sandboxes

        Example:
            # Find and stop all sandboxes with a tag
            sandboxes = await Sandbox.list(tags=["my-batch-job"])
            for sb in sandboxes:
                print(f"{sb.sandbox_id}: {sb.status}")
                await sb.stop()

            # Execute commands on discovered sandboxes
            running = await Sandbox.list(status="running")
            for sb in running:
                result = await sb.exec(["echo", "hello"])
        """
        auth = resolve_auth()
        effective_base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS

        status_enum = None
        if status is not None:
            status_enum = SandboxStatus(status)

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=auth.headers,
        ) as http_client:
            client = atc_connect.ATCServiceClient(
                address=effective_base_url,
                session=http_client,
                proto_json=True,
            )

            request_kwargs: dict[str, Any] = {}
            if tags:
                request_kwargs["tags"] = tags
            if status_enum:
                request_kwargs["status"] = status_enum.to_proto()
            if runway_ids:
                request_kwargs["runway_ids"] = runway_ids
            if tower_ids:
                request_kwargs["tower_ids"] = tower_ids

            request = atc_pb2.ListSandboxesRequest(**request_kwargs)
            response = await client.list(request)

            return [
                cls._from_sandbox_info(
                    sandbox_id=sb.sandbox_id,
                    sandbox_status=sb.sandbox_status,
                    started_at_time=sb.started_at_time,
                    tower_id=sb.tower_id,
                    tower_group_id=sb.tower_group_id,
                    runway_id=sb.runway_id,
                    base_url=effective_base_url,
                    timeout_seconds=timeout,
                )
                for sb in response.sandboxes
            ]

    @classmethod
    async def from_id(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Sandbox:
        """Attach to an existing sandbox by ID.

        Creates a Sandbox instance connected to an existing sandbox,
        allowing operations like exec(), stop(), get_status(), etc.

        Args:
            sandbox_id: The ID of the existing sandbox
            base_url: Override API URL (default: AVIATO_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)

        Returns:
            A Sandbox instance attached to the existing sandbox

        Raises:
            SandboxNotFoundError: If sandbox doesn't exist

        Example:
            # Reconnect to a sandbox from a previous session
            sb = await Sandbox.from_id("sandbox-abc123")
            result = await sb.exec(["python", "-c", "print('hello')"])
            await sb.stop()
        """
        auth = resolve_auth()
        effective_base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=auth.headers,
        ) as http_client:
            client = atc_connect.ATCServiceClient(
                address=effective_base_url,
                session=http_client,
                proto_json=True,
            )

            try:
                request = atc_pb2.GetSandboxRequest(sandbox_id=sandbox_id)
                response = await client.get(request)
            except ConnectError as e:
                if e.code == Code.NOT_FOUND:
                    raise SandboxNotFoundError(
                        f"Sandbox '{sandbox_id}' not found",
                        sandbox_id=sandbox_id,
                    ) from e
                raise

            return cls._from_sandbox_info(
                sandbox_id=response.sandbox_id,
                sandbox_status=response.sandbox_status,
                started_at_time=response.started_at_time,
                tower_id=response.tower_id,
                tower_group_id=response.tower_group_id,
                runway_id=response.runway_id,
                base_url=effective_base_url,
                timeout_seconds=timeout,
            )

    @classmethod
    async def delete(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> bool:
        """Delete a sandbox by ID without creating a Sandbox instance.

        This is a convenience method for cleanup scenarios where you
        don't need to perform other operations on the sandbox.

        Args:
            sandbox_id: The sandbox ID to delete
            base_url: Override API URL (default: AVIATO_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)
            missing_ok: If True, return False instead of raising when sandbox doesn't exist

        Returns:
            True if sandbox was deleted successfully, False if missing_ok=True and not found

        Raises:
            SandboxNotFoundError: If sandbox doesn't exist and missing_ok=False
            SandboxError: If deletion failed for other reasons

        Example:
            # Quick cleanup without creating Sandbox instances
            await Sandbox.delete("sandbox-abc123")

            # Ignore if already deleted
            await Sandbox.delete("sandbox-abc123", missing_ok=True)
        """
        auth = resolve_auth()
        effective_base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=auth.headers,
        ) as http_client:
            client = atc_connect.ATCServiceClient(
                address=effective_base_url,
                session=http_client,
                proto_json=True,
            )

            try:
                request = atc_pb2.DeleteSandboxRequest(sandbox_id=sandbox_id)
                response = await client.delete(request)
            except ConnectError as e:
                if e.code == Code.NOT_FOUND:
                    if missing_ok:
                        return False
                    raise SandboxNotFoundError(
                        f"Sandbox '{sandbox_id}' not found",
                        sandbox_id=sandbox_id,
                    ) from e
                raise

            if not response.success:
                raise SandboxError(f"Failed to delete sandbox: {response.error_message}")
            return True

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

    @property
    def status(self) -> SandboxStatus | None:
        """Last known status of the sandbox.

        This is the cached status from the most recent API interaction.

        Returns None only for sandboxes that haven't been started yet.

        Note: This value may be stale. Check status_updated_at for when it
        was last fetched. For guaranteed fresh status, use
        `await sandbox.get_status()` which always hits the API.
        """
        return self._status

    @property
    def status_updated_at(self) -> datetime | None:
        """Timestamp when status was last fetched from the API.

        Returns None only for sandboxes that haven't been started yet.
        """
        return self._status_updated_at

    @property
    def started_at(self) -> datetime | None:
        """Timestamp when the sandbox was started.

        Populated after start() completes or when obtained via list()/from_id().
        None only for sandboxes that haven't been started yet.
        """
        return self._started_at

    @property
    def tower_group_id(self) -> str | None:
        """Tower group ID where the sandbox is running."""
        return self._tower_group_id

    def __repr__(self) -> str:
        if self._status:
            status_str = self._status.value
        elif self._sandbox_id is None:
            status_str = "not_started"
        else:
            status_str = "unknown"
        return f"<Sandbox id={self._sandbox_id} status={status_str}>"

    async def get_status(self) -> SandboxStatus:
        """Get the current status of the sandbox from the backend.

        Returns:
            SandboxStatus enum value

        Raises:
            SandboxNotRunningError: If sandbox has not been started

        Example:
            sandbox = await Sandbox.create("sleep", "10")
            status = await sandbox.get_status()
            print(f"Sandbox is {status}")  # SandboxStatus.RUNNING

            await sandbox.wait()
            status = await sandbox.get_status()
            print(f"Sandbox is {status}")  # SandboxStatus.COMPLETED
        """
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("Sandbox has not been started")

        await self._ensure_client()
        assert self._client is not None

        request = atc_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
        response = await self._client.get(request)

        status = SandboxStatus.from_proto(response.sandbox_status)
        self._status = status
        self._status_updated_at = datetime.now(UTC)

        return status

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
        if hasattr(self, "_sandbox_id") and self._sandbox_id is not None and not self._stopped:
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

        auth = resolve_auth()
        logger.debug("Using %s auth strategy", auth.strategy)

        session = httpx.AsyncClient(
            timeout=httpx.Timeout(self._request_timeout_seconds),
            headers=auth.headers,
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

    async def _ensure_streaming_client(self) -> None:
        """Ensure the streaming Connect RPC client is initialized.

        The streaming client uses HTTP/2 which is required for bidirectional
        streaming with Connect RPC. Uses a lock to prevent race conditions
        when multiple coroutines call this concurrently.
        """

        if self._streaming_client is not None:
            return

        # Lazily create the lock (can't create in __init__ before event loop exists)
        if self._streaming_client_lock is None:
            self._streaming_client_lock = asyncio.Lock()

        async with self._streaming_client_lock:
            if self._streaming_client is not None:
                return

            auth = resolve_auth()
            logger.debug("Using %s auth strategy for streaming client", auth.strategy)

            session = httpx.AsyncClient(
                timeout=httpx.Timeout(self._request_timeout_seconds),
                headers=auth.headers,
                http2=True,
                http1=False,
            )
            self._streaming_client = streaming_connect.ATCStreamingServiceClient(
                address=self._base_url,
                session=session,
                proto_json=True,
            )
            logger.debug("Initialized streaming client for %s (HTTP/2)", self._base_url)

    async def _close_streaming_client(self) -> None:
        """Close the streaming HTTP client if open."""
        if self._streaming_client is not None:
            await self._streaming_client.close()
            self._streaming_client = None
            logger.debug("Closed streaming client")

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
            if self._stopped or self._client is None:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} was stopped while polling"
                )

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
                self._tower_group_id = response.tower_group_id or None
                self._runway_id = response.runway_id or None
                self._status = SandboxStatus.RUNNING
                self._status_updated_at = datetime.now(UTC)
                self._started_at = (
                    response.started_at_time.ToDatetime() if response.started_at_time else None
                )
                logger.info("Sandbox %s is running", self._sandbox_id)
            case atc_pb2.SANDBOX_STATUS_FAILED:
                self._status = SandboxStatus.FAILED
                self._status_updated_at = datetime.now(UTC)
                raise SandboxFailedError(f"Sandbox {self._sandbox_id} failed to start")
            case atc_pb2.SANDBOX_STATUS_TERMINATED:
                self._status = SandboxStatus.TERMINATED
                self._status_updated_at = datetime.now(UTC)
                raise SandboxTerminatedError(f"Sandbox {self._sandbox_id} was terminated")
            case atc_pb2.SANDBOX_STATUS_COMPLETED | atc_pb2.SANDBOX_STATUS_UNSPECIFIED:
                # COMPLETED: finished successfully
                # UNSPECIFIED: sandbox already cleaned up (fast command)
                # TODO: Adjust UNSPECIFIED behavior once backend has proper accounting
                self._tower_id = response.tower_id or None
                self._tower_group_id = response.tower_group_id or None
                self._runway_id = response.runway_id or None
                self._status = SandboxStatus.COMPLETED
                self._status_updated_at = datetime.now(UTC)
                self._started_at = (
                    response.started_at_time.ToDatetime() if response.started_at_time else None
                )
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
            await self._close_streaming_client()

    async def exec(
        self,
        command: list[str],
        *,
        check: bool = False,
        timeout_seconds: float | None = None,
        stream_output: bool = False,
        on_stdout: Callable[[bytes], None] | None = None,
        on_stderr: Callable[[bytes], None] | None = None,
    ) -> ExecResult:
        """Execute a command in the running sandbox.

        Args:
            command: Command and arguments to execute
            check: If True, raise SandboxExecutionError on non-zero returncode
            timeout_seconds: Timeout for the command execution
            stream_output: If True, stream output to stdout/stderr in real-time.
                This is useful for long-running commands like training scripts.
            on_stdout: Optional callback invoked with stdout bytes as they arrive.
                Overrides the default stream_output=True behavior of printing to stdout.
            on_stderr: Optional callback invoked with stderr bytes as they arrive.
                Overrides the default stream_output=True behavior of printing to stderr.

        Returns:
            ExecResult with stdout, stderr, and returncode

        Raises:
            SandboxNotRunningError: If sandbox is not running
            SandboxExecutionError: If check=True and command returns non-zero
            SandboxTimeoutError: If command exceeds timeout_seconds
            ConnectError: If an unexpected error occurs

        Example:
            # Basic execution (non-streaming)
            result = await sandbox.exec(["echo", "hello"])

            # Stream output to console in real-time
            result = await sandbox.exec(["python", "train.py"], stream_output=True)

            # Custom streaming callbacks for advanced use cases
            result = await sandbox.exec(
                ["python", "train.py"],
                on_stdout=lambda data: my_logger.info(data.decode()),
                on_stderr=lambda data: my_logger.error(data.decode()),
            )
        """
        if not command:
            raise ValueError("Command cannot be empty")

        # If streaming is enabled or callbacks are provided, use the streaming API
        if stream_output or on_stdout is not None or on_stderr is not None:
            # Default to printing to stdio if stream_output=True but no callbacks
            if stream_output:
                on_stdout = on_stdout or _default_stdout_callback
                on_stderr = on_stderr or _default_stderr_callback

            return await self._exec_with_streaming(
                command,
                check=check,
                timeout_seconds=timeout_seconds,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

        timeout = timeout_seconds or self._request_timeout_seconds

        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        await self._ensure_client()
        assert self._client is not None

        logger.debug("Executing command in sandbox %s: %s", self._sandbox_id, shlex.join(command))

        request = atc_pb2.ExecSandboxRequest(
            sandbox_id=self._sandbox_id,
            command=command,
            max_timeout_seconds=int(timeout),
        )

        try:
            response = await asyncio.wait_for(
                self._client.exec(request),
                timeout=timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
            )
        except ConnectError as e:
            # Many different errors throw `ConnectError` (auth, server, etc.), only catch a timeout
            if e.code == Code.DEADLINE_EXCEEDED:
                raise SandboxTimeoutError(
                    f"Command {shlex.join(command)} timed out after {timeout}s"
                ) from e
            raise

        logger.debug("Command completed with exit code %d", response.result.exit_code)

        stdout_raw = response.result.stdout
        stderr_raw = response.result.stderr

        result = ExecResult(
            stdout_bytes=stdout_raw if stdout_raw else b"",
            stderr_bytes=stderr_raw if stderr_raw else b"",
            returncode=response.result.exit_code,
            command=command,
        )

        if check and result.returncode != 0:
            raise SandboxExecutionError(
                f"Command {shlex.join(command)} failed with exit code {result.returncode}",
                exec_result=result,
            )

        return result

    async def _exec_with_streaming(
        self,
        command: list[str],
        *,
        check: bool = False,
        timeout_seconds: float | None = None,
        on_stdout: Callable[[bytes], None] | None = None,
        on_stderr: Callable[[bytes], None] | None = None,
    ) -> ExecResult:
        """Execute a command using streaming API with callbacks."""
        timeout = timeout_seconds or self._request_timeout_seconds

        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        await self._ensure_streaming_client()
        assert self._streaming_client is not None

        logger.debug("Streaming exec in sandbox %s: %s", self._sandbox_id, shlex.join(command))

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        exit_code: int | None = None

        # Queue decouples httpx iteration from our processing.
        # Without this, processing suspends the httpx stream and breaks HTTP/2.
        queue: asyncio.Queue[streaming_pb2.ExecStreamResponse | Exception | None] = asyncio.Queue()

        async def request_iterator() -> AsyncIterator[streaming_pb2.ExecStreamRequest]:
            yield streaming_pb2.ExecStreamRequest(
                init=streaming_pb2.ExecStreamInit(
                    sandbox_id=self._sandbox_id,
                    command=command,
                )
            )

        async def collect() -> None:
            """Collect responses from the stream into the queue."""
            try:
                stream = self._streaming_client.stream_exec(
                    request_iterator(),
                    timeout_ms=int(timeout * 1000) if timeout else None,
                )
                async for resp in stream:
                    await queue.put(resp)
                    if resp.HasField("exit") or resp.HasField("error"):
                        return
            except ConnectError as e:
                if e.code == Code.DEADLINE_EXCEEDED:
                    await queue.put(
                        SandboxTimeoutError(
                            f"Streaming exec {shlex.join(command)} timed out after {timeout}s"
                        )
                    )
                else:
                    await queue.put(e)
            except Exception as e:
                await queue.put(e)
            finally:
                await queue.put(None)

        task = asyncio.create_task(collect())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                # Process the proto response directly
                resp = item
                if resp.HasField("output"):
                    data = bytes(resp.output.data)
                    is_stderr = (
                        resp.output.stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR
                    )
                    if is_stderr:
                        stderr_chunks.append(data)
                        if on_stderr is not None:
                            on_stderr(data)
                    else:
                        stdout_chunks.append(data)
                        if on_stdout is not None:
                            on_stdout(data)
                elif resp.HasField("exit"):
                    exit_code = resp.exit.exit_code
                    logger.debug("Streaming exec exit code %d", exit_code)
                    break
                elif resp.HasField("error"):
                    err = resp.error
                    raise SandboxExecutionError(
                        f"Streaming exec error: {err.message} (code: {err.code})"
                    )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if exit_code is None:
            raise SandboxExecutionError(
                f"Stream ended without exit code for command: {shlex.join(command)}"
            )

        result = ExecResult(
            stdout_bytes=b"".join(stdout_chunks),
            stderr_bytes=b"".join(stderr_chunks),
            returncode=exit_code,
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
