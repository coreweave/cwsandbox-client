# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

from __future__ import annotations

import asyncio
import builtins
import contextlib
import logging
import os
import shlex
import time
import warnings
from collections.abc import AsyncIterator, Generator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import grpc
import grpc.aio
from coreweave.aviato.v1beta1 import (
    atc_pb2,
    atc_pb2_grpc,
    streaming_pb2,
    streaming_pb2_grpc,
)
from google.protobuf import timestamp_pb2

from aviato._auth import resolve_auth_metadata
from aviato._defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
    DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    DEFAULT_MAX_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_BACKOFF_FACTOR,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    STDIN_CHUNK_SIZE,
    SandboxDefaults,
)
from aviato._loop_manager import _LoopManager
from aviato._network import create_channel, parse_grpc_target
from aviato._types import (
    ExecOutcome,
    NetworkOptions,
    OperationRef,
    Process,
    ProcessResult,
    StreamReader,
    StreamWriter,
)
from aviato.exceptions import (
    AviatoAuthenticationError,
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
        """Convert protobuf status enum to SandboxStatus."""
        try:
            proto_name = atc_pb2.SandboxStatus.Name(proto_status)
            enum_name = proto_name.replace("SANDBOX_STATUS_", "")
            return cls[enum_name]
        except ValueError:
            logger.warning("Unknown sandbox status %s, treating as UNSPECIFIED", proto_status)
            return cls.UNSPECIFIED

    def to_proto(self) -> int:
        """Convert SandboxStatus to protobuf enum"""
        proto_name = f"SANDBOX_STATUS_{self.name}"
        return atc_pb2.SandboxStatus.Value(proto_name)


def _validate_cwd(cwd: str | None) -> None:
    """Validate cwd parameter for exec().

    Args:
        cwd: Working directory path to validate

    Raises:
        ValueError: If cwd is empty string or not an absolute path
    """
    if cwd is None:
        return
    if not cwd:
        raise ValueError("cwd cannot be empty string")
    if not cwd.startswith("/"):
        raise ValueError(f"cwd must be an absolute path, got: {cwd!r}")


def _wrap_command_with_cwd(command: Sequence[str], cwd: str) -> list[str]:
    """Wrap command with shell cd to change working directory.

    Args:
        command: Original command and arguments (must not be empty)
        cwd: Absolute path for working directory

    Returns:
        Wrapped command: ["/bin/sh", "-c", "cd /path && exec cmd arg1 arg2"]

    Raises:
        ValueError: If command is empty
    """
    if not command:
        raise ValueError("Command cannot be empty when wrapping with cwd")
    escaped_cwd = shlex.quote(cwd)
    escaped_command = " ".join(shlex.quote(arg) for arg in command)
    return ["/bin/sh", "-c", f"cd {escaped_cwd} && exec {escaped_command}"]


def _translate_rpc_error(
    e: grpc.RpcError,
    *,
    sandbox_id: str | None = None,
    operation: str = "operation",
) -> SandboxError | AviatoAuthenticationError:
    """Translate gRPC RpcError to appropriate Aviato exception.

    Args:
        e: The gRPC RpcError to translate
        sandbox_id: Optional sandbox ID for context in error messages
        operation: Description of the operation that failed

    Returns:
        An appropriate Aviato exception
    """
    code = e.code()
    details = e.details() or str(e)

    if code == grpc.StatusCode.NOT_FOUND:
        return SandboxNotFoundError(
            f"Sandbox '{sandbox_id}' not found" if sandbox_id else details,
            sandbox_id=sandbox_id,
        )
    elif code == grpc.StatusCode.CANCELLED:
        return SandboxNotRunningError(
            f"{operation} was cancelled"
            + (f" (sandbox {sandbox_id} connection closed)" if sandbox_id else "")
        )
    elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return SandboxTimeoutError(f"{operation} timed out: {details}")
    elif code == grpc.StatusCode.UNAVAILABLE:
        return SandboxNotRunningError(f"Service unavailable: {details}")
    elif code == grpc.StatusCode.PERMISSION_DENIED:
        return AviatoAuthenticationError(f"Permission denied: {details}")
    elif code == grpc.StatusCode.UNAUTHENTICATED:
        return AviatoAuthenticationError(f"Authentication failed: {details}")
    else:
        return SandboxError(f"{operation} failed: {details}")


class Sandbox:
    """Aviato sandbox client with sync/async hybrid API.

    All methods return immediately and can be used in both sync and async contexts.
    Operations are executed in a background event loop managed by _LoopManager.

    Examples:
        Factory method:
        ```python
        sb = Sandbox.run("echo", "hello")  # Returns immediately
        result = sb.exec(["echo", "more"]).result()  # Block for result
        sb.stop().result()  # Block for completion
        ```

        Context manager (recommended):
        ```python
        with Sandbox.run("sleep", "infinity") as sb:
            result = sb.exec(["echo", "hello"]).result()
        # Automatically stopped on exit
        ```

        Async context manager:
        ```python
        async with Sandbox.run("sleep", "infinity") as sb:
            result = await sb.exec(["echo", "hello"])
        ```

    Attributes:
        sandbox_id: Unique identifier for this sandbox.
        status: Cached status from last API call.
        tower_id: Tower ID where sandbox is running.
        runway_id: Runway ID for this sandbox.
        returncode: Exit code if sandbox completed.
        started_at: When sandbox started running.
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
        network: NetworkOptions | dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
        environment_variables: dict[str, str] | None = None,
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
            network: Network configuration (NetworkOptions dataclass)
            max_timeout_seconds: Maximum timeout for sandbox operations
            environment_variables: Environment variables to inject into the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive config only.
        """
        if network is not None:
            if isinstance(network, dict):
                network = NetworkOptions(**network)
            elif not isinstance(network, NetworkOptions):
                raise TypeError(
                    f"network must be NetworkOptions, dict, or None, got {type(network).__name__}"
                )

        self._defaults = defaults or SandboxDefaults()
        self._session = _session

        # Note: These can be None for sandboxes discovered via list()/from_id()
        self._command: str | None = command or self._defaults.command
        self._args: list[str] | None = args if args is not None else list(self._defaults.args)

        # Apply defaults with explicit values taking precedence
        self._container_image: str | None = container_image or self._defaults.container_image
        self._base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or self._defaults.base_url
        ).rstrip("/")
        self._request_timeout_seconds = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else self._defaults.request_timeout_seconds
        )
        self._max_lifetime_seconds = (
            max_lifetime_seconds
            if max_lifetime_seconds is not None
            else self._defaults.max_lifetime_seconds
        )

        self._tags: list[str] | None = self._defaults.merge_tags(tags)
        self._environment_variables = self._defaults.merge_environment_variables(
            environment_variables
        )

        self._runway_ids: list[str] | None
        if runway_ids is not None:
            self._runway_ids = list(runway_ids)
        elif self._defaults.runway_ids:
            self._runway_ids = list(self._defaults.runway_ids)
        else:
            self._runway_ids = None

        self._tower_ids: list[str] | None
        if tower_ids is not None:
            self._tower_ids = list(tower_ids)
        elif self._defaults.tower_ids:
            self._tower_ids = list(self._defaults.tower_ids)
        else:
            self._tower_ids = None

        self._start_kwargs: dict[str, Any] = {}
        # Use explicit resources or fall back to defaults
        effective_resources = resources if resources is not None else self._defaults.resources
        if effective_resources is not None:
            self._start_kwargs["resources"] = effective_resources
        if mounted_files is not None:
            self._start_kwargs["mounted_files"] = mounted_files
        if s3_mount is not None:
            self._start_kwargs["s3_mount"] = s3_mount
        if ports is not None:
            self._start_kwargs["ports"] = ports
        # Use explicit network or fall back to defaults
        effective_network = network if network is not None else self._defaults.network
        if effective_network is not None:
            self._start_kwargs["network"] = effective_network
        if max_timeout_seconds is not None:
            self._start_kwargs["max_timeout_seconds"] = max_timeout_seconds

        self._channel: grpc.aio.Channel | None = None
        self._stub: atc_pb2_grpc.ATCServiceStub | None = None
        self._auth_metadata: tuple[tuple[str, str], ...] = ()
        self._streaming_channel: grpc.aio.Channel | None = None
        self._streaming_channel_lock = asyncio.Lock()
        self._sandbox_id: str | None = None
        self._returncode: int | None = None
        self._tower_id: str | None = None
        self._runway_id: str | None = None
        # TODO: Remove _stopped once backend adds proper accounting for stopped/terminated
        # sandboxes. Currently backend stops tracking sandbox IDs after stop(), so we need
        # this client-side flag as a workaround. Once backend properly tracks stopped
        # sandboxes, we can remove this and query backend status directly.
        self._stopped = False
        self._start_lock = asyncio.Lock()

        # Shared polling task for _wait_until_running_async deduplication
        self._running_task: asyncio.Task[None] | None = None
        self._running_lock = asyncio.Lock()

        # Shared polling task for _wait_until_complete_async deduplication
        self._complete_task: asyncio.Task[SandboxStatus] | None = None
        self._complete_lock = asyncio.Lock()

        self._status: SandboxStatus | None = None
        self._status_updated_at: datetime | None = None
        self._started_at: datetime | None = None
        self._tower_group_id: str | None = None
        self._service_address: str | None = None
        self._exposed_ports: tuple[tuple[int, str], ...] | None = None
        self._applied_ingress_mode: str | None = None
        self._applied_egress_mode: str | None = None

        # Execution statistics for metrics
        self._exec_count = 0
        self._exec_completed_ok = 0
        self._exec_completed_nonzero = 0
        self._exec_failures = 0

        # Startup timing for metrics
        self._start_accepted_at: float | None = None
        self._startup_recorded: bool = False

        # Get the singleton loop manager for sync/async bridging
        self._loop_manager = _LoopManager.get()

    @classmethod
    def run(
        cls,
        *args: str,
        container_image: str | None = None,
        defaults: SandboxDefaults | None = None,
        request_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        tags: list[str] | None = None,
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        network: NetworkOptions | dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
        environment_variables: dict[str, str] | None = None,
    ) -> Sandbox:
        """Create and start a sandbox, return immediately once backend accepts.

        Does NOT wait for RUNNING status. Use .wait() to block until ready.
        If positional args are provided, the first is the command and the rest
        are its arguments. If no args are provided, uses defaults (tail -f /dev/null).

        Args:
            *args: Optional command and arguments (e.g., "echo", "hello", "world").
                If omitted, uses default command from SandboxDefaults.
            container_image: Container image to use
            defaults: Optional SandboxDefaults to apply
            request_timeout_seconds: Timeout for API requests (client-side)
            max_lifetime_seconds: Max sandbox lifetime (server-side)
            tags: Optional tags for the sandbox
            runway_ids: Optional list of runway IDs
            tower_ids: Optional list of tower IDs
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            network: Network configuration (NetworkOptions dataclass)
            max_timeout_seconds: Maximum timeout for sandbox operations
            environment_variables: Environment variables to inject into the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive config only.
        Returns:
            A Sandbox instance (start request sent, but may still be starting)

        Example:
            ```python
            # Using defaults (tail -f /dev/null)
            sb = Sandbox.run()

            # Fire and forget style
            sb = Sandbox.run("echo", "hello")
            # sb.sandbox_id is set, but sandbox may still be starting

            # Wait for ready if needed
            sb = Sandbox.run("sleep", "infinity").wait()
            result = sb.exec(["echo", "hello"]).result()

            # Or use context manager for automatic cleanup
            with Sandbox.run("sleep", "infinity") as sb:
                result = sb.exec(["echo", "hello"]).result()
            ```
        """
        if network is not None:
            if isinstance(network, dict):
                network = NetworkOptions(**network)
            elif not isinstance(network, NetworkOptions):
                raise TypeError(
                    f"network must be NetworkOptions, dict, or None, got {type(network).__name__}"
                )

        command = args[0] if args else None
        cmd_args = list(args[1:]) if len(args) > 1 else None

        sandbox = cls(
            command=command,
            args=cmd_args,
            container_image=container_image,
            defaults=defaults,
            request_timeout_seconds=request_timeout_seconds,
            max_lifetime_seconds=max_lifetime_seconds,
            tags=tags,
            runway_ids=runway_ids,
            tower_ids=tower_ids,
            resources=resources,
            mounted_files=mounted_files,
            s3_mount=s3_mount,
            ports=ports,
            network=network,
            max_timeout_seconds=max_timeout_seconds,
            environment_variables=environment_variables,
        )
        logger.debug("Creating sandbox with command: %s", command)
        sandbox.start().result()
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
            ```python
            session = Sandbox.session(defaults)
            sb = session.create(command="sleep", args=["infinity"])

            @session.function()
            def compute(x, y):
                return x + y

            await session.close()
            ```
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
        sandbox._environment_variables = {}
        sandbox._channel = None
        sandbox._stub = None
        sandbox._auth_metadata = ()
        sandbox._streaming_channel = None
        sandbox._streaming_channel_lock = asyncio.Lock()
        sandbox._stopped = False
        sandbox._returncode = None
        sandbox._session = None
        sandbox._defaults = SandboxDefaults()
        sandbox._start_kwargs = {}
        sandbox._start_lock = asyncio.Lock()
        sandbox._running_task = None
        sandbox._running_lock = asyncio.Lock()
        sandbox._complete_task = None
        sandbox._complete_lock = asyncio.Lock()
        sandbox._loop_manager = _LoopManager.get()
        sandbox._service_address = None
        sandbox._exposed_ports = None
        # TODO: Add applied_ingress_mode/applied_egress_mode once backend adds to GetSandboxResponse
        sandbox._applied_ingress_mode = None
        sandbox._applied_egress_mode = None
        # Exec stats for discovered sandboxes
        sandbox._exec_count = 0
        sandbox._exec_completed_ok = 0
        sandbox._exec_completed_nonzero = 0
        sandbox._exec_failures = 0
        # Startup already happened for discovered sandboxes
        sandbox._start_accepted_at = None
        sandbox._startup_recorded = True
        return sandbox

    @classmethod
    def list(
        cls,
        *,
        tags: list[str] | None = None,
        status: str | None = None,
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[builtins.list[Sandbox]]:
        """List existing sandboxes with optional filters.

        Returns OperationRef that resolves to Sandbox instances usable for
        operations like exec(), stop(), get_status(), read_file(), write_file().

        Args:
            tags: Filter by tags (sandboxes must have ALL specified tags)
            status: Filter by status ("running", "completed", "failed", etc.)
            runway_ids: Filter by runway IDs
            tower_ids: Filter by tower IDs
            base_url: Override API URL (default: AVIATO_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)

        Returns:
            OperationRef[list[Sandbox]]: Use .result() to block for results,
            or await directly in async contexts.

        Example:
            ```python
            # Sync usage
            sandboxes = Sandbox.list(tags=["my-batch-job"]).result()
            for sb in sandboxes:
                print(f"{sb.sandbox_id}: {sb.status}")
                sb.stop().result()

            # Async usage
            sandboxes = await Sandbox.list(status="running")
            for sb in sandboxes:
                result = await sb.exec(["echo", "hello"])
            ```
        """
        future = _LoopManager.get().run_async(
            cls._list_async(
                tags=tags,
                status=status,
                runway_ids=runway_ids,
                tower_ids=tower_ids,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _list_async(
        cls,
        *,
        tags: builtins.list[str] | None = None,
        status: str | None = None,
        runway_ids: builtins.list[str] | None = None,
        tower_ids: builtins.list[str] | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> builtins.list[Sandbox]:
        """Internal async: List existing sandboxes with optional filters."""
        effective_base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )

        status_enum = None
        if status is not None:
            status_enum = SandboxStatus(status)

        auth_metadata = resolve_auth_metadata()

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = atc_pb2_grpc.ATCServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request_kwargs: dict[str, Any] = {}
            if tags:
                request_kwargs["tags"] = tags
            if status_enum:
                request_kwargs["status"] = status_enum.to_proto()
            if runway_ids is not None:
                request_kwargs["runway_ids"] = runway_ids
            if tower_ids is not None:
                request_kwargs["tower_ids"] = tower_ids

            request = atc_pb2.ListSandboxesRequest(**request_kwargs)
            try:
                response = await stub.List(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, operation="List sandboxes") from e

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
        finally:
            await channel.close(grace=None)

    @classmethod
    def from_id(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[Sandbox]:
        """Attach to an existing sandbox by ID.

        Creates a Sandbox instance connected to an existing sandbox,
        allowing operations like exec(), stop(), get_status(), etc.

        Args:
            sandbox_id: The ID of the existing sandbox
            base_url: Override API URL (default: AVIATO_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)

        Returns:
            OperationRef[Sandbox]: Use .result() to block for the Sandbox instance,
            or await directly in async contexts.

        Raises:
            SandboxNotFoundError: If sandbox doesn't exist

        Example:
            ```python
            # Sync usage
            sb = Sandbox.from_id("sandbox-abc123").result()
            result = sb.exec(["python", "-c", "print('hello')"]).result()
            sb.stop().result()

            # Async usage
            sb = await Sandbox.from_id("sandbox-abc123")
            result = await sb.exec(["python", "-c", "print('hello')"])
            ```
        """
        future = _LoopManager.get().run_async(
            cls._from_id_async(
                sandbox_id,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _from_id_async(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Sandbox:
        """Internal async: Attach to an existing sandbox by ID."""
        effective_base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )

        auth_metadata = resolve_auth_metadata()

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = atc_pb2_grpc.ATCServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request = atc_pb2.GetSandboxRequest(sandbox_id=sandbox_id)
            try:
                response = await stub.Get(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, sandbox_id=sandbox_id, operation="Get sandbox") from e

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
        finally:
            await channel.close(grace=None)

    @classmethod
    def delete(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> OperationRef[None]:
        """Delete a sandbox by ID without creating a Sandbox instance.

        This is a convenience method for cleanup scenarios where you
        don't need to perform other operations on the sandbox.

        Args:
            sandbox_id: The sandbox ID to delete
            base_url: Override API URL (default: AVIATO_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)
            missing_ok: If True, suppress SandboxNotFoundError when sandbox
                doesn't exist.

        Returns:
            OperationRef[None]: Use .result() to block until complete.
            Raises SandboxNotFoundError if not found (unless missing_ok=True),
            SandboxError if deletion failed.

        Raises:
            SandboxNotFoundError: If sandbox doesn't exist and missing_ok=False
            SandboxError: If deletion failed for other reasons

        Example:
            ```python
            # Sync usage
            Sandbox.delete("sandbox-abc123").result()

            # Ignore if already deleted
            Sandbox.delete("sandbox-abc123", missing_ok=True).result()

            # Async usage
            await Sandbox.delete("sandbox-abc123")
            ```
        """
        future = _LoopManager.get().run_async(
            cls._delete_async(
                sandbox_id,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                missing_ok=missing_ok,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _delete_async(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> None:
        """Internal async: Delete a sandbox by ID."""
        effective_base_url = (
            base_url or os.environ.get("AVIATO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )

        auth_metadata = resolve_auth_metadata()

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = atc_pb2_grpc.ATCServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request = atc_pb2.DeleteSandboxRequest(sandbox_id=sandbox_id)
            try:
                response = await stub.Delete(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND and missing_ok:
                    return
                raise _translate_rpc_error(
                    e, sandbox_id=sandbox_id, operation="Delete sandbox"
                ) from e

            if not response.success:
                raise SandboxError(f"Failed to delete sandbox: {response.error_message}")
        finally:
            await channel.close(grace=None)

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

    @property
    def service_address(self) -> str | None:
        """External address for accessing sandbox services.

        Returns an address like '166.19.9.70:8080' for network-accessible sandboxes
        (SSH, web services). Availability depends on tower configuration.

        Returns None if:
        - Sandbox hasn't been started yet
        - Sandbox was obtained via from_id() or list()
        - Tower uses ClusterIP instead of LoadBalancer
        """
        return self._service_address

    @property
    def exposed_ports(self) -> tuple[tuple[int, str], ...] | None:
        """Exposed ports for the sandbox.

        Returns a tuple of (container_port, name) tuples for ports exposed by
        the sandbox. Useful for network-accessible sandboxes.

        Returns None if:
        - Sandbox hasn't been started yet
        - Sandbox was obtained via from_id() or list()
        - No ports were exposed
        """
        return self._exposed_ports

    @property
    def applied_ingress_mode(self) -> str | None:
        """The ingress mode applied by the backend (set after start)."""
        return self._applied_ingress_mode

    @property
    def applied_egress_mode(self) -> str | None:
        """The egress mode applied by the backend (set after start)."""
        return self._applied_egress_mode

    @property
    def exec_stats(self) -> dict[str, int]:
        """Execution statistics for this sandbox.

        Returns:
            Dictionary with execution counts:
            - exec_count: Total number of exec() calls
            - exec_completed_ok: Execs that completed with returncode 0
            - exec_completed_nonzero: Execs that completed with non-zero returncode
              (when check=False; with check=True, non-zero exits count as failures)
            - exec_failures: Execs that failed with an exception (including
              SandboxExecutionError from check=True with non-zero exit)
        """
        return {
            "exec_count": self._exec_count,
            "exec_completed_ok": self._exec_completed_ok,
            "exec_completed_nonzero": self._exec_completed_nonzero,
            "exec_failures": self._exec_failures,
        }

    def _on_exec_complete(
        self,
        result: ProcessResult | None,
        exception: BaseException | None,
    ) -> None:
        """Record exec completion outcome for metrics.

        Args:
            result: The ProcessResult if execution completed, None on failure
            exception: The exception if execution failed, None on success
        """
        if exception is not None:
            self._exec_failures += 1
            outcome = ExecOutcome.FAILURE
        elif result is not None:
            if result.returncode == 0:
                self._exec_completed_ok += 1
                outcome = ExecOutcome.COMPLETED_OK
            else:
                self._exec_completed_nonzero += 1
                outcome = ExecOutcome.COMPLETED_NONZERO
        else:
            return

        if self._session is not None:
            self._session._record_exec_outcome(outcome, self._sandbox_id)

    def __repr__(self) -> str:
        if self._status:
            status_str = self._status.value
        elif self._sandbox_id is None:
            status_str = "not_started"
        else:
            status_str = "unknown"
        return f"<Sandbox id={self._sandbox_id} status={status_str}>"

    async def _get_status_async(self) -> SandboxStatus:
        """Internal async: Get the current status from the backend."""
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("Sandbox has not been started")

        await self._ensure_client()
        assert self._stub is not None

        request = atc_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
        try:
            response = await self._stub.Get(
                request, timeout=self._request_timeout_seconds, metadata=self._auth_metadata
            )
        except grpc.RpcError as e:
            raise _translate_rpc_error(
                e, sandbox_id=self._sandbox_id, operation="Get status"
            ) from e

        status = SandboxStatus.from_proto(response.sandbox_status)
        self._status = status
        self._status_updated_at = datetime.now(UTC)

        return status

    def get_status(self) -> SandboxStatus:
        """Get the current status of the sandbox from the backend.

        Returns:
            SandboxStatus enum value

        Raises:
            SandboxNotRunningError: If sandbox has not been started

        Example:
            ```python
            sb = Sandbox.run("sleep", "10")
            status = sb.get_status()
            print(f"Sandbox is {status}")  # SandboxStatus.PENDING or RUNNING
            ```
        """
        return self._loop_manager.run_sync(self._get_status_async())

    # Context managers

    def __enter__(self) -> Sandbox:
        """Enter sync context manager.

        If sandbox not started, starts it. Returns self for use in with statement.
        """
        if self._sandbox_id is None:
            self.start().result()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager, stopping the sandbox.

        If an exception is in flight, suppresses stop errors to avoid masking
        the original exception. Stop errors are logged as warnings.
        """
        if self._sandbox_id is None or self._stopped:
            return
        try:
            self.stop().result()
        except Exception as stop_error:
            if exc_val is not None:
                logger.warning(
                    "Failed to stop sandbox %s during exception handling: %s",
                    self._sandbox_id,
                    stop_error,
                )
            else:
                raise

    async def __aenter__(self) -> Sandbox:
        """Enter async context manager.

        If sandbox not started, starts it. Returns self for use in async with.
        """
        if self._sandbox_id is None:
            future = self._loop_manager.run_async(self._start_async())
            await asyncio.wrap_future(future)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager, stopping the sandbox.

        If an exception is in flight, suppresses stop errors to avoid masking
        the original exception. Stop errors are logged as warnings.
        """
        if self._sandbox_id is None or self._stopped:
            return
        try:
            await self.stop()
        except Exception as stop_error:
            if exc_val is not None:
                logger.warning(
                    "Failed to stop sandbox %s during exception handling: %s",
                    self._sandbox_id,
                    stop_error,
                )
            else:
                raise

    def __del__(self) -> None:
        """Warn if sandbox was not properly stopped."""
        if hasattr(self, "_sandbox_id") and self._sandbox_id is not None and not self._stopped:
            warnings.warn(
                f"Sandbox {self._sandbox_id} was not stopped. "
                "Use 'sandbox.stop().result()' or the context manager pattern.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_client(self) -> None:
        """Ensure the gRPC channel and stub are initialized."""
        if self._channel is not None:
            return

        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(self._base_url)
        channel = create_channel(target, is_secure)
        stub = atc_pb2_grpc.ATCServiceStub(channel)  # type: ignore[no-untyped-call]
        self._channel = channel
        self._stub = stub
        self._auth_metadata = auth_metadata
        logger.debug("Initialized gRPC channel for %s", self._base_url)

    async def _get_or_create_streaming_channel(self) -> grpc.aio.Channel:
        """Get or create the cached streaming gRPC channel."""
        if self._streaming_channel is not None:
            return self._streaming_channel

        async with self._streaming_channel_lock:
            if self._streaming_channel is not None:
                return self._streaming_channel

            target, is_secure = parse_grpc_target(self._base_url)
            channel = create_channel(target, is_secure)

            try:
                await asyncio.wait_for(
                    channel.channel_ready(),
                    timeout=self._request_timeout_seconds,
                )
            except TimeoutError:
                await channel.close(grace=None)
                raise SandboxTimeoutError(
                    f"Timed out connecting to streaming service at {target}"
                ) from None

            self._streaming_channel = channel
            return channel

    async def _poll_until_stable(
        self,
        timeout_seconds: float | None = None,
        timeout_message: str = "",
    ) -> atc_pb2.GetSandboxResponse:
        """Poll sandbox status until a stable state is reached.

        Returns the response when sandbox reaches a stable state (RUNNING,
        COMPLETED, FAILED, TERMINATED, or UNSPECIFIED). Transient states
        like CREATING, PENDING, and PAUSED are polled through.

        Args:
            timeout_seconds: Maximum time to wait, or None for no timeout
                (relies on external cancellation via stop() or asyncio.wait_for).
            timeout_message: Message for SandboxTimeoutError if timeout occurs

        Returns:
            The GetSandboxResponse with a stable status
        """
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox ID available")

        await self._ensure_client()
        assert self._stub is not None

        start_time = time.monotonic()
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS

        while True:
            if self._stopped or self._channel is None:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} was stopped while polling"
                )

            if timeout_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout_seconds:
                    raise SandboxTimeoutError(timeout_message)

            request = atc_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
            try:
                response: atc_pb2.GetSandboxResponse = await self._stub.Get(
                    request,
                    timeout=self._request_timeout_seconds,
                    metadata=self._auth_metadata,
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(
                    e, sandbox_id=self._sandbox_id, operation="Poll sandbox status"
                ) from e

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

    async def _ensure_started_async(self) -> None:
        """Ensure sandbox has been started, starting it if needed."""
        if self._sandbox_id is None:
            await self._start_async()

    async def _start_async(self) -> str:
        """Internal async: Send StartSandbox to backend, return sandbox_id.

        Does NOT wait for RUNNING status. Idempotent - safe to call multiple times.
        """
        if self._sandbox_id is not None:
            return self._sandbox_id

        async with self._start_lock:
            if self._sandbox_id is not None:
                return self._sandbox_id
            if self._stopped:
                raise SandboxNotRunningError("Sandbox has been stopped")

            await self._ensure_client()
            assert self._stub is not None

            request_kwargs: dict[str, Any] = {
                "command": self._command,
                "args": self._args or [],
                "container_image": self._container_image,
                "tags": self._tags or [],
            }

            if self._max_lifetime_seconds is not None:
                request_kwargs["max_lifetime_seconds"] = int(self._max_lifetime_seconds)
            if self._runway_ids is not None:
                request_kwargs["runway_ids"] = self._runway_ids
            if self._tower_ids is not None:
                request_kwargs["tower_ids"] = self._tower_ids
            if self._environment_variables:
                request_kwargs["environment_variables"] = self._environment_variables

            request_kwargs.update(self._start_kwargs)

            # Convert NetworkOptions to dict for proto
            network = request_kwargs.get("network")
            if network is not None and isinstance(network, NetworkOptions):
                net_opts = network
                network_dict: dict[str, Any] = {}
                if net_opts.ingress_mode is not None:
                    network_dict["ingress_mode"] = net_opts.ingress_mode
                if net_opts.exposed_ports is not None:
                    network_dict["exposed_ports"] = list(net_opts.exposed_ports)
                if net_opts.egress_mode is not None:
                    network_dict["egress_mode"] = net_opts.egress_mode
                request_kwargs["network"] = network_dict

            logger.debug("Starting sandbox with image %s", self._container_image)

            request = atc_pb2.StartSandboxRequest(**request_kwargs)
            self._start_accepted_at = time.monotonic()
            try:
                response = await self._stub.Start(
                    request, timeout=self._request_timeout_seconds, metadata=self._auth_metadata
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, operation="Start sandbox") from e

            sandbox_id = str(response.sandbox_id)
            self._sandbox_id = sandbox_id
            self._status = SandboxStatus.PENDING
            self._status_updated_at = datetime.now(UTC)
            self._service_address = response.service_address or None
            self._exposed_ports = (
                tuple((p.container_port, p.name) for p in response.exposed_ports)
                if response.exposed_ports
                else None
            )
            self._applied_ingress_mode = response.applied_ingress_mode or None
            self._applied_egress_mode = response.applied_egress_mode or None

            logger.debug("Sandbox %s created (pending)", sandbox_id)
            return sandbox_id

    async def _do_poll_running(self) -> None:
        """Poll until sandbox reaches a stable state and update instance fields.

        Used as the body of the shared _running_task so multiple concurrent
        waiters share a single polling loop instead of each hitting the API.
        Polls indefinitely, relying on external cancellation via stop() for
        termination. Per-waiter timeouts allow individual waiters to give up
        without killing the shared poll (asyncio.shield protects the task).

        Raises directly on FAILED/TERMINATED so all waiters see the same
        exception. This differs from _do_poll_complete which returns
        SandboxStatus for per-waiter raise_on_termination control.
        """
        response = await self._poll_until_stable()

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
                if not self._startup_recorded and self._start_accepted_at is not None:
                    startup_time = time.monotonic() - self._start_accepted_at
                    self._startup_recorded = True
                    if self._session is not None:
                        self._session._record_startup_time(startup_time)
                logger.debug("Sandbox %s is running", self._sandbox_id)
            case atc_pb2.SANDBOX_STATUS_FAILED:
                self._status = SandboxStatus.FAILED
                self._status_updated_at = datetime.now(UTC)
                raise SandboxFailedError(f"Sandbox {self._sandbox_id} failed to start")
            case atc_pb2.SANDBOX_STATUS_TERMINATED:
                self._status = SandboxStatus.TERMINATED
                self._status_updated_at = datetime.now(UTC)
                raise SandboxTerminatedError(f"Sandbox {self._sandbox_id} was terminated")
            case atc_pb2.SANDBOX_STATUS_COMPLETED | atc_pb2.SANDBOX_STATUS_UNSPECIFIED:
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
            case _:
                raise SandboxError(f"Unexpected sandbox status: {response.sandbox_status}")

    def _on_poll_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback when _running_task completes.

        Retrieves and logs exceptions to prevent 'Task exception was never
        retrieved' warnings. Always clears the task reference so future
        waiters start a fresh poll instead of seeing a stale completed task.
        """
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.debug(
                "Polling task for sandbox %s failed: %s",
                self._sandbox_id,
                exc,
            )
        if self._running_task is task:
            self._running_task = None

    async def _wait_until_running_async(self, timeout: float | None = None) -> None:
        """Internal async: Wait until sandbox reaches RUNNING status.

        Multiple concurrent callers share a single polling task to avoid
        redundant GetSandbox API calls. asyncio.shield() prevents one
        caller's cancellation/timeout from killing the poll for others.
        """
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._status == SandboxStatus.RUNNING:
            return

        await self._ensure_started_async()
        effective_timeout = timeout if timeout is not None else self._request_timeout_seconds

        async with self._running_lock:
            if self._stopped:
                raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
            # Re-check after lock acquisition: another coroutine may have
            # completed polling between our first check and acquiring the lock.
            if self._status == SandboxStatus.RUNNING:  # type: ignore[comparison-overlap]
                return
            if self._running_task is None:
                self._running_task = asyncio.create_task(self._do_poll_running())
                self._running_task.add_done_callback(self._on_poll_task_done)
            task = self._running_task

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=effective_timeout)
        except TimeoutError:
            raise SandboxTimeoutError(
                f"Sandbox {self._sandbox_id} did not become ready within {effective_timeout}s"
            ) from None
        except asyncio.CancelledError:
            if self._stopped:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} has been stopped"
                ) from None
            raise

    async def _do_poll_complete(self) -> SandboxStatus:
        """Poll until sandbox reaches terminal state and return the status.

        Used as the body of the shared _complete_task so multiple concurrent
        waiters share a single polling loop instead of each hitting the API.
        Returns SandboxStatus instead of raising, so each waiter can apply
        its own raise_on_termination policy. Since all waiters share a single
        shielded task, raising here would force all to see the same exception,
        preventing per-waiter raise_on_termination control.

        Polls indefinitely, relying on external cancellation via stop() for
        termination. Per-waiter timeouts allow individual waiters to give up
        without killing the shared poll (asyncio.shield protects the task).
        """
        assert self._sandbox_id is not None
        sandbox_id = self._sandbox_id
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS
        while True:
            response = await self._poll_until_stable()

            match response.sandbox_status:
                case atc_pb2.SANDBOX_STATUS_COMPLETED | atc_pb2.SANDBOX_STATUS_UNSPECIFIED:
                    self._status = SandboxStatus.COMPLETED
                    self._status_updated_at = datetime.now(UTC)
                    self._returncode = 0
                    logger.info("Sandbox %s completed", sandbox_id)
                    return SandboxStatus.COMPLETED
                case atc_pb2.SANDBOX_STATUS_FAILED:
                    self._status = SandboxStatus.FAILED
                    self._status_updated_at = datetime.now(UTC)
                    return SandboxStatus.FAILED
                case atc_pb2.SANDBOX_STATUS_TERMINATED:
                    self._status = SandboxStatus.TERMINATED
                    self._status_updated_at = datetime.now(UTC)
                    self._returncode = None
                    logger.info("Sandbox %s was terminated", sandbox_id)
                    return SandboxStatus.TERMINATED
                case atc_pb2.SANDBOX_STATUS_RUNNING:
                    await asyncio.sleep(poll_interval)
                    poll_interval = min(
                        poll_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                        DEFAULT_MAX_POLL_INTERVAL_SECONDS,
                    )
                    continue
                case _:
                    raise SandboxError(f"Unexpected sandbox status: {response.sandbox_status}")

    def _on_complete_task_done(self, task: asyncio.Task[SandboxStatus]) -> None:
        """Callback when _complete_task completes.

        Retrieves and logs exceptions to prevent 'Task exception was never
        retrieved' warnings. Always clears the task reference so future
        waiters start a fresh poll instead of seeing a stale completed task.
        """
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.debug(
                "Complete-polling task for sandbox %s failed: %s",
                self._sandbox_id,
                exc,
            )
        if self._complete_task is task:
            self._complete_task = None

    async def _wait_until_complete_async(
        self,
        timeout: float | None = None,
        raise_on_termination: bool = True,
    ) -> None:
        """Internal async: Poll until sandbox reaches terminal state.

        Multiple concurrent callers share a single polling task to avoid
        redundant GetSandbox API calls. asyncio.shield() prevents one
        caller's cancellation/timeout from killing the poll for others.
        """
        await self._ensure_started_async()
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        if self._returncode is not None:
            return

        effective_timeout = timeout if timeout is not None else self._request_timeout_seconds

        async with self._complete_lock:
            if self._stopped:
                raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
            if self._returncode is not None:
                return
            if self._complete_task is None:
                self._complete_task = asyncio.create_task(self._do_poll_complete())
                self._complete_task.add_done_callback(self._on_complete_task_done)
            task = self._complete_task

        sandbox_id = self._sandbox_id
        try:
            status = await asyncio.wait_for(asyncio.shield(task), timeout=effective_timeout)
        except TimeoutError:
            raise SandboxTimeoutError(f"Timed out waiting for sandbox {sandbox_id}") from None
        except asyncio.CancelledError:
            if self._stopped:
                raise SandboxNotRunningError(f"Sandbox {sandbox_id} has been stopped") from None
            raise

        if status == SandboxStatus.FAILED:
            raise SandboxFailedError(f"Sandbox {sandbox_id} failed")
        if status == SandboxStatus.TERMINATED and raise_on_termination:
            raise SandboxTerminatedError(f"Sandbox {sandbox_id} was terminated")

    # Lifecycle methods

    def start(self) -> OperationRef[None]:
        """Send StartSandbox to backend, return OperationRef immediately.

        Does NOT wait for RUNNING status. Use wait() to block until ready.
        Call .result() to block until the start request is accepted.

        Returns:
            OperationRef[None]: Use .result() to block until backend accepts.

        Example:
            ```python
            sandbox = Sandbox(command="sleep", args=["infinity"])
            sandbox.start().result()
            print(f"Started sandbox: {sandbox.sandbox_id}")
            sandbox.wait()  # Block until RUNNING
            ```
        """

        async def _start_and_discard() -> None:
            await self._start_async()

        future = self._loop_manager.run_async(_start_and_discard())
        return OperationRef(future)

    def wait(self, timeout: float | None = None) -> Sandbox:
        """Block until sandbox reaches RUNNING or a terminal state.

        Returns when sandbox is RUNNING or has already completed (COMPLETED/UNSPECIFIED).

        Args:
            timeout: Maximum seconds to wait. None means use default timeout.

        Returns:
            Self for method chaining. Check .status to determine final state.

        Raises:
            SandboxFailedError: If sandbox fails to start
            SandboxTerminatedError: If sandbox was terminated externally
            SandboxTimeoutError: If timeout expires

        Example:
            ```python
            sb = Sandbox.run("sleep", "infinity").wait()
            result = sb.exec(["echo", "ready"]).result()
            ```
        """
        self._loop_manager.run_sync(self._wait_until_running_async(timeout))
        return self

    def wait_until_complete(
        self,
        timeout: float | None = None,
        *,
        raise_on_termination: bool = True,
    ) -> Sandbox:
        """Block until sandbox reaches terminal state (COMPLETED/FAILED/TERMINATED).

        After this returns successfully, returncode will be available.

        Args:
            timeout: Maximum seconds to wait. None means use default timeout.
            raise_on_termination: If True (default), raises SandboxTerminatedError
                if sandbox was terminated externally.

        Returns:
            Self for method chaining.

        Raises:
            SandboxTimeoutError: If timeout expires
            SandboxTerminatedError: If sandbox was terminated (and raise_on_termination=True)
            SandboxFailedError: If sandbox failed

        Example:
            ```python
            sb = Sandbox.run("python", "-c", "print('done')")
            sb.wait_until_complete()
            print(f"Exit code: {sb.returncode}")
            ```
        """
        self._loop_manager.run_sync(self._wait_until_complete_async(timeout, raise_on_termination))
        return self

    def __await__(self) -> Generator[Any, None, Sandbox]:
        """Make sandbox awaitable - await sandbox waits until RUNNING.

        Routes through _loop_manager to avoid cross-event-loop issues.
        Auto-starts if not already started.

        Example:
            ```python
            sb = Sandbox.run("sleep", "infinity")
            await sb  # Wait until RUNNING
            result = await sb.exec(["echo", "hello"])
            ```
        """

        async def _await_running() -> Sandbox:
            await self._ensure_started_async()
            await self._wait_until_running_async()
            return self

        future = self._loop_manager.run_async(_await_running())
        return asyncio.wrap_future(future).__await__()

    async def _stop_async(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
        missing_ok: bool = False,
    ) -> None:
        """Internal async: Stop the sandbox."""
        # Acquire _start_lock to wait for any in-flight start() to complete
        async with self._start_lock:
            if self._stopped:
                logger.debug("stop() called on already-stopped sandbox %s", self._sandbox_id)
                return
            if self._sandbox_id is None:
                logger.debug("stop() called on sandbox that was never started")
                self._stopped = True
                return
            self._stopped = True

        # Cancel the shared polling tasks so waiters get CancelledError
        if self._running_task is not None and not self._running_task.done():
            self._running_task.cancel()
            self._running_task = None
        if self._complete_task is not None and not self._complete_task.done():
            self._complete_task.cancel()
            self._complete_task = None

        await self._ensure_client()
        assert self._stub is not None

        logger.debug("Stopping sandbox %s", self._sandbox_id)

        max_timeout = int(graceful_shutdown_seconds) + int(DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS)
        request = atc_pb2.StopSandboxRequest(
            sandbox_id=self._sandbox_id,
            graceful_shutdown_seconds=int(graceful_shutdown_seconds),
            snapshot_on_stop=snapshot_on_stop,
            max_timeout_seconds=max_timeout,
        )

        try:
            try:
                response = await self._stub.Stop(
                    request,
                    timeout=max_timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
                    metadata=self._auth_metadata,
                )
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND and missing_ok:
                    logger.debug(
                        "Sandbox %s not found during stop (missing_ok=True)",
                        self._sandbox_id,
                    )
                    return
                raise _translate_rpc_error(
                    e, sandbox_id=self._sandbox_id, operation="Stop sandbox"
                ) from e

            if response.success:
                logger.info("Sandbox %s stopped successfully", self._sandbox_id)
            else:
                error_msg = response.error_message or "unknown error"
                raise SandboxError(f"Failed to stop sandbox: {error_msg}")
        finally:
            # Deregister from session regardless of outcome - the sandbox
            # is no longer usable either way
            if self._session is not None:
                self._session._deregister_sandbox(self)
            # Close channels to release resources
            if self._streaming_channel is not None:
                await self._streaming_channel.close(grace=None)
                self._streaming_channel = None
            if self._channel is not None:
                await self._channel.close(grace=None)
                self._channel = None
                self._stub = None

    def stop(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
        missing_ok: bool = False,
    ) -> OperationRef[None]:
        """Stop sandbox, return OperationRef immediately.

        The sandbox is deregistered from its session regardless of whether
        the stop was successful, since the sandbox is no longer usable.

        Args:
            snapshot_on_stop: If True, capture sandbox state before shutdown.
            graceful_shutdown_seconds: Time to wait for graceful shutdown.
            missing_ok: If True, suppress SandboxNotFoundError when sandbox
                doesn't exist.

        Returns:
            OperationRef[None]: Use .result() to block until complete.
            Raises SandboxError on failure, SandboxNotFoundError if not found
            (unless missing_ok=True).

        Example:
            ```python
            sb.stop().result()  # Block until stopped

            # Ignore if already deleted
            sb.stop(missing_ok=True).result()
            ```
        """
        future = self._loop_manager.run_async(
            self._stop_async(
                snapshot_on_stop=snapshot_on_stop,
                graceful_shutdown_seconds=graceful_shutdown_seconds,
                missing_ok=missing_ok,
            )
        )
        return OperationRef(future)

    async def _exec_streaming_async(
        self,
        command: Sequence[str],
        stdout_queue: asyncio.Queue[str | None],
        stderr_queue: asyncio.Queue[str | None],
        *,
        cwd: str | None = None,
        check: bool = False,
        timeout_seconds: float | None = None,
        stdin_queue: asyncio.Queue[bytes | None] | None = None,
        stdin_writer: StreamWriter | None = None,
    ) -> ProcessResult:
        """Internal async: Execute command using StreamExec RPC, push output to queues.

        Uses gRPC bidirectional streaming to receive stdout/stderr as they arrive.
        Buffers output while also pushing to queues for real-time streaming.
        Signals end-of-stream with None sentinel when command completes.

        When stdin_queue is provided, data from it is sent to the process's stdin.
        None in stdin_queue signals EOF. Uses done_writing() for proper half-close.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds

        if not command:
            raise ValueError("Command cannot be empty")

        await self._ensure_started_async()
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        # Wait for sandbox to be RUNNING before sending exec request
        await self._wait_until_running_async()

        await self._ensure_client()
        channel = await self._get_or_create_streaming_channel()
        stub = streaming_pb2_grpc.ATCStreamingServiceStub(channel)  # type: ignore[no-untyped-call]

        auth_metadata = self._auth_metadata

        # Wrap command with cwd if provided
        rpc_command = _wrap_command_with_cwd(command, cwd) if cwd else list(command)

        logger.debug(
            "Executing command (streaming) in sandbox %s: %s",
            self._sandbox_id,
            shlex.join(command),
        )

        stdout_buffer: list[bytes] = []
        stderr_buffer: list[bytes] = []
        exit_code: int | None = None

        # Shutdown event signals request generator to stop when process exits/times out
        shutdown_event = asyncio.Event()
        # Ready event signals that server is ready to receive stdin data
        ready_event = asyncio.Event()
        # Capture exceptions from request_generator (gRPC swallows them otherwise)
        request_error: Exception | None = None
        # Track exec start time for ready latency measurement (only used when stdin enabled)
        exec_start_time = time.monotonic()

        async def request_generator() -> AsyncIterator[streaming_pb2.ExecStreamRequest]:
            """Generate request messages for the bidirectional stream.

            Yields init message, then stdin data if enabled, then returns.
            The generator naturally completes when all messages are sent,
            which signals gRPC to half-close the send direction.
            """
            # Yield init message first
            yield streaming_pb2.ExecStreamRequest(
                init=streaming_pb2.ExecStreamInit(
                    sandbox_id=self._sandbox_id,
                    command=rpc_command,
                )
            )

            # If stdin is enabled, wait for ready signal before sending data
            if stdin_queue is not None:
                nonlocal request_error
                # Wait for ready signal with timeout
                ready_timeout = min(5.0, timeout) if timeout is not None else 5.0
                try:
                    await asyncio.wait_for(ready_event.wait(), timeout=ready_timeout)
                except TimeoutError:
                    # Capture error for propagation (gRPC swallows generator exceptions)
                    request_error = SandboxTimeoutError(
                        "stdin ready signal not received within timeout"
                    )
                    shutdown_event.set()
                    raise request_error from None

                # Now safe to send stdin data
                while not shutdown_event.is_set():
                    # Wait for either queue data or shutdown signal
                    get_task = asyncio.create_task(stdin_queue.get())
                    shutdown_task = asyncio.create_task(shutdown_event.wait())
                    try:
                        done, pending = await asyncio.wait(
                            [get_task, shutdown_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task

                        # Check if shutdown was triggered
                        if shutdown_task in done:
                            return

                        # Process queue data
                        data = get_task.result()
                        if data is None:  # EOF sentinel - close stdin
                            yield streaming_pb2.ExecStreamRequest(
                                close=streaming_pb2.ExecStreamClose()
                            )
                            return

                        # Chunk large data into 64KB pieces
                        for i in range(0, len(data), STDIN_CHUNK_SIZE):
                            chunk = data[i : i + STDIN_CHUNK_SIZE]
                            yield streaming_pb2.ExecStreamRequest(
                                stdin=streaming_pb2.ExecStreamData(data=chunk)
                            )
                    except asyncio.CancelledError:
                        return

        # Create the bidirectional streaming call with request iterator
        call_timeout = (
            timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS if timeout is not None else None
        )
        call: grpc.aio.StreamStreamCall[
            streaming_pb2.ExecStreamRequest, streaming_pb2.ExecStreamResponse
        ] = stub.StreamExec(
            request_iterator=request_generator(),
            timeout=call_timeout,
            metadata=auth_metadata,
        )

        # Queue decouples stream iteration from our processing.
        # Without this, processing suspends the stream and can cause issues.
        response_queue: asyncio.Queue[streaming_pb2.ExecStreamResponse | Exception | None] = (
            asyncio.Queue()
        )

        async def collect_responses() -> None:
            """Collect responses from gRPC streaming call into queue."""
            try:
                async for response in call:
                    await response_queue.put(response)
                    if response.HasField("exit") or response.HasField("error"):
                        return
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    await response_queue.put(
                        SandboxTimeoutError(
                            f"Command {shlex.join(command)} timed out after {timeout}s"
                        )
                    )
                else:
                    await response_queue.put(e)
            except Exception as e:
                await response_queue.put(e)
            finally:
                await response_queue.put(None)  # Sentinel

        # Start collector task (sender is handled by gRPC via the request_iterator)
        collect_task = asyncio.create_task(collect_responses())

        try:
            while True:
                item = await response_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                response = item
                if response.HasField("ready"):
                    # Server ready to receive stdin data
                    # Log latency only when stdin is enabled (no overhead when stdin=False)
                    if stdin_queue is not None:
                        ready_latency = time.monotonic() - exec_start_time
                        logger.debug(
                            "stdin ready signal received",
                            extra={
                                "sandbox_id": self._sandbox_id,
                                "ready_latency_ms": ready_latency * 1000,
                                "ready_at": response.ready.ready_at.ToDatetime().isoformat(),
                            },
                        )
                    ready_event.set()

                elif response.HasField("output"):
                    data = response.output.data
                    # Decode as UTF-8 for queue (replacing invalid chars)
                    text = data.decode("utf-8", errors="replace")
                    stream_type = response.output.stream_type

                    if stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT:
                        stdout_buffer.append(data)
                        await stdout_queue.put(text)
                    elif stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR:
                        stderr_buffer.append(data)
                        await stderr_queue.put(text)
                    else:
                        logger.warning(
                            "Received output with unexpected stream_type %s, treating as stdout",
                            stream_type,
                        )
                        stdout_buffer.append(data)
                        await stdout_queue.put(text)

                elif response.HasField("exit"):
                    ready_event.set()  # Unblock stdin sender on terminal message
                    exit_code = response.exit.exit_code
                    break

                elif response.HasField("error"):
                    ready_event.set()  # Unblock stdin sender on terminal message
                    raise SandboxExecutionError(
                        f"Exec stream error: {response.error.message}",
                    )
        finally:
            # Unblock stdin sender if still waiting for ready signal
            ready_event.set()
            # Signal request generator to stop consuming stdin queue
            shutdown_event.set()
            # Cancel collector task
            collect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collect_task
            # Signal stdin writer that process has exited (prevents writes to exited process)
            if stdin_writer is not None:
                stdin_writer.set_exception(SandboxExecutionError("Process has exited"))
            # Signal end-of-stream
            await stdout_queue.put(None)
            await stderr_queue.put(None)

        # Propagate any error from request_generator (gRPC swallows generator exceptions)
        if request_error is not None:
            raise request_error

        # Combine buffers into final output
        stdout_bytes = b"".join(stdout_buffer)
        stderr_bytes = b"".join(stderr_buffer)
        final_exit_code = exit_code if exit_code is not None else 0

        logger.debug("Command completed with exit code %d", final_exit_code)

        result = ProcessResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=final_exit_code,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            command=list(command),
        )

        if check and result.returncode != 0:
            raise SandboxExecutionError(
                f"Command {shlex.join(command)} failed with exit code {result.returncode}",
                exec_result=result,
            )

        return result

    def exec(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        check: bool = False,
        timeout_seconds: float | None = None,
        stdin: bool = False,
    ) -> Process:
        """Execute command, return Process immediately.

        Note: If sandbox is not yet RUNNING, this method waits for it first.
        The timeout_seconds parameter only applies to command execution, not to
        the initial wait for RUNNING status.

        Args:
            command: Command and arguments to execute
            cwd: Working directory for command execution. Must be an absolute path.
                When specified, the command is wrapped with a shell cd.
            check: If True, raise SandboxExecutionError on non-zero returncode
            timeout_seconds: Timeout for command execution (after sandbox is RUNNING).
                Does not include time waiting for sandbox to reach RUNNING status.
            stdin: If True, enable stdin streaming. Process.stdin will be a
                StreamWriter that can send input to the command. If False (default),
                stdin is closed immediately and Process.stdin is None.

        Returns:
            Process handle with streaming stdout/stderr. Call .result() to block
            for the final ProcessResult, or iterate over .stdout/.stderr for
            real-time output. When stdin=True, Process.stdin is a StreamWriter.

        Raises:
            ValueError: If command is empty or cwd is invalid (empty or relative path)

        Example:
            ```python
            # Get result directly
            process = sb.exec(["echo", "hello"])
            result = process.result()
            print(result.stdout)

            # With working directory
            result = sb.exec(["ls", "-la"], cwd="/app").result()

            # Stream output in real-time
            process = sb.exec(["python", "script.py"])
            for line in process.stdout:
                print(line)
            result = process.result()

            # With stdin streaming
            process = sb.exec(["cat"], stdin=True)
            process.stdin.write(b"hello world").result()
            process.stdin.close().result()
            result = process.result()

            # Async usage
            result = await sb.exec(["echo", "hello"])
            ```
        """
        if not command:
            raise ValueError("Command cannot be empty")
        _validate_cwd(cwd)

        # Track exec count for metrics
        self._exec_count += 1

        # Unbounded queues prevent data loss when producer fills queue before consumer iterates.
        # Bounded queues caused race conditions with HTTP/2 stream buffering.
        stdout_queue: asyncio.Queue[str | None] = asyncio.Queue()
        stderr_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Stdin queue is bounded to provide backpressure
        stdin_queue: asyncio.Queue[bytes | None] | None = None
        stdin_writer: StreamWriter | None = None
        if stdin:
            stdin_queue = asyncio.Queue(maxsize=StreamWriter.QUEUE_SIZE)
            stdin_writer = StreamWriter(stdin_queue, self._loop_manager)

        process_future = self._loop_manager.run_async(
            self._exec_streaming_async(
                command,
                stdout_queue,
                stderr_queue,
                cwd=cwd,
                check=check,
                timeout_seconds=timeout_seconds,
                stdin_queue=stdin_queue,
                stdin_writer=stdin_writer,
            )
        )

        return Process(
            future=process_future,
            command=list(command),
            stdout=StreamReader(stdout_queue, self._loop_manager),
            stderr=StreamReader(stderr_queue, self._loop_manager),
            stdin=stdin_writer,
            stats_callback=self._on_exec_complete,
        )

    async def _read_file_async(
        self,
        filepath: str,
        timeout: float,
    ) -> bytes:
        """Internal async: Read a file from the sandbox filesystem."""
        await self._ensure_started_async()
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        # Wait for sandbox to be running before file operations
        await self._wait_until_running_async()

        await self._ensure_client()
        assert self._stub is not None

        logger.debug("Reading file from sandbox %s: %s", self._sandbox_id, filepath)

        request = atc_pb2.RetrieveFileSandboxRequest(
            sandbox_id=self._sandbox_id,
            filepath=filepath,
            max_timeout_seconds=int(timeout),
        )

        try:
            response = await self._stub.RetrieveFile(
                request, timeout=timeout, metadata=self._auth_metadata
            )
        except grpc.RpcError as e:
            raise _translate_rpc_error(e, sandbox_id=self._sandbox_id, operation="Read file") from e

        if not response.success:
            logger.warning("Failed to read file %s from sandbox %s", filepath, self._sandbox_id)
            raise SandboxFileError(
                f"Failed to read file '{filepath}': {response.error_message}",
                filepath=filepath,
            )

        return bytes(response.file_contents)

    def read_file(
        self,
        filepath: str,
        *,
        timeout_seconds: float | None = None,
    ) -> OperationRef[bytes]:
        """Read file from sandbox, return OperationRef immediately.

        Args:
            filepath: Path to file in sandbox
            timeout_seconds: Timeout for the operation

        Returns:
            OperationRef[bytes]: Use .result() to block and retrieve contents.

        Example:
            ```python
            data = sb.read_file("/output/result.txt").result()
            ```
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        future = self._loop_manager.run_async(self._read_file_async(filepath, timeout))
        return OperationRef(future)

    async def _write_file_async(
        self,
        filepath: str,
        contents: bytes,
        timeout: float,
    ) -> None:
        """Internal async: Write a file to the sandbox filesystem."""
        await self._ensure_started_async()
        if self._stopped:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        # Wait for sandbox to be running before file operations
        await self._wait_until_running_async()

        await self._ensure_client()
        assert self._stub is not None

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

        try:
            response = await self._stub.AddFile(
                request, timeout=timeout, metadata=self._auth_metadata
            )
        except grpc.RpcError as e:
            raise _translate_rpc_error(
                e, sandbox_id=self._sandbox_id, operation="Write file"
            ) from e

        if not response.success:
            logger.warning("Failed to write file %s to sandbox %s", filepath, self._sandbox_id)
            raise SandboxFileError(
                f"Failed to write file '{filepath}'",
                filepath=filepath,
            )

    def write_file(
        self,
        filepath: str,
        contents: bytes,
        *,
        timeout_seconds: float | None = None,
    ) -> OperationRef[None]:
        """Write file to sandbox, return OperationRef immediately.

        Args:
            filepath: Path to file in sandbox
            contents: File contents as bytes
            timeout_seconds: Timeout for the operation

        Returns:
            OperationRef[None]: Use .result() to block until complete.

        Example:
            ```python
            sb.write_file("/input/data.txt", b"content").result()
            ```
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        future = self._loop_manager.run_async(self._write_file_async(filepath, contents, timeout))
        return OperationRef(future)
