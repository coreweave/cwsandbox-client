from __future__ import annotations

import asyncio
import builtins
import logging
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aviato._defaults import SandboxDefaults
from aviato._function import RemoteFunction
from aviato._loop_manager import _LoopManager
from aviato._types import OperationRef, Serialization
from aviato.exceptions import SandboxError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aviato._sandbox import Sandbox

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class Session:
    """Manages sandbox lifecycle and provides function execution.

    Use a session when:
    - Creating multiple sandboxes with shared configuration
    - Executing Python functions in sandboxes
    - You want automatic cleanup of orphaned sandboxes

    Example:
        ```python
        defaults = SandboxDefaults(container_image="python:3.11")

        # Sync context manager
        with Session(defaults) as session:
            # Create and start sandboxes with session defaults
            sb1 = session.sandbox(command="sleep", args=["infinity"])
            sb2 = session.sandbox(command="sleep", args=["infinity"])

            # Execute commands
            result = sb1.exec(["echo", "hello"]).result()

            # Execute functions in sandboxes
            @session.function()
            def compute(x: int, y: int) -> int:
                return x + y

            result = compute.remote(2, 3).result()  # Returns OperationRef
            print(result)  # 5

        # Session automatically cleans up all sandboxes on exit

        # Async context manager also supported
        async with Session(defaults) as session:
            sb = session.sandbox(command="sleep", args=["infinity"])
            result = await sb.exec(["echo", "hello"])
        ```
    """

    def __init__(self, defaults: SandboxDefaults | None = None) -> None:
        self._defaults = defaults or SandboxDefaults()
        self._sandboxes: dict[int, Sandbox] = {}
        self._closed = False
        self._loop_manager = _LoopManager.get()
        self._loop_manager.register_session(self)

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"<Session sandboxes={len(self._sandboxes)} status={status}>"

    @property
    def sandbox_count(self) -> int:
        """Number of sandboxes currently tracked by this session."""
        return len(self._sandboxes)

    def __enter__(self) -> Session:
        """Enter sync context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit sync context manager, stop all sandboxes."""
        self.close().result()

    async def __aenter__(self) -> Session:
        """Enter async context manager."""
        # TODO: Implement backend pre-warming optimizations
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager, stop all sandboxes."""
        # Route through close() which uses _LoopManager to ensure cleanup
        # runs in the same event loop where the httpx client was created
        await self.close()

    def close(self) -> OperationRef[None]:
        """Stop all managed sandboxes, return OperationRef immediately.

        Returns:
            OperationRef[None]: Use .result() to block until all sandboxes stopped.

        Raises:
            SandboxError: If one or more running sandboxes failed to stop.

        Example:
            ```python
            session.close().result()  # Block until all sandboxes stopped
            ```
        """
        future = self._loop_manager.run_async(self._close_async())
        return OperationRef(future)

    async def _close_async(self) -> None:
        """Internal async: Stop all managed sandboxes concurrently."""
        if self._closed:
            return

        self._closed = True

        if not self._sandboxes:
            return

        sandboxes = list(self._sandboxes.values())
        self._sandboxes.clear()

        results = await asyncio.gather(
            *[sandbox._stop_async() for sandbox in sandboxes],
            return_exceptions=True,
        )

        errors: list[Exception] = []
        for sandbox, result in zip(sandboxes, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to stop sandbox %s: %s",
                    id(sandbox),
                    result,
                    exc_info=result,
                )
                errors.append(result)

        if errors:
            raise SandboxError(
                f"Failed to stop {len(errors)} sandbox(es). Some sandboxes may still be running."
            ) from ExceptionGroup("Sandbox stop failures", errors)

    def _register_sandbox(self, sandbox: Sandbox) -> None:
        """Register a sandbox for tracking."""
        self._sandboxes[id(sandbox)] = sandbox

    def _deregister_sandbox(self, sandbox: Sandbox) -> None:
        """Deregister a sandbox from tracking."""
        self._sandboxes.pop(id(sandbox), None)

    def sandbox(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        container_image: str | None = None,
        tags: list[str] | None = None,
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
    ) -> Sandbox:
        """Create and start a sandbox with session defaults, return immediately.

        This is the recommended way to create sandboxes in the sync API.
        The sandbox is created and started, returning immediately once the
        backend accepts the start request (does NOT wait for RUNNING status).

        Args:
            command: Command to run in sandbox
            args: Arguments for the command
            container_image: Container image to use
            tags: Tags for the sandbox (merged with session defaults)
            runway_ids: Optional list of runway IDs
            tower_ids: Optional list of tower IDs
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            service: Service configuration for network access
            max_timeout_seconds: Maximum timeout for sandbox operations

        Returns:
            A started Sandbox instance. Use .wait() to block until RUNNING.

        Raises:
            SandboxError: If the session has been closed.

        Example:
            ```python
            with Session(defaults) as session:
                sb = session.sandbox(command="sleep", args=["infinity"])
                sb.wait()  # Optional: block until RUNNING
                result = sb.exec(["echo", "hello"]).result()
            ```
        """
        if self._closed:
            raise SandboxError(
                "Cannot create sandbox: session is closed. "
                "Create a new session or call sandbox() before close()."
            )

        from aviato._sandbox import Sandbox

        sandbox = Sandbox(
            command=command,
            args=args,
            container_image=container_image,
            tags=tags,
            runway_ids=runway_ids,
            tower_ids=tower_ids,
            resources=resources,
            mounted_files=mounted_files,
            s3_mount=s3_mount,
            ports=ports,
            service=service,
            max_timeout_seconds=max_timeout_seconds,
            defaults=self._defaults,
            _session=self,
        )
        self._register_sandbox(sandbox)
        sandbox.start()
        return sandbox

    def list(
        self,
        *,
        tags: builtins.list[str] | None = None,
        status: str | None = None,
        runway_ids: builtins.list[str] | None = None,
        tower_ids: builtins.list[str] | None = None,
        adopt: bool = False,
    ) -> OperationRef[builtins.list[Sandbox]]:
        """List sandboxes, optionally adopting them into this session.

        Automatically includes the session's default tags in the filter.
        This makes it easy to find sandboxes created by this session or
        a previous run with the same defaults.

        Args:
            tags: Additional tags to filter by (merged with session's default tags)
            status: Filter by status
            runway_ids: Filter by runway IDs (defaults to session's runway_ids if set)
            tower_ids: Filter by tower IDs (defaults to session's tower_ids if set)
            adopt: If True, register discovered sandboxes with this session
                   so they are stopped when the session closes

        Returns:
            OperationRef[list[Sandbox]]: Use .result() to block for results,
            or await directly in async contexts.

        Example:
            ```python
            # Session defaults include a tag for this application/run
            defaults = SandboxDefaults(tags=("my-app", "run-abc123"))

            with Session(defaults) as session:
                # Sync usage - automatically filters by ["my-app", "run-abc123"]
                orphans = session.list(adopt=True).result()

                # Can add additional filters
                running = session.list(status="running").result()

            # Async usage
            async with Session(defaults) as session:
                orphans = await session.list(adopt=True)
            ```
        """
        future = self._loop_manager.run_async(
            self._list_async(
                tags=tags,
                status=status,
                runway_ids=runway_ids,
                tower_ids=tower_ids,
                adopt=adopt,
            )
        )
        return OperationRef(future)

    async def _list_async(
        self,
        *,
        tags: builtins.list[str] | None = None,
        status: str | None = None,
        runway_ids: builtins.list[str] | None = None,
        tower_ids: builtins.list[str] | None = None,
        adopt: bool = False,
    ) -> builtins.list[Sandbox]:
        """Internal async: List sandboxes, optionally adopting them into this session."""
        from aviato._sandbox import Sandbox

        merged_tags = self._defaults.merge_tags(tags)

        # Use session's default runway/tower IDs if not overridden
        if runway_ids is not None:
            effective_runway_ids = list(runway_ids)
        elif self._defaults.runway_ids:
            effective_runway_ids = list(self._defaults.runway_ids)
        else:
            effective_runway_ids = None

        if tower_ids is not None:
            effective_tower_ids = list(tower_ids)
        elif self._defaults.tower_ids:
            effective_tower_ids = list(self._defaults.tower_ids)
        else:
            effective_tower_ids = None

        sandboxes = await Sandbox._list_async(
            tags=merged_tags if merged_tags else None,
            status=status,
            runway_ids=effective_runway_ids,
            tower_ids=effective_tower_ids,
            base_url=self._defaults.base_url,
            timeout_seconds=self._defaults.request_timeout_seconds,
        )

        if adopt:
            for sb in sandboxes:
                self._register_sandbox(sb)
                sb._session = self

        return sandboxes

    def from_id(
        self,
        sandbox_id: str,
        *,
        adopt: bool = True,
    ) -> OperationRef[Sandbox]:
        """Attach to an existing sandbox, optionally adopting it into this session.

        Args:
            sandbox_id: The ID of the existing sandbox
            adopt: If True (default), register the sandbox with this session

        Returns:
            OperationRef[Sandbox]: Use .result() to block for the Sandbox instance,
            or await directly in async contexts.

        Example:
            ```python
            with Session(defaults) as session:
                # Sync usage - reconnect to a sandbox
                sb = session.from_id("sandbox-abc123").result()
                result = sb.exec(["echo", "hello"]).result()
            # sb is stopped when session exits

            # Async usage
            async with Session(defaults) as session:
                sb = await session.from_id("sandbox-abc123")
                result = await sb.exec(["echo", "hello"])
            ```
        """
        future = self._loop_manager.run_async(self._from_id_async(sandbox_id, adopt=adopt))
        return OperationRef(future)

    async def _from_id_async(
        self,
        sandbox_id: str,
        *,
        adopt: bool = True,
    ) -> Sandbox:
        """Internal async: Attach to an existing sandbox, optionally adopting it."""
        from aviato._sandbox import Sandbox

        sandbox = await Sandbox._from_id_async(
            sandbox_id,
            base_url=self._defaults.base_url,
            timeout_seconds=self._defaults.request_timeout_seconds,
        )

        if adopt:
            self._register_sandbox(sandbox)
            sandbox._session = self

        return sandbox

    def adopt(self, sandbox: Sandbox) -> None:
        """Adopt an existing Sandbox instance into this session for cleanup tracking.

        Use this when you have a Sandbox from Sandbox.list() or Sandbox.from_id()
        that you want to be automatically stopped when the session closes.

        Args:
            sandbox: A Sandbox instance to track

        Raises:
            SandboxError: If the session is closed
            ValueError: If the sandbox has no sandbox_id

        Example:
            ```python
            with Session(defaults) as session:
                # Get sandboxes via class method
                sandboxes = Sandbox.list(tags=["my-job"]).result()

                # Adopt them into the session
                for sb in sandboxes:
                    session.adopt(sb)

                # Now they'll be stopped when session closes
            ```
        """
        if self._closed:
            raise SandboxError("Cannot adopt sandbox: session is closed")
        if sandbox.sandbox_id is None:
            raise ValueError("Cannot adopt sandbox without sandbox_id")

        self._register_sandbox(sandbox)
        sandbox._session = self

    def function(
        self,
        *,
        container_image: str | None = None,
        serialization: Serialization = Serialization.JSON,
        temp_dir: str | None = None,
        runway_ids: builtins.list[str] | None = None,
        tower_ids: builtins.list[str] | None = None,
        resources: dict[str, Any] | None = None,
        mounted_files: Sequence[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: Sequence[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
    ) -> Callable[[Callable[P, R]], RemoteFunction[P, R]]:
        """Decorator to execute a Python function in a sandbox.

        Each function call creates an ephemeral sandbox, executes the function,
        and returns the result. The sandbox is automatically cleaned up.

        The decorated function must be synchronous. Async functions are not supported.

        Args:
            container_image: Override session's default image for this function
            serialization: How to serialize arguments and return values.
                Defaults to JSON for safety. Use PICKLE for complex types,
                but only in trusted environments.
            temp_dir: Override temp directory for payload/result files in sandbox.
                Defaults to session default. Created if missing.
            runway_ids: Optional list of runway IDs
            tower_ids: Optional list of tower IDs
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            service: Service configuration for network access
            max_timeout_seconds: Maximum timeout for sandbox operations

        Returns:
            A decorator that wraps a function as a RemoteFunction

        Example:
            ```python
            with Session(defaults) as session:
                @session.function()
                def compute(x: int, y: int) -> int:
                    return x + y

                @session.function(serialization=Serialization.PICKLE)
                def process_complex(data: MyClass) -> MyClass:
                    return data.transform()

                # Call .remote() to execute in sandbox
                ref = compute.remote(2, 3)  # Returns OperationRef immediately
                result = ref.result()       # Block for result
                print(result)  # 5

                # Or use await in async context
                result = await compute.remote(2, 3)

                # Execute locally for testing
                result = compute.local(2, 3)

                # Map over multiple inputs in parallel
                refs = compute.map([(1, 2), (3, 4), (5, 6)])
                results = [ref.result() for ref in refs]
            ```
        """

        def decorator(f: Callable[P, R]) -> RemoteFunction[P, R]:
            return RemoteFunction(
                f,
                session=self,
                container_image=container_image,
                serialization=serialization,
                temp_dir=temp_dir or self._defaults.temp_dir,
                runway_ids=runway_ids,
                tower_ids=tower_ids,
                resources=resources,
                mounted_files=list(mounted_files) if mounted_files else None,
                s3_mount=s3_mount,
                ports=list(ports) if ports else None,
                service=service,
                max_timeout_seconds=max_timeout_seconds,
            )

        return decorator
