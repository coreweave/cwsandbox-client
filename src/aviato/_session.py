from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aviato._defaults import SandboxDefaults
from aviato._function import create_function_wrapper
from aviato._types import Serialization
from aviato.exceptions import SandboxError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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
        defaults = SandboxDefaults(container_image="python:3.11")

        async with Session(defaults) as session:
            # Create sandboxes with session defaults
            sb1 = session.create(command="sleep", args=["infinity"])
            sb2 = session.create(command="sleep", args=["infinity"])

            async with sb1, sb2:
                await sb1.exec(["echo", "hello"])

            # Execute functions in sandboxes
            @session.function()
            def compute(x: int, y: int) -> int:
                return x + y

            result = await compute(2, 3)
            print(result)  # 5

        # Session automatically cleans up all sandboxes on exit
    """

    def __init__(self, defaults: SandboxDefaults | None = None) -> None:
        self._defaults = defaults or SandboxDefaults()
        self._sandboxes: dict[int, Sandbox] = {}
        self._closed = False

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"<Session sandboxes={len(self._sandboxes)} status={status}>"

    @property
    def sandbox_count(self) -> int:
        """Number of sandboxes currently tracked by this session."""
        return len(self._sandboxes)

    async def __aenter__(self) -> Session:
        """Initialize session resources."""
        # TODO: Implement backend pre-warming optimizations
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop any orphaned sandboxes and close shared resources."""
        await self.close()

    async def close(self) -> None:
        """Stop all managed sandboxes concurrently.

        Raises:
            SandboxError: If one or more running sandboxes failed to stop.
        """
        if self._closed:
            return

        self._closed = True

        if not self._sandboxes:
            return

        sandboxes = list(self._sandboxes.values())
        self._sandboxes.clear()

        results = await asyncio.gather(
            *[sandbox.stop() for sandbox in sandboxes],
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

    def create(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        container_image: str | None = None,
        tags: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
    ) -> Sandbox:
        """Create a sandbox with session defaults applied.

        Raises:
            SandboxError: If the session has been closed.
        """
        if self._closed:
            raise SandboxError(
                "Cannot create sandbox: session is closed. "
                "Create a new session or call create() before close()."
            )

        from aviato._sandbox import Sandbox

        sandbox = Sandbox(
            command=command,
            args=args,
            container_image=container_image,
            tags=tags,
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
        return sandbox

    async def list(
        self,
        *,
        tags: list[str] | None = None,
        status: str | None = None,  # Accept str for now to avoid circular import
        runway_ids: list[str] | None = None,
        tower_ids: list[str] | None = None,
        adopt: bool = False,
    ) -> list[Sandbox]:
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
            List of Sandbox instances

        Example:
            # Session defaults include a tag for this application/run
            defaults = SandboxDefaults(tags=("my-app", "run-abc123"))

            async with Session(defaults) as session:
                # Automatically filters by ["my-app", "run-abc123"]
                # No need to pass tags explicitly!
                orphans = await session.list(adopt=True)

                # Can add additional filters
                running = await session.list(status="running")
        """
        from aviato._sandbox import Sandbox

        merged_tags = self._defaults.merge_tags(tags)

        # Use session's default runway/tower IDs if not overridden
        effective_runway_ids = runway_ids or (
            list(self._defaults.runway_ids) if self._defaults.runway_ids else None
        )
        effective_tower_ids = tower_ids or (
            list(self._defaults.tower_ids) if self._defaults.tower_ids else None
        )

        sandboxes = await Sandbox.list(
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

    async def from_id(
        self,
        sandbox_id: str,
        *,
        adopt: bool = True,
    ) -> Sandbox:
        """Attach to an existing sandbox, optionally adopting it into this session.

        Args:
            sandbox_id: The ID of the existing sandbox
            adopt: If True (default), register the sandbox with this session

        Returns:
            A Sandbox instance attached to the existing sandbox

        Example:
            async with Session(defaults) as session:
                # Reconnect to a sandbox and have it cleaned up with the session
                sb = await session.from_id("sandbox-abc123")
                await sb.exec(["echo", "hello"])
            # sb is stopped when session exits
        """
        from aviato._sandbox import Sandbox

        sandbox = await Sandbox.from_id(
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
            async with Session(defaults) as session:
                # Get sandboxes via class method
                sandboxes = await Sandbox.list(tags=["my-job"])

                # Adopt them into the session
                for sb in sandboxes:
                    session.adopt(sb)

                # Now they'll be stopped when session closes
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
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, Awaitable[R]]]:
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
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            service: Service configuration for network access
            max_timeout_seconds: Maximum timeout for sandbox operations

        Returns:
            A decorator that wraps a function for async execution in a sandbox

        Example:
            async with Session(defaults) as session:
                @session.function()
                def compute(x: int, y: int) -> int:
                    return x + y

                @session.function(serialization=Serialization.PICKLE)
                def process_complex(data: MyClass) -> MyClass:
                    return data.transform()

                result = await compute(2, 3)
                print(result)  # 5
        """

        def decorator(f: Callable[P, R]) -> Callable[P, Awaitable[R]]:
            return create_function_wrapper(
                f,
                session=self,
                container_image=container_image,
                serialization=serialization,
                temp_dir=temp_dir or self._defaults.temp_dir,
                resources=resources,
                mounted_files=mounted_files,
                s3_mount=s3_mount,
                ports=ports,
                service=service,
                max_timeout_seconds=max_timeout_seconds,
            )

        return decorator
