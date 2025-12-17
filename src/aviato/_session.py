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
                f"Failed to stop {len(errors)} sandbox(es). " "Some sandboxes may still be running."
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
        command: str,
        args: list[str] | None = None,
        container_image: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
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
            defaults=self._defaults,
            _session=self,
            **kwargs,
        )
        self._register_sandbox(sandbox)
        return sandbox

    def function(
        self,
        *,
        container_image: str | None = None,
        serialization: Serialization = Serialization.JSON,
        temp_dir: str | None = None,
        **sandbox_kwargs: Any,
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
            **sandbox_kwargs: Additional kwargs passed to sandbox creation

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
                **sandbox_kwargs,
            )

        return decorator
