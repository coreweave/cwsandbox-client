from __future__ import annotations

import ast
import dis
import inspect
import json
import logging
import pickle
import textwrap
import types
import uuid
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError

from aviato._defaults import DEFAULT_TEMP_DIR
from aviato._types import OperationRef, Serialization
from aviato.exceptions import (
    AsyncFunctionError,
    FunctionSerializationError,
    SandboxExecutionError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from aviato._session import Session

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class RemoteFunction(Generic[P, R]):
    """Wrapper for remote function execution in sandboxes.

    RemoteFunction wraps a Python function for execution in a sandbox.
    It provides methods to execute remotely, locally (for testing), and
    map over multiple inputs in parallel.

    The wrapped function must be synchronous. Async functions are not supported.

    Type Parameters:
        P: The parameter spec of the wrapped function.
        R: The return type of the wrapped function.

    Examples:
        Basic usage with decorator:
            with Session(defaults) as session:
                @session.function()
                def compute(x: int, y: int) -> int:
                    return x + y

                # Call .remote() to execute in sandbox
                ref = compute.remote(2, 3)
                result = ref.get()  # Block for result
                print(result)  # 5

        Using .map() for parallel execution:
            refs = compute.map([(1, 2), (3, 4), (5, 6)])
            results = [ref.get() for ref in refs]

        Using .local() for testing:
            result = compute.local(2, 3)  # Runs locally, no sandbox
    """

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        session: Session,
        container_image: str | None = None,
        serialization: Serialization = Serialization.JSON,
        temp_dir: str = DEFAULT_TEMP_DIR,
        resources: dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        service: dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
    ) -> None:
        """Initialize RemoteFunction with function and execution configuration.

        Args:
            fn: The function to wrap for remote execution
            session: The sandbox session to use for execution
            container_image: Override container image for this function
            serialization: Serialization mode (JSON by default for safety)
            temp_dir: Directory for temporary payload/result files in sandbox
            resources: Resource requests (CPU, memory, GPU)
            mounted_files: Files to mount into the sandbox
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            service: Service configuration for network access
            max_timeout_seconds: Maximum timeout for sandbox operations
        """
        unwrapped = fn
        while hasattr(unwrapped, "__wrapped__"):
            unwrapped = unwrapped.__wrapped__

        if inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped):
            raise AsyncFunctionError(
                f"Function '{fn.__name__}' is async, but @session.function() only supports "
                "synchronous functions. The sandbox executes Python synchronously. "
                "If you need async behavior, run your async code inside the sync function "
                "using asyncio.run()."
            )

        self._fn = fn
        self._session = session
        self._container_image = container_image
        self._serialization = serialization
        self._temp_dir = temp_dir
        self._resources = resources
        self._mounted_files = mounted_files
        self._s3_mount = s3_mount
        self._ports = ports
        self._service = service
        self._max_timeout_seconds = max_timeout_seconds

        # Preserve function metadata
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__module__ = fn.__module__
        self.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
        self.__annotations__ = getattr(fn, "__annotations__", {})

    def remote(self, *args: P.args, **kwargs: P.kwargs) -> OperationRef[R]:
        """Execute the function in a sandbox, return OperationRef immediately.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            OperationRef[R]: Use .get() to block until result is ready.

        Example:
            ref = compute.remote(2, 3)
            result = ref.get()  # Block for result
            # Or in async context:
            result = await ref
        """
        future = self._session._loop_manager.run_async(self._execute_async(*args, **kwargs))
        return OperationRef(future)

    def map(self, items: Iterable[tuple[Any, ...]]) -> list[OperationRef[R]]:
        """Execute the function for each item, return OperationRefs immediately.

        Each item should be a tuple of positional arguments for the function.
        All executions are launched in parallel.

        Args:
            items: Iterable of argument tuples to pass to the function.

        Returns:
            List of OperationRef[R], one for each item.

        Example:
            # Execute add(1, 2), add(3, 4), add(5, 6) in parallel
            refs = add.map([(1, 2), (3, 4), (5, 6)])
            results = [ref.get() for ref in refs]  # [3, 7, 11]
        """
        # Type ignore: ParamSpec doesn't support tuple unpacking validation
        return [self.remote(*item) for item in items]  # type: ignore[call-arg]

    def local(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the function locally without a sandbox.

        Useful for testing and debugging. Runs the original function
        directly in the current Python process.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function execution.

        Example:
            # Test without sandbox overhead
            result = compute.local(2, 3)
            assert result == 5
        """
        return self._fn(*args, **kwargs)

    async def _execute_async(self, *args: Any, **kwargs: Any) -> R:
        """Internal async execution logic.

        Creates a sandbox, serializes the function and arguments,
        executes in the sandbox, and deserializes the result.
        """
        logger.debug("Executing function %s in sandbox", self._fn.__name__)

        source = _get_function_source_for_sandbox(self._fn)
        closure_vars = _extract_closure_variables(self._fn)
        global_vars = _extract_global_variables(self._fn)

        merged_vars = {**global_vars, **closure_vars}

        file_id = uuid.uuid4()
        payload_file = f"{self._temp_dir}/sandbox_payload_{file_id.hex}.bin"
        result_file = f"{self._temp_dir}/sandbox_result_{file_id.hex}.bin"

        mode_config = _SERIALIZATION_MODES[self._serialization.value]

        payload_bytes = mode_config.create_payload(
            source, self._fn.__name__, merged_vars, args, kwargs
        )

        execution_script = mode_config.template.format(
            payload_file=payload_file,
            result_file=result_file,
            temp_dir=self._temp_dir,
        )

        sandbox_kwargs: dict[str, Any] = {}
        if self._resources is not None:
            sandbox_kwargs["resources"] = self._resources
        if self._mounted_files is not None:
            sandbox_kwargs["mounted_files"] = self._mounted_files
        if self._s3_mount is not None:
            sandbox_kwargs["s3_mount"] = self._s3_mount
        if self._ports is not None:
            sandbox_kwargs["ports"] = self._ports
        if self._service is not None:
            sandbox_kwargs["service"] = self._service
        if self._max_timeout_seconds is not None:
            sandbox_kwargs["max_timeout_seconds"] = self._max_timeout_seconds

        # Import here to avoid circular import
        from aviato._sandbox import Sandbox

        # Create sandbox directly and use async start to avoid deadlock.
        # session.sandbox() uses sync APIs which would deadlock when called
        # from the daemon thread running this async method.
        sandbox = Sandbox(
            container_image=self._container_image,
            defaults=self._session._defaults,
            _session=self._session,
            **sandbox_kwargs,
        )
        self._session._register_sandbox(sandbox)
        await sandbox._start_async()

        logger.debug("Sandbox started for function %s", self._fn.__name__)

        async with sandbox:
            await sandbox.write_file(payload_file, payload_bytes)

            logger.debug(
                "Executing function %s in sandbox %s",
                self._fn.__name__,
                sandbox.sandbox_id,
            )
            exec_result = await sandbox.exec(["python", "-c", execution_script])

            if exec_result.returncode != 0:
                stderr = exec_result.stderr
                exception_type, exception_message = _parse_exception_from_stderr(stderr)

                logger.error(
                    "Function %s failed in sandbox: %s",
                    self._fn.__name__,
                    exception_message or stderr[:200],
                )

                error_detail = exception_message or stderr
                raise SandboxExecutionError(
                    f"Function '{self._fn.__name__}' execution failed in sandbox: {error_detail}",
                    exec_result=exec_result,
                    exception_type=exception_type,
                    exception_message=exception_message,
                )

            result_content = await sandbox.read_file(result_file)
            result_value = mode_config.parse_result(result_content)

            logger.debug("Function %s completed successfully", self._fn.__name__)
            return result_value  # type: ignore[no-any-return]


# Bytecode opcodes that reference global variables used when serializing functions.
_GLOBAL_OPS = frozenset(
    (
        dis.opmap.get("LOAD_GLOBAL"),  # Reading: x = SOME_GLOBAL
        dis.opmap.get("STORE_GLOBAL"),  # Writing: global x; x = value
        dis.opmap.get("DELETE_GLOBAL"),  # Deleting: del SOME_GLOBAL
    )
)


def _get_function_source_for_sandbox(func: Callable[..., Any]) -> str:
    """Extract function source code for remote execution in sandbox.

    Gets the source of the original function and removes the @session.function
    decorator. Other decorators are preserved.

    Args:
        func: The function to extract source from

    Returns:
        Function source code with @session.function decorator removed

    Raises:
        OSError: If the source file cannot be found (e.g., REPL/notebook)
        TypeError: If the function has no source (e.g., lambdas, built-ins)
    """
    unwrapped = func
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__

    source = inspect.getsource(unwrapped)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    func_def = tree.body[0]

    if not isinstance(func_def, ast.FunctionDef | ast.AsyncFunctionDef):
        raise ValueError(f"Expected function definition, got {type(func_def).__name__}")

    # Remove @session.function decorator
    if func_def.decorator_list:
        func_def.decorator_list = [
            d for d in func_def.decorator_list if not _is_session_function_decorator(d)
        ]

    return ast.unparse(func_def)


def _is_session_function_decorator(node: ast.expr) -> bool:
    """Check if an AST node represents @<obj>.function() pattern.

    Note: At AST level we can't verify the object is a Session instance,
    so this matches any @<identifier>.function() pattern.
    """
    if isinstance(node, ast.Call):
        return _is_session_function_decorator(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr == "function"
    return False


def _extract_closure_variables(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract closure variables from a function."""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    closure_vars: dict[str, Any] = {}
    if hasattr(func, "__closure__") and func.__closure__:
        closure_names = func.__code__.co_freevars
        for name, cell in zip(closure_names, func.__closure__, strict=True):
            closure_vars[name] = cell.cell_contents
    return closure_vars


def _extract_global_variables(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract global variables referenced by the function.

    A function's __globals__ contains everything in its module, but
    we only want to serialize the globals the function actually uses.
    This function inspects the function's bytecode to find which global names it
    references, then extracts only those values.

    Example:
        MODULE_CONFIG = {"key": "value"}  # Used by func
        UNUSED_GLOBAL = "not needed"       # Not used by func

        def func(x):
            return x + MODULE_CONFIG["key"]

        # Returns {"MODULE_CONFIG": {"key": "value"}}
        # Does NOT include UNUSED_GLOBAL
    """
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    referenced_names = _get_referenced_globals(func.__code__)

    global_vars: dict[str, Any] = {}
    func_globals = func.__globals__

    for name in referenced_names:
        if name in func_globals:
            value = func_globals[name]
            # Skip modules (like 'os') and builtins (like 'print') since
            # they're available in the sandbox without serialization
            is_builtin_or_module = isinstance(value, types.ModuleType | type) or (
                hasattr(value, "__module__") and value.__module__ == "builtins"
            )
            if not is_builtin_or_module:
                global_vars[name] = value

    return global_vars


def _get_referenced_globals(code: Any) -> set[str]:
    """Get all global variable names referenced by a code object.

    Walks the bytecode instructions looking for LOAD_GLOBAL, STORE_GLOBAL,
    and DELETE_GLOBAL operations. Also recursively checks nested functions.
    """
    names: set[str] = set()

    for instr in dis.get_instructions(code):
        if instr.opcode in _GLOBAL_OPS:
            names.add(instr.argval)  # argval is the variable name

    # Handle nested functions: their code objects are stored in co_consts.
    # Example: def outer(): def inner(): return SOME_GLOBAL
    # SOME_GLOBAL is referenced in inner's bytecode, not outer's.
    if code.co_consts:
        for const in code.co_consts:
            if hasattr(const, "co_code"):  # It's a code object (nested function)
                names.update(_get_referenced_globals(const))

    return names


def _parse_exception_from_stderr(stderr: str) -> tuple[str | None, str | None]:
    """Parse exception type and message from sandbox stderr.

    Returns:
        Tuple of (exception_type, exception_message), either may be None
    """
    exception_type = None
    exception_message = None

    for line in stderr.split("\n"):
        if line.startswith("EXCEPTION:"):
            exception_message = line[len("EXCEPTION:") :].strip()
        elif ": " in line and not line.startswith(" "):
            parts = line.split(": ", 1)
            if parts[0].endswith("Error") or parts[0].endswith("Exception"):
                exception_type = parts[0]
                if len(parts) > 1:
                    exception_message = parts[1]

    return exception_type, exception_message


def _create_function_payload(
    source: str,
    func_name: str,
    closure_vars: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bytes:
    """Create a pickled function execution payload."""
    payload = {
        "source": source,
        "name": func_name,
        "closure_vars": closure_vars,
        "args": args,
        "kwargs": kwargs,
    }
    try:
        return pickle.dumps(payload)
    except (pickle.PickleError, TypeError) as e:
        raise FunctionSerializationError(
            f"Cannot serialize function '{func_name}' using PICKLE mode: {e}\n\n"
            f"Avoid lambdas, thread locks, file handles, and other non-picklable objects "
            f"in arguments, referenced globals, or closures"
        ) from e


def _parse_sandbox_result(result_content: bytes) -> Any:
    """Parse function execution result from sandbox result file."""
    return pickle.loads(result_content)


def _create_json_payload(
    source: str,
    func_name: str,
    closure_vars: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bytes:
    """Create JSON-serializable payload using Pydantic."""
    payload = {
        "source": source,
        "name": func_name,
        "closure_vars": closure_vars,
        "args": list(args),
        "kwargs": kwargs,
    }
    adapter = TypeAdapter(dict[str, Any])
    try:
        return adapter.dump_json(payload)
    except PydanticSerializationError as e:
        raise FunctionSerializationError(
            f"Cannot serialize function '{func_name}' using JSON mode: {e}\n\n"
            f"Try serialization=Serialization.PICKLE or use only JSON-compatible types "
            f"(str, int, float, dict, list, etc.) in arguments, referenced globals, and closures"
        ) from e


def _parse_json_result(result_content: bytes) -> Any:
    """Parse JSON-serialized result."""
    return json.loads(result_content)


_FUNCTION_EXECUTION_TEMPLATE = """
import os
import pickle
import sys

temp_dir = "{temp_dir}"
payload_file = "{payload_file}"
result_file = "{result_file}"

os.makedirs(temp_dir, exist_ok=True)

with open(payload_file, "rb") as f:
    payload = pickle.load(f)

exec_globals = payload["closure_vars"].copy()
exec(payload["source"], exec_globals)
func = exec_globals[payload["name"]]

try:
    result = func(*payload["args"], **payload["kwargs"])
    with open(result_file, "wb") as f:
        pickle.dump(result, f)
except Exception as e:
    import traceback
    print("EXCEPTION:" + str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""

_JSON_EXECUTION_TEMPLATE = """
import json
import os
import sys

temp_dir = "{temp_dir}"
payload_file = "{payload_file}"
result_file = "{result_file}"

os.makedirs(temp_dir, exist_ok=True)

with open(payload_file, "rb") as f:
    payload = json.load(f)

closure_vars = payload["closure_vars"]
args = payload["args"]
kwargs = payload["kwargs"]

exec_globals = closure_vars.copy()
exec(payload["source"], exec_globals)
func = exec_globals[payload["name"]]

try:
    result = func(*args, **kwargs)
    with open(result_file, "w") as f:
        json.dump(result, f)
except Exception as e:
    import traceback
    print("EXCEPTION:" + str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""


class _SerializationMode:
    """Configuration for a serialization mode."""

    def __init__(
        self,
        create_payload: Callable[..., bytes],
        parse_result: Callable[[bytes], Any],
        template: str,
    ) -> None:
        self.create_payload = create_payload
        self.parse_result = parse_result
        self.template = template


# TODO: Investigate cloudpickle as an alternative to source extraction.
_SERIALIZATION_MODES: dict[str, _SerializationMode] = {
    Serialization.PICKLE.value: _SerializationMode(
        create_payload=_create_function_payload,
        parse_result=_parse_sandbox_result,
        template=_FUNCTION_EXECUTION_TEMPLATE,
    ),
    Serialization.JSON.value: _SerializationMode(
        create_payload=_create_json_payload,
        parse_result=_parse_json_result,
        template=_JSON_EXECUTION_TEMPLATE,
    ),
}
