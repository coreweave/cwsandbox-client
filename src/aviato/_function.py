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
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pydantic import TypeAdapter

from aviato._defaults import DEFAULT_TEMP_DIR
from aviato._types import Serialization
from aviato.exceptions import (
    AsyncFunctionError,
    SandboxExecutionError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aviato._session import Session

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

# Bytecode opcodes that reference global variables used when serializing functions.
_GLOBAL_OPS = frozenset(
    (
        dis.opmap.get("LOAD_GLOBAL"),  # Reading: x = SOME_GLOBAL
        dis.opmap.get("STORE_GLOBAL"),  # Writing: global x; x = value
        dis.opmap.get("DELETE_GLOBAL"),  # Deleting: del SOME_GLOBAL
    )
)


def create_function_wrapper(
    func: Callable[P, R],
    *,
    session: Session,
    container_image: str | None = None,
    serialization: Serialization = Serialization.JSON,
    temp_dir: str = DEFAULT_TEMP_DIR,
    **sandbox_kwargs: Any,
) -> Callable[P, Awaitable[R]]:
    """Create an async wrapper that executes the function in a sandbox.

    Args:
        func: The function to wrap
        session: The sandbox session to use for execution
        container_image: Override container image for this function
        serialization: Serialization mode (JSON by default for safety)
        temp_dir: Directory for temporary payload/result files in sandbox
        **sandbox_kwargs: Additional kwargs passed to sandbox creation
    """
    unwrapped = func
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__

    if inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped):
        raise AsyncFunctionError(
            f"Function '{func.__name__}' is async, but @session.function() only supports "
            "synchronous functions. The sandbox executes Python synchronously. "
            "If you need async behavior, run your async code inside the sync function "
            "using asyncio.run()."
        )

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        logger.debug("Executing function %s in sandbox", func.__name__)

        source = _get_function_source_for_sandbox(func)
        closure_vars = _extract_closure_variables(func)
        global_vars = _extract_global_variables(func)

        merged_vars = {**global_vars, **closure_vars}

        file_id = uuid.uuid4()
        payload_file = f"{temp_dir}/sandbox_payload_{file_id.hex}.bin"
        result_file = f"{temp_dir}/sandbox_result_{file_id.hex}.bin"

        mode_config = _SERIALIZATION_MODES[serialization.value]

        payload_bytes = mode_config.create_payload(source, func.__name__, merged_vars, args, kwargs)

        execution_script = mode_config.template.format(
            payload_file=payload_file,
            result_file=result_file,
            temp_dir=temp_dir,
        )

        sandbox = session.create(
            command="tail",
            args=["-f", "/dev/null"],
            container_image=container_image,
            **sandbox_kwargs,
        )

        logger.debug("Starting sandbox for function %s", func.__name__)

        async with sandbox:
            await sandbox.write_file(payload_file, payload_bytes)

            logger.debug("Executing function %s in sandbox %s", func.__name__, sandbox.sandbox_id)
            exec_result = await sandbox.exec(["python", "-c", execution_script])

            if exec_result.returncode != 0:
                stderr = exec_result.stderr
                exception_type, exception_message = _parse_exception_from_stderr(stderr)

                logger.error(
                    "Function %s failed in sandbox: %s",
                    func.__name__,
                    exception_message or stderr[:200],
                )

                error_detail = exception_message or stderr
                raise SandboxExecutionError(
                    f"Function '{func.__name__}' execution failed in sandbox: {error_detail}",
                    exec_result=exec_result,
                    exception_type=exception_type,
                    exception_message=exception_message,
                )

            result_content = await sandbox.read_file(result_file)
            result_value = mode_config.parse_result(result_content)

            logger.debug("Function %s completed successfully", func.__name__)
            return result_value  # type: ignore[no-any-return]

    return wrapper


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
    """Check if an AST node represents @session.function() or similar."""
    if isinstance(node, ast.Call):
        return _is_session_function_decorator(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr == "function"
    if isinstance(node, ast.Name):
        return node.id == "function"
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
    return pickle.dumps(payload)


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
    return adapter.dump_json(payload)


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
