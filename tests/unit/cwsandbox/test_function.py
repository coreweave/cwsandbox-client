# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._function module."""

import ast
import json
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cwsandbox._function import (
    RemoteFunction,
    _create_function_payload,
    _extract_closure_variables,
    _extract_global_variables,
    _get_function_source_for_sandbox,
    _is_session_function_decorator,
    _parse_exception_from_stderr,
)
from cwsandbox._types import OperationRef
from tests.unit.cwsandbox.conftest import make_operation_ref, make_process

T = TypeVar("T", bound=Callable[..., Any])


def _make_session_function_decorator(f: T) -> T:
    """A decorator that mimics @session.function() for testing."""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return f(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# Alias to match pattern recognition
class _MockSession:
    @staticmethod
    def function(f: T) -> T:
        return _make_session_function_decorator(f)


session = _MockSession()


class TestGetFunctionSource:
    """Tests for _get_function_source_for_sandbox."""

    def test_removes_session_function_decorator(self) -> None:
        """Test that the @session.function decorator is removed."""

        @session.function
        def decorated_func(x: int) -> int:
            return x * 2

        source = _get_function_source_for_sandbox(decorated_func)

        tree = ast.parse(source)
        func_def = tree.body[0]

        assert isinstance(func_def, ast.FunctionDef)
        assert func_def.name == "decorated_func"
        assert len(func_def.decorator_list) == 0

    def test_preserves_inner_decorators(self) -> None:
        """Test that inner decorators are preserved."""

        @session.function
        @lru_cache(maxsize=128)
        def cached_func(x: int) -> int:
            return x * 2

        source = _get_function_source_for_sandbox(cached_func)

        tree = ast.parse(source)
        func_def = tree.body[0]

        assert isinstance(func_def, ast.FunctionDef)
        assert func_def.name == "cached_func"
        assert len(func_def.decorator_list) == 1

    def test_handles_no_decorators(self) -> None:
        """Test function without decorators."""

        def plain_func(x: int) -> int:
            return x * 2

        source = _get_function_source_for_sandbox(plain_func)

        tree = ast.parse(source)
        func_def = tree.body[0]

        assert isinstance(func_def, ast.FunctionDef)
        assert func_def.name == "plain_func"

    def test_removes_session_function_from_middle(self) -> None:
        """Test that @session.function is removed even when not outermost."""

        def other_decorator(f: T) -> T:
            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return f(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        @other_decorator
        @session.function
        def func_with_session_in_middle(x: int) -> int:
            return x * 2

        source = _get_function_source_for_sandbox(func_with_session_in_middle)

        tree = ast.parse(source)
        func_def = tree.body[0]

        assert isinstance(func_def, ast.FunctionDef)
        assert func_def.name == "func_with_session_in_middle"
        # Only @other_decorator should remain
        assert len(func_def.decorator_list) == 1

    def test_preserves_all_other_decorators(self) -> None:
        """Test that all non-session.function decorators are preserved."""

        def decorator_a(f: T) -> T:
            return f

        def decorator_b(f: T) -> T:
            return f

        @decorator_a
        @session.function
        @decorator_b
        def multi_decorated(x: int) -> int:
            return x * 2

        source = _get_function_source_for_sandbox(multi_decorated)

        tree = ast.parse(source)
        func_def = tree.body[0]

        assert isinstance(func_def, ast.FunctionDef)
        # @decorator_a and @decorator_b should remain, @session.function removed
        assert len(func_def.decorator_list) == 2


class TestIsSessionFunctionDecorator:
    """Tests for _is_session_function_decorator."""

    def test_recognizes_session_function_call(self) -> None:
        """Test recognition of @session.function()."""
        source = "@session.function()\ndef f(): pass"
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        decorator = func_def.decorator_list[0]
        assert _is_session_function_decorator(decorator)

    def test_recognizes_session_function_no_parens(self) -> None:
        """Test recognition of @session.function without parens."""
        source = "@session.function\ndef f(): pass"
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        decorator = func_def.decorator_list[0]
        assert _is_session_function_decorator(decorator)

    def test_recognizes_alternative_session_names(self) -> None:
        """Test recognition of @*.function() with different session variable names."""
        for session_name in ["session", "my_session", "sess", "s"]:
            source = f"@{session_name}.function()\ndef f(): pass"
            tree = ast.parse(source)
            func_def = tree.body[0]
            assert isinstance(func_def, ast.FunctionDef)
            decorator = func_def.decorator_list[0]
            assert _is_session_function_decorator(decorator), (
                f"Should match @{session_name}.function()"
            )

    def test_rejects_bare_function_decorator(self) -> None:
        """Test that @function() alone is NOT matched (critical for avoiding false positives)."""
        source = "@function()\ndef f(): pass"
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        decorator = func_def.decorator_list[0]
        assert not _is_session_function_decorator(decorator), "@function() should not match"

    def test_rejects_bare_function_no_parens(self) -> None:
        """Test that @function alone is NOT matched."""
        source = "@function\ndef f(): pass"
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        decorator = func_def.decorator_list[0]
        assert not _is_session_function_decorator(decorator), "@function should not match"

    def test_rejects_other_decorators(self) -> None:
        """Test that other decorators are not recognized."""
        source = "@other_decorator\ndef f(): pass"
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        decorator = func_def.decorator_list[0]
        assert not _is_session_function_decorator(decorator)

    def test_rejects_other_attribute_methods(self) -> None:
        """Test that @obj.other_method() is not matched."""
        for method_name in ["execute", "run", "process", "handle"]:
            source = f"@session.{method_name}()\ndef f(): pass"
            tree = ast.parse(source)
            func_def = tree.body[0]
            assert isinstance(func_def, ast.FunctionDef)
            decorator = func_def.decorator_list[0]
            assert not _is_session_function_decorator(decorator), (
                f"Should not match @session.{method_name}()"
            )


class TestExtractClosureVariables:
    """Tests for _extract_closure_variables."""

    def test_extracts_closure_variables(self) -> None:
        """Test extracting closure variables from a function."""
        outer_var = 42
        another_var = "hello"

        def func_with_closure(x: int) -> tuple[int, str]:
            return x + outer_var, another_var

        closure_vars = _extract_closure_variables(func_with_closure)

        assert closure_vars == {"outer_var": 42, "another_var": "hello"}

    def test_no_closure(self) -> None:
        """Test function without closure."""

        def func_no_closure(x: int) -> int:
            return x * 2

        closure_vars = _extract_closure_variables(func_no_closure)

        assert closure_vars == {}

    def test_unwraps_decorated_function(self) -> None:
        """Test closure extraction unwraps decorated functions."""
        outer_var = 100

        def decorator(f: T) -> T:
            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return f(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        @decorator
        def wrapped_func(x: int) -> int:
            return x + outer_var

        closure_vars = _extract_closure_variables(wrapped_func)

        assert closure_vars == {"outer_var": 100}


# Module-level global for testing
_TEST_GLOBAL_VAR = 42


class TestExtractGlobalVariables:
    """Tests for _extract_global_variables."""

    def test_extracts_referenced_globals(self) -> None:
        """Test extraction of globals referenced in function."""

        def func_with_global(x: int) -> int:
            return x + _TEST_GLOBAL_VAR

        global_vars = _extract_global_variables(func_with_global)

        assert "_TEST_GLOBAL_VAR" in global_vars
        assert global_vars["_TEST_GLOBAL_VAR"] == 42

    def test_excludes_modules(self) -> None:
        """Test that module imports are excluded."""
        from pathlib import Path

        def func_using_module() -> Path:
            from pathlib import Path

            return Path.cwd()

        global_vars = _extract_global_variables(func_using_module)

        assert "Path" not in global_vars

    def test_no_globals(self) -> None:
        """Test function without global references."""

        def func_no_globals(x: int) -> int:
            return x * 2

        global_vars = _extract_global_variables(func_no_globals)

        # May contain some globals from scope, but not our test global
        assert "_TEST_GLOBAL_VAR" not in global_vars or global_vars == {}

    def test_extracts_globals_from_nested_functions(self) -> None:
        """Test that globals referenced in nested functions are extracted."""

        def outer_func() -> int:
            def inner_func() -> int:
                return _TEST_GLOBAL_VAR

            return inner_func()

        global_vars = _extract_global_variables(outer_func)

        assert "_TEST_GLOBAL_VAR" in global_vars
        assert global_vars["_TEST_GLOBAL_VAR"] == 42


class TestParseExceptionFromStderr:
    """Tests for _parse_exception_from_stderr."""

    def test_parses_standard_exception_format(self) -> None:
        """Test parsing standard Python exception format."""
        stderr = "ValueError: invalid literal for int()"

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type == "ValueError"
        assert exc_msg == "invalid literal for int()"

    def test_parses_exception_marker(self) -> None:
        """Test parsing EXCEPTION: marker format."""
        stderr = "EXCEPTION: Something went wrong"

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type is None
        assert exc_msg == "Something went wrong"

    def test_parses_exception_with_traceback(self) -> None:
        """Test parsing exception from full traceback output."""
        stderr = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise RuntimeError("test error")
RuntimeError: test error"""

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type == "RuntimeError"
        assert exc_msg == "test error"

    def test_returns_none_for_no_exception(self) -> None:
        """Test returns None tuple when no exception found."""
        stderr = "Some regular output\nAnother line"

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type is None
        assert exc_msg is None

    def test_parses_custom_exception_type(self) -> None:
        """Test parsing custom exception types ending in Exception."""
        stderr = "CustomException: custom error message"

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type == "CustomException"
        assert exc_msg == "custom error message"

    def test_exception_marker_takes_precedence(self) -> None:
        """Test EXCEPTION: marker message is captured alongside type."""
        stderr = """EXCEPTION: marker message
TypeError: type error message"""

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type == "TypeError"
        assert exc_msg == "type error message"

    def test_ignores_indented_lines(self) -> None:
        """Test that indented lines are not parsed as exceptions."""
        stderr = """Traceback (most recent call last):
  File "test.py", line 10, in func
    ValueError: this is in code, not an exception
ValueError: actual exception"""

        exc_type, exc_msg = _parse_exception_from_stderr(stderr)

        assert exc_type == "ValueError"
        assert exc_msg == "actual exception"

    def test_empty_stderr(self) -> None:
        """Test empty stderr returns None tuple."""
        exc_type, exc_msg = _parse_exception_from_stderr("")

        assert exc_type is None
        assert exc_msg is None


class TestCreateFunctionPayload:
    """Tests for _create_function_payload (JSON serialization)."""

    def test_creates_valid_json_payload(self) -> None:
        """Test payload is valid JSON data."""
        source = "def add(x, y): return x + y"
        payload = _create_function_payload(
            source=source,
            func_name="add",
            closure_vars={"multiplier": 2},
            args=(1, 2),
            kwargs={"z": 3},
        )

        parsed = json.loads(payload)

        assert parsed["source"] == source
        assert parsed["name"] == "add"
        assert parsed["closure_vars"] == {"multiplier": 2}
        assert parsed["args"] == [1, 2]
        assert parsed["kwargs"] == {"z": 3}

    def test_converts_tuple_to_list(self) -> None:
        """Test JSON payload converts args tuple to list."""
        payload = _create_function_payload(
            source="def f(): pass",
            func_name="f",
            closure_vars={},
            args=(1, 2, 3),
            kwargs={},
        )

        parsed = json.loads(payload)
        assert parsed["args"] == [1, 2, 3]
        assert isinstance(parsed["args"], list)

    def test_handles_nested_closure_vars(self) -> None:
        """Test payload handles nested JSON-compatible closure variable types."""
        complex_vars = {
            "numbers": [1, 2, 3],
            "nested": {"a": {"b": "c"}},
        }

        payload = _create_function_payload(
            source="def f(): pass",
            func_name="f",
            closure_vars=complex_vars,
            args=(),
            kwargs={},
        )

        parsed = json.loads(payload)
        assert parsed["closure_vars"] == complex_vars


class TestRemoteFunction:
    """Tests for RemoteFunction class."""

    def test_remote_returns_operation_ref(self) -> None:
        """Test that remote() returns an OperationRef."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(add, session=session)

        # Mock the loop manager to avoid actual async execution
        mock_future = MagicMock()
        mock_future.result.return_value = 5

        def mock_run_async(coro: Any) -> MagicMock:
            coro.close()  # Close coroutine to prevent unawaited warning
            return mock_future

        session._loop_manager.run_async = MagicMock(side_effect=mock_run_async)

        ref = remote_fn.remote(2, 3)

        assert isinstance(ref, OperationRef)

    def test_local_executes_without_sandbox(self) -> None:
        """Test that local() executes the function directly."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(add, session=session)

        result = remote_fn.local(2, 3)

        assert result == 5

    def test_map_returns_list_of_operation_refs(self) -> None:
        """Test that map() returns a list of OperationRefs."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(add, session=session)

        mock_future = MagicMock()

        def mock_run_async(coro: Any) -> MagicMock:
            coro.close()  # Close coroutine to prevent unawaited warning
            return mock_future

        session._loop_manager.run_async = MagicMock(side_effect=mock_run_async)

        refs = remote_fn.map([(1, 2), (3, 4), (5, 6)])

        assert len(refs) == 3
        assert all(isinstance(ref, OperationRef) for ref in refs)

    def test_preserves_function_name(self) -> None:
        """Test that RemoteFunction preserves function metadata."""
        from cwsandbox import Session

        session = Session()

        def my_special_function(x: int) -> int:
            """My docstring."""
            return x * 2

        remote_fn = RemoteFunction(my_special_function, session=session)

        assert remote_fn.__name__ == "my_special_function"
        assert remote_fn.__doc__ == "My docstring."

    def test_rejects_async_function(self) -> None:
        """Test that RemoteFunction rejects async functions."""
        from cwsandbox import Session
        from cwsandbox.exceptions import AsyncFunctionError

        session = Session()

        async def async_func(x: int) -> int:
            return x * 2

        with pytest.raises(AsyncFunctionError, match="async"):
            RemoteFunction(async_func, session=session)

    def test_rejects_async_generator(self) -> None:
        """Test that RemoteFunction rejects async generator functions."""
        from cwsandbox import Session
        from cwsandbox.exceptions import AsyncFunctionError

        session = Session()

        async def async_gen() -> Any:
            yield 1

        with pytest.raises(AsyncFunctionError, match="async"):
            RemoteFunction(async_gen, session=session)

    @pytest.mark.asyncio
    async def test_execute_async_runs_in_sandbox(self) -> None:
        """Test that _execute_async creates sandbox and runs function."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(add, session=session)

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(return_value=make_process(returncode=0))

        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = MagicMock(return_value=make_operation_ref(result_json))

        with patch("cwsandbox._sandbox.Sandbox", return_value=mock_sandbox):
            result = await remote_fn._execute_async(2, 3)

            assert result == 5
            mock_sandbox._start_async.assert_called_once()
            mock_sandbox.write_file.assert_called_once()
            mock_sandbox.exec.assert_called_once()
            mock_sandbox.read_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_raises_on_failure(self) -> None:
        """Test _execute_async raises SandboxExecutionError on non-zero exit."""
        from cwsandbox import Session
        from cwsandbox.exceptions import SandboxExecutionError

        session = Session()

        def failing_func() -> None:
            raise RuntimeError("boom")

        remote_fn = RemoteFunction(failing_func, session=session)

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(
            return_value=make_process(returncode=1, stderr="RuntimeError: boom")
        )

        with patch("cwsandbox._sandbox.Sandbox", return_value=mock_sandbox):
            with pytest.raises(SandboxExecutionError, match="execution failed"):
                await remote_fn._execute_async()


class TestSessionFunctionDecorator:
    """Tests for session.function() decorator returning RemoteFunction."""

    def test_decorator_returns_remote_function(self) -> None:
        """Test that @session.function() returns RemoteFunction."""
        from cwsandbox import Session

        session = Session()

        @session.function()
        def compute(x: int, y: int) -> int:
            return x + y

        assert isinstance(compute, RemoteFunction)

    def test_decorated_function_has_remote_method(self) -> None:
        """Test that decorated function has .remote() method."""
        from cwsandbox import Session

        session = Session()

        @session.function()
        def compute(x: int, y: int) -> int:
            return x + y

        assert hasattr(compute, "remote")
        assert callable(compute.remote)

    def test_decorated_function_has_local_method(self) -> None:
        """Test that decorated function has .local() method."""
        from cwsandbox import Session

        session = Session()

        @session.function()
        def compute(x: int, y: int) -> int:
            return x + y

        assert hasattr(compute, "local")
        result = compute.local(2, 3)
        assert result == 5

    def test_decorated_function_has_map_method(self) -> None:
        """Test that decorated function has .map() method."""
        from cwsandbox import Session

        session = Session()

        @session.function()
        def compute(x: int, y: int) -> int:
            return x + y

        assert hasattr(compute, "map")
        assert callable(compute.map)

    def test_decorated_function_preserves_name(self) -> None:
        """Test that decorated function preserves original name."""
        from cwsandbox import Session

        session = Session()

        @session.function()
        def my_special_function(x: int) -> int:
            return x * 2

        assert my_special_function.__name__ == "my_special_function"


class TestRemoteFunctionAnnotations:
    """Tests for annotations passthrough in RemoteFunction."""

    def test_function_with_annotations(self) -> None:
        """Test RemoteFunction stores annotations."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(
            add,
            session=session,
            annotations={"team": "platform"},
        )

        assert remote_fn._annotations == {"team": "platform"}

    @pytest.mark.asyncio
    async def test_function_annotations_in_sandbox_kwargs(self) -> None:
        """Test annotations are passed to sandbox creation in _execute_async."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(
            add,
            session=session,
            annotations={"team": "platform"},
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(return_value=make_process(returncode=0))

        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = MagicMock(return_value=make_operation_ref(result_json))

        with patch("cwsandbox._sandbox.Sandbox") as MockSandbox:
            MockSandbox.return_value = mock_sandbox
            await remote_fn._execute_async(2, 3)

            call_kwargs = MockSandbox.call_args[1]
            assert call_kwargs["annotations"] == {"team": "platform"}

    def test_session_function_decorator_with_annotations(self) -> None:
        """Test @session.function(annotations=...) passes annotations to RemoteFunction."""
        from cwsandbox import Session

        session = Session()

        @session.function(annotations={"team": "platform"})
        def compute(x: int, y: int) -> int:
            return x + y

        assert isinstance(compute, RemoteFunction)
        assert compute._annotations == {"team": "platform"}


class TestRemoteFunctionResourceOptions:
    """Tests for ResourceOptions passthrough in RemoteFunction."""

    @pytest.mark.asyncio
    async def test_resource_options_passed_to_sandbox(self) -> None:
        """Test that ResourceOptions flows through .remote() to Sandbox.__init__."""
        from cwsandbox import Session
        from cwsandbox._types import ResourceOptions

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        resource_opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )

        remote_fn = RemoteFunction(
            add,
            session=session,
            resources=resource_opts,
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(return_value=make_process(returncode=0))

        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = MagicMock(return_value=make_operation_ref(result_json))

        with patch("cwsandbox._sandbox.Sandbox") as MockSandbox:
            MockSandbox.return_value = mock_sandbox
            await remote_fn._execute_async(2, 3)

            call_kwargs = MockSandbox.call_args[1]
            assert call_kwargs["resources"] is resource_opts
            assert isinstance(call_kwargs["resources"], ResourceOptions)
            assert call_kwargs["resources"].requests == {"cpu": "1", "memory": "256Mi"}
            assert call_kwargs["resources"].limits == {"cpu": "8", "memory": "2Gi"}


class TestRemoteFunctionProfileNames:
    """Tests for profile_names passthrough in RemoteFunction."""

    def test_function_stores_profile_names(self) -> None:
        """RemoteFunction stores profile_names passed to constructor."""
        from cwsandbox import Session

        session = Session()

        def f(x: int) -> int:
            return x

        remote_fn = RemoteFunction(f, session=session, profile_names=["prod"])
        assert remote_fn._profile_names == ["prod"]

    @pytest.mark.asyncio
    async def test_execute_async_passes_profile_names_to_sandbox(self) -> None:
        """_execute_async wires profile_names into sandbox_kwargs for Sandbox construction."""
        from cwsandbox import Session

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        remote_fn = RemoteFunction(
            add,
            session=session,
            profile_ids=["id-1"],
            profile_names=["name-1"],
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(return_value=make_process(returncode=0))
        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = MagicMock(return_value=make_operation_ref(result_json))

        with patch("cwsandbox._sandbox.Sandbox") as MockSandbox:
            MockSandbox.return_value = mock_sandbox
            await remote_fn._execute_async(2, 3)

            call_kwargs = MockSandbox.call_args[1]
            assert call_kwargs["profile_ids"] == ["id-1"]
            assert call_kwargs["profile_names"] == ["name-1"]

    @pytest.mark.asyncio
    async def test_session_defaults_profile_names_flow_through_function(
        self, mock_api_key: str
    ) -> None:
        """Session(profile_names=...) + @session.function() reaches StartSandboxRequest.

        Exercises the full seam: RemoteFunction has no explicit profile_names, so
        sandbox_kwargs omits the key. Sandbox.__init__ must then fall back to
        SandboxDefaults.profile_names and surface them on the wire request.
        """
        from cwsandbox import SandboxDefaults, Session

        defaults = SandboxDefaults(profile_names=("prod",))
        session = Session(defaults)

        @session.function()
        def foo() -> int:
            return 1

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()

        class _StopTest(Exception):
            pass

        captured: list[Any] = []

        async def capture_start(request: Any, *args: Any, **kwargs: Any) -> Any:
            captured.append(request)
            raise _StopTest("captured Start request")

        mock_stub.Start = AsyncMock(side_effect=capture_start)

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
            pytest.raises(_StopTest),
        ):
            await foo._execute_async()

        assert len(captured) == 1
        assert list(captured[0].profile_names) == ["prod"]


class TestJsonReturnContract:
    """Sandbox-side JSON return handling.

    These tests run the real ``_FUNCTION_EXECUTION_TEMPLATE`` script in a
    subprocess to validate the round-trip without booting a sandbox.
    """

    @staticmethod
    def _run_template(
        tmp_path: Any,
        source: str,
        func_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> tuple[int, str, str | None]:
        """Run _FUNCTION_EXECUTION_TEMPLATE for the given function and capture result."""
        import subprocess
        import sys

        from cwsandbox._function import (
            _FUNCTION_EXECUTION_TEMPLATE,
            _create_function_payload,
        )

        payload_file = tmp_path / "payload.json"
        result_file = tmp_path / "result.json"
        payload_file.write_bytes(
            _create_function_payload(
                source=source,
                func_name=func_name,
                closure_vars={},
                args=tuple(args),
                kwargs=kwargs,
            )
        )

        script = _FUNCTION_EXECUTION_TEMPLATE.format(
            payload_file=str(payload_file),
            result_file=str(result_file),
            temp_dir=str(tmp_path),
        )

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        result_text = result_file.read_text() if result_file.exists() else None
        return proc.returncode, proc.stderr, result_text

    def test_json_return_value_round_trips(self, tmp_path: Any) -> None:
        """A JSON-serializable return value round-trips through the template."""
        returncode, _stderr, result_text = self._run_template(
            tmp_path,
            source="def echo(value): return value",
            func_name="echo",
            args=[{"a": 1, "b": [2, 3]}],
            kwargs={},
        )
        assert returncode == 0
        assert result_text is not None
        assert json.loads(result_text) == {"a": 1, "b": [2, 3]}

    def test_non_json_return_value_surfaces_failure(self, tmp_path: Any) -> None:
        """A non-JSON-serializable return value fails inside the sandbox script.

        Returning values like ``set`` is expected to surface as a non-zero exit
        code with a ``RESULT_SERIALIZATION_ERROR:`` marker on stderr that the
        outer SDK distinguishes from a user-code exception.
        """
        returncode, stderr, _result_text = self._run_template(
            tmp_path,
            source="def returns_set(): return {1, 2, 3}",
            func_name="returns_set",
            args=[],
            kwargs={},
        )
        assert returncode != 0
        assert "RESULT_SERIALIZATION_ERROR:" in stderr
        assert "TypeError" in stderr

    def test_serialization_marker_parses_into_clear_message(self, tmp_path: Any) -> None:
        """``RESULT_SERIALIZATION_ERROR:`` is parsed distinctly from user exceptions."""
        returncode, stderr, _result_text = self._run_template(
            tmp_path,
            source="def returns_set(): return {1, 2, 3}",
            func_name="returns_set",
            args=[],
            kwargs={},
        )
        assert returncode != 0
        exc_type, exc_msg = _parse_exception_from_stderr(stderr)
        assert exc_type == "TypeError"
        assert exc_msg is not None
        assert "return value is not JSON-serializable" in exc_msg


class TestJsonReturnCoercionContract:
    """Pin known silent JSON coercions on the return path.

    These document behaviors users must be aware of: stdlib ``json.dumps``
    rewrites tuples as lists, coerces dict integer keys to strings, and
    encodes ``NaN`` as a non-strict literal.
    """

    @staticmethod
    def _round_trip(tmp_path: Any, source: str, func_name: str) -> tuple[int, str, str | None]:
        return TestJsonReturnContract._run_template(
            tmp_path,
            source=source,
            func_name=func_name,
            args=[],
            kwargs={},
        )

    def test_tuple_return_becomes_list(self, tmp_path: Any) -> None:
        returncode, _stderr, result_text = self._round_trip(
            tmp_path,
            source="def f(): return (1, 2)",
            func_name="f",
        )
        assert returncode == 0
        assert result_text is not None
        assert json.loads(result_text) == [1, 2]

    def test_int_dict_keys_become_strings(self, tmp_path: Any) -> None:
        returncode, _stderr, result_text = self._round_trip(
            tmp_path,
            source='def f(): return {1: "a"}',
            func_name="f",
        )
        assert returncode == 0
        assert result_text is not None
        assert json.loads(result_text) == {"1": "a"}

    def test_nan_return_is_not_strict_json(self, tmp_path: Any) -> None:
        returncode, _stderr, result_text = self._round_trip(
            tmp_path,
            source='def f():\n    return float("nan")',
            func_name="f",
        )
        assert returncode == 0
        assert result_text is not None
        with pytest.raises(json.JSONDecodeError):
            json.loads(result_text, parse_constant=_reject)


def _reject(_token: str) -> Any:
    raise json.JSONDecodeError("strict mode rejects non-finite", "", 0)


class TestCreateFunctionPayloadKeyValidation:
    """Pre-flight validation for non-string mapping keys."""

    def test_non_string_key_in_closure_var_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="closure_vars"):
            _create_function_payload(
                source="def f(): pass",
                func_name="f",
                closure_vars={"LOOKUP": {(1, 2): "ok"}},
                args=(),
                kwargs={},
            )

    def test_non_string_key_in_args_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="args"):
            _create_function_payload(
                source="def f(x): pass",
                func_name="f",
                closure_vars={},
                args=({1: "a"},),
                kwargs={},
            )

    def test_non_string_key_in_kwargs_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="kwargs"):
            _create_function_payload(
                source="def f(**kw): pass",
                func_name="f",
                closure_vars={},
                args=(),
                kwargs={"data": {1: "a"}},
            )

    def test_non_serializable_argument_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="not JSON-serializable"):
            _create_function_payload(
                source="def f(x): pass",
                func_name="f",
                closure_vars={},
                args=({1, 2, 3},),
                kwargs={},
            )
