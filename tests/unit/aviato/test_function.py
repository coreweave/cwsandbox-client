"""Unit tests for aviato._function module."""

import ast
import json
import pickle
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato._function import (
    _create_function_payload,
    _create_json_payload,
    _extract_closure_variables,
    _extract_global_variables,
    _get_function_source_for_sandbox,
    _is_session_function_decorator,
    _parse_exception_from_stderr,
    _parse_json_result,
    _parse_sandbox_result,
    create_function_wrapper,
)

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
            assert _is_session_function_decorator(
                decorator
            ), f"Should match @{session_name}.function()"

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
            assert not _is_session_function_decorator(
                decorator
            ), f"Should not match @session.{method_name}()"


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


class TestParseSandboxResult:
    """Tests for _parse_sandbox_result."""

    def test_parses_valid_result(self) -> None:
        """Test parsing valid sandbox result."""
        result_value = {"key": "value", "numbers": [1, 2, 3]}
        result_bytes = pickle.dumps(result_value)

        parsed = _parse_sandbox_result(result_bytes)

        assert parsed == result_value

    def test_parses_simple_types(self) -> None:
        """Test parsing various simple types."""
        for value in [42, "hello", [1, 2, 3], None, True]:
            pickled = pickle.dumps(value)

            parsed = _parse_sandbox_result(pickled)

            assert parsed == value

    def test_invalid_format_raises_error(self) -> None:
        """Test parsing invalid pickle data raises error."""
        with pytest.raises(pickle.UnpicklingError):
            _parse_sandbox_result(b"Invalid pickle content")


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
    """Tests for _create_function_payload (pickle mode)."""

    def test_creates_valid_pickle_payload(self) -> None:
        """Test payload is valid pickle data."""
        source = "def test_func(x): return x * 2"
        payload = _create_function_payload(
            source=source,
            func_name="test_func",
            closure_vars={"y": 10},
            args=(5,),
            kwargs={"extra": "value"},
        )

        unpickled = pickle.loads(payload)

        assert unpickled["source"] == source
        assert unpickled["name"] == "test_func"
        assert unpickled["closure_vars"] == {"y": 10}
        assert unpickled["args"] == (5,)
        assert unpickled["kwargs"] == {"extra": "value"}

    def test_handles_complex_closure_vars(self) -> None:
        """Test payload handles complex closure variable types."""
        complex_vars = {
            "numbers": [1, 2, 3],
            "nested": {"a": {"b": "c"}},
            "tuple_data": (1, "two", 3.0),
        }

        payload = _create_function_payload(
            source="def f(): pass",
            func_name="f",
            closure_vars=complex_vars,
            args=(),
            kwargs={},
        )

        unpickled = pickle.loads(payload)
        assert unpickled["closure_vars"] == complex_vars


class TestJsonSerialization:
    """Tests for JSON serialization functions."""

    def test_create_json_payload_valid(self) -> None:
        """Test _create_json_payload creates valid JSON."""
        source = "def add(x, y): return x + y"
        payload = _create_json_payload(
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

    def test_create_json_payload_converts_tuple_to_list(self) -> None:
        """Test JSON payload converts args tuple to list."""
        payload = _create_json_payload(
            source="def f(): pass",
            func_name="f",
            closure_vars={},
            args=(1, 2, 3),
            kwargs={},
        )

        parsed = json.loads(payload)
        assert parsed["args"] == [1, 2, 3]
        assert isinstance(parsed["args"], list)

    def test_parse_json_result_simple_types(self) -> None:
        """Test _parse_json_result parses simple types."""
        for value in [42, "hello", [1, 2, 3], None, True, {"key": "value"}]:
            json_bytes = json.dumps(value).encode()

            parsed = _parse_json_result(json_bytes)

            assert parsed == value

    def test_parse_json_result_nested(self) -> None:
        """Test _parse_json_result parses nested structures."""
        value = {"outer": {"inner": [1, 2, {"deep": True}]}}
        json_bytes = json.dumps(value).encode()

        parsed = _parse_json_result(json_bytes)

        assert parsed == value

    def test_parse_json_result_invalid_raises(self) -> None:
        """Test _parse_json_result raises on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            _parse_json_result(b"not valid json {")


class TestCreateFunctionWrapper:
    """Tests for create_function_wrapper."""

    @pytest.mark.asyncio
    async def test_rejects_async_function(self) -> None:
        """Test wrapper rejects async functions with clear error."""
        from aviato import Session
        from aviato.exceptions import AsyncFunctionError

        session = Session()

        async def async_func(x: int) -> int:
            return x * 2

        with pytest.raises(AsyncFunctionError, match="async"):
            create_function_wrapper(async_func, session=session)

    @pytest.mark.asyncio
    async def test_rejects_async_generator(self) -> None:
        """Test wrapper rejects async generator functions."""
        from aviato import Session
        from aviato.exceptions import AsyncFunctionError

        session = Session()

        async def async_gen() -> Any:
            yield 1

        with pytest.raises(AsyncFunctionError, match="async"):
            create_function_wrapper(async_gen, session=session)

    @pytest.mark.asyncio
    async def test_wrapper_is_async(self) -> None:
        """Test wrapper returns an async callable."""
        import asyncio

        from aviato import Session

        session = Session()

        def sync_func(x: int) -> int:
            return x * 2

        wrapper = create_function_wrapper(sync_func, session=session)

        assert asyncio.iscoroutinefunction(wrapper)

    @pytest.mark.asyncio
    async def test_wrapper_preserves_function_name(self) -> None:
        """Test wrapper preserves the original function name."""
        from aviato import Session

        session = Session()

        def my_special_function(x: int) -> int:
            return x

        wrapper = create_function_wrapper(my_special_function, session=session)

        assert wrapper.__name__ == "my_special_function"

    @pytest.mark.asyncio
    async def test_wrapper_executes_in_sandbox(self) -> None:
        """Test wrapper creates sandbox and executes function."""
        from aviato import Session
        from aviato._types import Serialization

        session = Session()

        def add(x: int, y: int) -> int:
            return x + y

        wrapper = create_function_wrapper(
            add,
            session=session,
            serialization=Serialization.JSON,
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = AsyncMock()

        mock_exec_result = MagicMock()
        mock_exec_result.returncode = 0
        mock_exec_result.stderr = ""
        mock_sandbox.exec = AsyncMock(return_value=mock_exec_result)

        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = AsyncMock(return_value=result_json)

        with patch.object(session, "create", return_value=mock_sandbox):
            result = await wrapper(2, 3)

            assert result == 5
            session.create.assert_called_once()
            mock_sandbox.write_file.assert_called_once()
            mock_sandbox.exec.assert_called_once()
            mock_sandbox.read_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_wrapper_raises_on_execution_failure(self) -> None:
        """Test wrapper raises SandboxExecutionError on non-zero exit."""
        from aviato import Session
        from aviato._types import Serialization
        from aviato.exceptions import SandboxExecutionError

        session = Session()

        def failing_func() -> None:
            raise RuntimeError("boom")

        wrapper = create_function_wrapper(
            failing_func,
            session=session,
            serialization=Serialization.JSON,
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = AsyncMock()

        mock_exec_result = MagicMock()
        mock_exec_result.returncode = 1
        mock_exec_result.stderr = "RuntimeError: boom"
        mock_sandbox.exec = AsyncMock(return_value=mock_exec_result)

        with patch.object(session, "create", return_value=mock_sandbox):
            with pytest.raises(SandboxExecutionError, match="execution failed"):
                await wrapper()

    @pytest.mark.asyncio
    async def test_wrapper_uses_pickle_serialization(self) -> None:
        """Test wrapper uses pickle when specified."""
        from aviato import Session
        from aviato._types import Serialization

        session = Session()

        def compute(data: list[int]) -> int:
            return sum(data)

        wrapper = create_function_wrapper(
            compute,
            session=session,
            serialization=Serialization.PICKLE,
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = AsyncMock()

        mock_exec_result = MagicMock()
        mock_exec_result.returncode = 0
        mock_exec_result.stderr = ""
        mock_sandbox.exec = AsyncMock(return_value=mock_exec_result)

        result_pickle = pickle.dumps(10)
        mock_sandbox.read_file = AsyncMock(return_value=result_pickle)

        with patch.object(session, "create", return_value=mock_sandbox):
            result = await wrapper([1, 2, 3, 4])

            assert result == 10

            write_call = mock_sandbox.write_file.call_args
            payload_bytes = write_call[0][1]
            payload = pickle.loads(payload_bytes)
            assert "source" in payload
            assert payload["args"] == ([1, 2, 3, 4],)


class TestCreateFunctionWrapperKwargsValidation:
    """Tests for kwargs validation in create_function_wrapper."""

    def test_valid_sandbox_kwargs(self) -> None:
        """Test create_function_wrapper accepts valid sandbox_kwargs."""
        from aviato import Session

        session = Session()

        def compute(x: int) -> int:
            return x * 2

        wrapper = create_function_wrapper(
            compute,
            session=session,
            resources={"cpu": "100m"},
            ports=[{"container_port": 8080}],
        )

        assert callable(wrapper)

    def test_invalid_sandbox_kwargs(self) -> None:
        """Test create_function_wrapper rejects invalid sandbox_kwargs."""
        from aviato import Session

        session = Session()

        def compute(x: int) -> int:
            return x * 2

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_function_wrapper(
                compute,
                session=session,
                invalid_param="value",
            )

    def test_mixed_valid_invalid_sandbox_kwargs(self) -> None:
        """Test create_function_wrapper rejects if any sandbox_kwargs are invalid."""
        from aviato import Session

        session = Session()

        def compute(x: int) -> int:
            return x * 2

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_function_wrapper(
                compute,
                session=session,
                resources={"cpu": "100m"},
                invalid_param="value",
            )
