"""Tests for MBPP reward calculation utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add examples to path for importing
examples_path = Path(__file__).parent.parent.parent.parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from rl_training.art.rewards import (  # noqa: E402
    MBPPProblem,
    extract_code,
    load_mbpp_problems,
    passes_tests,
)


class TestMBPPProblem:
    """Tests for MBPPProblem dataclass."""

    def test_create_problem(self) -> None:
        """MBPPProblem can be created with required fields."""
        problem = MBPPProblem(
            task_id=1,
            prompt="Write a function to add two numbers.",
            test_cases=("assert add(1, 2) == 3",),
        )

        assert problem.task_id == 1
        assert problem.prompt == "Write a function to add two numbers."
        assert problem.test_cases == ("assert add(1, 2) == 3",)

    def test_problem_is_frozen(self) -> None:
        """MBPPProblem is immutable (frozen)."""
        problem = MBPPProblem(
            task_id=1,
            prompt="Test prompt",
            test_cases=("assert True",),
        )

        with pytest.raises(AttributeError):
            problem.task_id = 2  # type: ignore[misc]

    def test_test_cases_converted_to_tuple(self) -> None:
        """test_cases list is converted to tuple."""
        problem = MBPPProblem(
            task_id=1,
            prompt="Test prompt",
            test_cases=["assert True", "assert 1 == 1"],  # type: ignore[arg-type]
        )

        assert isinstance(problem.test_cases, tuple)
        assert problem.test_cases == ("assert True", "assert 1 == 1")

    def test_multiple_test_cases(self) -> None:
        """MBPPProblem can have multiple test cases."""
        test_cases = (
            "assert add(1, 2) == 3",
            "assert add(0, 0) == 0",
            "assert add(-1, 1) == 0",
        )
        problem = MBPPProblem(
            task_id=42,
            prompt="Add two numbers",
            test_cases=test_cases,
        )

        assert len(problem.test_cases) == 3
        assert problem.test_cases == test_cases


class TestExtractCode:
    """Tests for extract_code function."""

    def test_extract_from_python_fence(self) -> None:
        """Extracts code from ```python fence."""
        completion = """Here's the solution:

```python
def add(a, b):
    return a + b
```

This function adds two numbers."""

        code = extract_code(completion)

        assert code == "def add(a, b):\n    return a + b"

    def test_extract_from_plain_fence(self) -> None:
        """Extracts code from ``` fence without language specifier."""
        completion = """Solution:

```
def multiply(x, y):
    return x * y
```"""

        code = extract_code(completion)

        assert code == "def multiply(x, y):\n    return x * y"

    def test_extract_from_code_tags(self) -> None:
        """Extracts code from <code> tags."""
        completion = "The answer is: <code>print('hello')</code>"

        code = extract_code(completion)

        assert code == "print('hello')"

    def test_extract_raw_code(self) -> None:
        """Returns raw code when no delimiters present."""
        completion = """def greet(name):
    return f"Hello, {name}!"
"""

        code = extract_code(completion)

        assert "def greet(name):" in code
        assert 'return f"Hello, {name}!"' in code

    def test_extract_empty_completion(self) -> None:
        """Returns empty string for empty completion."""
        assert extract_code("") == ""

    def test_extract_whitespace_only(self) -> None:
        """Returns empty string for whitespace-only completion."""
        assert extract_code("   \n\n   ") == ""

    def test_filters_interpreter_prompts(self) -> None:
        """Filters out >>> and ... interpreter prompts."""
        completion = """>>> def foo():
...     return 42
>>> print(foo())
result = foo()
print(result)"""

        code = extract_code(completion)

        # Should not contain >>> or ... lines
        assert ">>>" not in code
        assert "..." not in code
        assert "result = foo()" in code

    def test_extract_first_code_block(self) -> None:
        """Extracts first code block when multiple present."""
        completion = """First block:
```python
first = 1
```

Second block:
```python
second = 2
```"""

        code = extract_code(completion)

        assert code == "first = 1"

    def test_preserves_indentation(self) -> None:
        """Preserves indentation in extracted code."""
        completion = """```python
def nested():
    if True:
        for i in range(3):
            print(i)
```"""

        code = extract_code(completion)

        assert "    if True:" in code
        assert "        for i in range(3):" in code
        assert "            print(i)" in code

    def test_handles_empty_code_block(self) -> None:
        """Handles empty code block."""
        completion = "```python\n```"

        code = extract_code(completion)

        assert code == ""


class TestPassesTests:
    """Tests for passes_tests function."""

    def test_passes_with_clean_output(self) -> None:
        """Returns True for clean execution (no errors)."""
        assert passes_tests("42\n", "", []) is True

    def test_passes_with_empty_output(self) -> None:
        """Returns True for empty output (assertions pass silently)."""
        assert passes_tests("", "", ["assert True"]) is True

    def test_fails_with_assertion_error(self) -> None:
        """Returns False when stderr contains AssertionError."""
        stderr = "Traceback (most recent call last):\n  AssertionError"
        assert passes_tests("", stderr, ["assert False"]) is False

    def test_fails_with_assertion_error_in_stdout(self) -> None:
        """Returns False when stdout contains AssertionError."""
        stdout = "AssertionError: 1 != 2"
        assert passes_tests(stdout, "", ["assert 1 == 2"]) is False

    def test_fails_with_syntax_error(self) -> None:
        """Returns False when stderr contains SyntaxError."""
        stderr = "SyntaxError: invalid syntax"
        assert passes_tests("", stderr, []) is False

    def test_fails_with_name_error(self) -> None:
        """Returns False when stderr contains NameError."""
        stderr = "NameError: name 'undefined' is not defined"
        assert passes_tests("", stderr, []) is False

    def test_fails_with_type_error(self) -> None:
        """Returns False when stderr contains TypeError."""
        stderr = "TypeError: unsupported operand type(s)"
        assert passes_tests("", stderr, []) is False

    def test_fails_with_value_error(self) -> None:
        """Returns False when stderr contains ValueError."""
        stderr = "ValueError: invalid literal"
        assert passes_tests("", stderr, []) is False

    def test_fails_with_traceback(self) -> None:
        """Returns False when stderr contains Traceback."""
        stderr = "Traceback (most recent call last):\n  File..."
        assert passes_tests("", stderr, []) is False

    def test_fails_with_traceback_in_stdout(self) -> None:
        """Returns False when stdout contains Traceback."""
        stdout = "Traceback (most recent call last):\n  File..."
        assert passes_tests(stdout, "", []) is False

    def test_fails_with_zero_division_error(self) -> None:
        """Returns False when stderr contains ZeroDivisionError."""
        stderr = "ZeroDivisionError: division by zero"
        assert passes_tests("", stderr, []) is False

    def test_case_insensitive_error_detection(self) -> None:
        """Error detection is case-insensitive."""
        stderr = "ASSERTIONERROR: test failed"
        assert passes_tests("", stderr, []) is False

    def test_passes_with_regular_output(self) -> None:
        """Returns True for normal program output."""
        stdout = "Hello, World!\nProcessing complete.\n"
        assert passes_tests(stdout, "", []) is True


class TestLoadMBPPProblems:
    """Tests for load_mbpp_problems function."""

    def test_raises_import_error_without_datasets(self) -> None:
        """Raises ImportError when datasets library not installed."""
        with patch.dict(sys.modules, {"datasets": None}):
            # Clear the cached import
            with pytest.raises(ImportError, match="datasets library required"):
                # Force re-import by reloading
                import importlib

                from rl_training.art import rewards

                importlib.reload(rewards)
                rewards.load_mbpp_problems()

    def test_loads_problems_from_dataset(self) -> None:
        """Loads problems from HuggingFace dataset."""
        mock_dataset = [
            {
                "task_id": 1,
                "text": "Write a function to find the maximum.",
                "test_list": ["assert max([1,2,3]) == 3"],
                "code": "def max(lst): return max(lst)",
            },
            {
                "task_id": 2,
                "text": "Write a function to find the minimum.",
                "test_list": ["assert min([1,2,3]) == 1", "assert min([5]) == 5"],
                "code": "def min(lst): return min(lst)",
            },
        ]

        mock_load_dataset = MagicMock(return_value=mock_dataset)

        with patch.dict(sys.modules, {"datasets": MagicMock()}):
            sys.modules["datasets"].load_dataset = mock_load_dataset

            problems = load_mbpp_problems(split="test")

            assert len(problems) == 2
            assert problems[0].task_id == 1
            assert problems[0].prompt == "Write a function to find the maximum."
            assert problems[0].test_cases == ("assert max([1,2,3]) == 3",)
            assert problems[1].task_id == 2
            assert len(problems[1].test_cases) == 2

    def test_respects_limit_parameter(self) -> None:
        """Respects limit parameter when loading problems."""
        mock_dataset = [
            {
                "task_id": i,
                "text": f"Problem {i}",
                "test_list": [f"assert test_{i}()"],
                "code": f"def test_{i}(): pass",
            }
            for i in range(10)
        ]

        mock_load_dataset = MagicMock(return_value=mock_dataset)

        with patch.dict(sys.modules, {"datasets": MagicMock()}):
            sys.modules["datasets"].load_dataset = mock_load_dataset

            problems = load_mbpp_problems(split="test", limit=3)

            assert len(problems) == 3
            assert problems[0].task_id == 0
            assert problems[2].task_id == 2

    def test_passes_split_to_load_dataset(self) -> None:
        """Passes split parameter to load_dataset."""
        mock_load_dataset = MagicMock(return_value=[])

        with patch.dict(sys.modules, {"datasets": MagicMock()}):
            sys.modules["datasets"].load_dataset = mock_load_dataset

            load_mbpp_problems(split="validation")

            mock_load_dataset.assert_called_once_with(
                "google-research-datasets/mbpp", split="validation"
            )
