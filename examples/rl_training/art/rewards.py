"""MBPP reward calculation utilities for RL training.

This module provides functions for loading MBPP problems from HuggingFace
and calculating rewards based on test case execution results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MBPPProblem:
    """A problem from the MBPP (Mostly Basic Python Problems) benchmark.

    Attributes:
        task_id: Unique identifier for the problem.
        prompt: The problem description/instruction text.
        test_cases: List of test assertions (e.g., ["assert add(1, 2) == 3"]).
    """

    task_id: int
    prompt: str
    test_cases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.test_cases, tuple):
            object.__setattr__(self, "test_cases", tuple(self.test_cases))


def load_mbpp_problems(
    split: str = "test",
    limit: int | None = None,
) -> list[MBPPProblem]:
    """Load MBPP problems from HuggingFace.

    Args:
        split: Dataset split to load ("train", "test", "validation", "prompt").
        limit: Maximum number of problems to load. None for all.

    Returns:
        List of MBPPProblem instances.

    Raises:
        ImportError: If datasets library is not installed.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("datasets library required: pip install datasets") from e

    dataset = load_dataset("google-research-datasets/mbpp", split=split)

    problems = []
    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        problems.append(
            MBPPProblem(
                task_id=example["task_id"],
                prompt=example["text"],
                test_cases=tuple(example["test_list"]),
            )
        )

    return problems


def extract_code(completion: str) -> str:
    """Extract Python code from a model completion.

    Handles various formats:
    - Markdown code blocks with ```python or ``` delimiters
    - <code> tags
    - Raw code without any delimiters

    Args:
        completion: The raw model completion text.

    Returns:
        Extracted Python code, stripped of whitespace.
    """
    if not completion:
        return ""

    # Try markdown code block with language specifier
    match = re.search(r"```python\s*\n?(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try markdown code block without language specifier
    match = re.search(r"```\s*\n?(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try <code> tags
    match = re.search(r"<code>(.*?)</code>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as raw code, filtering out common non-code patterns
    lines = completion.strip().split("\n")
    code_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Skip interactive interpreter prompts
        if stripped.startswith(">>>") or stripped.startswith("..."):
            continue
        code_lines.append(line)

    return "\n".join(code_lines).strip()


def passes_tests(
    stdout: str,
    stderr: str,
    test_cases: list[str] | tuple[str, ...],
) -> bool:
    """Check if code execution passed all test cases.

    This function determines pass/fail based on execution output:
    - Pass: No assertion errors, no exceptions, clean execution
    - Fail: Any AssertionError, exception, or syntax error

    Args:
        stdout: Standard output from code execution.
        stderr: Standard error from code execution.

    Returns:
        True if all tests passed, False otherwise.
    """
    # Check for explicit failure indicators in stderr
    if stderr:
        stderr_lower = stderr.lower()
        failure_indicators = [
            "assertionerror",
            "error",
            "exception",
            "traceback",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "valueerror",
            "attributeerror",
            "indexerror",
            "keyerror",
            "zerodivisionerror",
        ]
        for indicator in failure_indicators:
            if indicator in stderr_lower:
                return False

    # Check for explicit failure indicators in stdout (some errors print there)
    if stdout:
        stdout_lower = stdout.lower()
        if "assertionerror" in stdout_lower or "traceback" in stdout_lower:
            return False

    # If we have test cases but no output and no errors, execution was likely clean
    # (assertions don't produce output when they pass)
    return True
