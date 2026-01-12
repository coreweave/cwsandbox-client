"""Shared fixtures for aviato unit tests."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TypeVar
from unittest.mock import MagicMock

import pytest

from aviato import OperationRef
from aviato._types import Process, ProcessResult, StreamReader

T = TypeVar("T")

# Environment variables that affect authentication behavior.
# These are cleared before each test to ensure isolation.
AUTH_ENV_VARS = (
    "AVIATO_API_KEY",
    "AVIATO_BASE_URL",
    "WANDB_API_KEY",
    "WANDB_ENTITY_NAME",
    "WANDB_PROJECT_NAME",
)


@pytest.fixture(autouse=True)
def clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all auth-related env vars before each test.

    This runs automatically for every test (autouse=True) and ensures:
    1. Tests start with a clean environment (no leakage from local setup)
    2. The original environment is restored after each test (even on failure)
    3. Tests are deterministic regardless of the developer's local env
    """
    for var in AUTH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_aviato_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock AVIATO_API_KEY for the test."""
    test_key = "test-api-key"
    monkeypatch.setenv("AVIATO_API_KEY", test_key)
    return test_key


@pytest.fixture
def mock_aviato_base_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock AVIATO_BASE_URL for the test."""
    test_url = "http://test-api.example.com"
    monkeypatch.setenv("AVIATO_BASE_URL", test_url)
    return test_url


@pytest.fixture
def mock_wandb_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock WANDB_API_KEY for the test."""
    test_key = "test-wandb-api-key"
    monkeypatch.setenv("WANDB_API_KEY", test_key)
    return test_key


@pytest.fixture
def mock_wandb_entity_name(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock WANDB_ENTITY_NAME for the test."""
    test_entity = "test-entity"
    monkeypatch.setenv("WANDB_ENTITY_NAME", test_entity)
    return test_entity


# Helper functions for creating real OperationRef/Process objects in tests


def make_operation_ref(value: T) -> OperationRef[T]:
    """Create an OperationRef with a pre-completed future.

    Use this instead of MagicMock to exercise the real .result() method.
    """
    future: concurrent.futures.Future[T] = concurrent.futures.Future()
    future.set_result(value)
    return OperationRef(future)


def make_process(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    command: list[str] | None = None,
) -> Process:
    """Create a Process with pre-completed result.

    Use this instead of MagicMock to exercise the real .result() method.
    """
    result = ProcessResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        stdout_bytes=stdout.encode(),
        stderr_bytes=stderr.encode(),
        command=command or [],
    )
    future: concurrent.futures.Future[ProcessResult] = concurrent.futures.Future()
    future.set_result(result)

    # Create StreamReaders with queued content
    stdout_queue: asyncio.Queue[str | None] = asyncio.Queue()
    stdout_queue.put_nowait(stdout if stdout else None)
    if stdout:
        stdout_queue.put_nowait(None)  # Sentinel

    stderr_queue: asyncio.Queue[str | None] = asyncio.Queue()
    stderr_queue.put_nowait(stderr if stderr else None)
    if stderr:
        stderr_queue.put_nowait(None)  # Sentinel

    mock_lm = MagicMock()
    stdout_reader = StreamReader(stdout_queue, mock_lm)
    stderr_reader = StreamReader(stderr_queue, mock_lm)

    return Process(future, command or [], stdout_reader, stderr_reader)
