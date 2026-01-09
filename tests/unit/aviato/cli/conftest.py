"""Fixtures for CLI unit tests."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock Sandbox object."""
    sb = MagicMock()
    sb.sandbox_id = "sb-abc123"
    sb.status = "running"
    sb.started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
    sb.container_image = "python:3.11"
    return sb


@pytest.fixture
def mock_sandboxes(mock_sandbox: MagicMock) -> list[MagicMock]:
    """Create a list of mock Sandbox objects."""
    sb1 = mock_sandbox

    sb2 = MagicMock()
    sb2.sandbox_id = "sb-def456"
    sb2.status = "completed"
    sb2.started_at = datetime(2025, 1, 7, 10, 0, 0, tzinfo=UTC)
    sb2.container_image = "ubuntu:22.04"

    return [sb1, sb2]
