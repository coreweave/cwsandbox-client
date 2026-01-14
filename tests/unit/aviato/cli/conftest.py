"""Fixtures for CLI unit tests."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from aviato import SandboxStatus


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock Sandbox object."""
    sb = MagicMock()
    sb.sandbox_id = "sb-abc123"
    sb.status = SandboxStatus.RUNNING
    sb.started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
    sb.tower_id = "tower-abc"
    sb.runway_id = "runway-123"
    return sb


@pytest.fixture
def mock_sandboxes(mock_sandbox: MagicMock) -> list[MagicMock]:
    """Create a list of mock Sandbox objects."""
    sb1 = mock_sandbox

    sb2 = MagicMock()
    sb2.sandbox_id = "sb-def456"
    sb2.status = SandboxStatus.COMPLETED
    sb2.started_at = datetime(2025, 1, 7, 10, 0, 0, tzinfo=UTC)
    sb2.tower_id = "tower-def"
    sb2.runway_id = "runway-456"

    return [sb1, sb2]
