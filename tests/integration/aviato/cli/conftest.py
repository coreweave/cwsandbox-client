"""Fixtures for CLI integration tests."""

import pytest
import pytest_asyncio

from aviato import Sandbox


@pytest.fixture(scope="module")
def cli_test_tag() -> str:
    """Tag for CLI integration test sandboxes."""
    return "cli-integration-test"


@pytest_asyncio.fixture
async def test_sandbox(cli_test_tag: str) -> Sandbox:
    """Create a real sandbox for testing, stop on cleanup."""
    sandbox = await Sandbox.create(
        "sleep", "infinity",
        container_image="python:3.11",
        tags=[cli_test_tag],
        max_lifetime_seconds=120,
    )
    yield sandbox
    await sandbox.stop()

