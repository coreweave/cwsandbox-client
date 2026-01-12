"""Fixtures for CLI integration tests."""

from collections.abc import Generator

import pytest

from aviato import Sandbox


@pytest.fixture(scope="module")
def cli_test_tag() -> str:
    """Tag for CLI integration test sandboxes."""
    return "cli-integration-test"


@pytest.fixture
def test_sandbox(cli_test_tag: str) -> Generator[Sandbox, None, None]:
    """Create a real sandbox for testing, stop on cleanup."""
    sandbox = Sandbox.run(
        "sleep",
        "infinity",
        container_image="python:3.11",
        tags=[cli_test_tag],
        max_lifetime_seconds=120,
    ).wait()
    yield sandbox
    sandbox.stop().result()
