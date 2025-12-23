"""Shared fixtures for integration tests."""

import pytest

from aviato import SandboxDefaults


@pytest.fixture(scope="module")
def sandbox_defaults():
    """Module-scoped defaults for creating test sandboxes"""
    return SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=300,
        tags=("integration-test",),
    )
