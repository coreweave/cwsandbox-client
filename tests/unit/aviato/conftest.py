"""Shared fixtures for aviato unit tests."""

import os
from collections.abc import Generator

import pytest


@pytest.fixture
def mock_aviato_api_key() -> Generator[str, None, None]:
    """Set a mock AVIATO_API_KEY for test, restore original after."""
    original = os.environ.get("AVIATO_API_KEY")
    test_key = "test-api-key"
    os.environ["AVIATO_API_KEY"] = test_key

    yield test_key

    if original is not None:
        os.environ["AVIATO_API_KEY"] = original
    else:
        os.environ.pop("AVIATO_API_KEY", None)


@pytest.fixture
def mock_aviato_base_url() -> Generator[str, None, None]:
    """Set a mock AVIATO_BASE_URL for test, restore original after."""
    original = os.environ.get("AVIATO_BASE_URL")
    test_url = "http://test-api.example.com"
    os.environ["AVIATO_BASE_URL"] = test_url

    yield test_url

    if original is not None:
        os.environ["AVIATO_BASE_URL"] = original
    else:
        os.environ.pop("AVIATO_BASE_URL", None)
