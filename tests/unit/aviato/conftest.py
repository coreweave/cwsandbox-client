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


@pytest.fixture
def mock_wandb_api_key() -> Generator[str, None, None]:
    """Set a mock WANDB_API_KEY for test, restore original after."""
    original = os.environ.get("WANDB_API_KEY")
    test_key = "test-wandb-api-key"
    os.environ["WANDB_API_KEY"] = test_key

    yield test_key

    if original is not None:
        os.environ["WANDB_API_KEY"] = original
    else:
        os.environ.pop("WANDB_API_KEY", None)


@pytest.fixture
def mock_wandb_entity_name() -> Generator[str, None, None]:
    """Set a mock WANDB_ENTITY_NAME for test, restore original after."""
    original = os.environ.get("WANDB_ENTITY_NAME")
    test_entity = "test-entity"
    os.environ["WANDB_ENTITY_NAME"] = test_entity

    yield test_entity

    if original is not None:
        os.environ["WANDB_ENTITY_NAME"] = original
    else:
        os.environ.pop("WANDB_ENTITY_NAME", None)
