"""Shared fixtures for aviato unit tests."""

import pytest

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
