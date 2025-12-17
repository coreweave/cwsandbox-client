"""Integration tests for W&B authentication.

These tests require:
1. A running Aviato backend that accepts W&B auth
2. Valid W&B credentials via WANDB_API_KEY and WANDB_ENTITY_NAME env vars

Set AVIATO_BASE_URL if not using the default endpoint.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aviato import Sandbox
from aviato._auth import WANDB_NETRC_HOST

# Skip all tests in this module if W&B credentials are not configured
pytestmark = pytest.mark.skipif(
    not (os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_ENTITY_NAME")),
    reason="WANDB_API_KEY and WANDB_ENTITY_NAME required for W&B auth integration tests",
)


@pytest.fixture
def wandb_env_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up W&B auth via environment variables."""
    monkeypatch.delenv("AVIATO_API_KEY", raising=False)


@pytest.fixture
def wandb_netrc_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set up W&B auth via netrc file.

    Uses the real WANDB_API_KEY value to create a temporary netrc file,
    then removes the env var so the SDK falls back to netrc.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        pytest.skip("WANDB_API_KEY required for netrc tests")

    # Create temp netrc file with the real credentials
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password {api_key}\n")

    # Remove AVIATO_API_KEY and WANDB_API_KEY so we fall back to netrc
    monkeypatch.delenv("AVIATO_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    # Mock Path.home to return our temp directory
    with patch("aviato._auth.Path.home", return_value=tmp_path):
        yield


@pytest.mark.asyncio
async def test_wandb_auth_via_env_vars(wandb_env_auth: None) -> None:
    """Test W&B authentication via environment variables."""
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        assert sandbox.sandbox_id is not None

        result = await sandbox.exec(["echo", "hello from wandb"])
        assert result.returncode == 0
        assert "hello from wandb" in result.stdout


@pytest.mark.asyncio
async def test_wandb_auth_via_netrc(wandb_netrc_auth: None) -> None:
    """Test W&B authentication via netrc file."""
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        assert sandbox.sandbox_id is not None

        result = await sandbox.exec(["echo", "hello from netrc"])
        assert result.returncode == 0
        assert "hello from netrc" in result.stdout
