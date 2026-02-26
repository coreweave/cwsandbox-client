# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for W&B authentication.

These tests require:
1. A running CWSandbox backend that accepts W&B auth
2. Valid W&B credentials (see individual test docstrings for requirements)

Set CWSANDBOX_BASE_URL if not using the default endpoint.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cwsandbox import Sandbox
from cwsandbox._auth import _read_api_key_from_netrc
from cwsandbox._defaults import WANDB_NETRC_HOST


@pytest.fixture
def wandb_env_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up W&B auth via environment variables.

    Requires WANDB_API_KEY and WANDB_ENTITY_NAME env vars to be set.
    Skips if credentials are not available.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    entity = os.environ.get("WANDB_ENTITY_NAME")

    if not api_key:
        pytest.skip("WANDB_API_KEY env var required for env var auth test")
    if not entity:
        pytest.skip("WANDB_ENTITY_NAME env var required for env var auth test")

    # Ensure we're not using API key auth
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)


@pytest.fixture
def wandb_netrc_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set up W&B auth via netrc file.

    Uses WANDB_API_KEY env var OR existing ~/.netrc credentials to create
    a temporary netrc file, then unsets the env var so the SDK falls back
    to reading from netrc.

    Requires WANDB_ENTITY_NAME env var to be set.
    Skips if no API key source is available.
    """
    # Get API key from env var or existing netrc
    api_key = os.environ.get("WANDB_API_KEY") or _read_api_key_from_netrc()
    entity = os.environ.get("WANDB_ENTITY_NAME")

    if not api_key:
        pytest.skip("WANDB_API_KEY env var or ~/.netrc credentials required for netrc auth test")
    if not entity:
        pytest.skip("WANDB_ENTITY_NAME env var required for netrc auth test")

    # Create temp netrc file with the credentials
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password {api_key}\n")

    # Remove CWSANDBOX_API_KEY and WANDB_API_KEY so we fall back to netrc
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    # Mock Path.home to return our temp directory
    with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
        yield


def test_wandb_auth_via_env_vars(wandb_env_auth: None) -> None:
    """Test W&B authentication via environment variables.

    Requires: WANDB_API_KEY and WANDB_ENTITY_NAME env vars.
    """
    with Sandbox.run("sleep", "infinity", container_image="python:3.11") as sandbox:
        assert sandbox.sandbox_id is not None

        result = sandbox.exec(["echo", "hello from wandb"]).result()
        assert result.returncode == 0
        assert "hello from wandb" in result.stdout


def test_wandb_auth_via_netrc(wandb_netrc_auth: None) -> None:
    """Test W&B authentication via netrc file.

    Requires: WANDB_ENTITY_NAME env var, plus either WANDB_API_KEY env var
    or existing ~/.netrc credentials for api.wandb.ai.
    """
    with Sandbox.run("sleep", "infinity", container_image="python:3.11") as sandbox:
        assert sandbox.sandbox_id is not None

        result = sandbox.exec(["echo", "hello from netrc"]).result()
        assert result.returncode == 0
        assert "hello from netrc" in result.stdout
