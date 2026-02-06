# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Shared fixtures for integration tests.

Credentials must be set via environment variables before running tests.
See .env.example for configuration options. A .env file in the project root
is automatically loaded via python-dotenv.
"""

from __future__ import annotations

import asyncio
import os
import uuid
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

from aviato import Sandbox, SandboxDefaults
from aviato._auth import resolve_auth
from aviato.exceptions import AviatoError, WandbAuthError

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="module", autouse=True)
def require_auth(request: pytest.FixtureRequest) -> None:
    """Skip integration tests if no auth credentials are configured.

    This fixture validates authentication upfront rather than letting tests
    fail with opaque RPC errors. Runs automatically for all integration tests
    except test_auth.py which manages its own auth fixtures.

    Raises pytest.skip for:
    - No auth configured at all (strategy == "none")
    - Incomplete W&B auth (WANDB_API_KEY found but missing WANDB_ENTITY_NAME)
    """
    # Skip for test_auth.py which manages its own auth via fixtures
    if request.path.name == "test_auth.py":
        return

    try:
        auth = resolve_auth()
    except WandbAuthError as e:
        pytest.skip(f"W&B credentials incomplete: {e}\nSet WANDB_ENTITY_NAME environment variable.")

    if auth.strategy == "none":
        pytest.skip(
            "Integration tests require authentication. Configure one of:\n"
            "  1. AVIATO_API_KEY environment variable\n"
            "  2. WANDB_API_KEY + WANDB_ENTITY_NAME environment variables\n"
            "  3. ~/.netrc (api.wandb.ai) + WANDB_ENTITY_NAME\n"
            "  4. .env file in project root (auto-loaded, see .env.example)"
        )


# Generate unique session ID for this test run
_SESSION_ID = uuid.uuid4().hex[:8]
_SESSION_TAG = f"test-session-{_SESSION_ID}"


@pytest.fixture(scope="session", autouse=True)
def cleanup_session_sandboxes() -> Generator[None, None, None]:
    """Clean up any sandboxes created during this test session.

    Runs at session end to delete sandboxes that may have been orphaned
    due to test failures, timeouts, or interrupts. Uses the unique session
    tag to identify sandboxes from this specific test run.
    """
    yield  # Run tests first

    async def cleanup() -> None:
        try:
            # Find all sandboxes with this session's tag
            sandboxes = await Sandbox.list(tags=[_SESSION_TAG])
            if not sandboxes:
                return

            for sandbox in sandboxes:
                try:
                    if sandbox.sandbox_id:
                        await Sandbox.delete(sandbox.sandbox_id, missing_ok=True)
                except AviatoError:
                    # Sandbox may already be deleted or unreachable
                    pass
                except Exception as e:
                    warnings.warn(
                        f"Failed to cleanup sandbox {sandbox.sandbox_id}: {e}",
                        stacklevel=2,
                    )
        except Exception as e:
            warnings.warn(f"Sandbox cleanup failed: {e}", stacklevel=2)

    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(cleanup())
        finally:
            loop.close()
    except Exception as e:
        warnings.warn(f"Session cleanup failed to run: {e}", stacklevel=2)


@pytest.fixture(scope="module")
def sandbox_defaults() -> SandboxDefaults:
    """Module-scoped defaults for creating test sandboxes.

    Uses reduced resources and short lifetime for local Kind cluster
    compatibility and faster cleanup of orphaned sandboxes.

    Includes a unique session tag to identify sandboxes from this test run,
    enabling cleanup of orphaned sandboxes at session end.

    Respects AVIATO_BASE_URL environment variable for testing against
    local backends (e.g., Tilt/Kind development setup).
    """
    base_url = os.environ.get("AVIATO_BASE_URL")

    # Build kwargs, only including base_url if explicitly set
    kwargs: dict[str, object] = {
        "container_image": "python:3.11",
        "max_lifetime_seconds": 60,  # Short lifetime for faster cleanup
        "tags": ("integration-test", _SESSION_TAG),
        "resources": {"cpu": "500m", "memory": "256Mi"},
    }
    if base_url:
        kwargs["base_url"] = base_url

    return SandboxDefaults(**kwargs)  # type: ignore[arg-type]
