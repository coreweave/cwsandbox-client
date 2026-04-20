# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

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

from cwsandbox import Sandbox, SandboxDefaults, list_runners
from cwsandbox._auth import resolve_auth
from cwsandbox.exceptions import CWSandboxError

if TYPE_CHECKING:
    pass


_ENV_RUNNER_IDS = "CWSANDBOX_TEST_RUNNER_IDS"
_CLI_RUNNER_IDS = "--cwsandbox-runner-ids"


def _parse_runner_ids(raw: str) -> tuple[str, ...] | None:
    """Parse a comma-separated runner ID string.

    Contract: split on ``,``, strip whitespace, drop empty tokens, dedupe
    preserving first-seen order. Returns ``None`` if nothing remains.
    """
    seen: set[str] = set()
    result: list[str] = []
    for token in raw.split(","):
        trimmed = token.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        result.append(trimmed)
    return tuple(result) if result else None


def _resolve_runner_ids(
    cli_value: str | None, env_value: str | None
) -> tuple[tuple[str, ...] | None, str | None]:
    """Resolve configured runner IDs. CLI wins; empty CLI clears env.

    ``cli_value`` is ``None`` when the flag was not passed, ``""`` when
    passed explicitly empty (clears), or a string value. Returns
    ``(runner_ids, source_label)`` where ``source_label`` is ``None`` when
    ``runner_ids`` is ``None``.
    """
    if cli_value is not None:
        ids = _parse_runner_ids(cli_value)
        source = f"CLI ({_CLI_RUNNER_IDS})" if ids is not None else None
        return ids, source
    if env_value is not None:
        ids = _parse_runner_ids(env_value)
        source = f"env ({_ENV_RUNNER_IDS})" if ids is not None else None
        return ids, source
    return None, None


@pytest.fixture(scope="session", autouse=True)
def _validate_runner_ids(pytestconfig: pytest.Config) -> None:
    """Fail fast when configured runner IDs are unknown.

    Only performs a discovery call when ``runner_ids`` resolves to a
    non-empty tuple; the default path (no flag, no env) is zero-cost. Also
    skips validation when no auth is configured, since ``require_auth``
    will skip the tests anyway.
    """
    cli_value = pytestconfig.getoption(_CLI_RUNNER_IDS)
    env_value = os.environ.get(_ENV_RUNNER_IDS)
    runner_ids, source_label = _resolve_runner_ids(cli_value, env_value)

    if runner_ids is None:
        return

    if resolve_auth().strategy == "none":
        return

    try:
        runners = list_runners()
    except CWSandboxError as exc:
        raise pytest.UsageError(
            f"Failed to verify runner IDs via discovery service: {exc}"
        ) from exc

    known_ids = {r.runner_id for r in runners}
    missing = [rid for rid in runner_ids if rid not in known_ids]
    if missing:
        available = ", ".join(sorted(known_ids)) if known_ids else "(none)"
        raise pytest.UsageError(
            f"Unknown runner ID(s) from {source_label}: {', '.join(missing)}. "
            f"Available runner IDs: {available}"
        )


@pytest.fixture(scope="module", autouse=True)
def require_auth() -> None:
    """Skip integration tests if no auth credentials are configured.

    This fixture validates authentication upfront rather than letting tests
    fail with opaque RPC errors. Runs automatically for all integration tests.
    """
    auth = resolve_auth()

    if auth.strategy == "none":
        pytest.skip(
            "Integration tests require authentication. Configure one of:\n"
            "  1. CWSANDBOX_API_KEY environment variable\n"
            "  2. .env file in project root (auto-loaded, see .env.example)"
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
                except CWSandboxError:
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
def sandbox_defaults(request: pytest.FixtureRequest) -> SandboxDefaults:
    """Module-scoped defaults for creating test sandboxes.

    Uses reduced resources and short lifetime for local Kind cluster
    compatibility and faster cleanup of orphaned sandboxes.

    Includes a unique session tag to identify sandboxes from this test run,
    enabling cleanup of orphaned sandboxes at session end.

    Respects ``CWSANDBOX_BASE_URL`` for pointing at local backends (e.g.,
    Tilt/Kind). Respects ``CWSANDBOX_TEST_RUNNER_IDS`` or
    ``--cwsandbox-runner-ids`` to pin sandboxes to specific runner(s) --
    CLI wins, empty CLI clears the env var. Only ``runner_ids`` is touched
    by this targeting; all other defaults are unchanged.
    """
    base_url = os.environ.get("CWSANDBOX_BASE_URL")
    cli_value = request.config.getoption(_CLI_RUNNER_IDS)
    env_value = os.environ.get(_ENV_RUNNER_IDS)
    runner_ids, _ = _resolve_runner_ids(cli_value, env_value)

    # Build kwargs, only including base_url if explicitly set
    kwargs: dict[str, object] = {
        "container_image": "python:3.11",
        "max_lifetime_seconds": 60,  # Short lifetime for faster cleanup
        "tags": ("integration-test", _SESSION_TAG),
        "resources": {"cpu": "500m", "memory": "256Mi"},
    }
    if base_url:
        kwargs["base_url"] = base_url
    if runner_ids is not None:
        kwargs["runner_ids"] = runner_ids

    return SandboxDefaults(**kwargs)  # type: ignore[arg-type]
