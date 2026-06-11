# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for the shared transient-retry mechanism (cwsandbox._retry)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cwsandbox._retry import classify, is_retryable, retry_transient_async
from cwsandbox.exceptions import (
    CWSandboxError,
    DiscoveryError,
    SandboxCommandTimeoutError,
    SandboxError,
    SandboxNotFoundError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxUnavailableError,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the retry backoff instant so tests run on mock timing alone."""
    monkeypatch.setattr("cwsandbox._retry.asyncio.sleep", AsyncMock())


class TestClassify:
    """The single classification point: transient-vs-fatal by exception class."""

    @pytest.mark.parametrize(
        "exc",
        [
            SandboxUnavailableError("x"),
            SandboxRequestTimeoutError("x"),
            SandboxResourceExhaustedError("x"),
        ],
    )
    def test_retryable(self, exc: CWSandboxError) -> None:
        assert classify(exc) == "retryable"
        assert is_retryable(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            SandboxNotFoundError("x"),
            SandboxCommandTimeoutError("x"),  # command's own timeout: not transient
            SandboxError("x"),
            DiscoveryError("x"),  # generic discovery error (e.g. non-UNAVAILABLE)
            CWSandboxError("x"),
        ],
    )
    def test_fatal(self, exc: CWSandboxError) -> None:
        assert classify(exc) == "fatal"
        assert is_retryable(exc) is False


class TestRetryTransientAsync:
    """The shared bounded-retry backoff core."""

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        attempts = 0

        async def attempt(_rpc_timeout: float | None) -> str:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise SandboxUnavailableError("transient")
            return "ok"

        result = await retry_transient_async(
            attempt, budget_seconds=30.0, rpc_timeout_seconds=5.0, label="t"
        )
        assert result == "ok"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_budget_zero_single_attempt(self) -> None:
        attempts = 0

        async def attempt(_rpc_timeout: float | None) -> str:
            nonlocal attempts
            attempts += 1
            raise SandboxUnavailableError("transient")

        with pytest.raises(SandboxUnavailableError):
            await retry_transient_async(
                attempt, budget_seconds=0.0, rpc_timeout_seconds=5.0, label="t"
            )
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_fatal_not_retried(self) -> None:
        attempts = 0

        async def attempt(_rpc_timeout: float | None) -> str:
            nonlocal attempts
            attempts += 1
            raise SandboxNotFoundError("gone")

        with pytest.raises(SandboxNotFoundError):
            await retry_transient_async(
                attempt, budget_seconds=30.0, rpc_timeout_seconds=5.0, label="t"
            )
        assert attempts == 1  # fatal: no retry

    @pytest.mark.asyncio
    async def test_should_retry_override_excludes_a_code(self) -> None:
        """A narrower should_retry can exclude a normally-retryable type
        (e.g. read_file excludes RESOURCE_EXHAUSTED for its streaming fallback)."""
        attempts = 0

        async def attempt(_rpc_timeout: float | None) -> str:
            nonlocal attempts
            attempts += 1
            raise SandboxResourceExhaustedError("oversized")

        with pytest.raises(SandboxResourceExhaustedError):
            await retry_transient_async(
                attempt,
                budget_seconds=30.0,
                rpc_timeout_seconds=5.0,
                label="t",
                should_retry=lambda exc: is_retryable(exc)
                and not isinstance(exc, SandboxResourceExhaustedError),
            )
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_clamp_passes_budget_bounded_timeout(self) -> None:
        """With clamping on, the first attempt's rpc timeout is min(rpc, budget)."""
        seen: list[float | None] = []

        async def attempt(rpc_timeout: float | None) -> str:
            seen.append(rpc_timeout)
            return "ok"

        await retry_transient_async(
            attempt, budget_seconds=2.0, rpc_timeout_seconds=15.0, label="t"
        )
        assert seen == [2.0]  # clamped to the smaller budget

    @pytest.mark.asyncio
    async def test_no_clamp_passes_full_timeout(self) -> None:
        seen: list[float | None] = []

        async def attempt(rpc_timeout: float | None) -> str:
            seen.append(rpc_timeout)
            return "ok"

        await retry_transient_async(
            attempt,
            budget_seconds=2.0,
            rpc_timeout_seconds=15.0,
            label="t",
            clamp_rpc_timeout_to_budget=False,
        )
        assert seen == [15.0]
