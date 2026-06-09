# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Shared transient-retry mechanism for idempotent RPCs.

This module is the single home for client-side retry of transient gRPC
failures (e.g. ``UNAVAILABLE`` during a gateway pod restart/rollout). It
provides:

- :func:`is_retryable` - the one classification point for transient-vs-fatal
  (dispatched on the translated exception class), and
- :func:`retry_transient_async` - a shared bounded-retry backoff core used by
  every idempotent RPC path instead of per-function retry loops.

Only idempotent operations should be wrapped (status polling, ``get_status``,
``list``, ``read_file``). Mutating/streaming RPCs (``start``, ``exec``) must NOT
be auto-retried - they are translated to typed exceptions by their call sites
but never re-driven, because retrying them can duplicate side effects.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

import grpc

from cwsandbox._defaults import (
    DEFAULT_MAX_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_BACKOFF_FACTOR,
    DEFAULT_POLL_INTERVAL_SECONDS,
)
from cwsandbox.exceptions import (
    CWSandboxError,
    SandboxNotFoundError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxUnavailableError,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Maximum time to honor for a server-hinted retry_delay (AIP-193 RetryInfo).
# Ensures one hinted sleep cannot consume the entire retry budget in a single
# sleep - the remaining budget is also a ceiling, so a misconfigured server
# emitting a large hint still only stalls a retry by at most
# min(hint, budget, MAX_RETRY_HINTED_DELAY_SECONDS).
MAX_RETRY_HINTED_DELAY_SECONDS: float = 10.0

# Translated exception classes that represent a transient condition safe to
# retry for an IDEMPOTENT operation.
_RETRYABLE_EXCEPTIONS: tuple[type[CWSandboxError], ...] = (
    SandboxUnavailableError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
)

# Classes that are always fatal even if they were to subclass a retryable
# type. SandboxNotFoundError is an authoritative "gone" signal and must never
# be retried regardless of the transport code that produced it.
_FATAL_EXCEPTIONS: tuple[type[CWSandboxError], ...] = (SandboxNotFoundError,)

_RetryClassification = Literal["retryable", "fatal"]


def classify(exc: CWSandboxError) -> _RetryClassification:
    """Classify a translated exception as ``"retryable"`` or ``"fatal"``.

    The single classification point shared by every retrying call site. Fatal
    overrides win first, then membership in the retryable registry; everything
    else is fatal.
    """
    if isinstance(exc, _FATAL_EXCEPTIONS):
        return "fatal"
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return "retryable"
    return "fatal"


def is_retryable(exc: CWSandboxError) -> bool:
    """Return True if ``exc`` is a transient error safe to retry (idempotent ops)."""
    return classify(exc) == "retryable"


async def retry_transient_async(
    attempt: Callable[[float | None], Awaitable[_T]],
    *,
    budget_seconds: float,
    rpc_timeout_seconds: float,
    label: str,
    sandbox_id: str | None = None,
    clamp_rpc_timeout_to_budget: bool = True,
    should_retry: Callable[[CWSandboxError], bool] = is_retryable,
) -> _T:
    """Run ``attempt(rpc_timeout_override)`` with bounded retry on transient errors.

    The wall-clock ``budget_seconds`` caps time spent retrying *after* the first transient
    failure (a healthy first attempt never consumes it) and resets after any
    success. AIP-193 ``RetryInfo`` hints (``exc.retry_delay``) are honored
    literally; otherwise AWS-style decorrelated jitter is used to avoid
    fleet-scale thundering herd during a regional outage or gateway rollout.

    ``attempt`` receives an rpc-timeout override (seconds, or ``None`` to use
    its own default) and MUST translate gRPC errors into ``CWSandboxError`` so
    :func:`classify` can dispatch on them. ``SandboxNotFoundError`` is fatal and
    is never retried.

    Set ``clamp_rpc_timeout_to_budget=False`` for calls whose per-attempt
    timeout must not shrink to the retry budget (e.g. a large file read or a
    paginated list that legitimately runs longer than the budget); the budget
    then only governs the spacing/number of retries, not a single attempt's
    deadline.

    ``should_retry`` defaults to the centralized :func:`is_retryable` classifier
    and should be left as-is for almost all call sites. A call site may pass a
    narrower predicate when a normally-retryable code carries a different,
    context-specific meaning - e.g. ``read_file`` excludes
    ``SandboxResourceExhaustedError`` because there it signals an oversized
    message that must fall back to streaming rather than be retried.

    Raises:
        CWSandboxError: The last translated exception once the budget is
            exhausted, or immediately for a fatal (non-retryable) error.
    """
    if clamp_rpc_timeout_to_budget and budget_seconds > 0:
        rpc_timeout_override: float | None = min(rpc_timeout_seconds, budget_seconds)
    elif clamp_rpc_timeout_to_budget:
        rpc_timeout_override = None
    else:
        rpc_timeout_override = rpc_timeout_seconds

    retry_deadline: float | None = None
    last_exc: CWSandboxError | None = None
    prev_sleep = DEFAULT_POLL_INTERVAL_SECONDS
    attempts = 0

    while True:
        try:
            return await attempt(rpc_timeout_override)
        except CWSandboxError as exc:
            last_exc = exc
            if not should_retry(exc) or budget_seconds <= 0:
                raise
            # First retryable failure: arm the deadline timer. Healthy first
            # attempts must not consume the budget.
            if retry_deadline is None:
                retry_deadline = time.monotonic() + budget_seconds
            attempts += 1
            now = time.monotonic()
            if now >= retry_deadline:
                logger.debug(
                    "%s retry budget exhausted for sandbox %s after %d attempt(s)",
                    label,
                    sandbox_id,
                    attempts,
                )
                raise
            remaining = retry_deadline - now
            hinted_delay = exc.retry_delay.total_seconds() if exc.retry_delay else None
            if hinted_delay is not None and hinted_delay > 0:
                sleep_for = min(hinted_delay, remaining, MAX_RETRY_HINTED_DELAY_SECONDS)
                source = "hinted"
            else:
                base = DEFAULT_POLL_INTERVAL_SECONDS
                cap = DEFAULT_MAX_POLL_INTERVAL_SECONDS
                jitter_ceiling = max(
                    base, min(cap, prev_sleep * DEFAULT_POLL_BACKOFF_FACTOR, remaining)
                )
                sleep_for = min(random.uniform(base, jitter_ceiling), remaining)
                source = "computed-jittered"
            cause = exc.__cause__ if isinstance(exc.__cause__, grpc.RpcError) else None
            code = cause.code() if cause is not None else None
            logger.debug(
                "%s retry for sandbox %s: code=%s sleep=%.2fs source=%s remaining=%.2fs",
                label,
                sandbox_id,
                code,
                sleep_for,
                source,
                remaining,
            )
        await asyncio.sleep(sleep_for)
        prev_sleep = sleep_for
        # Re-check the deadline after sleeping: a long hinted delay can exhaust
        # the budget mid-sleep. Re-raise the last translated exception rather
        # than issuing an RPC that would overrun the ceiling.
        assert retry_deadline is not None
        now = time.monotonic()
        if now >= retry_deadline:
            assert last_exc is not None
            raise last_exc
        if clamp_rpc_timeout_to_budget:
            # Clamp the next RPC timeout to the remaining budget (floored at
            # 0.1s to avoid degenerate zero-timeout RPCs).
            rpc_timeout_override = min(rpc_timeout_seconds, max(0.1, retry_deadline - now))
