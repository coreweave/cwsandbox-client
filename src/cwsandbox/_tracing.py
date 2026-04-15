# SPDX-FileCopyrightText: 2024 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optional distributed tracing support for cwsandbox.

Call ``configure_tracing()`` once per process before creating a Session.
When tracing is not configured, all helpers are safe no-ops.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_configured = False
_lock = threading.Lock()

# Lazy imports — only resolved when configure_tracing() is called.
_tracer: Any = None  # opentelemetry.trace.Tracer | None
_propagator: Any = None  # opentelemetry.context.propagation.TextMapPropagator | None


def configure_tracing(
    *,
    endpoint: str,
    headers: dict[str, str] | None = None,
    service_name: str = "cwsandbox-client",
) -> None:
    """One-time, idempotent process-global tracing setup.

    Args:
        endpoint: OTLP HTTP endpoint (e.g. ``https://jaeger-otlp.example.com``).
        headers: Optional HTTP headers for auth (e.g. ``{"Authorization": "Basic ..."}``)
        service_name: ``service.name`` resource attribute.
    """
    global _configured, _tracer, _propagator  # noqa: PLW0603

    with _lock:
        if _configured:
            logger.debug("Tracing already configured, skipping")
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.propagate import get_global_textmap
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": service_name})
            exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(
                exporter,
                max_queue_size=65536,
                max_export_batch_size=4096,
            ))
            trace.set_tracer_provider(provider)

            _tracer = trace.get_tracer("cwsandbox")
            _propagator = get_global_textmap()
            _configured = True
            logger.info("Distributed tracing configured: endpoint=%s service=%s", endpoint, service_name)

        except ImportError as exc:
            logger.warning(
                "opentelemetry packages not installed (%s). "
                "Install with: pip install cwsandbox[tracing]",
                exc,
            )
        except Exception:
            logger.exception("Failed to configure tracing")


def get_tracer() -> Any:
    """Return the configured tracer, or a no-op proxy if tracing is not set up."""
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        return trace.get_tracer("cwsandbox")
    except ImportError:
        return _NoOpTracer()


def inject_trace_context(
    metadata: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Merge current span's trace context into gRPC metadata.

    Replaces any existing ``traceparent``/``tracestate`` entries to avoid
    duplicates, then appends the current context's headers.
    """
    if _propagator is None:
        return metadata

    # Collect current trace context headers.
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    if not carrier:
        return metadata

    # Strip old trace headers, append new ones.
    trace_keys = frozenset(carrier.keys())
    filtered = tuple((k, v) for k, v in metadata if k not in trace_keys)
    return filtered + tuple(carrier.items())


class _NoOpTracer:
    """Fallback when opentelemetry is not installed."""

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Any:
        return _NoOpContextManager()


class _NoOpContextManager:
    def __enter__(self) -> "_NoOpContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any) -> None:
        pass
