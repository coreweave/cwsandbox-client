# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Discovery types and gRPC client for runner capability introspection."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import grpc
import grpc.aio

from cwsandbox._auth import AuthConfig, resolve_auth_metadata
from cwsandbox._defaults import DEFAULT_BASE_URL, DEFAULT_DISCOVERY_TIMEOUT_SECONDS
from cwsandbox._error_info import (
    CWSANDBOX_RUNNER_NOT_FOUND,
    is_not_found,
    parse_error_info,
)
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import (
    create_channel,
    paginate_async,
    parse_grpc_target,
    translate_grpc_error,
)
from cwsandbox._proto import discovery_pb2, discovery_pb2_grpc, sandbox_pb2
from cwsandbox.exceptions import DiscoveryError, RunnerNotFoundError

logger = logging.getLogger(__name__)


def format_bytes(value: int) -> str:
    """Format bytes as a human-readable string using binary units."""
    if value == 0:
        return "0 B"
    for unit, threshold in (
        ("TiB", 1 << 40),
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
    ):
        if value >= threshold:
            return f"{value / threshold:.1f} {unit}"
    return f"{value} B"


def format_cpu(millicores: int) -> str:
    """Format CPU millicores as a human-readable string."""
    return f"{millicores / 1000:.1f} vCPU"


@dataclass(frozen=True, kw_only=True)
class RunnerResources:
    """Live resource availability for a runner.

    Attributes:
        available_cpu_millicores: Unreserved CPU millicores.
        available_memory_bytes: Unreserved memory bytes.
        available_gpu_count: Unreserved GPU count.
        running_sandboxes: Number of sandboxes currently running.
    """

    available_cpu_millicores: int
    available_memory_bytes: int
    available_gpu_count: int
    running_sandboxes: int


@dataclass(frozen=True, kw_only=True)
class Runner:
    """A runner registered with the discovery service.

    Attributes:
        runner_id: Unique identifier for the runner within its organization.
        organization_id: Organization that owns the runner.
        runner_group_id: Group this runner belongs to.
        tags: Tags associated with the runner.
        healthy: Whether the runner is currently healthy.
        is_shared: True when the runner belongs to a shared organization.
        connected_at: When the runner connected, as a UTC-aware datetime.
        max_cpu_millicores: Maximum CPU capacity in millicores.
        max_memory_bytes: Maximum memory capacity in bytes.
        max_gpu_count: Maximum GPU count.
        supported_gpu_types: GPU types supported by this runner.
        supported_architectures: CPU architectures supported (e.g. ``"amd64"``).
        supports_privileged: Whether privileged containers are allowed.
        available_storage_classes: Kubernetes storage classes available.
        supported_service_visibilities: Typed service visibilities this runner
            can enforce (``public`` / ``private`` / ``custom``).
        resources: Live resource availability, or None if not reported.
    """

    runner_id: str
    organization_id: str
    runner_group_id: str
    tags: tuple[str, ...]
    healthy: bool
    is_shared: bool
    connected_at: datetime
    max_cpu_millicores: int
    max_memory_bytes: int
    max_gpu_count: int
    supported_gpu_types: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    supports_privileged: bool
    available_storage_classes: tuple[str, ...]
    supported_service_visibilities: tuple[str, ...]
    resources: RunnerResources | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "tags",
            "supported_gpu_types",
            "supported_architectures",
            "available_storage_classes",
            "supported_service_visibilities",
        ):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))

    def __repr__(self) -> str:
        return (
            f"Runner("
            f"runner_id={self.runner_id!r}, "
            f"organization_id={self.organization_id!r}, "
            f"healthy={self.healthy!r}, "
            f"cpu={format_cpu(self.max_cpu_millicores)}, "
            f"memory={format_bytes(self.max_memory_bytes)}, "
            f"gpus={self.max_gpu_count}, "
            f"visibilities={list(self.supported_service_visibilities)!r})"
        )


def _visibility_name(value: int) -> str:
    try:
        return sandbox_pb2.Visibility.Name(value).replace("VISIBILITY_", "").lower()
    except ValueError:
        return "unspecified"


def _runner_from_proto(proto: discovery_pb2.AvailableRunner) -> Runner:
    """Convert an ``AvailableRunner`` proto to a ``Runner`` dataclass."""
    connected_at = proto.connected_at.ToDatetime().replace(tzinfo=UTC)

    if proto.HasField("capabilities"):
        caps = proto.capabilities
        max_cpu_millicores = caps.max_cpu_millicores
        max_memory_bytes = caps.max_memory_bytes
        max_gpu_count = caps.max_gpu_count
        supported_gpu_types = tuple(caps.supported_gpu_types)
        supported_architectures = tuple(caps.supported_architectures)
        supports_privileged = caps.supports_privileged
        available_storage_classes = tuple(caps.available_storage_classes)
        supported_service_visibilities = tuple(
            _visibility_name(v) for v in caps.supported_service_visibilities
        )
    else:
        max_cpu_millicores = 0
        max_memory_bytes = 0
        max_gpu_count = 0
        supported_gpu_types = ()
        supported_architectures = ()
        supports_privileged = False
        available_storage_classes = ()
        supported_service_visibilities = ()

    resources = None
    if proto.HasField("resources"):
        res = proto.resources
        resources = RunnerResources(
            available_cpu_millicores=res.available_cpu_millicores,
            available_memory_bytes=res.available_memory_bytes,
            available_gpu_count=res.available_gpu_count,
            running_sandboxes=res.running_sandboxes,
        )

    return Runner(
        runner_id=proto.runner_id,
        organization_id=proto.organization_id,
        runner_group_id=proto.runner_group_id,
        tags=tuple(proto.tags),
        healthy=proto.healthy,
        is_shared=proto.is_shared,
        connected_at=connected_at,
        max_cpu_millicores=max_cpu_millicores,
        max_memory_bytes=max_memory_bytes,
        max_gpu_count=max_gpu_count,
        supported_gpu_types=supported_gpu_types,
        supported_architectures=supported_architectures,
        supports_privileged=supports_privileged,
        available_storage_classes=available_storage_classes,
        supported_service_visibilities=supported_service_visibilities,
        resources=resources,
    )


def _parse_service_visibility(value: str | None) -> sandbox_pb2.Visibility:
    if value is None:
        return sandbox_pb2.VISIBILITY_UNSPECIFIED
    name = value.strip().upper()
    if not name.startswith("VISIBILITY_"):
        name = f"VISIBILITY_{name}"
    try:
        return cast(sandbox_pb2.Visibility, sandbox_pb2.Visibility.Value(name))
    except ValueError as e:
        raise ValueError(f"unknown service_visibility: {value!r}") from e


async def _list_runners_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    view: discovery_pb2.RunnerView,
    runner_group_id: str | None = None,
    gpu_type: str | None = None,
    architecture: str | None = None,
    healthy_only: bool = False,
    min_available_cpu_millicores: int | None = None,
    min_available_memory_bytes: int | None = None,
    min_available_gpu_count: int | None = None,
    service_visibility: str | None = None,
) -> list[Runner]:
    """Async implementation of :func:`list_runners`."""
    deadline = time.monotonic() + timeout
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.ListAvailableRunnersRequest(
            view=view,
            page_size=100,
            healthy_only=healthy_only,
            service_visibility=_parse_service_visibility(service_visibility),
        )
        if runner_group_id is not None:
            request.runner_group_id = runner_group_id
        if gpu_type is not None:
            request.gpu_type = gpu_type
        if architecture is not None:
            request.architecture = architecture

        protos = await paginate_async(
            stub.ListAvailableRunners,
            request,
            "runners",
            metadata,
            deadline - time.monotonic(),
            operation="List runners",
        )
        results = [_runner_from_proto(r) for r in protos]

        if min_available_cpu_millicores is not None:
            results = [
                r
                for r in results
                if r.resources is not None
                and r.resources.available_cpu_millicores >= min_available_cpu_millicores
            ]
        if min_available_memory_bytes is not None:
            results = [
                r
                for r in results
                if r.resources is not None
                and r.resources.available_memory_bytes >= min_available_memory_bytes
            ]
        if min_available_gpu_count is not None:
            results = [
                r
                for r in results
                if r.resources is not None
                and r.resources.available_gpu_count >= min_available_gpu_count
            ]
        return results
    except grpc.aio.AioRpcError as e:
        raise translate_grpc_error(e, operation="List runners", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def list_runners(
    *,
    auth: AuthConfig | None = None,
    runner_group_id: str | None = None,
    gpu_type: str | None = None,
    architecture: str | None = None,
    healthy_only: bool = False,
    include_resources: bool = False,
    min_available_cpu_millicores: int | None = None,
    min_available_memory_bytes: int | None = None,
    min_available_gpu_count: int | None = None,
    service_visibility: str | None = None,
) -> list[Runner]:
    """List available runners, optionally filtered.

    Args:
        auth: Authentication strategy, resolved headers, or provider for this request.
        runner_group_id: Restrict results to this runner group.
        gpu_type: Only return runners that support this GPU type.
        architecture: Only return runners that support this CPU architecture.
        healthy_only: Only return healthy runners.
        include_resources: If ``True``, include live resource availability.
        min_available_cpu_millicores: Client-side filter on unreserved CPU.
        min_available_memory_bytes: Client-side filter on unreserved memory.
        min_available_gpu_count: Client-side filter on unreserved GPUs.
        service_visibility: Require a runner that can enforce this typed
            visibility (``public`` / ``private`` / ``custom``).

    Returns:
        List of ``Runner`` objects matching the filters.
    """
    if any(
        v is not None
        for v in (min_available_cpu_millicores, min_available_memory_bytes, min_available_gpu_count)
    ):
        include_resources = True

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata(auth, base_url=base_url)
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS
    view = discovery_pb2.RUNNER_VIEW_FULL if include_resources else discovery_pb2.RUNNER_VIEW_BASIC

    return (
        _LoopManager.get()
        .run_async(
            _list_runners_async(
                base_url,
                metadata,
                timeout,
                view=view,
                runner_group_id=runner_group_id,
                gpu_type=gpu_type,
                architecture=architecture,
                healthy_only=healthy_only,
                min_available_cpu_millicores=min_available_cpu_millicores,
                min_available_memory_bytes=min_available_memory_bytes,
                min_available_gpu_count=min_available_gpu_count,
                service_visibility=service_visibility,
            )
        )
        .result()
    )


async def _get_runner_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    runner_id: str,
    organization_id: str,
) -> Runner:
    """Async implementation of :func:`get_runner`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.GetAvailableRunnerRequest(
            runner_id=runner_id,
            organization_id=organization_id,
            view=discovery_pb2.RUNNER_VIEW_FULL,
        )
        proto = await stub.GetAvailableRunner(request, metadata=metadata, timeout=timeout)
        return _runner_from_proto(proto)
    except grpc.aio.AioRpcError as e:
        parsed = parse_error_info(e)
        if is_not_found(e, parsed, CWSANDBOX_RUNNER_NOT_FOUND):
            raise RunnerNotFoundError(
                f"Runner not found: {runner_id!r}",
                runner_id=runner_id,
                reason=parsed.reason if parsed is not None else None,
                metadata=parsed.metadata if parsed is not None else None,
                retry_delay=parsed.retry_delay if parsed is not None else None,
            ) from e
        raise translate_grpc_error(
            e, operation="Get runner", fallback_cls=DiscoveryError, parsed=parsed
        ) from e
    finally:
        await channel.close(grace=None)


def get_runner(
    runner_id: str,
    *,
    organization_id: str,
    auth: AuthConfig | None = None,
) -> Runner:
    """Get a single runner by ``(organization_id, runner_id)``.

    Always returns full details including resource availability when owned.

    Args:
        runner_id: Runner identifier (not globally unique alone).
        organization_id: Organization that owns the runner.
        auth: Authentication strategy, resolved headers, or provider for this request.

    Returns:
        ``Runner`` with full details.
    """
    if not runner_id or not runner_id.strip():
        raise ValueError("runner_id must not be empty")
    if not organization_id or not organization_id.strip():
        raise ValueError("organization_id must not be empty")

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata(auth, base_url=base_url)
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS

    return (
        _LoopManager.get()
        .run_async(
            _get_runner_async(
                base_url,
                metadata,
                timeout,
                runner_id=runner_id,
                organization_id=organization_id,
            )
        )
        .result()
    )
