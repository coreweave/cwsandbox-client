# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Discovery types and gRPC client for tower and runway introspection."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import grpc
import grpc.aio

from cwsandbox._auth import resolve_auth_metadata
from cwsandbox._defaults import DEFAULT_BASE_URL, DEFAULT_DISCOVERY_TIMEOUT_SECONDS
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import create_channel, parse_grpc_target, translate_grpc_error
from cwsandbox._proto import discovery_pb2, discovery_pb2_grpc
from cwsandbox.exceptions import (
    CWSandboxError,
    DiscoveryError,
    RunwayNotFoundError,
    TowerNotFoundError,
)

logger = logging.getLogger(__name__)


def format_bytes(value: int) -> str:
    """Format bytes as a human-readable string using binary units.

    Args:
        value: Number of bytes.

    Returns:
        Human-readable string with appropriate unit.

    Examples:
        >>> format_bytes(17179869184)
        '16.0 GiB'
        >>> format_bytes(0)
        '0 B'
    """
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
    """Format CPU millicores as a human-readable string.

    Args:
        millicores: CPU capacity in millicores.

    Returns:
        Human-readable string in cores.

    Examples:
        >>> format_cpu(4000)
        '4.0 vCPU'
        >>> format_cpu(500)
        '0.5 vCPU'
    """
    return f"{millicores / 1000:.1f} vCPU"


@dataclass(frozen=True, kw_only=True)
class IngressMode:
    """An ingress mode supported by a runway.

    Attributes:
        name: The ingress mode identifier (e.g. ``"public"``).
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class EgressMode:
    """An egress mode supported by a runway.

    Attributes:
        name: The egress mode identifier (e.g. ``"internet"``).
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class TowerResources:
    """Live resource availability for a tower.

    Attributes:
        available_cpu_millicores: Unreserved CPU capacity in millicores.
        available_memory_bytes: Unreserved memory in bytes.
        available_gpu_count: Unreserved GPU count.
        running_sandboxes: Number of sandboxes currently running on the tower.
    """

    available_cpu_millicores: int
    available_memory_bytes: int
    available_gpu_count: int
    running_sandboxes: int


@dataclass(frozen=True, kw_only=True)
class Tower:
    """A tower registered with the discovery service.

    Towers are the compute nodes that run sandboxes. Each tower advertises
    its capabilities (CPU, memory, GPU) and the runways it supports.

    Attributes:
        tower_id: Unique identifier for the tower.
        tower_group_id: Group this tower belongs to.
        tags: Tags associated with the tower.
        healthy: Whether the tower is currently healthy.
        runway_names: Names of runways available on this tower.
        connected_at: When the tower connected, as a UTC-aware datetime.
        max_cpu_millicores: Maximum CPU capacity in millicores.
        max_memory_bytes: Maximum memory capacity in bytes.
        max_gpu_count: Maximum GPU count.
        supported_gpu_types: GPU types supported by this tower.
        supported_architectures: CPU architectures supported (e.g. ``"amd64"``).
        supports_privileged: Whether privileged containers are allowed.
        available_storage_classes: Kubernetes storage classes available.
        resources: Live resource availability, or None if not reported.
    """

    tower_id: str
    tower_group_id: str
    tags: tuple[str, ...]
    healthy: bool
    runway_names: tuple[str, ...]
    connected_at: datetime
    max_cpu_millicores: int
    max_memory_bytes: int
    max_gpu_count: int
    supported_gpu_types: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    supports_privileged: bool
    available_storage_classes: tuple[str, ...]
    resources: TowerResources | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "tags",
            "runway_names",
            "supported_gpu_types",
            "supported_architectures",
            "available_storage_classes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))

    def __repr__(self) -> str:
        runways = list(self.runway_names)
        return (
            f"Tower("
            f"tower_id={self.tower_id!r}, "
            f"healthy={self.healthy!r}, "
            f"cpu={format_cpu(self.max_cpu_millicores)}, "
            f"memory={format_bytes(self.max_memory_bytes)}, "
            f"gpus={self.max_gpu_count}, "
            f"runways={runways!r})"
        )


@dataclass(frozen=True, kw_only=True)
class Runway:
    """A runway configuration on a specific tower.

    Runways define execution environments with specific ingress/egress
    modes and hardware capabilities.

    Attributes:
        runway_name: Name of the runway.
        tower_id: Tower this runway belongs to.
        supported_gpu_types: GPU types available on this runway.
        supported_architectures: CPU architectures supported.
        ingress_modes: Ingress modes this runway supports.
        egress_modes: Egress modes this runway supports.
    """

    runway_name: str
    tower_id: str
    supported_gpu_types: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    ingress_modes: tuple[IngressMode, ...]
    egress_modes: tuple[EgressMode, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "supported_gpu_types",
            "supported_architectures",
            "ingress_modes",
            "egress_modes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _paginate_async(
    rpc_method: Any,
    request: Any,
    items_field: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
) -> list[Any]:
    """Auto-paginate a list RPC.

    Follows ``next_page_token`` until the server returns an empty token or
    the overall deadline is reached.

    Args:
        rpc_method: Bound stub method (e.g. ``stub.ListAvailableTowers``).
        request: The protobuf request message. Its ``page_token`` field is
            mutated in-place between pages.
        items_field: Name of the repeated field on the response that holds
            the result items (e.g. ``"towers"``).
        metadata: gRPC call metadata (auth headers).
        timeout: Total wall-clock seconds allowed for all pages.

    Returns:
        Flat list of proto items collected across all pages.

    Raises:
        CWSandboxError: On timeout, pagination loop, or exceeding page limit.
    """
    all_items: list[Any] = []
    deadline = time.monotonic() + timeout
    max_pages = 100
    seen_tokens: set[str] = set()

    for _ in range(max_pages):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CWSandboxError("Discovery request timed out during pagination")

        response = await rpc_method(request, metadata=metadata, timeout=remaining)
        items = getattr(response, items_field)
        all_items.extend(items)

        next_token = response.next_page_token
        if not next_token:
            break
        if next_token in seen_tokens:
            raise CWSandboxError("Discovery pagination loop detected: repeated page token")
        seen_tokens.add(next_token)
        request.page_token = next_token
    else:
        raise CWSandboxError(f"Discovery pagination exceeded {max_pages} pages")

    return all_items


# ---------------------------------------------------------------------------
# Proto-to-dataclass conversion
# ---------------------------------------------------------------------------


def _tower_from_proto(proto: discovery_pb2.AvailableTower) -> Tower:
    """Convert an ``AvailableTower`` proto to a ``Tower`` dataclass."""
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
    else:
        max_cpu_millicores = 0
        max_memory_bytes = 0
        max_gpu_count = 0
        supported_gpu_types = ()
        supported_architectures = ()
        supports_privileged = False
        available_storage_classes = ()

    resources = None
    if proto.HasField("resources"):
        res = proto.resources
        resources = TowerResources(
            available_cpu_millicores=res.available_cpu_millicores,
            available_memory_bytes=res.available_memory_bytes,
            available_gpu_count=res.available_gpu_count,
            running_sandboxes=res.running_sandboxes,
        )

    return Tower(
        tower_id=proto.tower_id,
        tower_group_id=proto.tower_group_id,
        tags=tuple(proto.tags),
        healthy=proto.healthy,
        runway_names=tuple(proto.runway_names),
        connected_at=connected_at,
        max_cpu_millicores=max_cpu_millicores,
        max_memory_bytes=max_memory_bytes,
        max_gpu_count=max_gpu_count,
        supported_gpu_types=supported_gpu_types,
        supported_architectures=supported_architectures,
        supports_privileged=supports_privileged,
        available_storage_classes=available_storage_classes,
        resources=resources,
    )


def _runway_from_proto(proto: discovery_pb2.RunwaySummary) -> Runway:
    """Convert a ``RunwaySummary`` proto to a ``Runway`` dataclass."""
    return Runway(
        runway_name=proto.runway_name,
        tower_id=proto.tower_id,
        supported_gpu_types=tuple(proto.supported_gpu_types),
        supported_architectures=tuple(proto.supported_architectures),
        ingress_modes=tuple(IngressMode(name=m.name) for m in proto.service_exposure_modes),
        egress_modes=tuple(EgressMode(name=m.name) for m in proto.egress_modes),
    )


# ---------------------------------------------------------------------------
# Public discovery functions
# ---------------------------------------------------------------------------


async def _list_towers_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    view: discovery_pb2.TowerView,
    tower_group_id: str | None = None,
    runway_name: str | None = None,
    gpu_type: str | None = None,
    architecture: str | None = None,
    min_available_cpu_millicores: int | None = None,
    min_available_memory_bytes: int | None = None,
    min_available_gpu_count: int | None = None,
    ingress_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Tower]:
    """Async implementation of :func:`list_towers`."""
    deadline = time.monotonic() + timeout
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.ListAvailableTowersRequest(
            view=view,
            page_size=100,
        )
        if tower_group_id is not None:
            request.tower_group_id = tower_group_id
        if runway_name is not None:
            request.runway_name = runway_name
        if gpu_type is not None:
            request.gpu_type = gpu_type
        if architecture is not None:
            request.architecture = architecture

        protos = await _paginate_async(
            stub.ListAvailableTowers,
            request,
            "towers",
            metadata,
            deadline - time.monotonic(),
        )
        results = [_tower_from_proto(t) for t in protos]

        if min_available_cpu_millicores is not None:
            results = [
                t
                for t in results
                if t.resources is not None
                and t.resources.available_cpu_millicores >= min_available_cpu_millicores
            ]
        if min_available_memory_bytes is not None:
            results = [
                t
                for t in results
                if t.resources is not None
                and t.resources.available_memory_bytes >= min_available_memory_bytes
            ]
        if min_available_gpu_count is not None:
            results = [
                t
                for t in results
                if t.resources is not None
                and t.resources.available_gpu_count >= min_available_gpu_count
            ]

        if ingress_mode is not None or egress_mode is not None:
            if not results:
                return results
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiscoveryError("Discovery request timed out before runway fetch")
            all_runways = await _list_runways_async(base_url, metadata, remaining)
            tower_ingress: dict[str, set[str]] = {}
            tower_egress: dict[str, set[str]] = {}
            for rw in all_runways:
                tower_ingress.setdefault(rw.tower_id, set()).update(
                    m.name for m in rw.ingress_modes
                )
                tower_egress.setdefault(rw.tower_id, set()).update(m.name for m in rw.egress_modes)
            if ingress_mode is not None:
                results = [
                    t for t in results if ingress_mode in tower_ingress.get(t.tower_id, set())
                ]
            if egress_mode is not None:
                results = [t for t in results if egress_mode in tower_egress.get(t.tower_id, set())]

        return results
    except grpc.aio.AioRpcError as e:
        raise translate_grpc_error(e, operation="List towers", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def list_towers(
    *,
    tower_group_id: str | None = None,
    runway_name: str | None = None,
    gpu_type: str | None = None,
    architecture: str | None = None,
    include_resources: bool = False,
    min_available_cpu_millicores: int | None = None,
    min_available_memory_bytes: int | None = None,
    min_available_gpu_count: int | None = None,
    ingress_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Tower]:
    """List available towers, optionally filtered.

    Creates a gRPC channel, issues the RPC(s), and closes the channel before
    returning.  Automatically paginates when the server returns a
    ``next_page_token``.

    Args:
        tower_group_id: Restrict results to this tower group.
        runway_name: Only return towers that have this runway.
        gpu_type: Only return towers that support this GPU type.
        architecture: Only return towers that support this CPU architecture.
        include_resources: If ``True``, include live resource availability
            on each tower.  Defaults to ``False`` (basic view).
        min_available_cpu_millicores: Only return towers with at least this
            many unreserved CPU millicores.  Automatically enables
            ``include_resources``.  Filtered client-side.
        min_available_memory_bytes: Only return towers with at least this
            many unreserved memory bytes.  Automatically enables
            ``include_resources``.  Filtered client-side.
        min_available_gpu_count: Only return towers with at least this many
            unreserved GPUs.  Automatically enables ``include_resources``.
            Filtered client-side.
        ingress_mode: Only return towers whose runways support this ingress
            mode.  Requires an additional runway fetch.  Filtered
            client-side.
        egress_mode: Only return towers whose runways support this egress
            mode.  Requires an additional runway fetch.  Filtered
            client-side.

    Note:
        The ``ingress_mode`` and ``egress_mode`` filters check across **all**
        runways on each tower, not only the runway specified by
        ``runway_name``.  A tower will match if *any* of its runways provides
        the requested mode, even if the runway selected by ``runway_name``
        does not.

    Returns:
        List of ``Tower`` objects matching the filters.

    Raises:
        CWSandboxAuthenticationError: If credentials are invalid.
        CWSandboxError: On network or service errors.
    """
    if any(
        v is not None
        for v in (min_available_cpu_millicores, min_available_memory_bytes, min_available_gpu_count)
    ):
        include_resources = True

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata()
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS
    view = discovery_pb2.TOWER_VIEW_FULL if include_resources else discovery_pb2.TOWER_VIEW_BASIC

    return (
        _LoopManager.get()
        .run_async(
            _list_towers_async(
                base_url,
                metadata,
                timeout,
                view=view,
                tower_group_id=tower_group_id,
                runway_name=runway_name,
                gpu_type=gpu_type,
                architecture=architecture,
                min_available_cpu_millicores=min_available_cpu_millicores,
                min_available_memory_bytes=min_available_memory_bytes,
                min_available_gpu_count=min_available_gpu_count,
                ingress_mode=ingress_mode,
                egress_mode=egress_mode,
            )
        )
        .result()
    )


async def _get_tower_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    tower_id: str,
) -> Tower:
    """Async implementation of :func:`get_tower`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.GetAvailableTowerRequest(
            tower_id=tower_id,
            view=discovery_pb2.TOWER_VIEW_FULL,
        )
        proto = await stub.GetAvailableTower(request, metadata=metadata, timeout=timeout)
        return _tower_from_proto(proto)
    except grpc.aio.AioRpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise TowerNotFoundError(
                f"Tower not found: {tower_id!r}",
                tower_id=tower_id,
            ) from e
        raise translate_grpc_error(e, operation="Get tower", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def get_tower(tower_id: str) -> Tower:
    """Get a single tower by ID.

    Always returns full details including resource availability.

    Args:
        tower_id: Unique identifier of the tower.

    Returns:
        ``Tower`` with full details.

    Raises:
        ValueError: If *tower_id* is empty.
        TowerNotFoundError: If no tower exists with the given ID.
        CWSandboxAuthenticationError: If credentials are invalid.
        CWSandboxError: On network or service errors.
    """
    if not tower_id or not tower_id.strip():
        raise ValueError("tower_id must not be empty")

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata()
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS

    return (
        _LoopManager.get()
        .run_async(_get_tower_async(base_url, metadata, timeout, tower_id=tower_id))
        .result()
    )


async def _list_runways_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    gpu_type: str | None = None,
    architecture: str | None = None,
    tower_id: str | None = None,
    ingress_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Runway]:
    """Async implementation of :func:`list_runways`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.ListRunwaysRequest(page_size=100)
        if gpu_type is not None:
            request.gpu_type = gpu_type
        if architecture is not None:
            request.architecture = architecture
        if tower_id is not None:
            request.tower_id = tower_id

        protos = await _paginate_async(
            stub.ListRunways,
            request,
            "runways",
            metadata,
            timeout,
        )
        results = [_runway_from_proto(r) for r in protos]
        if ingress_mode is not None:
            results = [r for r in results if any(m.name == ingress_mode for m in r.ingress_modes)]
        if egress_mode is not None:
            results = [r for r in results if any(m.name == egress_mode for m in r.egress_modes)]
        return results
    except grpc.aio.AioRpcError as e:
        raise translate_grpc_error(e, operation="List runways", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def list_runways(
    *,
    gpu_type: str | None = None,
    architecture: str | None = None,
    tower_id: str | None = None,
    ingress_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Runway]:
    """List available runways, optionally filtered.

    Creates a gRPC channel, issues the RPC(s), and closes the channel before
    returning.  Automatically paginates when the server returns a
    ``next_page_token``.

    Args:
        gpu_type: Only return runways that support this GPU type.
        architecture: Only return runways that support this CPU architecture.
        tower_id: Only return runways belonging to this tower.
        ingress_mode: Only return runways that support this ingress mode.
            Filtered client-side after fetching results from the backend.
        egress_mode: Only return runways that support this egress mode.
            Filtered client-side after fetching results from the backend.

    Returns:
        List of ``Runway`` objects matching the filters.

    Raises:
        CWSandboxAuthenticationError: If credentials are invalid.
        CWSandboxError: On network or service errors.
    """
    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata()
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS

    return (
        _LoopManager.get()
        .run_async(
            _list_runways_async(
                base_url,
                metadata,
                timeout,
                gpu_type=gpu_type,
                architecture=architecture,
                tower_id=tower_id,
                ingress_mode=ingress_mode,
                egress_mode=egress_mode,
            )
        )
        .result()
    )


async def _get_runway_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    runway_name: str,
    tower_id: str | None = None,
) -> Runway:
    """Async implementation of :func:`get_runway`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.GetRunwayRequest(runway_name=runway_name)
        if tower_id is not None:
            request.tower_id = tower_id
        proto = await stub.GetRunway(request, metadata=metadata, timeout=timeout)
        return _runway_from_proto(proto)
    except grpc.aio.AioRpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise RunwayNotFoundError(
                f"Runway not found: {runway_name!r}",
                runway_name=runway_name,
                tower_id=tower_id,
            ) from e
        raise translate_grpc_error(e, operation="Get runway", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def get_runway(runway_name: str, *, tower_id: str | None = None) -> Runway:
    """Get a single runway by name.

    Args:
        runway_name: Name of the runway to retrieve.
        tower_id: Optionally scope the lookup to a specific tower.

    Returns:
        ``Runway`` matching the given name.

    Raises:
        ValueError: If *runway_name* is empty.
        RunwayNotFoundError: If no matching runway is found.
        CWSandboxAuthenticationError: If credentials are invalid.
        CWSandboxError: On network or service errors.
    """
    if not runway_name or not runway_name.strip():
        raise ValueError("runway_name must not be empty")

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata()
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS

    return (
        _LoopManager.get()
        .run_async(
            _get_runway_async(
                base_url, metadata, timeout, runway_name=runway_name, tower_id=tower_id
            )
        )
        .result()
    )
