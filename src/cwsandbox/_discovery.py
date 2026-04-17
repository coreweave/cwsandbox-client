# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Discovery types and gRPC client for runner and profile introspection."""

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
from cwsandbox._error_info import (
    CWSANDBOX_PROFILE_NOT_FOUND,
    CWSANDBOX_RUNNER_NOT_FOUND,
    is_not_found,
    parse_error_info,
)
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import create_channel, parse_grpc_target, translate_grpc_error
from cwsandbox._proto import discovery_pb2, discovery_pb2_grpc
from cwsandbox.exceptions import (
    CWSandboxError,
    DiscoveryError,
    ProfileNotFoundError,
    RunnerNotFoundError,
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
class ServiceExposureMode:
    """A service exposure mode supported by a profile.

    Attributes:
        name: The service exposure mode identifier (e.g. ``"public"``).
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class EgressMode:
    """An egress mode supported by a profile.

    Attributes:
        name: The egress mode identifier (e.g. ``"internet"``).
    """

    name: str


@dataclass(frozen=True, kw_only=True)
class RunnerResources:
    """Live resource availability for a runner.

    Attributes:
        available_cpu_millicores: Unreserved CPU capacity in millicores.
        available_memory_bytes: Unreserved memory in bytes.
        available_gpu_count: Unreserved GPU count.
        running_sandboxes: Number of sandboxes currently running on the runner.
    """

    available_cpu_millicores: int
    available_memory_bytes: int
    available_gpu_count: int
    running_sandboxes: int


@dataclass(frozen=True, kw_only=True)
class Runner:
    """A runner registered with the discovery service.

    Runners are the compute nodes that run sandboxes. Each runner advertises
    its capabilities (CPU, memory, GPU) and the profiles it supports.

    Attributes:
        runner_id: Unique identifier for the runner.
        runner_group_id: Group this runner belongs to.
        tags: Tags associated with the runner.
        healthy: Whether the runner is currently healthy.
        profile_names: Names of profiles available on this runner.
        connected_at: When the runner connected, as a UTC-aware datetime.
        max_cpu_millicores: Maximum CPU capacity in millicores.
        max_memory_bytes: Maximum memory capacity in bytes.
        max_gpu_count: Maximum GPU count.
        supported_gpu_types: GPU types supported by this runner.
        supported_architectures: CPU architectures supported (e.g. ``"amd64"``).
        supports_privileged: Whether privileged containers are allowed.
        available_storage_classes: Kubernetes storage classes available.
        resources: Live resource availability, or None if not reported.
    """

    runner_id: str
    runner_group_id: str
    tags: tuple[str, ...]
    healthy: bool
    profile_names: tuple[str, ...]
    connected_at: datetime
    max_cpu_millicores: int
    max_memory_bytes: int
    max_gpu_count: int
    supported_gpu_types: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    supports_privileged: bool
    available_storage_classes: tuple[str, ...]
    resources: RunnerResources | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "tags",
            "profile_names",
            "supported_gpu_types",
            "supported_architectures",
            "available_storage_classes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))

    def __repr__(self) -> str:
        profiles = list(self.profile_names)
        return (
            f"Runner("
            f"runner_id={self.runner_id!r}, "
            f"healthy={self.healthy!r}, "
            f"cpu={format_cpu(self.max_cpu_millicores)}, "
            f"memory={format_bytes(self.max_memory_bytes)}, "
            f"gpus={self.max_gpu_count}, "
            f"profiles={profiles!r})"
        )


@dataclass(frozen=True, kw_only=True)
class Profile:
    """A profile configuration on a specific runner.

    Profiles define execution environments with specific service exposure/egress
    modes and hardware capabilities.

    Attributes:
        profile_name: Name of the profile.
        runner_id: Runner this profile belongs to.
        supported_gpu_types: GPU types available on this profile.
        supported_architectures: CPU architectures supported.
        service_exposure_modes: Service exposure modes this profile supports.
        egress_modes: Egress modes this profile supports.
    """

    profile_name: str
    runner_id: str
    supported_gpu_types: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    service_exposure_modes: tuple[ServiceExposureMode, ...]
    egress_modes: tuple[EgressMode, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "supported_gpu_types",
            "supported_architectures",
            "service_exposure_modes",
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
        rpc_method: Bound stub method (e.g. ``stub.ListAvailableRunners``).
        request: The protobuf request message. Its ``page_token`` field is
            mutated in-place between pages.
        items_field: Name of the repeated field on the response that holds
            the result items (e.g. ``"runners"``).
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
        resources = RunnerResources(
            available_cpu_millicores=res.available_cpu_millicores,
            available_memory_bytes=res.available_memory_bytes,
            available_gpu_count=res.available_gpu_count,
            running_sandboxes=res.running_sandboxes,
        )

    return Runner(
        runner_id=proto.runner_id,
        runner_group_id=proto.runner_group_id,
        tags=tuple(proto.tags),
        healthy=proto.healthy,
        profile_names=tuple(proto.profile_names),
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


def _profile_from_proto(proto: discovery_pb2.ProfileSummary) -> Profile:
    """Convert a ``ProfileSummary`` proto to a ``Profile`` dataclass."""
    return Profile(
        profile_name=proto.profile_name,
        runner_id=proto.runner_id,
        supported_gpu_types=tuple(proto.supported_gpu_types),
        supported_architectures=tuple(proto.supported_architectures),
        service_exposure_modes=tuple(
            ServiceExposureMode(name=m.name) for m in proto.service_exposure_modes
        ),
        egress_modes=tuple(EgressMode(name=m.name) for m in proto.egress_modes),
    )


# ---------------------------------------------------------------------------
# Public discovery functions
# ---------------------------------------------------------------------------


async def _list_runners_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    view: discovery_pb2.RunnerView,
    runner_group_id: str | None = None,
    profile_name: str | None = None,
    gpu_type: str | None = None,
    architecture: str | None = None,
    min_available_cpu_millicores: int | None = None,
    min_available_memory_bytes: int | None = None,
    min_available_gpu_count: int | None = None,
    service_exposure_mode: str | None = None,
    egress_mode: str | None = None,
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
        )
        if runner_group_id is not None:
            request.runner_group_id = runner_group_id
        if profile_name is not None:
            request.profile_name = profile_name
        if gpu_type is not None:
            request.gpu_type = gpu_type
        if architecture is not None:
            request.architecture = architecture

        protos = await _paginate_async(
            stub.ListAvailableRunners,
            request,
            "runners",
            metadata,
            deadline - time.monotonic(),
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

        if service_exposure_mode is not None or egress_mode is not None:
            if not results:
                return results
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiscoveryError("Discovery request timed out before profile fetch")
            all_profiles = await _list_profiles_async(base_url, metadata, remaining)
            runner_exposure: dict[str, set[str]] = {}
            runner_egress: dict[str, set[str]] = {}
            for p in all_profiles:
                runner_exposure.setdefault(p.runner_id, set()).update(
                    m.name for m in p.service_exposure_modes
                )
                runner_egress.setdefault(p.runner_id, set()).update(m.name for m in p.egress_modes)
            if service_exposure_mode is not None:
                results = [
                    r
                    for r in results
                    if service_exposure_mode in runner_exposure.get(r.runner_id, set())
                ]
            if egress_mode is not None:
                results = [
                    r for r in results if egress_mode in runner_egress.get(r.runner_id, set())
                ]

        return results
    except grpc.aio.AioRpcError as e:
        raise translate_grpc_error(e, operation="List runners", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def list_runners(
    *,
    runner_group_id: str | None = None,
    profile_name: str | None = None,
    gpu_type: str | None = None,
    architecture: str | None = None,
    include_resources: bool = False,
    min_available_cpu_millicores: int | None = None,
    min_available_memory_bytes: int | None = None,
    min_available_gpu_count: int | None = None,
    service_exposure_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Runner]:
    """List available runners, optionally filtered.

    Creates a gRPC channel, issues the RPC(s), and closes the channel before
    returning.  Automatically paginates when the server returns a
    ``next_page_token``.

    Args:
        runner_group_id: Restrict results to this runner group.
        profile_name: Only return runners that have this profile.
        gpu_type: Only return runners that support this GPU type.
        architecture: Only return runners that support this CPU architecture.
        include_resources: If ``True``, include live resource availability
            on each runner.  Defaults to ``False`` (basic view).
        min_available_cpu_millicores: Only return runners with at least this
            many unreserved CPU millicores.  Automatically enables
            ``include_resources``.  Filtered client-side.
        min_available_memory_bytes: Only return runners with at least this
            many unreserved memory bytes.  Automatically enables
            ``include_resources``.  Filtered client-side.
        min_available_gpu_count: Only return runners with at least this many
            unreserved GPUs.  Automatically enables ``include_resources``.
            Filtered client-side.
        service_exposure_mode: Only return runners whose profiles support this
            service exposure mode.  Requires an additional profile fetch.
            Filtered client-side.
        egress_mode: Only return runners whose profiles support this egress
            mode.  Requires an additional profile fetch.  Filtered
            client-side.

    Note:
        The ``service_exposure_mode`` and ``egress_mode`` filters check across
        **all** profiles on each runner, not only the profile specified by
        ``profile_name``.  A runner will match if *any* of its profiles
        provides the requested mode, even if the profile selected by
        ``profile_name`` does not.

    Returns:
        List of ``Runner`` objects matching the filters.

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
                profile_name=profile_name,
                gpu_type=gpu_type,
                architecture=architecture,
                min_available_cpu_millicores=min_available_cpu_millicores,
                min_available_memory_bytes=min_available_memory_bytes,
                min_available_gpu_count=min_available_gpu_count,
                service_exposure_mode=service_exposure_mode,
                egress_mode=egress_mode,
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
) -> Runner:
    """Async implementation of :func:`get_runner`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.GetAvailableRunnerRequest(
            runner_id=runner_id,
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


def get_runner(runner_id: str) -> Runner:
    """Get a single runner by ID.

    Always returns full details including resource availability.

    Args:
        runner_id: Unique identifier of the runner.

    Returns:
        ``Runner`` with full details.

    Raises:
        ValueError: If *runner_id* is empty.
        RunnerNotFoundError: If no runner exists with the given ID.
        CWSandboxAuthenticationError: If credentials are invalid.
        CWSandboxError: On network or service errors.
    """
    if not runner_id or not runner_id.strip():
        raise ValueError("runner_id must not be empty")

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata()
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS

    return (
        _LoopManager.get()
        .run_async(_get_runner_async(base_url, metadata, timeout, runner_id=runner_id))
        .result()
    )


async def _list_profiles_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    gpu_type: str | None = None,
    architecture: str | None = None,
    runner_id: str | None = None,
    service_exposure_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Profile]:
    """Async implementation of :func:`list_profiles`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.ListProfilesRequest(page_size=100)
        if gpu_type is not None:
            request.gpu_type = gpu_type
        if architecture is not None:
            request.architecture = architecture
        if runner_id is not None:
            request.runner_id = runner_id

        protos = await _paginate_async(
            stub.ListProfiles,
            request,
            "profiles",
            metadata,
            timeout,
        )
        results = [_profile_from_proto(p) for p in protos]
        if service_exposure_mode is not None:
            results = [
                p
                for p in results
                if any(m.name == service_exposure_mode for m in p.service_exposure_modes)
            ]
        if egress_mode is not None:
            results = [p for p in results if any(m.name == egress_mode for m in p.egress_modes)]
        return results
    except grpc.aio.AioRpcError as e:
        raise translate_grpc_error(e, operation="List profiles", fallback_cls=DiscoveryError) from e
    finally:
        await channel.close(grace=None)


def list_profiles(
    *,
    gpu_type: str | None = None,
    architecture: str | None = None,
    runner_id: str | None = None,
    service_exposure_mode: str | None = None,
    egress_mode: str | None = None,
) -> list[Profile]:
    """List available profiles, optionally filtered.

    Creates a gRPC channel, issues the RPC(s), and closes the channel before
    returning.  Automatically paginates when the server returns a
    ``next_page_token``.

    Args:
        gpu_type: Only return profiles that support this GPU type.
        architecture: Only return profiles that support this CPU architecture.
        runner_id: Only return profiles belonging to this runner.
        service_exposure_mode: Only return profiles that support this service
            exposure mode. Filtered client-side after fetching results from
            the backend.
        egress_mode: Only return profiles that support this egress mode.
            Filtered client-side after fetching results from the backend.

    Returns:
        List of ``Profile`` objects matching the filters.

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
            _list_profiles_async(
                base_url,
                metadata,
                timeout,
                gpu_type=gpu_type,
                architecture=architecture,
                runner_id=runner_id,
                service_exposure_mode=service_exposure_mode,
                egress_mode=egress_mode,
            )
        )
        .result()
    )


async def _get_profile_async(
    base_url: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    profile_name: str,
    runner_id: str | None = None,
) -> Profile:
    """Async implementation of :func:`get_profile`."""
    target, is_secure = parse_grpc_target(base_url)
    channel = create_channel(target, is_secure)
    try:
        stub = discovery_pb2_grpc.DiscoveryServiceStub(channel)  # type: ignore[no-untyped-call]
        request = discovery_pb2.GetProfileRequest(profile_name=profile_name)
        if runner_id is not None:
            request.runner_id = runner_id
        proto = await stub.GetProfile(request, metadata=metadata, timeout=timeout)
        return _profile_from_proto(proto)
    except grpc.aio.AioRpcError as e:
        parsed = parse_error_info(e)
        if is_not_found(e, parsed, CWSANDBOX_PROFILE_NOT_FOUND):
            raise ProfileNotFoundError(
                f"Profile not found: {profile_name!r}",
                profile_name=profile_name,
                runner_id=runner_id,
                reason=parsed.reason if parsed is not None else None,
                metadata=parsed.metadata if parsed is not None else None,
                retry_delay=parsed.retry_delay if parsed is not None else None,
            ) from e
        raise translate_grpc_error(
            e, operation="Get profile", fallback_cls=DiscoveryError, parsed=parsed
        ) from e
    finally:
        await channel.close(grace=None)


def get_profile(profile_name: str, *, runner_id: str | None = None) -> Profile:
    """Get a single profile by name.

    Args:
        profile_name: Name of the profile to retrieve.
        runner_id: Optionally scope the lookup to a specific runner.

    Returns:
        ``Profile`` matching the given name.

    Raises:
        ValueError: If *profile_name* is empty.
        ProfileNotFoundError: If no matching profile is found.
        CWSandboxAuthenticationError: If credentials are invalid.
        CWSandboxError: On network or service errors.
    """
    if not profile_name or not profile_name.strip():
        raise ValueError("profile_name must not be empty")

    base_url = os.environ.get("CWSANDBOX_BASE_URL", DEFAULT_BASE_URL)
    metadata = resolve_auth_metadata()
    timeout = DEFAULT_DISCOVERY_TIMEOUT_SECONDS

    return (
        _LoopManager.get()
        .run_async(
            _get_profile_async(
                base_url, metadata, timeout, profile_name=profile_name, runner_id=runner_id
            )
        )
        .result()
    )
