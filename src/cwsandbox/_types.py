# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from cwsandbox.exceptions import SandboxExecutionError

if TYPE_CHECKING:
    from cwsandbox._loop_manager import _LoopManager

T = TypeVar("T")
_S = TypeVar("_S")


class OperationRef(Generic[T]):
    """Generic ref for async operations with lazy result retrieval.

    OperationRef wraps a concurrent.futures.Future and provides a unified interface
    for both synchronous and asynchronous result retrieval. This enables the
    sync/async hybrid API where operations return immediately and results are
    retrieved lazily.

    Type Parameters:
        T: The type of the result this operation will return.

    Examples:
        Synchronous usage:
        ```python
        ref = sandbox.read_file("/path/to/file")  # Returns OperationRef[bytes]
        data = ref.result()  # Block until complete
        ```

        With timeout:
        ```python
        try:
            data = ref.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            print("Operation timed out")
        ```

        Async usage:
        ```python
        data = await ref  # Awaitable in async context
        ```
    """

    def __init__(self, future: concurrent.futures.Future[T]) -> None:
        """Initialize with a concurrent.futures.Future.

        Args:
            future: The underlying future that will contain the result.
        """
        self._future = future

    def result(self, timeout: float | None = None) -> T:
        """Block until the result is ready and return it.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The result of the operation.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires before completion.
            concurrent.futures.CancelledError: If the operation was cancelled.
            Exception: Any exception raised by the operation.
        """
        return self._future.result(timeout)

    def __await__(self) -> Generator[Any, None, T]:
        """Make this ref awaitable for async contexts.

        Bridges the concurrent.futures.Future to asyncio, allowing the ref
        to be awaited in async code.

        Returns:
            Generator that yields the result when complete.

        Examples:
            ```python
            async def example():
                ref = sandbox.read_file("/path")
                data = await ref  # Works in async context
            ```
        """
        return asyncio.wrap_future(self._future).__await__()


class ExecOutcome(StrEnum):
    """Outcome classification for exec() calls.

    Taxonomy:
    - COMPLETED_OK: returncode == 0
    - COMPLETED_NONZERO: returncode != 0 (process completed but returned error)
    - FAILURE: SandboxTimeoutError, cancellation, transport failures
    """

    COMPLETED_OK = "completed_ok"
    COMPLETED_NONZERO = "completed_nonzero"
    FAILURE = "failure"


class PlacementMode(StrEnum):
    """Placement candidate set for sandbox scheduling.

    Attributes:
        UNSPECIFIED: Leave unset; the backend defaults to serverless.
        SERVERLESS: CoreWeave-managed serverless pool.
        CKS: Caller's own CKS cluster. Required when ``runner_ids`` is set.
    """

    UNSPECIFIED = "unspecified"
    SERVERLESS = "serverless"
    CKS = "cks"


class PlacementSpillover(StrEnum):
    """Client-side create retry across placement modes on capacity / constraint failure.

    ``placement_mode`` remains the primary (first-attempt) mode. Spillover modes
    retry CreateSandbox once with the alternate mode when the first attempt
    cannot place the request (capacity, no suitable runner, runner
    unavailable/overloaded, or a placement constraint). Template creates
    allow only ``STRICT``. ``SERVERLESS_THEN_CKS`` cannot be combined with
    ``runner_ids``.

    Attributes:
        STRICT: No spill (default). Honor ``placement_mode`` only.
        CKS_THEN_SERVERLESS: Attempt CKS first, then serverless. Unset
            ``placement_mode`` is treated as CKS for attempt 1. Explicit
            ``placement_mode=serverless`` raises ``ValueError``.
        SERVERLESS_THEN_CKS: Attempt serverless first, then CKS. Unset
            ``placement_mode`` is treated as serverless for attempt 1. Explicit
            ``placement_mode=cks`` or a non-empty ``runner_ids`` pin raises
            ``ValueError``.
    """

    STRICT = "strict"
    CKS_THEN_SERVERLESS = "cks_then_serverless"
    SERVERLESS_THEN_CKS = "serverless_then_cks"


class DataPlaneMode(StrEnum):
    """Transport policy for sandbox data operations.

    Lifecycle and management operations always use the CoreWeave Sandbox API.
    This policy applies only to exec, logs, and file operations after a sandbox
    is running.

    Attributes:
        AUTO: Prefer a sandbox-scoped direct mTLS connection and transparently
            use the API gateway when direct access is unavailable.
        GATEWAY: Always route data operations through the API gateway.
        DIRECT: Require the direct mTLS connection and surface an error when it
            cannot be established.
    """

    AUTO = "auto"
    GATEWAY = "gateway"
    DIRECT = "direct"


class ServiceVisibility(StrEnum):
    """Reachability intent for a typed sandbox service port."""

    UNSPECIFIED = "unspecified"
    PUBLIC = "public"
    PRIVATE = "private"
    CUSTOM = "custom"


class ServiceProtocol(StrEnum):
    """L4 protocol for a typed sandbox service port."""

    UNSPECIFIED = "unspecified"
    TCP = "tcp"
    UDP = "udp"
    SCTP = "sctp"


class EndpointKind(StrEnum):
    """HTTPS is the only supported endpoint kind."""

    HTTPS = "https"


class EndpointAuth(StrEnum):
    """OPEN is the only supported endpoint auth (no token required)."""

    OPEN = "open"


@dataclass(frozen=True, kw_only=True)
class Endpoint:
    """Public HTTPS URL for a service. Set at create time; CoreWeave terminates TLS.

    ``kind`` and ``auth`` are required. Only HTTPS + OPEN is supported.
    A URL in ``Sandbox.service_urls`` means the hostname was assigned, not
    that the app is listening yet.

    ``request_timeout_seconds`` is the server-side HTTPS request clock on
    this product endpoint (504 while the sandbox stays alive). It is not
    ``Sandbox.run(request_timeout_seconds=...)``, which is the client RPC
    deadline. This client only requires an ``int`` (or ``None``).

    Attributes:
        kind: ``HTTPS``.
        auth: ``OPEN`` (no platform token required).
        request_timeout_seconds: Seconds before the platform closes an
            in-flight HTTPS request. ``None`` or ``0`` selects the
            platform default (15s on serverless). The server accepts
            ``0`` or ``[15, 900]``. On create-from-template, ``0`` is
            replace-on-presence and does not clear a template timeout
            back to the platform default.
    """

    kind: EndpointKind | str
    auth: EndpointAuth | str
    request_timeout_seconds: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", EndpointKind(self.kind.lower()))
        if isinstance(self.auth, str):
            object.__setattr__(self, "auth", EndpointAuth(self.auth.lower()))
        timeout = self.request_timeout_seconds
        if timeout is None:
            return
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            raise TypeError(
                "Endpoint.request_timeout_seconds must be an int or None, "
                f"got {type(timeout).__name__}"
            )


@dataclass(frozen=True, kw_only=True)
class HttpsEndpointStatus:
    """Applied HTTPS product endpoint echoed on ``Sandbox.service_endpoints``.

    ``request_timeout_seconds`` is the effective server-side clock (15 when
    create omitted or sent ``0``). ``url`` is empty when the API suppresses
    it (terminal sandboxes). This is not ``Sandbox.run(request_timeout_seconds=...)``.

    Attributes:
        port: Container port for this service.
        name: Service name from status (may be empty).
        kind: ``HTTPS``.
        auth: ``OPEN``.
        url: Assigned HTTPS URL, or empty when suppressed.
        request_timeout_seconds: Applied HTTPS request timeout in seconds.
    """

    port: int
    name: str
    kind: EndpointKind
    auth: EndpointAuth
    url: str
    request_timeout_seconds: int


@dataclass(frozen=True, kw_only=True)
class Service:
    """Typed service port exposed by a sandbox.

    Replaces the beta string ``NetworkOptions`` ingress/egress mode model.

    Attributes:
        port: Container port the workload listens on.
        name: Optional service name.
        protocol: L4 protocol (defaults to TCP when unset). Must be unset or
            TCP when ``endpoint`` is set.
        visibility: Who may reach this port (PUBLIC/PRIVATE/CUSTOM).
            CUSTOM means the fleet assigns reachability; ``service_urls``
            stays empty unless the API reports a URL. The service still
            appears in ``exposed_ports``. Must be PUBLIC when ``endpoint``
            is set.
        endpoint: Optional HTTPS URL (HTTPS/OPEN, optional
            ``request_timeout_seconds``). Omit for a plain TCP/UDP port.
    """

    port: int
    name: str | None = None
    protocol: ServiceProtocol | str | None = None
    visibility: ServiceVisibility | str | None = None
    endpoint: Endpoint | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Service.port must be 1-65535, got {self.port}")
        if isinstance(self.protocol, str):
            object.__setattr__(self, "protocol", ServiceProtocol(self.protocol.lower()))
        if isinstance(self.visibility, str):
            object.__setattr__(self, "visibility", ServiceVisibility(self.visibility.lower()))
        if isinstance(self.endpoint, dict):
            object.__setattr__(self, "endpoint", Endpoint(**self.endpoint))
        elif self.endpoint is not None and not isinstance(self.endpoint, Endpoint):
            raise TypeError(
                f"Service.endpoint must be Endpoint or dict, got {type(self.endpoint).__name__}"
            )
        if self.endpoint is not None:
            if self.visibility != ServiceVisibility.PUBLIC:
                raise ValueError("Service.visibility must be PUBLIC when endpoint is set")
            if self.protocol not in (None, ServiceProtocol.UNSPECIFIED, ServiceProtocol.TCP):
                raise ValueError("Service.protocol must be unset or TCP when endpoint is set")


# DNS-1123 subdomain, matching k8s IsDNS1123Subdomain used by the gateway.
_DNS1123_LABEL = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
_DNS1123_LABEL_RE = re.compile(rf"^{_DNS1123_LABEL}$")
_DNS1123_LABEL_MAX = 63
_DNS1123_SUBDOMAIN_RE = re.compile(rf"^{_DNS1123_LABEL}(?:\.{_DNS1123_LABEL})*$")
_DNS1123_SUBDOMAIN_MAX = 253

# Closed reserved kubelet names (and restore- prefix) from the v1 multi-container contract.
_RESERVED_CONTAINER_NAMES = frozenset(
    {
        "cw-object-store-agent",
        "cw-object-store-agent-restore",
        "dns-egress",
        "dns-egress-probe",
    }
)
_RESERVED_CONTAINER_NAME_PREFIX = "cw-object-store-agent-restore-"


def _is_dns1123_subdomain(name: str) -> bool:
    return len(name) <= _DNS1123_SUBDOMAIN_MAX and _DNS1123_SUBDOMAIN_RE.fullmatch(name) is not None


def _is_dns1123_label(name: str) -> bool:
    return len(name) <= _DNS1123_LABEL_MAX and _DNS1123_LABEL_RE.fullmatch(name) is not None


def _validate_absolute_mount_path(path: str, *, field: str) -> None:
    if not path:
        raise ValueError(f"{field} cannot be empty")
    if not path.startswith("/"):
        raise ValueError(f"{field} must be an absolute path, got: {path!r}")
    if path == "/":
        raise ValueError(f"{field} cannot be '/'")


def _validate_container_name(name: str) -> None:
    if not _is_dns1123_label(name):
        raise ValueError(
            "Container.name must be a DNS-1123 label (lowercase alphanumeric and "
            f"hyphens, at most {_DNS1123_LABEL_MAX} characters), got: {name!r}"
        )
    if name in _RESERVED_CONTAINER_NAMES or name.startswith(_RESERVED_CONTAINER_NAME_PREFIX):
        raise ValueError(f"Container.name {name!r} is reserved by the platform")


class TenantScope(StrEnum):
    """Relational selector for other sandboxes."""

    UNSPECIFIED = "unspecified"
    SAME_USER = "same_user"
    SAME_ORG = "same_org"
    SANDBOX_NETWORK = "sandbox_network"


class StorageMedium(StrEnum):
    """Backing store for a scratch volume."""

    UNSPECIFIED = "unspecified"
    DISK = "disk"
    MEMORY = "memory"


class ObjectStoragePermission(StrEnum):
    """Permission for minted object-storage credentials."""

    UNSPECIFIED = "unspecified"
    READ = "read"
    READ_WRITE = "read_write"


@dataclass(frozen=True, kw_only=True)
class CidrBlock:
    """An IP range with optional carved-out sub-ranges."""

    cidr: str
    except_cidrs: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.cidr:
            raise ValueError("CidrBlock.cidr cannot be empty")
        object.__setattr__(self, "except_cidrs", tuple(self.except_cidrs))


@dataclass(frozen=True, kw_only=True)
class SelectorBlock:
    """Label selector for cluster workloads (matchLabels only)."""

    pod_labels: Mapping[str, str]
    namespace_labels: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.pod_labels:
            raise ValueError("SelectorBlock.pod_labels must contain at least one label")
        object.__setattr__(self, "pod_labels", dict(self.pod_labels))
        if self.namespace_labels is not None:
            object.__setattr__(self, "namespace_labels", dict(self.namespace_labels))


@dataclass(frozen=True, kw_only=True)
class PortRange:
    """A single port or inclusive port range."""

    port: int
    end_port: int | None = None
    protocol: str | None = None

    def __post_init__(self) -> None:
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"PortRange.port must be 1-65535, got {self.port}")
        if self.end_port is not None and (self.end_port < self.port or self.end_port > 65535):
            raise ValueError(f"PortRange.end_port must be {self.port}-65535, got {self.end_port}")


def _normalize_dns_name(name: str, *, field: str, allow_star: bool = False) -> str:
    name = name.strip().lower()
    if not name:
        raise ValueError(f"{field} cannot be empty")
    if name == "*":
        if allow_star:
            return name
        raise ValueError(f'{field} cannot be "*"; that is a policy ceiling, not a sandbox grant')
    if name.startswith("*."):
        valid = len(name) <= _DNS1123_SUBDOMAIN_MAX and _is_dns1123_subdomain(name[2:])
    else:
        valid = _is_dns1123_subdomain(name)
    if not valid:
        raise ValueError(
            f"{field} must be a DNS-1123 subdomain or a single leftmost wildcard (*.example.com)"
        )
    return name


def _coerce_cidr(value: CidrBlock | Mapping[str, Any] | str) -> CidrBlock:
    if isinstance(value, CidrBlock):
        return value
    if isinstance(value, str):
        return CidrBlock(cidr=value)
    if isinstance(value, Mapping):
        except_cidrs = value.get("except_cidrs", value.get("except", ()))
        return CidrBlock(cidr=value["cidr"], except_cidrs=except_cidrs)
    raise TypeError(f"cidr must be CidrBlock, dict, or str, got {type(value).__name__}")


def _coerce_port_range(value: PortRange | Mapping[str, Any] | int) -> PortRange:
    if isinstance(value, PortRange):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return PortRange(port=value)
    if isinstance(value, Mapping):
        return PortRange(**value)
    raise TypeError(f"ports entries must be PortRange, dict, or int, got {type(value).__name__}")


@dataclass(frozen=True, kw_only=True)
class EgressRule:
    """One create-time egress destination.

    Exactly one destination must be set: ``dns_name``, ``cidr``, ``tenant``,
    ``any``, or ``selector``. DNS names are HTTPS (TCP 443) grants. Exact
    names (``pypi.org``) or a single leftmost wildcard (``*.pypi.org``).
    ``"*"`` is a policy ceiling, not a sandbox grant.

    Attributes:
        dns_name: Hostname to grant for HTTPS (TCP 443).
        cidr: IP range (``CidrBlock``, dict, or CIDR string).
        tenant: Other sandboxes selected relationally.
        any: All destinations except platform-internal ranges.
        selector: Cluster workloads selected by pod/namespace labels.
        ports: Optional destination port filter. Empty = all ports
            on non-DNS rules. DNS-name grants omit ports or set
            exactly TCP 443.
        dns_name_except: Policy-only carve-outs. Rejected on this
            create-time type.
    """

    dns_name: str | None = None
    cidr: CidrBlock | Mapping[str, Any] | str | None = None
    tenant: TenantScope | str | None = None
    any: bool = False
    selector: SelectorBlock | Mapping[str, Any] | None = None
    ports: Sequence[PortRange | Mapping[str, Any] | int] | None = None
    dns_name_except: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.dns_name is not None:
            object.__setattr__(
                self,
                "dns_name",
                _normalize_dns_name(self.dns_name, field="EgressRule.dns_name"),
            )
        if self.cidr is not None:
            object.__setattr__(self, "cidr", _coerce_cidr(self.cidr))
        if self.tenant is not None:
            tenant = self.tenant
            if isinstance(tenant, str):
                tenant = TenantScope(tenant.lower())
            object.__setattr__(self, "tenant", tenant)
        if self.selector is not None and not isinstance(self.selector, SelectorBlock):
            if not isinstance(self.selector, Mapping):
                raise TypeError(
                    "EgressRule.selector must be SelectorBlock or dict, "
                    f"got {type(self.selector).__name__}"
                )
            object.__setattr__(self, "selector", SelectorBlock(**self.selector))
        ports: tuple[PortRange, ...] | None = None
        if self.ports is not None:
            if isinstance(self.ports, (str, bytes)):
                raise TypeError("ports must be a sequence of PortRange, dict, or int")
            ports = tuple(_coerce_port_range(p) for p in self.ports)
            object.__setattr__(self, "ports", ports)
        if self.dns_name_except is not None:
            raise ValueError(
                "EgressRule.dns_name_except is policy-only and is not valid "
                "on a sandbox create grant"
            )
        destinations = sum(
            (
                self.dns_name is not None,
                self.cidr is not None,
                self.tenant is not None,
                self.any is True,
                self.selector is not None,
            )
        )
        if destinations != 1:
            raise ValueError(
                "EgressRule requires exactly one destination: "
                "dns_name, cidr, tenant, any, or selector"
            )
        if self.dns_name is not None:
            _validate_sandbox_dns_ports(ports)


def _is_https_443_port(port: PortRange) -> bool:
    """Return True if ``port`` is a single TCP 443 grant."""
    if port.port != 443:
        return False
    if port.end_port is not None and port.end_port != 443:
        return False
    if port.protocol and port.protocol.upper() != "TCP":
        return False
    return True


def _validate_sandbox_dns_ports(ports: Sequence[PortRange] | None) -> None:
    if not ports:
        return
    if len(ports) == 1 and _is_https_443_port(ports[0]):
        return
    raise ValueError(
        "EgressRule.dns_name grants only HTTPS (TCP 443); omit ports or set ports to 443"
    )


def _coerce_egress_rule(value: EgressRule | Mapping[str, Any]) -> EgressRule:
    if isinstance(value, EgressRule):
        return value
    if isinstance(value, Mapping):
        return EgressRule(**value)
    raise TypeError(f"egress entries must be EgressRule or dict, got {type(value).__name__}")


@dataclass(frozen=True, kw_only=True)
class IngressRule:
    """One create-time CUSTOM-port ingress source.

    Exactly one source must be set: ``cidr``, ``tenant``, or ``any``.

    Attributes:
        cidr: Source IP range.
        tenant: Other sandboxes selected relationally.
        any: Any source, including the public internet.
        ports: Optional port filter. Empty = all CUSTOM-visibility ports.
    """

    cidr: CidrBlock | Mapping[str, Any] | str | None = None
    tenant: TenantScope | str | None = None
    any: bool = False
    ports: Sequence[PortRange | Mapping[str, Any] | int] | None = None

    def __post_init__(self) -> None:
        if self.cidr is not None:
            object.__setattr__(self, "cidr", _coerce_cidr(self.cidr))
        if self.tenant is not None:
            tenant = self.tenant
            if isinstance(tenant, str):
                tenant = TenantScope(tenant.lower())
            object.__setattr__(self, "tenant", tenant)
        if self.ports is not None:
            if isinstance(self.ports, (str, bytes)):
                raise TypeError("ports must be a sequence of PortRange, dict, or int")
            object.__setattr__(self, "ports", tuple(_coerce_port_range(p) for p in self.ports))
        sources = sum((self.cidr is not None, self.tenant is not None, self.any is True))
        if sources != 1:
            raise ValueError("IngressRule requires exactly one source: cidr, tenant, or any")


def _coerce_ingress_rule(value: IngressRule | Mapping[str, Any]) -> IngressRule:
    if isinstance(value, IngressRule):
        return value
    if isinstance(value, Mapping):
        return IngressRule(**value)
    raise TypeError(f"ingress entries must be IngressRule or dict, got {type(value).__name__}")


@dataclass(frozen=True, kw_only=True)
class NetworkOptions:
    """Network deny flags and create-time ingress/egress grants.

    Port exposure uses typed ``services=``. ``egress`` and ``ingress`` are
    allow-only rule sets over default-deny; empty/None leaves the runner
    policy default in effect.

    Attributes:
        deny_egress: When True, deny all declared egress (policy default unused).
            Mutually exclusive with a non-empty ``egress`` list.
        deny_ingress: When True, deny CUSTOM ingress (policy default unused).
        egress: Egress grants (``EgressRule`` or dict).
        ingress: CUSTOM-port ingress sources (``IngressRule`` or dict).

    Examples:
        Grant PyPI over HTTPS::

            NetworkOptions(
                egress=[
                    EgressRule(dns_name="pypi.org"),
                    EgressRule(dns_name="*.pypi.org"),
                ],
            )
    """

    deny_egress: bool | None = None
    deny_ingress: bool | None = None
    egress: Sequence[EgressRule | Mapping[str, Any]] | None = None
    ingress: Sequence[IngressRule | Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.egress is not None:
            if isinstance(self.egress, (str, bytes)):
                raise TypeError("egress must be a sequence of EgressRule or dict, not a string")
            rules = tuple(_coerce_egress_rule(rule) for rule in self.egress)
            object.__setattr__(self, "egress", rules)
            if self.deny_egress is True and rules:
                raise ValueError("NetworkOptions.deny_egress cannot be combined with egress rules")
        if self.ingress is not None:
            if isinstance(self.ingress, (str, bytes)):
                raise TypeError("ingress must be a sequence of IngressRule or dict, not a string")
            object.__setattr__(
                self, "ingress", tuple(_coerce_ingress_rule(rule) for rule in self.ingress)
            )


def _validate_sub_path(value: str | None, *, field: str) -> str | None:
    """Validate a VolumeMount.sub_path: relative and canonical, or None."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None, got {type(value).__name__}")
    parts = value.split("/")
    if value.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{field} must be relative and canonical (no '.' / '..'), got {value!r}")
    return value


def _validate_optional_uid(value: int | None, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int or None, got {type(value).__name__}")
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{field} must be an unsigned 32-bit integer, got {value}")
    return value


def _validate_optional_bool(value: bool | None, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a bool or None, got {type(value).__name__}")
    return value


def _coerce_string_sequence(value: Sequence[str] | None, *, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of strings, not a string")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{field} must contain only strings")
    return items


@dataclass(frozen=True, kw_only=True)
class ScratchVolumeOptions:
    """Named scratch volume for snapshot/restore workflows.

    Volumes are sandbox-level. ``mount_path`` is a convenience that mounts
    this volume on the primary container. For multi-container sandboxes,
    omit ``mount_path`` and declare mounts on ``Container.volume_mounts``.

    Attributes:
        name: Volume name within the sandbox (referenced by mounts/snapshots).
        mount_path: Absolute path to mount into the primary container. None
            declares the volume without mounting it.
        size: Volume size (e.g. ``"10Gi"``). None uses the platform default.
        restore_from_snapshot_id: When set, restore this snapshot at create.
        medium: Backing store. Unset uses disk. ``MEMORY`` is tmpfs and
            counts against container memory.
        sub_path: Relative path inside the volume to mount instead of its root.
        read_only: Mount the volume read-only.
    """

    name: str
    mount_path: str | None = None
    size: str | None = None
    restore_from_snapshot_id: str | None = None
    medium: StorageMedium | str | None = None
    sub_path: str | None = None
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ScratchVolumeOptions.name cannot be empty")
        if self.mount_path is not None:
            _validate_absolute_mount_path(self.mount_path, field="ScratchVolumeOptions.mount_path")
        if self.size is not None and not self.size:
            object.__setattr__(self, "size", None)
        if self.restore_from_snapshot_id is not None and not self.restore_from_snapshot_id:
            object.__setattr__(self, "restore_from_snapshot_id", None)
        if self.medium is not None:
            medium = self.medium
            if isinstance(medium, str):
                medium = StorageMedium(medium.lower())
            object.__setattr__(self, "medium", medium)
        object.__setattr__(
            self,
            "sub_path",
            _validate_sub_path(self.sub_path, field="ScratchVolumeOptions.sub_path"),
        )
        if not isinstance(self.read_only, bool):
            raise TypeError(
                "ScratchVolumeOptions.read_only must be a bool, "
                f"got {type(self.read_only).__name__}"
            )


@dataclass(frozen=True, kw_only=True)
class RegisteredVolumeOptions:
    """Mount a registered Volume into the sandbox.

    Attributes:
        name: Volume name within the sandbox (referenced by the mount).
        volume_id: Registered Volume ID.
        mount_path: Absolute path to mount into the primary container.
        sub_path: Relative path inside the volume to mount instead of its root.
            Combined with any volume-level prefix set at registration.
        read_only: Mount the volume read-only.
    """

    name: str
    volume_id: str
    mount_path: str
    sub_path: str | None = None
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RegisteredVolumeOptions.name cannot be empty")
        if not self.volume_id:
            raise ValueError("RegisteredVolumeOptions.volume_id cannot be empty")
        if not self.mount_path:
            raise ValueError("RegisteredVolumeOptions.mount_path cannot be empty")
        if not self.mount_path.startswith("/"):
            raise ValueError(
                f"RegisteredVolumeOptions.mount_path must be absolute, got: {self.mount_path!r}"
            )
        if self.mount_path == "/":
            raise ValueError("RegisteredVolumeOptions.mount_path cannot be '/'")
        object.__setattr__(
            self,
            "sub_path",
            _validate_sub_path(self.sub_path, field="RegisteredVolumeOptions.sub_path"),
        )
        if not isinstance(self.read_only, bool):
            raise TypeError(
                "RegisteredVolumeOptions.read_only must be a bool, "
                f"got {type(self.read_only).__name__}"
            )


def _coerce_volume_options(
    vol: ScratchVolumeOptions | RegisteredVolumeOptions | Mapping[str, Any],
) -> ScratchVolumeOptions | RegisteredVolumeOptions:
    if isinstance(vol, (ScratchVolumeOptions, RegisteredVolumeOptions)):
        return vol
    if isinstance(vol, Mapping):
        if "volume_id" in vol:
            return RegisteredVolumeOptions(**vol)
        return ScratchVolumeOptions(**vol)
    raise TypeError(
        "volumes entries must be ScratchVolumeOptions, RegisteredVolumeOptions, "
        f"or dict, got {type(vol).__name__}"
    )


@dataclass(frozen=True, kw_only=True)
class VolumeMount:
    """Mount of a sandbox-level volume into one container.

    Attributes:
        volume: Name of a ``ScratchVolumeOptions`` / ``SandboxSpec`` volume.
        mount_path: Absolute path inside the container.
        read_only: When True, mount the volume read-only.
        sub_path: Optional path within the volume to mount.
    """

    volume: str
    mount_path: str
    read_only: bool = False
    sub_path: str | None = None

    def __post_init__(self) -> None:
        if not self.volume:
            raise ValueError("VolumeMount.volume cannot be empty")
        _validate_absolute_mount_path(self.mount_path, field="VolumeMount.mount_path")
        if self.sub_path is not None and not self.sub_path:
            object.__setattr__(self, "sub_path", None)


def _coerce_volume_mount(value: VolumeMount | Mapping[str, Any]) -> VolumeMount:
    if isinstance(value, VolumeMount):
        return value
    if isinstance(value, Mapping):
        return VolumeMount(**value)
    raise TypeError(
        f"volume_mounts entries must be VolumeMount or dict, got {type(value).__name__}"
    )


@dataclass(frozen=True, kw_only=True)
class ImagePullCredentials:
    """Private-registry pull credentials resolved from a secret store.

    Attributes:
        registry: Registry authority (e.g. ``"ghcr.io"``).
        store: Secret store name.
        name: Secret path/name within the store.
        field: Optional structured-secret field.
    """

    registry: str
    store: str
    name: str
    field: str = ""

    def __post_init__(self) -> None:
        if not self.registry:
            raise ValueError("ImagePullCredentials.registry cannot be empty")
        if not self.store:
            raise ValueError("ImagePullCredentials.store cannot be empty")
        if not self.name:
            raise ValueError("ImagePullCredentials.name cannot be empty")


@dataclass(frozen=True, kw_only=True)
class ObjectStorageAccess:
    """Temporary object-storage credentials injected into every user container.

    Attributes:
        buckets: Bucket names the minted credential may access.
        permission: ``READ`` or ``READ_WRITE``. Unset uses the platform default.
        object_prefix: Optional key prefix that scopes the credential.
    """

    buckets: Sequence[str] = ()
    permission: ObjectStoragePermission | str | None = None
    object_prefix: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.buckets, (str, bytes)):
            raise TypeError(
                "ObjectStorageAccess.buckets must be a sequence of non-empty "
                "strings, not a bare string"
            )
        buckets = tuple(self.buckets)
        if not buckets:
            raise ValueError("ObjectStorageAccess.buckets cannot be empty")
        for bucket in buckets:
            if not isinstance(bucket, str):
                raise TypeError(
                    "ObjectStorageAccess.buckets entries must be strings, "
                    f"got {type(bucket).__name__}"
                )
            if not bucket:
                raise ValueError("ObjectStorageAccess.buckets entries cannot be empty")
        object.__setattr__(self, "buckets", buckets)
        if self.permission is not None:
            permission = self.permission
            if isinstance(permission, str):
                permission = ObjectStoragePermission(permission.lower())
            object.__setattr__(self, "permission", permission)
        if self.object_prefix is not None and not self.object_prefix:
            object.__setattr__(self, "object_prefix", None)


def _coerce_object_storage_access(
    value: ObjectStorageAccess | Mapping[str, Any] | None,
) -> ObjectStorageAccess | None:
    if value is None:
        return None
    if isinstance(value, ObjectStorageAccess):
        return value
    if isinstance(value, Mapping):
        return ObjectStorageAccess(**value)
    raise TypeError(
        "object_storage_access must be ObjectStorageAccess, dict, or None, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True, kw_only=True)
class SecurityContext:
    """In-guest container privilege, clamped by the runner policy.

    Host-reaching settings (host network, host PID, hostPath) are not
    expressible here. ``privileged``, capabilities, run-as, and seccomp
    apply inside the sandbox isolation boundary.

    Attributes:
        run_as_user: UID for the entrypoint. Unset uses the image user.
        run_as_group: GID for the entrypoint. Unset uses the image group.
        privileged: Run privileged inside the isolation boundary.
        allow_privilege_escalation: Allow a process to gain more privileges
            than its parent.
        read_only_root_filesystem: Mount the container root filesystem read-only.
        capabilities_add: Linux capabilities to add (e.g. ``"SYS_PTRACE"``).
        capabilities_drop: Linux capabilities to drop.
        seccomp_profile: ``"RuntimeDefault"`` or ``"Unconfined"``.
    """

    run_as_user: int | None = None
    run_as_group: int | None = None
    privileged: bool | None = None
    allow_privilege_escalation: bool | None = None
    read_only_root_filesystem: bool | None = None
    capabilities_add: Sequence[str] | None = None
    capabilities_drop: Sequence[str] | None = None
    seccomp_profile: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_as_user",
            _validate_optional_uid(self.run_as_user, field="SecurityContext.run_as_user"),
        )
        object.__setattr__(
            self,
            "run_as_group",
            _validate_optional_uid(self.run_as_group, field="SecurityContext.run_as_group"),
        )
        object.__setattr__(
            self,
            "privileged",
            _validate_optional_bool(self.privileged, field="SecurityContext.privileged"),
        )
        object.__setattr__(
            self,
            "allow_privilege_escalation",
            _validate_optional_bool(
                self.allow_privilege_escalation,
                field="SecurityContext.allow_privilege_escalation",
            ),
        )
        object.__setattr__(
            self,
            "read_only_root_filesystem",
            _validate_optional_bool(
                self.read_only_root_filesystem,
                field="SecurityContext.read_only_root_filesystem",
            ),
        )
        object.__setattr__(
            self,
            "capabilities_add",
            _coerce_string_sequence(
                self.capabilities_add, field="SecurityContext.capabilities_add"
            ),
        )
        object.__setattr__(
            self,
            "capabilities_drop",
            _coerce_string_sequence(
                self.capabilities_drop, field="SecurityContext.capabilities_drop"
            ),
        )
        profile = self.seccomp_profile
        if profile is not None and not profile:
            object.__setattr__(self, "seccomp_profile", None)


def _coerce_security_context(
    value: SecurityContext | Mapping[str, Any] | None,
) -> SecurityContext | None:
    if value is None:
        return None
    if isinstance(value, SecurityContext):
        return value
    if isinstance(value, Mapping):
        return SecurityContext(**value)
    raise TypeError(
        f"security_context must be SecurityContext, dict, or None, got {type(value).__name__}"
    )


@dataclass(frozen=True, kw_only=True)
class Secret:
    """A secret to inject from a store into a sandbox environment variable.

    All fields are keyword-only. When ``env_var`` is not specified it defaults
    to ``name``. Plain dicts with matching keys are also accepted and
    converted automatically.

    Attributes:
        store: Name of the secret store (e.g. "wandb").
        name: Name of the secret in the store.
        field: Specific field within a structured secret (optional).
        env_var: Environment variable the secret is injected as (defaults to name).

    Examples:
        Minimal usage (env_var defaults to name)::

            Secret(store="wandb", name="HF_TOKEN")

        Extracting a field from a structured secret::

            Secret(
                store="wandb",
                name="db-credentials",
                field="password",
                env_var="DB_PASS",
            )
    """

    store: str
    name: str
    field: str = ""
    env_var: str | None = None

    def __post_init__(self) -> None:
        # Default env_var to name and validate required fields.
        if self.env_var is None:
            object.__setattr__(self, "env_var", self.name)
        if not self.store:
            raise ValueError("Secret.store cannot be empty")
        if not self.name:
            raise ValueError("Secret.name cannot be empty")
        if not self.env_var:
            raise ValueError("Secret.env_var cannot be empty")


def _unique_secrets_by_env_var(secrets: Sequence[Secret]) -> tuple[Secret, ...]:
    """Keep one secret per env_var; raise if two distinct sources conflict."""
    seen: dict[str, Secret] = {}
    for secret in secrets:
        env_var = secret.env_var
        assert env_var is not None  # guaranteed by Secret.__post_init__
        if env_var in seen and secret != seen[env_var]:
            raise ValueError(
                f"Conflicting secrets for env_var {env_var!r}: "
                f"Secret(store={seen[env_var].store!r}, name={seen[env_var].name!r}, "
                f"field={seen[env_var].field!r}) vs "
                f"Secret(store={secret.store!r}, name={secret.name!r}, "
                f"field={secret.field!r})"
            )
        seen[env_var] = secret
    return tuple(seen.values())


@dataclass(frozen=True, kw_only=True)
class ResourceOptions:
    """Resource configuration for sandbox CPU, memory, and GPU.

    Supports separate requests and limits for Burstable QoS pods.
    GPU is a separate top-level field because GPU overcommit is not
    supported by the backend.

    Shallow immutability: dict fields prevent hashing but preserve
    the frozen dataclass pattern used by NetworkOptions and Secret.

    Attributes:
        requests: CPU/memory resource requests (e.g. {"cpu": "1", "memory": "256Mi"}).
        limits: CPU/memory resource limits (e.g. {"cpu": "8", "memory": "2Gi"}).
        gpu: GPU configuration (e.g. {"count": 1, "type": "A100"}).
    """

    requests: dict[str, str] | None = None
    limits: dict[str, str] | None = None
    gpu: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.requests is not None and len(self.requests) == 0:
            object.__setattr__(self, "requests", None)
        if self.limits is not None and len(self.limits) == 0:
            object.__setattr__(self, "limits", None)
        if isinstance(self.gpu, dict) and len(self.gpu) == 0:
            object.__setattr__(self, "gpu", None)


@dataclass(frozen=True, kw_only=True)
class Container:
    """One user container in a sandbox.

    Pass a list to ``Sandbox.run(containers=[...])``. That form is mutually
    exclusive with the single-container kwargs (``container_image``,
    ``command``/``args``, ``resources``, ``mounted_files``, ``secrets``,
    ``image_pull_credentials``, ``environment_variables``,
    ``security_context``, ``working_dir``) and does not inherit those
    same fields from ``SandboxDefaults``.

    One container: ``primary`` may be omitted or False (that row is primary).
    More than one: exactly one row must set ``primary=True``, and every row
    needs a name and resources. GPU is allowed only on the primary.

    Attributes:
        image: OCI image to run.
        name: DNS-1123 label. Optional for a single container; required when
            more than one container is specified.
        command: Entrypoint. Empty/None uses the image entrypoint. Args may
            be set without command.
        args: Arguments to the command or image entrypoint.
        environment_variables: Env vars injected into this container only.
        resources: CPU/memory/GPU for this container. Required when more
            than one container is specified.
        mounted_files: Files written into this container at startup.
        volume_mounts: Sandbox-level volumes to mount into this container.
        secrets: Secret-store inject for this container only.
        working_dir: Working directory for the command. Must be absolute
            when set.
        image_pull_credentials: Private-registry pull credentials.
        primary: When True, this container owns sandbox lifecycle and is
            the default exec/logs/files target.
    """

    image: str
    name: str | None = None
    command: str | None = None
    args: Sequence[str] | None = None
    environment_variables: Mapping[str, str] | None = None
    resources: ResourceOptions | dict[str, Any] | None = None
    mounted_files: Sequence[Mapping[str, Any]] | None = None
    volume_mounts: Sequence[VolumeMount | Mapping[str, Any]] | None = None
    secrets: Sequence[Secret | Mapping[str, Any]] | None = None
    working_dir: str | None = None
    image_pull_credentials: ImagePullCredentials | Mapping[str, Any] | None = None
    primary: bool = False
    # Status echo only. Create-time name/cwd/image checks do not apply to
    # server-authored rows (reserved platform sidecars, working_dir="/").
    _observed: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.image and not self._observed:
            raise ValueError("Container.image cannot be empty")
        if self.name is not None:
            if not self.name:
                object.__setattr__(self, "name", None)
            elif not self._observed:
                _validate_container_name(self.name)
        if self.command is not None and not self.command:
            object.__setattr__(self, "command", None)
        if self.args is not None:
            if isinstance(self.args, (str, bytes)):
                raise TypeError("Container.args must be a sequence of strings, not a string")
            coerced_args = tuple(self.args)
            for i, arg in enumerate(coerced_args):
                if not isinstance(arg, str):
                    raise TypeError(f"Container.args[{i}] must be str, got {type(arg).__name__}")
            object.__setattr__(self, "args", coerced_args)
        if self.environment_variables is not None:
            object.__setattr__(self, "environment_variables", dict(self.environment_variables))
        if self.working_dir is not None:
            if not self.working_dir:
                object.__setattr__(self, "working_dir", None)
            elif not self._observed:
                _validate_absolute_mount_path(self.working_dir, field="Container.working_dir")
        if self.volume_mounts is not None:
            object.__setattr__(
                self,
                "volume_mounts",
                tuple(_coerce_volume_mount(m) for m in self.volume_mounts),
            )
        if self.secrets is not None:
            object.__setattr__(
                self,
                "secrets",
                _unique_secrets_by_env_var(
                    tuple(s if isinstance(s, Secret) else Secret(**s) for s in self.secrets)
                ),
            )
        if self.image_pull_credentials is not None and not isinstance(
            self.image_pull_credentials, ImagePullCredentials
        ):
            object.__setattr__(
                self,
                "image_pull_credentials",
                ImagePullCredentials(**self.image_pull_credentials),
            )
        if self.mounted_files is not None:
            object.__setattr__(self, "mounted_files", tuple(self.mounted_files))

    @classmethod
    def _from_observed(cls, **kwargs: Any) -> Container:
        """Build a Container from a Get/list spec echo.

        Skips reserved-name, DNS-1123, working_dir, and empty-image checks.
        ``replace()`` keeps ``_observed`` so inferred ``primary`` stays valid.
        """
        return cls(_observed=True, **kwargs)


def _coerce_container(value: Container | Mapping[str, Any]) -> Container:
    if isinstance(value, Container):
        return value
    if isinstance(value, Mapping):
        return Container(**value)
    raise TypeError(f"containers entries must be Container or dict, got {type(value).__name__}")


def _validate_containers(containers: Sequence[Container]) -> tuple[Container, ...]:
    """Validate a create-time container list (names, primary flag)."""
    if not containers:
        raise ValueError("containers cannot be empty")
    rows = tuple(containers)
    if len(rows) > 1:
        primary_count = sum(1 for row in rows if row.primary)
        if primary_count != 1:
            raise ValueError("containers with more than one entry require exactly one primary=True")
        missing = [i for i, row in enumerate(rows) if not row.name]
        if missing:
            raise ValueError("every container must have a name when more than one is specified")
    seen: set[str] = set()
    for row in rows:
        if not row.name:
            continue
        if row.name in seen:
            raise ValueError(f"duplicate container name: {row.name!r}")
        seen.add(row.name)
    return rows


class FileSystemSnapshotStatus(StrEnum):
    """Lifecycle status of a file-system snapshot (FSS).

    Lifecycle: CREATING -> READY | FAILED. DELETING is reported while a
    snapshot is being removed.

    Attributes:
        UNSPECIFIED: Status not reported by the backend.
        CREATING: Snapshot archive is being captured.
        READY: Snapshot is complete and available for restore.
        FAILED: Snapshot capture failed (see ``status_reason``).
        DELETING: Snapshot is being deleted.
    """

    UNSPECIFIED = "unspecified"
    CREATING = "creating"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class FileSystemSnapshotTrigger(StrEnum):
    """What triggered the creation of a file-system snapshot.

    Attributes:
        UNSPECIFIED: Trigger not reported by the backend.
        ON_DELETE: Captured during ``stop(snapshot_on_stop=True)`` / DeleteSandbox.
        MANUAL: Captured via ``snapshot()`` (CreateFileSystemSnapshot).
    """

    UNSPECIFIED = "unspecified"
    ON_DELETE = "on_delete"
    MANUAL = "manual"


class FileSystemSnapshotBucketMode(StrEnum):
    """Object-storage bucket ownership mode for FSS archives.

    Attributes:
        UNSPECIFIED: Mode not reported by the backend.
        CW_MANAGED: Snapshots are archived to a CoreWeave-managed bucket.
        BRING_YOUR_OWN: Snapshots are archived to a customer-owned bucket.
    """

    UNSPECIFIED = "unspecified"
    CW_MANAGED = "cw_managed"
    BRING_YOUR_OWN = "bring_your_own"


@dataclass(frozen=True, kw_only=True)
class FileSystemSnapshotOptions:
    """Convenience single-mount wrapper over a named scratch volume.

    Prefer ``ScratchVolumeOptions`` / ``volumes=`` for multi-volume sandboxes.
    This helper maps to a scratch volume named ``workspace`` (or ``name``).
    ``mount_path`` is optional: omit it to declare the volume without
    mounting, and attach it via ``Container.volume_mounts``.

    Attributes:
        mount_path: Absolute directory to mount (e.g. "/workspace"). None
            declares the volume without a convenience mount.
        size: Mount size as a Kubernetes resource quantity (e.g. "10Gi").
        file_system_snapshot_id: When set, restore this snapshot at start.
        name: Scratch volume name (default ``"workspace"``).
    """

    mount_path: str | None = None
    size: str | None = None
    file_system_snapshot_id: str | None = None
    name: str = "workspace"

    def __post_init__(self) -> None:
        if self.mount_path is not None:
            _validate_absolute_mount_path(
                self.mount_path, field="FileSystemSnapshotOptions.mount_path"
            )
        if not self.name:
            raise ValueError("FileSystemSnapshotOptions.name cannot be empty")
        if self.size is not None and not self.size:
            object.__setattr__(self, "size", None)
        if self.file_system_snapshot_id is not None and not self.file_system_snapshot_id:
            object.__setattr__(self, "file_system_snapshot_id", None)

    def to_scratch_volume(self) -> ScratchVolumeOptions:
        """Convert to the named scratch-volume model."""
        return ScratchVolumeOptions(
            name=self.name,
            mount_path=self.mount_path,
            size=self.size,
            restore_from_snapshot_id=self.file_system_snapshot_id,
        )


@dataclass(frozen=True, kw_only=True)
class FileSystemSnapshot:
    """An immutable, org-scoped file-system snapshot record.

    Returned by ``Sandbox.get_snapshot()`` and ``Sandbox.list_snapshots()``
    (``Sandbox.snapshot()`` returns just the snapshot ID). To restore, pass the
    ``file_system_snapshot_id`` to ``ScratchVolumeOptions`` /
    ``FileSystemSnapshotOptions`` on create.

    Attributes:
        file_system_snapshot_id: Unique snapshot identifier.
        status: Current lifecycle status.
        status_reason: Human-readable detail, typically set for FAILED snapshots.
        size_bytes: Archive size in bytes (0 until READY).
        source_sandbox_id: The sandbox the snapshot was captured from.
        trigger: Whether the snapshot was taken on delete or via a MANUAL request.
        request_id: Client-supplied create request id, if any.
        object_bucket: The object-storage bucket the archive landed in.
        source_volume_name: Scratch volume name the snapshot was taken from.
        created_at: When the snapshot record was created (UTC), or None.
        updated_at: When the snapshot record was last updated (UTC), or None.
        completed_at: When the snapshot reached a terminal status (UTC), or None.
    """

    file_system_snapshot_id: str
    status: FileSystemSnapshotStatus
    status_reason: str = ""
    size_bytes: int = 0
    source_sandbox_id: str = ""
    trigger: FileSystemSnapshotTrigger = FileSystemSnapshotTrigger.UNSPECIFIED
    request_id: str = ""
    object_bucket: str = ""
    source_volume_name: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None

    def __repr__(self) -> str:
        parts = [
            f"file_system_snapshot_id={self.file_system_snapshot_id!r}",
            f"status={self.status.value}",
        ]
        if self.size_bytes:
            parts.append(f"size_bytes={self.size_bytes}")
        if self.source_sandbox_id:
            parts.append(f"source_sandbox_id={self.source_sandbox_id!r}")
        if self.status_reason:
            parts.append(f"status_reason={self.status_reason!r}")
        return f"FileSystemSnapshot({', '.join(parts)})"


@dataclass(frozen=True, kw_only=True)
class FileSystemSnapshotBucketConfig:
    """Per-organization object-storage bucket configuration for FSS archives.

    Returned by ``Sandbox.get_snapshot_bucket_config()`` and
    ``Sandbox.set_snapshot_bucket_config()``.

    Attributes:
        mode: Bucket ownership mode (CW-managed or bring-your-own).
        bucket_name: The configured bucket name (empty when CW-managed).
        region: The configured bucket region (empty when CW-managed).
        effective_bucket_name: The bucket snapshots are actually archived to,
            resolved server-side from ``mode`` and ``bucket_name``.
    """

    mode: FileSystemSnapshotBucketMode = FileSystemSnapshotBucketMode.UNSPECIFIED
    bucket_name: str = ""
    region: str = ""
    effective_bucket_name: str = ""


@dataclass
class ProcessResult:
    """Result from a completed streaming exec operation.

    Contains both the raw bytes and decoded strings for stdout/stderr,
    along with the exit code and original command.

    Attributes:
        stdout: Decoded stdout as UTF-8 string
        stderr: Decoded stderr as UTF-8 string
        returncode: Exit code from the command (0 = success)
        stdout_bytes: Raw stdout bytes
        stderr_bytes: Raw stderr bytes
        command: The command that was executed

    Examples:
        ```python
        result = process.result()
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
        ```
    """

    stdout: str
    stderr: str
    returncode: int
    stdout_bytes: bytes = field(default=b"")
    stderr_bytes: bytes = field(default=b"")
    command: list[str] = field(default_factory=list)


class StreamReader(Generic[_S]):
    """Sync and async iterable stream reader.

    StreamReader wraps an asyncio.Queue and provides both synchronous and
    asynchronous iteration interfaces. This enables streaming output to be
    consumed in both sync and async contexts.

    Used as ``StreamReader[str]`` for text streams (exec stdout/stderr, logs)
    and ``StreamReader[bytes]`` for raw byte streams (TTY output).

    The stream uses None as a sentinel value to signal end-of-stream.
    Exception instances in the queue are re-raised to the consumer.

    Examples:
        Synchronous iteration:
        ```python
        for line in process.stdout:
            print(line)
        ```

        Asynchronous iteration:
        ```python
        async for line in process.stdout:
            print(line)
        ```
    """

    def __init__(
        self,
        queue: asyncio.Queue[_S | Exception | None],
        loop_manager: _LoopManager,
        *,
        cancel: Callable[[], object] | None = None,
    ) -> None:
        """Initialize with a queue and loop manager.

        Args:
            queue: The asyncio.Queue to read from. Supports items of the
                parameterized type, None as end-of-stream sentinel, and
                Exception instances which are re-raised to the consumer.
            loop_manager: The _LoopManager for executing async operations.
            cancel: Optional callback to cancel the background producer.
                Called by ``close()`` to stop the stream.
        """
        self._queue = queue
        self._loop_manager = loop_manager
        self._exhausted = False
        self._cancel = cancel

    def __iter__(self) -> StreamReader[_S]:
        """Return self as iterator for sync iteration."""
        return self

    def __next__(self) -> _S:
        """Get next item from stream (blocking).

        Returns:
            The next line from the stream.

        Raises:
            StopIteration: When the stream is exhausted (None sentinel).
            Exception: Re-raised if an Exception instance is in the queue.
        """
        if self._exhausted:
            raise StopIteration
        item = self._loop_manager.run_sync(self._queue.get())
        if item is None:
            self._exhausted = True
            raise StopIteration
        if isinstance(item, Exception):
            self._exhausted = True
            raise item
        return item

    def __aiter__(self) -> StreamReader[_S]:
        """Return self as async iterator for async iteration."""
        return self

    async def __anext__(self) -> _S:
        """Get next item from stream (async).

        Returns:
            The next line from the stream.

        Raises:
            StopAsyncIteration: When the stream is exhausted (None sentinel).
            Exception: Re-raised if an Exception instance is in the queue.
        """
        if self._exhausted:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            self._exhausted = True
            raise StopAsyncIteration
        if isinstance(item, Exception):
            self._exhausted = True
            raise item
        return item

    def close(self) -> None:
        """Cancel the background producer and mark the stream as exhausted.

        Safe to call multiple times. After close(), iteration will raise
        StopIteration/StopAsyncIteration on the next call.

        Puts a None sentinel on the queue to unblock any consumer currently
        waiting inside ``queue.get()``.
        """
        if self._exhausted:
            return
        self._exhausted = True
        if self._cancel is not None:
            self._cancel()
        # Wake any consumer blocked on queue.get() — put_nowait is fine here
        # because we only need *one* sentinel and the queue may have space;
        # if it's full the consumer will drain an item and see _exhausted.
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass


class StreamWriter:
    """Sync and async writer for streaming input to a process.

    StreamWriter wraps a bounded asyncio.Queue and provides both synchronous and
    asynchronous write interfaces. This enables streaming input to be sent in
    both sync and async contexts.

    The stream uses None as a sentinel value to signal end-of-stream (EOF).
    The queue is bounded (~16 items for ~1MB with 64KB chunks) to provide
    backpressure.

    Examples:
        Synchronous write:
        ```python
        process.stdin.write(b"data").result()
        process.stdin.writeline("hello").result()
        process.stdin.close().result()
        ```

        Asynchronous write:
        ```python
        await process.stdin.write(b"data")
        await process.stdin.writeline("hello")
        await process.stdin.close()
        ```
    """

    QUEUE_SIZE = 16  # ~1MB with 64KB chunks

    def __init__(self, queue: asyncio.Queue[bytes | None], loop_manager: _LoopManager) -> None:
        """Initialize with a queue and loop manager.

        Args:
            queue: The bounded asyncio.Queue to write to.
            loop_manager: The _LoopManager for executing async operations.
        """
        self._queue = queue
        self._loop_manager = loop_manager
        self._closed = False
        self._exception: BaseException | None = None

    @property
    def closed(self) -> bool:
        """True if close() has been called."""
        return self._closed

    def _check_writable(self) -> None:
        """Check if the stream is writable.

        Raises:
            SandboxExecutionError: If the stream is closed or has failed.
        """
        if self._exception is not None:
            raise SandboxExecutionError(
                "Cannot write to stdin: stream has failed"
            ) from self._exception
        if self._closed:
            raise SandboxExecutionError("Cannot write to stdin: stream is closed")

    def write(self, data: bytes) -> OperationRef[None]:
        """Write raw bytes to the stream.

        Queues the data to be sent to the process stdin. Blocks (via OperationRef.result())
        if the queue is full, providing backpressure.

        Args:
            data: The bytes to write.

        Returns:
            An OperationRef that completes when the data is queued.

        Raises:
            SandboxExecutionError: If the stream is closed or has failed.
        """
        self._check_writable()

        async def _write() -> None:
            await self._queue.put(data)

        future = self._loop_manager.run_async(_write())
        return OperationRef(future)

    def writeline(self, text: str, encoding: str = "utf-8") -> OperationRef[None]:
        """Write a line of text to the stream.

        Encodes the text, appends a newline, and queues it for sending.

        Args:
            text: The text to write.
            encoding: The text encoding to use. Defaults to "utf-8".

        Returns:
            An OperationRef that completes when the data is queued.

        Raises:
            SandboxExecutionError: If the stream is closed or has failed.
        """
        data = (text + "\n").encode(encoding)
        return self.write(data)

    def close(self) -> OperationRef[None]:
        """Close the stream, sending EOF sentinel.

        The EOF sentinel is queued at the end, so pending writes complete first.
        Multiple calls to close() are idempotent and return immediately.

        Returns:
            An OperationRef that completes when EOF is queued.
        """
        if self._closed:
            # Idempotent: return immediately-completed operation
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return OperationRef(future)

        self._closed = True

        async def _close() -> None:
            await self._queue.put(None)

        future = self._loop_manager.run_async(_close())
        return OperationRef(future)

    def set_exception(self, exception: BaseException) -> None:
        """Store an exception to be raised on subsequent writes.

        Called internally when the stream fails (e.g., process exits).

        Args:
            exception: The exception to store.
        """
        self._exception = exception


class Process(OperationRef[ProcessResult]):
    """Handle for a running process with streaming stdout/stderr.

    Process inherits from OperationRef[ProcessResult] and adds streaming
    capabilities and process-specific methods. It wraps an async operation
    that executes a command in a sandbox.

    The process's output streams (stdout, stderr) can be iterated either
    synchronously or asynchronously. The result() method blocks until
    completion and returns the full ProcessResult.

    Attributes:
        stdout: StreamReader for standard output.
        stderr: StreamReader for standard error.
        stdin: StreamWriter for standard input, or None if stdin streaming is disabled.
        returncode: Exit code from the command, or None if not yet complete.
        command: The command that was executed.

    Examples:
        Basic execution with result:
        ```python
        process = sandbox.exec(["echo", "hello"])
        result = process.result()
        print(result.stdout)  # hello
        ```

        Streaming output:
        ```python
        process = sandbox.exec(["python", "-c", "print('line1'); print('line2')"])
        for line in process.stdout:
            print(f"Got: {line}")
        ```

        Async streaming:
        ```python
        async for line in process.stdout:
            print(f"Got: {line}")
        ```

        Waiting with timeout:
        ```python
        try:
            exit_code = process.wait(timeout=10.0)
        except concurrent.futures.TimeoutError:
            process.cancel()
        ```
    """

    def __init__(
        self,
        future: concurrent.futures.Future[ProcessResult],
        command: list[str],
        stdout: StreamReader[str],
        stderr: StreamReader[str],
        stdin: StreamWriter | None = None,
        stats_callback: Callable[[ProcessResult | None, BaseException | None], None] | None = None,
    ) -> None:
        """Initialize with a future and stream readers.

        Args:
            future: Future that will contain the ProcessResult when complete.
            command: The command being executed.
            stdout: StreamReader for stdout.
            stderr: StreamReader for stderr.
            stdin: StreamWriter for stdin, or None if stdin streaming is disabled.
            stats_callback: Optional callback invoked once when result is available.
                Called with (result, None) on success or (None, exception) on failure.
        """
        super().__init__(future)
        self._command = command
        self._returncode: int | None = None
        self._result: ProcessResult | None = None
        self._exception: BaseException | None = None
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = stdin
        self._stats_callback = stats_callback
        self._stats_recorded = False
        self._stats_lock = threading.Lock()

        # Ensure stats are recorded even if user only streams without calling result()
        if stats_callback is not None:
            future.add_done_callback(self._on_future_done)

    def poll(self) -> int | None:
        """Check if the process has completed without blocking.

        Returns:
            The exit code if the process has completed, None otherwise.
        """
        if self._future.done():
            self._ensure_result()
            return self._returncode
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process completes.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The process exit code.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            concurrent.futures.CancelledError: If the operation was cancelled.
            Exception: Any exception from the execution.
        """
        self._ensure_result(timeout)
        if self._exception is not None:
            raise self._exception
        assert self._returncode is not None
        return self._returncode

    def result(self, timeout: float | None = None) -> ProcessResult:
        """Block until complete and return the full ProcessResult.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The ProcessResult containing stdout, stderr, and exit code.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            concurrent.futures.CancelledError: If the operation was cancelled.
            Exception: Any exception from the execution.
        """
        self._ensure_result(timeout)
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    @property
    def returncode(self) -> int | None:
        """The process exit code, or None if not yet complete."""
        return self._returncode

    @property
    def command(self) -> list[str]:
        """The command that was executed."""
        return self._command

    def cancel(self) -> bool:
        """Attempt to cancel the process.

        Returns:
            True if successfully cancelled, False otherwise.
        """
        return self._future.cancel()

    def _ensure_result(self, timeout: float | None = None) -> None:
        """Ensure result is available, fetching if necessary.

        Args:
            timeout: Maximum seconds to wait.
        """
        if self._result is None and self._exception is None:
            try:
                self._result = self._future.result(timeout)
                self._returncode = self._result.returncode
                self._record_stats()
            except concurrent.futures.TimeoutError:
                # Do not cache timeouts: allow callers to retry with a longer timeout.
                raise
            except Exception as e:
                self._exception = e
                self._record_stats()

    def _record_stats(self) -> None:
        """Record stats via callback exactly once.

        Thread-safe: uses lock to prevent double-counting when callback
        and result() race on different threads. The callback is invoked
        inside the lock to guarantee that when the main thread's
        _record_stats() returns (seeing _stats_recorded=True), the
        callback has already completed. Without this, a race exists
        where _on_future_done sets the flag but hasn't called the
        callback yet, causing result() to return before metrics update.
        """
        if self._stats_callback is None:
            return

        with self._stats_lock:
            if self._stats_recorded:
                return
            self._stats_recorded = True
            self._stats_callback(self._result, self._exception)

    def _on_future_done(self, future: concurrent.futures.Future[ProcessResult]) -> None:
        """Callback invoked when future completes, ensures stats are recorded.

        This handles the case where users only stream stdout/stderr without
        calling result()/wait()/await.
        """
        if self._stats_recorded:
            return

        try:
            result = future.result()
            self._result = result
            self._returncode = result.returncode
        except Exception as e:
            self._exception = e

        self._record_stats()

    def __await__(self) -> Generator[Any, None, ProcessResult]:
        """Make this process awaitable for async contexts.

        Ensures stats are recorded when awaited in async code, including
        on failure.

        Returns:
            Generator that yields the ProcessResult when complete.
        """

        async def _await_and_record() -> ProcessResult:
            try:
                result = await asyncio.wrap_future(self._future)
                self._result = result
                self._returncode = result.returncode
                self._record_stats()
                return result
            except Exception as e:
                self._exception = e
                self._record_stats()
                raise

        return _await_and_record().__await__()


@dataclass(frozen=True)
class TerminalResult:
    """Result from a completed terminal session.

    Unlike ProcessResult, does not contain captured stdout/stderr because
    TTY sessions do not buffer output.

    Attributes:
        returncode: Exit code from the shell process.
        command: The command that was executed.
    """

    returncode: int
    command: list[str] = field(default_factory=list)


class TerminalSession(OperationRef["TerminalResult"]):
    """Handle for an interactive TTY session in a sandbox.

    TerminalSession is purpose-built for interactive use cases where a local
    terminal is connected to a remote shell. Unlike Process:

    - Output is raw bytes (StreamReader[bytes]), preserving ANSI sequences
    - No output buffering -- safe for long-running sessions
    - result() returns TerminalResult (exit code only, no captured output)

    Attributes:
        output: StreamReader[bytes] for raw byte output (merged stdout/stderr)
        stdin: StreamWriter for standard input (always present)

    Examples:
        SDK-level interactive session:
        ```python
        import sys

        session = sandbox.shell(width=80, height=24)
        for chunk in session.output:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        exit_code = session.wait()
        ```

        Async usage:
        ```python
        import sys

        session = sandbox.shell()
        async for chunk in session.output:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        result = await session
        ```
    """

    def __init__(
        self,
        future: concurrent.futures.Future[TerminalResult],
        command: list[str],
        output: StreamReader[bytes],
        stdin: StreamWriter,
        resize_queue: asyncio.Queue[tuple[int, int] | None],
    ) -> None:
        """Initialize with a future and stream components.

        Args:
            future: Future that will contain the TerminalResult when complete.
            command: The command being executed.
            output: StreamReader[bytes] for raw byte output.
            stdin: StreamWriter for stdin input.
            resize_queue: Queue for sending terminal resize messages.
        """
        super().__init__(future)
        self._command = command
        self.output = output
        self.stdin = stdin
        self._resize_queue = resize_queue
        self._result: TerminalResult | None = None
        self._exception: BaseException | None = None

    @property
    def command(self) -> list[str]:
        """The command that was executed."""
        return self._command

    @property
    def returncode(self) -> int | None:
        """The exit code, or None if the session is still active."""
        if self._result is not None:
            return self._result.returncode
        if self._future.done():
            self._fetch_result()
            return self._result.returncode if self._result else None
        return None

    def resize(self, width: int, height: int) -> None:
        """Send terminal resize. Fire-and-forget.

        Args:
            width: New terminal width in columns.
            height: New terminal height in rows.

        Raises:
            SandboxExecutionError: If the session has ended.
        """
        if self._future.done():
            raise SandboxExecutionError("Cannot resize: terminal session has ended")

        from cwsandbox._loop_manager import _LoopManager

        _LoopManager.get().get_loop().call_soon_threadsafe(
            self._resize_queue.put_nowait, (width, height)
        )

    def result(self, timeout: float | None = None) -> TerminalResult:
        """Block until the terminal session ends and return the result.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            TerminalResult with the exit code.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            Exception: Any exception from the session.
        """
        self._fetch_result(timeout)
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    def wait(self, timeout: float | None = None) -> int:
        """Block until the session ends and return exit code.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The process exit code.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            Exception: Any exception from the session.
        """
        return self.result(timeout).returncode

    def _fetch_result(self, timeout: float | None = None) -> None:
        """Fetch result from future if not already cached."""
        if self._result is None and self._exception is None:
            try:
                self._result = self._future.result(timeout)
            except concurrent.futures.TimeoutError:
                raise
            except Exception as e:
                self._exception = e

    def __await__(self) -> Generator[Any, None, TerminalResult]:
        """Make this session awaitable for async contexts.

        Returns:
            Generator that yields the TerminalResult when complete.
        """

        async def _await() -> TerminalResult:
            try:
                result = await asyncio.wrap_future(self._future)
                self._result = result
                return result
            except Exception as e:
                self._exception = e
                raise

        return _await().__await__()
