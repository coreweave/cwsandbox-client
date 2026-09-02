# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Create-spec helpers: SDK dataclasses <-> v1 sandbox protobuf."""

from __future__ import annotations

from typing import Any

from cwsandbox._proto import sandbox_pb2
from cwsandbox._types import (
    CidrBlock,
    EgressRule,
    IngressRule,
    NetworkOptions,
    ObjectStorageAccess,
    ObjectStoragePermission,
    PortRange,
    ScratchVolumeOptions,
    SecurityContext,
    SelectorBlock,
    StorageMedium,
    TenantScope,
    _coerce_security_context,
    _coerce_volume_options,
    _is_https_443_port,
    _validate_sub_path,
)

_TENANT_TO_PROTO = {
    TenantScope.UNSPECIFIED: sandbox_pb2.TENANT_SCOPE_UNSPECIFIED,
    TenantScope.SAME_USER: sandbox_pb2.TENANT_SCOPE_SAME_USER,
    TenantScope.SAME_ORG: sandbox_pb2.TENANT_SCOPE_SAME_ORG,
    TenantScope.SANDBOX_NETWORK: sandbox_pb2.TENANT_SCOPE_SANDBOX_NETWORK,
}
_TENANT_FROM_PROTO = {value: key for key, value in _TENANT_TO_PROTO.items()}

_MEDIUM_TO_PROTO = {
    StorageMedium.UNSPECIFIED: sandbox_pb2.STORAGE_MEDIUM_UNSPECIFIED,
    StorageMedium.DISK: sandbox_pb2.STORAGE_MEDIUM_DISK,
    StorageMedium.MEMORY: sandbox_pb2.STORAGE_MEDIUM_MEMORY,
}

_OSA_PERM_TO_PROTO = {
    ObjectStoragePermission.UNSPECIFIED: sandbox_pb2.OBJECT_STORAGE_PERMISSION_UNSPECIFIED,
    ObjectStoragePermission.READ: sandbox_pb2.OBJECT_STORAGE_PERMISSION_READ,
    ObjectStoragePermission.READ_WRITE: sandbox_pb2.OBJECT_STORAGE_PERMISSION_READ_WRITE,
}


def security_context_to_proto(ctx: SecurityContext) -> sandbox_pb2.SecurityContext:
    proto = sandbox_pb2.SecurityContext()
    if ctx.run_as_user is not None:
        proto.run_as_user = ctx.run_as_user
    if ctx.run_as_group is not None:
        proto.run_as_group = ctx.run_as_group
    if ctx.privileged is not None:
        proto.privileged = ctx.privileged
    if ctx.allow_privilege_escalation is not None:
        proto.allow_privilege_escalation = ctx.allow_privilege_escalation
    if ctx.read_only_root_filesystem is not None:
        proto.read_only_root_filesystem = ctx.read_only_root_filesystem
    if ctx.capabilities_add:
        proto.capabilities_add.extend(ctx.capabilities_add)
    if ctx.capabilities_drop:
        proto.capabilities_drop.extend(ctx.capabilities_drop)
    if ctx.seccomp_profile:
        proto.seccomp_profile = ctx.seccomp_profile
    return proto


def volume_mount_to_proto(
    volume: str,
    mount_path: str,
    *,
    sub_path: str | None = None,
    read_only: bool = False,
) -> sandbox_pb2.VolumeMount:
    mount = sandbox_pb2.VolumeMount(volume=volume, mount_path=mount_path)
    if sub_path:
        mount.sub_path = sub_path
    if read_only:
        mount.read_only = True
    return mount


def scratch_volume_to_proto(vol: ScratchVolumeOptions) -> sandbox_pb2.SandboxVolume:
    scratch: dict[str, Any] = {}
    if vol.size:
        scratch["size"] = vol.size
    if vol.restore_from_snapshot_id:
        scratch["restore_from_snapshot_id"] = vol.restore_from_snapshot_id
    if vol.medium not in (None, StorageMedium.UNSPECIFIED):
        scratch["medium"] = _MEDIUM_TO_PROTO[vol.medium]  # type: ignore[index]
    return sandbox_pb2.SandboxVolume(name=vol.name, scratch=scratch)


def volumes_to_proto(
    volumes_arg: list[Any],
) -> tuple[list[sandbox_pb2.SandboxVolume], list[sandbox_pb2.VolumeMount], tuple[str, ...]]:
    volumes: list[sandbox_pb2.SandboxVolume] = []
    mounts: list[sandbox_pb2.VolumeMount] = []
    scratch_names: list[str] = []
    for raw in volumes_arg:
        if isinstance(raw, dict) and "volume_id" in raw and not raw.get("mount_path"):
            volumes.append(sandbox_pb2.SandboxVolume(name=raw["name"], volume_id=raw["volume_id"]))
            continue
        vol = _coerce_volume_options(raw)
        if isinstance(vol, ScratchVolumeOptions):
            volumes.append(scratch_volume_to_proto(vol))
            if vol.mount_path:
                mounts.append(
                    volume_mount_to_proto(
                        vol.name, vol.mount_path, sub_path=vol.sub_path, read_only=vol.read_only
                    )
                )
            scratch_names.append(vol.name)
        else:
            volumes.append(sandbox_pb2.SandboxVolume(name=vol.name, volume_id=vol.volume_id))
            if vol.mount_path:
                mounts.append(
                    volume_mount_to_proto(
                        vol.name, vol.mount_path, sub_path=vol.sub_path, read_only=vol.read_only
                    )
                )
    return volumes, mounts, tuple(scratch_names)


def _ports_to_proto(ports: tuple[PortRange, ...] | None) -> list[sandbox_pb2.PortRange]:
    if not ports:
        return []
    out: list[sandbox_pb2.PortRange] = []
    for port in ports:
        proto = sandbox_pb2.PortRange(port=port.port)
        if port.end_port is not None:
            proto.end_port = port.end_port
        if port.protocol:
            proto.protocol = port.protocol
        out.append(proto)
    return out


def _ports_from_proto(ports: Any) -> tuple[PortRange, ...] | None:
    if not ports:
        return None
    return tuple(
        PortRange(
            port=p.port,
            end_port=p.end_port or None,
            protocol=p.protocol or None,
        )
        for p in ports
    )


def _cidr_to_proto(block: CidrBlock) -> sandbox_pb2.CidrBlock:
    proto = sandbox_pb2.CidrBlock(cidr=block.cidr)
    if block.except_cidrs:
        getattr(proto, "except").extend(block.except_cidrs)
    return proto


def _cidr_from_proto(block: sandbox_pb2.CidrBlock) -> CidrBlock:
    return CidrBlock(cidr=block.cidr, except_cidrs=tuple(getattr(block, "except")))


def egress_rule_to_proto(rule: EgressRule) -> sandbox_pb2.EgressRule:
    proto = sandbox_pb2.EgressRule()
    if rule.dns_name:
        proto.dns_name = rule.dns_name
    elif rule.cidr is not None:
        proto.cidr.CopyFrom(_cidr_to_proto(rule.cidr))  # type: ignore[arg-type]
    elif rule.tenant is not None:
        proto.tenant = _TENANT_TO_PROTO[rule.tenant]  # type: ignore[index]
    elif rule.any:
        proto.any = True
    elif rule.selector is not None:
        selector = rule.selector
        assert isinstance(selector, SelectorBlock)
        proto.selector.pod_labels.update(selector.pod_labels)
        if selector.namespace_labels:
            proto.selector.namespace_labels.update(selector.namespace_labels)
    proto.ports.extend(_ports_to_proto(rule.ports))  # type: ignore[arg-type]
    if rule.dns_name_except:
        proto.dns_name_except.extend(rule.dns_name_except)
    return proto


def egress_rule_from_proto(rule: sandbox_pb2.EgressRule) -> EgressRule:
    dest = rule.WhichOneof("destination")
    ports = _ports_from_proto(rule.ports)
    if dest == "dns_name" and ports and not (len(ports) == 1 and _is_https_443_port(ports[0])):
        ports = None
    kwargs: dict[str, Any] = {"ports": ports}
    if dest == "dns_name":
        kwargs["dns_name"] = rule.dns_name
    elif dest == "cidr":
        kwargs["cidr"] = _cidr_from_proto(rule.cidr)
    elif dest == "tenant":
        kwargs["tenant"] = _TENANT_FROM_PROTO.get(rule.tenant, TenantScope.UNSPECIFIED)
    elif dest == "any":
        kwargs["any"] = True
    elif dest == "selector":
        kwargs["selector"] = SelectorBlock(
            pod_labels=dict(rule.selector.pod_labels),
            namespace_labels=dict(rule.selector.namespace_labels) or None,
        )
    else:
        raise ValueError("effective egress rule is missing a destination")
    return EgressRule(**kwargs)


def ingress_rule_to_proto(rule: IngressRule) -> sandbox_pb2.IngressRule:
    proto = sandbox_pb2.IngressRule()
    if rule.cidr is not None:
        proto.cidr.CopyFrom(_cidr_to_proto(rule.cidr))  # type: ignore[arg-type]
    elif rule.tenant is not None:
        proto.tenant = _TENANT_TO_PROTO[rule.tenant]  # type: ignore[index]
    elif rule.any:
        proto.any = True
    proto.ports.extend(_ports_to_proto(rule.ports))  # type: ignore[arg-type]
    return proto


def ingress_rule_from_proto(rule: sandbox_pb2.IngressRule) -> IngressRule:
    source = rule.WhichOneof("source")
    kwargs: dict[str, Any] = {"ports": _ports_from_proto(rule.ports)}
    if source == "cidr":
        kwargs["cidr"] = _cidr_from_proto(rule.cidr)
    elif source == "tenant":
        kwargs["tenant"] = _TENANT_FROM_PROTO.get(rule.tenant, TenantScope.UNSPECIFIED)
    elif source == "any":
        kwargs["any"] = True
    else:
        raise ValueError("effective ingress rule is missing a source")
    return IngressRule(**kwargs)


def network_to_proto(network: NetworkOptions) -> sandbox_pb2.NetworkOptions:
    proto = sandbox_pb2.NetworkOptions()
    if network.deny_egress is not None:
        proto.deny_egress = network.deny_egress
    if network.deny_ingress is not None:
        proto.deny_ingress = network.deny_ingress
    for egress_rule in network.egress or ():
        egress = egress_rule if isinstance(egress_rule, EgressRule) else EgressRule(**egress_rule)
        proto.egress.append(egress_rule_to_proto(egress))
    for ingress_rule in network.ingress or ():
        ingress = (
            ingress_rule if isinstance(ingress_rule, IngressRule) else IngressRule(**ingress_rule)
        )
        proto.ingress.append(ingress_rule_to_proto(ingress))
    return proto


def object_storage_to_proto(access: ObjectStorageAccess) -> sandbox_pb2.ObjectStorageAccess:
    proto = sandbox_pb2.ObjectStorageAccess(buckets=list(access.buckets))
    if access.permission not in (None, ObjectStoragePermission.UNSPECIFIED):
        proto.permission = _OSA_PERM_TO_PROTO[access.permission]  # type: ignore[index]
    if access.object_prefix:
        proto.object_prefix = access.object_prefix
    return proto


def coerce_security_context(
    value: SecurityContext | dict[str, Any] | None,
) -> SecurityContext | None:
    return _coerce_security_context(value)


def coerce_sub_path(value: str | None, *, field: str) -> str | None:
    return _validate_sub_path(value, field=field)
