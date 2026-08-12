# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SandboxMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SANDBOX_MODE_UNSPECIFIED: _ClassVar[SandboxMode]
    SANDBOX_MODE_SERVERLESS: _ClassVar[SandboxMode]
    SANDBOX_MODE_CKS: _ClassVar[SandboxMode]

class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATE_UNSPECIFIED: _ClassVar[State]
    STATE_CREATING: _ClassVar[State]
    STATE_RUNNING: _ClassVar[State]
    STATE_COMPLETED: _ClassVar[State]
    STATE_FAILED: _ClassVar[State]
    STATE_TERMINATING: _ClassVar[State]
    STATE_TERMINATED: _ClassVar[State]
    STATE_PENDING: _ClassVar[State]
    STATE_PAUSED: _ClassVar[State]

class ServiceProtocol(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SERVICE_PROTOCOL_UNSPECIFIED: _ClassVar[ServiceProtocol]
    SERVICE_PROTOCOL_TCP: _ClassVar[ServiceProtocol]
    SERVICE_PROTOCOL_UDP: _ClassVar[ServiceProtocol]
    SERVICE_PROTOCOL_SCTP: _ClassVar[ServiceProtocol]

class EndpointKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENDPOINT_KIND_UNSPECIFIED: _ClassVar[EndpointKind]
    ENDPOINT_KIND_HTTPS: _ClassVar[EndpointKind]
    ENDPOINT_KIND_TLS_PASSTHROUGH: _ClassVar[EndpointKind]

class EndpointAuth(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENDPOINT_AUTH_UNSPECIFIED: _ClassVar[EndpointAuth]
    ENDPOINT_AUTH_OPEN: _ClassVar[EndpointAuth]
    ENDPOINT_AUTH_TOKEN: _ClassVar[EndpointAuth]

class Visibility(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VISIBILITY_UNSPECIFIED: _ClassVar[Visibility]
    VISIBILITY_PUBLIC: _ClassVar[Visibility]
    VISIBILITY_PRIVATE: _ClassVar[Visibility]
    VISIBILITY_CUSTOM: _ClassVar[Visibility]

class TenantScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TENANT_SCOPE_UNSPECIFIED: _ClassVar[TenantScope]
    TENANT_SCOPE_SAME_USER: _ClassVar[TenantScope]
    TENANT_SCOPE_SAME_ORG: _ClassVar[TenantScope]
    TENANT_SCOPE_SANDBOX_NETWORK: _ClassVar[TenantScope]

class StorageMedium(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STORAGE_MEDIUM_UNSPECIFIED: _ClassVar[StorageMedium]
    STORAGE_MEDIUM_DISK: _ClassVar[StorageMedium]
    STORAGE_MEDIUM_MEMORY: _ClassVar[StorageMedium]

class ObjectStoragePermission(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECT_STORAGE_PERMISSION_UNSPECIFIED: _ClassVar[ObjectStoragePermission]
    OBJECT_STORAGE_PERMISSION_READ: _ClassVar[ObjectStoragePermission]
    OBJECT_STORAGE_PERMISSION_READ_WRITE: _ClassVar[ObjectStoragePermission]

class SnapshotState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SNAPSHOT_STATE_UNSPECIFIED: _ClassVar[SnapshotState]
    SNAPSHOT_STATE_CREATING: _ClassVar[SnapshotState]
    SNAPSHOT_STATE_READY: _ClassVar[SnapshotState]
    SNAPSHOT_STATE_FAILED: _ClassVar[SnapshotState]
    SNAPSHOT_STATE_DELETING: _ClassVar[SnapshotState]

class SnapshotTrigger(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SNAPSHOT_TRIGGER_UNSPECIFIED: _ClassVar[SnapshotTrigger]
    SNAPSHOT_TRIGGER_MANUAL: _ClassVar[SnapshotTrigger]
    SNAPSHOT_TRIGGER_ON_DELETE: _ClassVar[SnapshotTrigger]
SANDBOX_MODE_UNSPECIFIED: SandboxMode
SANDBOX_MODE_SERVERLESS: SandboxMode
SANDBOX_MODE_CKS: SandboxMode
STATE_UNSPECIFIED: State
STATE_CREATING: State
STATE_RUNNING: State
STATE_COMPLETED: State
STATE_FAILED: State
STATE_TERMINATING: State
STATE_TERMINATED: State
STATE_PENDING: State
STATE_PAUSED: State
SERVICE_PROTOCOL_UNSPECIFIED: ServiceProtocol
SERVICE_PROTOCOL_TCP: ServiceProtocol
SERVICE_PROTOCOL_UDP: ServiceProtocol
SERVICE_PROTOCOL_SCTP: ServiceProtocol
ENDPOINT_KIND_UNSPECIFIED: EndpointKind
ENDPOINT_KIND_HTTPS: EndpointKind
ENDPOINT_KIND_TLS_PASSTHROUGH: EndpointKind
ENDPOINT_AUTH_UNSPECIFIED: EndpointAuth
ENDPOINT_AUTH_OPEN: EndpointAuth
ENDPOINT_AUTH_TOKEN: EndpointAuth
VISIBILITY_UNSPECIFIED: Visibility
VISIBILITY_PUBLIC: Visibility
VISIBILITY_PRIVATE: Visibility
VISIBILITY_CUSTOM: Visibility
TENANT_SCOPE_UNSPECIFIED: TenantScope
TENANT_SCOPE_SAME_USER: TenantScope
TENANT_SCOPE_SAME_ORG: TenantScope
TENANT_SCOPE_SANDBOX_NETWORK: TenantScope
STORAGE_MEDIUM_UNSPECIFIED: StorageMedium
STORAGE_MEDIUM_DISK: StorageMedium
STORAGE_MEDIUM_MEMORY: StorageMedium
OBJECT_STORAGE_PERMISSION_UNSPECIFIED: ObjectStoragePermission
OBJECT_STORAGE_PERMISSION_READ: ObjectStoragePermission
OBJECT_STORAGE_PERMISSION_READ_WRITE: ObjectStoragePermission
SNAPSHOT_STATE_UNSPECIFIED: SnapshotState
SNAPSHOT_STATE_CREATING: SnapshotState
SNAPSHOT_STATE_READY: SnapshotState
SNAPSHOT_STATE_FAILED: SnapshotState
SNAPSHOT_STATE_DELETING: SnapshotState
SNAPSHOT_TRIGGER_UNSPECIFIED: SnapshotTrigger
SNAPSHOT_TRIGGER_MANUAL: SnapshotTrigger
SNAPSHOT_TRIGGER_ON_DELETE: SnapshotTrigger

class Sandbox(_message.Message):
    __slots__ = ("sandbox_id", "spec", "status", "source_template_id", "source_template_revision")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TEMPLATE_REVISION_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    spec: SandboxSpec
    status: SandboxStatus
    source_template_id: str
    source_template_revision: int
    def __init__(self, sandbox_id: _Optional[str] = ..., spec: _Optional[_Union[SandboxSpec, _Mapping]] = ..., status: _Optional[_Union[SandboxStatus, _Mapping]] = ..., source_template_id: _Optional[str] = ..., source_template_revision: _Optional[int] = ...) -> None: ...

class SandboxSpec(_message.Message):
    __slots__ = ("containers", "primary_container", "volumes", "services", "max_lifetime_seconds", "network_ids", "object_storage_access", "tags", "runner_ids", "network", "annotations", "init_containers", "instance_type", "mode", "runtime_class")
    class AnnotationsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CONTAINERS_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_CONTAINER_FIELD_NUMBER: _ClassVar[int]
    VOLUMES_FIELD_NUMBER: _ClassVar[int]
    SERVICES_FIELD_NUMBER: _ClassVar[int]
    MAX_LIFETIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_IDS_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STORAGE_ACCESS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    RUNNER_IDS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    INIT_CONTAINERS_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_CLASS_FIELD_NUMBER: _ClassVar[int]
    containers: _containers.RepeatedCompositeFieldContainer[Container]
    primary_container: str
    volumes: _containers.RepeatedCompositeFieldContainer[SandboxVolume]
    services: _containers.RepeatedCompositeFieldContainer[Service]
    max_lifetime_seconds: int
    network_ids: _containers.RepeatedScalarFieldContainer[str]
    object_storage_access: ObjectStorageAccess
    tags: _containers.RepeatedScalarFieldContainer[str]
    runner_ids: _containers.RepeatedScalarFieldContainer[str]
    network: NetworkOptions
    annotations: _containers.ScalarMap[str, str]
    init_containers: _containers.RepeatedCompositeFieldContainer[Container]
    instance_type: str
    mode: SandboxMode
    runtime_class: str
    def __init__(self, containers: _Optional[_Iterable[_Union[Container, _Mapping]]] = ..., primary_container: _Optional[str] = ..., volumes: _Optional[_Iterable[_Union[SandboxVolume, _Mapping]]] = ..., services: _Optional[_Iterable[_Union[Service, _Mapping]]] = ..., max_lifetime_seconds: _Optional[int] = ..., network_ids: _Optional[_Iterable[str]] = ..., object_storage_access: _Optional[_Union[ObjectStorageAccess, _Mapping]] = ..., tags: _Optional[_Iterable[str]] = ..., runner_ids: _Optional[_Iterable[str]] = ..., network: _Optional[_Union[NetworkOptions, _Mapping]] = ..., annotations: _Optional[_Mapping[str, str]] = ..., init_containers: _Optional[_Iterable[_Union[Container, _Mapping]]] = ..., instance_type: _Optional[str] = ..., mode: _Optional[_Union[SandboxMode, str]] = ..., runtime_class: _Optional[str] = ...) -> None: ...

class SecurityContext(_message.Message):
    __slots__ = ("run_as_user", "run_as_group", "privileged", "allow_privilege_escalation", "read_only_root_filesystem", "capabilities_add", "capabilities_drop", "seccomp_profile")
    RUN_AS_USER_FIELD_NUMBER: _ClassVar[int]
    RUN_AS_GROUP_FIELD_NUMBER: _ClassVar[int]
    PRIVILEGED_FIELD_NUMBER: _ClassVar[int]
    ALLOW_PRIVILEGE_ESCALATION_FIELD_NUMBER: _ClassVar[int]
    READ_ONLY_ROOT_FILESYSTEM_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_ADD_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_DROP_FIELD_NUMBER: _ClassVar[int]
    SECCOMP_PROFILE_FIELD_NUMBER: _ClassVar[int]
    run_as_user: int
    run_as_group: int
    privileged: bool
    allow_privilege_escalation: bool
    read_only_root_filesystem: bool
    capabilities_add: _containers.RepeatedScalarFieldContainer[str]
    capabilities_drop: _containers.RepeatedScalarFieldContainer[str]
    seccomp_profile: str
    def __init__(self, run_as_user: _Optional[int] = ..., run_as_group: _Optional[int] = ..., privileged: bool = ..., allow_privilege_escalation: bool = ..., read_only_root_filesystem: bool = ..., capabilities_add: _Optional[_Iterable[str]] = ..., capabilities_drop: _Optional[_Iterable[str]] = ..., seccomp_profile: _Optional[str] = ...) -> None: ...

class Container(_message.Message):
    __slots__ = ("name", "image", "command", "args", "environment_variables", "resources", "files", "volume_mounts", "secret_stores", "working_dir", "resource_requirements", "security_context", "image_pull_credentials")
    class EnvironmentVariablesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    VOLUME_MOUNTS_FIELD_NUMBER: _ClassVar[int]
    SECRET_STORES_FIELD_NUMBER: _ClassVar[int]
    WORKING_DIR_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    SECURITY_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_PULL_CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    name: str
    image: str
    command: str
    args: _containers.RepeatedScalarFieldContainer[str]
    environment_variables: _containers.ScalarMap[str, str]
    resources: Resources
    files: _containers.RepeatedCompositeFieldContainer[FileMount]
    volume_mounts: _containers.RepeatedCompositeFieldContainer[VolumeMount]
    secret_stores: _containers.RepeatedCompositeFieldContainer[SecretStoreReference]
    working_dir: str
    resource_requirements: ResourceRequirements
    security_context: SecurityContext
    image_pull_credentials: ImagePullCredentials
    def __init__(self, name: _Optional[str] = ..., image: _Optional[str] = ..., command: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ..., environment_variables: _Optional[_Mapping[str, str]] = ..., resources: _Optional[_Union[Resources, _Mapping]] = ..., files: _Optional[_Iterable[_Union[FileMount, _Mapping]]] = ..., volume_mounts: _Optional[_Iterable[_Union[VolumeMount, _Mapping]]] = ..., secret_stores: _Optional[_Iterable[_Union[SecretStoreReference, _Mapping]]] = ..., working_dir: _Optional[str] = ..., resource_requirements: _Optional[_Union[ResourceRequirements, _Mapping]] = ..., security_context: _Optional[_Union[SecurityContext, _Mapping]] = ..., image_pull_credentials: _Optional[_Union[ImagePullCredentials, _Mapping]] = ...) -> None: ...

class PartialContainer(_message.Message):
    __slots__ = ("name", "image", "command", "args", "environment_variables", "resources", "files", "volume_mounts", "secret_stores", "working_dir", "resource_requirements", "security_context")
    class EnvironmentVariablesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    VOLUME_MOUNTS_FIELD_NUMBER: _ClassVar[int]
    SECRET_STORES_FIELD_NUMBER: _ClassVar[int]
    WORKING_DIR_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    SECURITY_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    name: str
    image: str
    command: str
    args: _containers.RepeatedScalarFieldContainer[str]
    environment_variables: _containers.ScalarMap[str, str]
    resources: Resources
    files: _containers.RepeatedCompositeFieldContainer[FileMount]
    volume_mounts: _containers.RepeatedCompositeFieldContainer[VolumeMount]
    secret_stores: _containers.RepeatedCompositeFieldContainer[SecretStoreReference]
    working_dir: str
    resource_requirements: ResourceRequirements
    security_context: SecurityContext
    def __init__(self, name: _Optional[str] = ..., image: _Optional[str] = ..., command: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ..., environment_variables: _Optional[_Mapping[str, str]] = ..., resources: _Optional[_Union[Resources, _Mapping]] = ..., files: _Optional[_Iterable[_Union[FileMount, _Mapping]]] = ..., volume_mounts: _Optional[_Iterable[_Union[VolumeMount, _Mapping]]] = ..., secret_stores: _Optional[_Iterable[_Union[SecretStoreReference, _Mapping]]] = ..., working_dir: _Optional[str] = ..., resource_requirements: _Optional[_Union[ResourceRequirements, _Mapping]] = ..., security_context: _Optional[_Union[SecurityContext, _Mapping]] = ...) -> None: ...

class PartialSandboxSpec(_message.Message):
    __slots__ = ("containers", "primary_container", "volumes", "services", "max_lifetime_seconds", "network_ids", "object_storage_access", "tags", "runner_ids", "network", "annotations", "init_containers", "instance_type", "mode", "runtime_class")
    class AnnotationsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CONTAINERS_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_CONTAINER_FIELD_NUMBER: _ClassVar[int]
    VOLUMES_FIELD_NUMBER: _ClassVar[int]
    SERVICES_FIELD_NUMBER: _ClassVar[int]
    MAX_LIFETIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_IDS_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STORAGE_ACCESS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    RUNNER_IDS_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    INIT_CONTAINERS_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_CLASS_FIELD_NUMBER: _ClassVar[int]
    containers: _containers.RepeatedCompositeFieldContainer[PartialContainer]
    primary_container: str
    volumes: _containers.RepeatedCompositeFieldContainer[SandboxVolume]
    services: _containers.RepeatedCompositeFieldContainer[Service]
    max_lifetime_seconds: int
    network_ids: _containers.RepeatedScalarFieldContainer[str]
    object_storage_access: ObjectStorageAccess
    tags: _containers.RepeatedScalarFieldContainer[str]
    runner_ids: _containers.RepeatedScalarFieldContainer[str]
    network: NetworkOptions
    annotations: _containers.ScalarMap[str, str]
    init_containers: _containers.RepeatedCompositeFieldContainer[PartialContainer]
    instance_type: str
    mode: SandboxMode
    runtime_class: str
    def __init__(self, containers: _Optional[_Iterable[_Union[PartialContainer, _Mapping]]] = ..., primary_container: _Optional[str] = ..., volumes: _Optional[_Iterable[_Union[SandboxVolume, _Mapping]]] = ..., services: _Optional[_Iterable[_Union[Service, _Mapping]]] = ..., max_lifetime_seconds: _Optional[int] = ..., network_ids: _Optional[_Iterable[str]] = ..., object_storage_access: _Optional[_Union[ObjectStorageAccess, _Mapping]] = ..., tags: _Optional[_Iterable[str]] = ..., runner_ids: _Optional[_Iterable[str]] = ..., network: _Optional[_Union[NetworkOptions, _Mapping]] = ..., annotations: _Optional[_Mapping[str, str]] = ..., init_containers: _Optional[_Iterable[_Union[PartialContainer, _Mapping]]] = ..., instance_type: _Optional[str] = ..., mode: _Optional[_Union[SandboxMode, str]] = ..., runtime_class: _Optional[str] = ...) -> None: ...

class SandboxStatus(_message.Message):
    __slots__ = ("state", "state_reason", "create_time", "start_time", "end_time", "services", "resource_usage", "exit_code", "effective_resources", "effective_max_lifetime_seconds", "container_statuses", "runner_id", "runner_group_id", "effective_ingress", "effective_egress", "effective_resource_requirements", "attached_volume_ids", "effective_runtime_class")
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATE_REASON_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    SERVICES_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_RESOURCES_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_MAX_LIFETIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_STATUSES_FIELD_NUMBER: _ClassVar[int]
    RUNNER_ID_FIELD_NUMBER: _ClassVar[int]
    RUNNER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_INGRESS_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_EGRESS_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_RESOURCE_REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    ATTACHED_VOLUME_IDS_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_RUNTIME_CLASS_FIELD_NUMBER: _ClassVar[int]
    state: State
    state_reason: str
    create_time: _timestamp_pb2.Timestamp
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    services: _containers.RepeatedCompositeFieldContainer[ServiceStatus]
    resource_usage: ResourceUsage
    exit_code: int
    effective_resources: Resources
    effective_max_lifetime_seconds: int
    container_statuses: _containers.RepeatedCompositeFieldContainer[ContainerStatus]
    runner_id: str
    runner_group_id: str
    effective_ingress: _containers.RepeatedCompositeFieldContainer[IngressRule]
    effective_egress: _containers.RepeatedCompositeFieldContainer[EgressRule]
    effective_resource_requirements: ResourceRequirements
    attached_volume_ids: _containers.RepeatedScalarFieldContainer[str]
    effective_runtime_class: str
    def __init__(self, state: _Optional[_Union[State, str]] = ..., state_reason: _Optional[str] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., services: _Optional[_Iterable[_Union[ServiceStatus, _Mapping]]] = ..., resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., exit_code: _Optional[int] = ..., effective_resources: _Optional[_Union[Resources, _Mapping]] = ..., effective_max_lifetime_seconds: _Optional[int] = ..., container_statuses: _Optional[_Iterable[_Union[ContainerStatus, _Mapping]]] = ..., runner_id: _Optional[str] = ..., runner_group_id: _Optional[str] = ..., effective_ingress: _Optional[_Iterable[_Union[IngressRule, _Mapping]]] = ..., effective_egress: _Optional[_Iterable[_Union[EgressRule, _Mapping]]] = ..., effective_resource_requirements: _Optional[_Union[ResourceRequirements, _Mapping]] = ..., attached_volume_ids: _Optional[_Iterable[str]] = ..., effective_runtime_class: _Optional[str] = ...) -> None: ...

class ContainerStatus(_message.Message):
    __slots__ = ("name", "state", "exit_code", "restart_count")
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    RESTART_COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    state: State
    exit_code: int
    restart_count: int
    def __init__(self, name: _Optional[str] = ..., state: _Optional[_Union[State, str]] = ..., exit_code: _Optional[int] = ..., restart_count: _Optional[int] = ...) -> None: ...

class Resources(_message.Message):
    __slots__ = ("cpu", "memory", "gpu")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    cpu: str
    memory: str
    gpu: Gpu
    def __init__(self, cpu: _Optional[str] = ..., memory: _Optional[str] = ..., gpu: _Optional[_Union[Gpu, _Mapping]] = ...) -> None: ...

class ResourceRequirements(_message.Message):
    __slots__ = ("requests", "limits")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    requests: Resources
    limits: Resources
    def __init__(self, requests: _Optional[_Union[Resources, _Mapping]] = ..., limits: _Optional[_Union[Resources, _Mapping]] = ...) -> None: ...

class Gpu(_message.Message):
    __slots__ = ("count", "type", "memory_gb")
    COUNT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    count: int
    type: str
    memory_gb: int
    def __init__(self, count: _Optional[int] = ..., type: _Optional[str] = ..., memory_gb: _Optional[int] = ...) -> None: ...

class Service(_message.Message):
    __slots__ = ("port", "name", "protocol", "visibility", "endpoint")
    PORT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    port: int
    name: str
    protocol: ServiceProtocol
    visibility: Visibility
    endpoint: EndpointSpec
    def __init__(self, port: _Optional[int] = ..., name: _Optional[str] = ..., protocol: _Optional[_Union[ServiceProtocol, str]] = ..., visibility: _Optional[_Union[Visibility, str]] = ..., endpoint: _Optional[_Union[EndpointSpec, _Mapping]] = ...) -> None: ...

class EndpointSpec(_message.Message):
    __slots__ = ("kind", "auth")
    KIND_FIELD_NUMBER: _ClassVar[int]
    AUTH_FIELD_NUMBER: _ClassVar[int]
    kind: EndpointKind
    auth: EndpointAuth
    def __init__(self, kind: _Optional[_Union[EndpointKind, str]] = ..., auth: _Optional[_Union[EndpointAuth, str]] = ...) -> None: ...

class ServiceStatus(_message.Message):
    __slots__ = ("port", "name", "protocol", "visibility", "endpoint", "url")
    PORT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    port: int
    name: str
    protocol: ServiceProtocol
    visibility: Visibility
    endpoint: EndpointStatus
    url: str
    def __init__(self, port: _Optional[int] = ..., name: _Optional[str] = ..., protocol: _Optional[_Union[ServiceProtocol, str]] = ..., visibility: _Optional[_Union[Visibility, str]] = ..., endpoint: _Optional[_Union[EndpointStatus, _Mapping]] = ..., url: _Optional[str] = ...) -> None: ...

class EndpointStatus(_message.Message):
    __slots__ = ("kind", "auth", "url")
    KIND_FIELD_NUMBER: _ClassVar[int]
    AUTH_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    kind: EndpointKind
    auth: EndpointAuth
    url: str
    def __init__(self, kind: _Optional[_Union[EndpointKind, str]] = ..., auth: _Optional[_Union[EndpointAuth, str]] = ..., url: _Optional[str] = ...) -> None: ...

class NetworkOptions(_message.Message):
    __slots__ = ("ingress", "egress", "deny_egress", "deny_ingress")
    INGRESS_FIELD_NUMBER: _ClassVar[int]
    EGRESS_FIELD_NUMBER: _ClassVar[int]
    DENY_EGRESS_FIELD_NUMBER: _ClassVar[int]
    DENY_INGRESS_FIELD_NUMBER: _ClassVar[int]
    ingress: _containers.RepeatedCompositeFieldContainer[IngressRule]
    egress: _containers.RepeatedCompositeFieldContainer[EgressRule]
    deny_egress: bool
    deny_ingress: bool
    def __init__(self, ingress: _Optional[_Iterable[_Union[IngressRule, _Mapping]]] = ..., egress: _Optional[_Iterable[_Union[EgressRule, _Mapping]]] = ..., deny_egress: bool = ..., deny_ingress: bool = ...) -> None: ...

class EgressRule(_message.Message):
    __slots__ = ("cidr", "dns_name", "tenant", "any", "selector", "ports")
    CIDR_FIELD_NUMBER: _ClassVar[int]
    DNS_NAME_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    ANY_FIELD_NUMBER: _ClassVar[int]
    SELECTOR_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    cidr: CidrBlock
    dns_name: str
    tenant: TenantScope
    any: bool
    selector: SelectorBlock
    ports: _containers.RepeatedCompositeFieldContainer[PortRange]
    def __init__(self, cidr: _Optional[_Union[CidrBlock, _Mapping]] = ..., dns_name: _Optional[str] = ..., tenant: _Optional[_Union[TenantScope, str]] = ..., any: bool = ..., selector: _Optional[_Union[SelectorBlock, _Mapping]] = ..., ports: _Optional[_Iterable[_Union[PortRange, _Mapping]]] = ...) -> None: ...

class IngressRule(_message.Message):
    __slots__ = ("cidr", "tenant", "any", "ports")
    CIDR_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    ANY_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    cidr: CidrBlock
    tenant: TenantScope
    any: bool
    ports: _containers.RepeatedCompositeFieldContainer[PortRange]
    def __init__(self, cidr: _Optional[_Union[CidrBlock, _Mapping]] = ..., tenant: _Optional[_Union[TenantScope, str]] = ..., any: bool = ..., ports: _Optional[_Iterable[_Union[PortRange, _Mapping]]] = ...) -> None: ...

class CidrBlock(_message.Message):
    __slots__ = ("cidr",)
    CIDR_FIELD_NUMBER: _ClassVar[int]
    EXCEPT_FIELD_NUMBER: _ClassVar[int]
    cidr: str
    def __init__(self, cidr: _Optional[str] = ..., **kwargs) -> None: ...

class SelectorBlock(_message.Message):
    __slots__ = ("pod_labels", "namespace_labels")
    class PodLabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class NamespaceLabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    POD_LABELS_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_LABELS_FIELD_NUMBER: _ClassVar[int]
    pod_labels: _containers.ScalarMap[str, str]
    namespace_labels: _containers.ScalarMap[str, str]
    def __init__(self, pod_labels: _Optional[_Mapping[str, str]] = ..., namespace_labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PortRange(_message.Message):
    __slots__ = ("protocol", "port", "end_port")
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    END_PORT_FIELD_NUMBER: _ClassVar[int]
    protocol: str
    port: int
    end_port: int
    def __init__(self, protocol: _Optional[str] = ..., port: _Optional[int] = ..., end_port: _Optional[int] = ...) -> None: ...

class FileMount(_message.Message):
    __slots__ = ("path", "content")
    PATH_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    path: str
    content: bytes
    def __init__(self, path: _Optional[str] = ..., content: _Optional[bytes] = ...) -> None: ...

class SandboxVolume(_message.Message):
    __slots__ = ("name", "scratch", "volume_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SCRATCH_FIELD_NUMBER: _ClassVar[int]
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    scratch: ScratchVolume
    volume_id: str
    def __init__(self, name: _Optional[str] = ..., scratch: _Optional[_Union[ScratchVolume, _Mapping]] = ..., volume_id: _Optional[str] = ...) -> None: ...

class ScratchVolume(_message.Message):
    __slots__ = ("size", "medium", "restore_from_snapshot_id")
    SIZE_FIELD_NUMBER: _ClassVar[int]
    MEDIUM_FIELD_NUMBER: _ClassVar[int]
    RESTORE_FROM_SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    size: str
    medium: StorageMedium
    restore_from_snapshot_id: str
    def __init__(self, size: _Optional[str] = ..., medium: _Optional[_Union[StorageMedium, str]] = ..., restore_from_snapshot_id: _Optional[str] = ...) -> None: ...

class VolumeMount(_message.Message):
    __slots__ = ("volume", "mount_path", "read_only", "sub_path")
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    MOUNT_PATH_FIELD_NUMBER: _ClassVar[int]
    READ_ONLY_FIELD_NUMBER: _ClassVar[int]
    SUB_PATH_FIELD_NUMBER: _ClassVar[int]
    volume: str
    mount_path: str
    read_only: bool
    sub_path: str
    def __init__(self, volume: _Optional[str] = ..., mount_path: _Optional[str] = ..., read_only: bool = ..., sub_path: _Optional[str] = ...) -> None: ...

class SecretStoreReference(_message.Message):
    __slots__ = ("store_name", "secrets")
    STORE_NAME_FIELD_NUMBER: _ClassVar[int]
    SECRETS_FIELD_NUMBER: _ClassVar[int]
    store_name: str
    secrets: _containers.RepeatedCompositeFieldContainer[SecretMapping]
    def __init__(self, store_name: _Optional[str] = ..., secrets: _Optional[_Iterable[_Union[SecretMapping, _Mapping]]] = ...) -> None: ...

class SecretMapping(_message.Message):
    __slots__ = ("path", "field", "env_var")
    PATH_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    ENV_VAR_FIELD_NUMBER: _ClassVar[int]
    path: str
    field: str
    env_var: str
    def __init__(self, path: _Optional[str] = ..., field: _Optional[str] = ..., env_var: _Optional[str] = ...) -> None: ...

class ImagePullCredentials(_message.Message):
    __slots__ = ("registry", "credentials")
    REGISTRY_FIELD_NUMBER: _ClassVar[int]
    CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    registry: str
    credentials: SecretSource
    def __init__(self, registry: _Optional[str] = ..., credentials: _Optional[_Union[SecretSource, _Mapping]] = ...) -> None: ...

class SecretSource(_message.Message):
    __slots__ = ("store_name", "path", "field")
    STORE_NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    store_name: str
    path: str
    field: str
    def __init__(self, store_name: _Optional[str] = ..., path: _Optional[str] = ..., field: _Optional[str] = ...) -> None: ...

class ObjectStorageAccess(_message.Message):
    __slots__ = ("buckets", "permission", "object_prefix")
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    PERMISSION_FIELD_NUMBER: _ClassVar[int]
    OBJECT_PREFIX_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedScalarFieldContainer[str]
    permission: ObjectStoragePermission
    object_prefix: str
    def __init__(self, buckets: _Optional[_Iterable[str]] = ..., permission: _Optional[_Union[ObjectStoragePermission, str]] = ..., object_prefix: _Optional[str] = ...) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("cpu_millicores", "memory_mb", "gpu_count")
    CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores: int
    memory_mb: int
    gpu_count: int
    def __init__(self, cpu_millicores: _Optional[int] = ..., memory_mb: _Optional[int] = ..., gpu_count: _Optional[int] = ...) -> None: ...

class CreateSandboxRequest(_message.Message):
    __slots__ = ("sandbox", "request_id")
    SANDBOX_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    sandbox: Sandbox
    request_id: str
    def __init__(self, sandbox: _Optional[_Union[Sandbox, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class CreateSandboxFromTemplateRequest(_message.Message):
    __slots__ = ("template_id", "overrides", "request_id")
    TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    template_id: str
    overrides: PartialSandboxSpec
    request_id: str
    def __init__(self, template_id: _Optional[str] = ..., overrides: _Optional[_Union[PartialSandboxSpec, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class GetSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id",)
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    def __init__(self, sandbox_id: _Optional[str] = ...) -> None: ...

class ListSandboxesRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "tags", "state", "show_terminated", "runner_ids", "volume_ids")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    SHOW_TERMINATED_FIELD_NUMBER: _ClassVar[int]
    RUNNER_IDS_FIELD_NUMBER: _ClassVar[int]
    VOLUME_IDS_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    state: State
    show_terminated: bool
    runner_ids: _containers.RepeatedScalarFieldContainer[str]
    volume_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., state: _Optional[_Union[State, str]] = ..., show_terminated: bool = ..., runner_ids: _Optional[_Iterable[str]] = ..., volume_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ListSandboxesResponse(_message.Message):
    __slots__ = ("sandboxes", "next_page_token")
    SANDBOXES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    sandboxes: _containers.RepeatedCompositeFieldContainer[Sandbox]
    next_page_token: str
    def __init__(self, sandboxes: _Optional[_Iterable[_Union[Sandbox, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class DeleteSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "grace_period_seconds", "snapshot_volumes", "request_id", "allow_missing")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    GRACE_PERIOD_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_VOLUMES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    grace_period_seconds: int
    snapshot_volumes: _containers.RepeatedScalarFieldContainer[str]
    request_id: str
    allow_missing: bool
    def __init__(self, sandbox_id: _Optional[str] = ..., grace_period_seconds: _Optional[int] = ..., snapshot_volumes: _Optional[_Iterable[str]] = ..., request_id: _Optional[str] = ..., allow_missing: bool = ...) -> None: ...

class DeleteSandboxResponse(_message.Message):
    __slots__ = ("sandbox", "file_system_snapshot_ids")
    SANDBOX_FIELD_NUMBER: _ClassVar[int]
    FILE_SYSTEM_SNAPSHOT_IDS_FIELD_NUMBER: _ClassVar[int]
    sandbox: Sandbox
    file_system_snapshot_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, sandbox: _Optional[_Union[Sandbox, _Mapping]] = ..., file_system_snapshot_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class PurgeSandboxesRequest(_message.Message):
    __slots__ = ("tags", "dry_run", "request_id", "purge_all")
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PURGE_ALL_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.RepeatedScalarFieldContainer[str]
    dry_run: bool
    request_id: str
    purge_all: bool
    def __init__(self, tags: _Optional[_Iterable[str]] = ..., dry_run: bool = ..., request_id: _Optional[str] = ..., purge_all: bool = ...) -> None: ...

class PurgeSandboxesResponse(_message.Message):
    __slots__ = ("purge_count", "failed_sandbox_ids", "retryable_failed_sandbox_ids")
    PURGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_SANDBOX_IDS_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FAILED_SANDBOX_IDS_FIELD_NUMBER: _ClassVar[int]
    purge_count: int
    failed_sandbox_ids: _containers.RepeatedScalarFieldContainer[str]
    retryable_failed_sandbox_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, purge_count: _Optional[int] = ..., failed_sandbox_ids: _Optional[_Iterable[str]] = ..., retryable_failed_sandbox_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ExecRequest(_message.Message):
    __slots__ = ("sandbox_id", "command", "env", "max_output_bytes", "container")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    MAX_OUTPUT_BYTES_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    command: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.ScalarMap[str, str]
    max_output_bytes: int
    container: str
    def __init__(self, sandbox_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., env: _Optional[_Mapping[str, str]] = ..., max_output_bytes: _Optional[int] = ..., container: _Optional[str] = ...) -> None: ...

class ExecResponse(_message.Message):
    __slots__ = ("stdout", "stderr", "exit_code", "stdout_truncated", "stderr_truncated", "stdout_bytes_produced", "stderr_bytes_produced")
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    STDOUT_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    STDERR_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    STDOUT_BYTES_PRODUCED_FIELD_NUMBER: _ClassVar[int]
    STDERR_BYTES_PRODUCED_FIELD_NUMBER: _ClassVar[int]
    stdout: bytes
    stderr: bytes
    exit_code: int
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_bytes_produced: int
    stderr_bytes_produced: int
    def __init__(self, stdout: _Optional[bytes] = ..., stderr: _Optional[bytes] = ..., exit_code: _Optional[int] = ..., stdout_truncated: bool = ..., stderr_truncated: bool = ..., stdout_bytes_produced: _Optional[int] = ..., stderr_bytes_produced: _Optional[int] = ...) -> None: ...

class ExecStreamRequest(_message.Message):
    __slots__ = ("init", "stdin", "resize", "close")
    INIT_FIELD_NUMBER: _ClassVar[int]
    STDIN_FIELD_NUMBER: _ClassVar[int]
    RESIZE_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    init: ExecStreamInit
    stdin: bytes
    resize: ExecStreamResize
    close: ExecStreamClose
    def __init__(self, init: _Optional[_Union[ExecStreamInit, _Mapping]] = ..., stdin: _Optional[bytes] = ..., resize: _Optional[_Union[ExecStreamResize, _Mapping]] = ..., close: _Optional[_Union[ExecStreamClose, _Mapping]] = ...) -> None: ...

class ExecStreamClose(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ExecStreamInit(_message.Message):
    __slots__ = ("sandbox_id", "command", "tty", "tty_width", "tty_height", "env", "container")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TTY_FIELD_NUMBER: _ClassVar[int]
    TTY_WIDTH_FIELD_NUMBER: _ClassVar[int]
    TTY_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    command: _containers.RepeatedScalarFieldContainer[str]
    tty: bool
    tty_width: int
    tty_height: int
    env: _containers.ScalarMap[str, str]
    container: str
    def __init__(self, sandbox_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., tty: bool = ..., tty_width: _Optional[int] = ..., tty_height: _Optional[int] = ..., env: _Optional[_Mapping[str, str]] = ..., container: _Optional[str] = ...) -> None: ...

class ExecStreamResize(_message.Message):
    __slots__ = ("width", "height")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class ExecStreamResponse(_message.Message):
    __slots__ = ("ready", "output", "exit", "error")
    READY_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    EXIT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ready: ExecStreamReady
    output: ExecStreamOutput
    exit: ExecStreamExit
    error: ExecStreamError
    def __init__(self, ready: _Optional[_Union[ExecStreamReady, _Mapping]] = ..., output: _Optional[_Union[ExecStreamOutput, _Mapping]] = ..., exit: _Optional[_Union[ExecStreamExit, _Mapping]] = ..., error: _Optional[_Union[ExecStreamError, _Mapping]] = ...) -> None: ...

class ExecStreamReady(_message.Message):
    __slots__ = ("ready_time",)
    READY_TIME_FIELD_NUMBER: _ClassVar[int]
    ready_time: _timestamp_pb2.Timestamp
    def __init__(self, ready_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExecStreamOutput(_message.Message):
    __slots__ = ("stream", "data")
    class Stream(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STREAM_UNSPECIFIED: _ClassVar[ExecStreamOutput.Stream]
        STREAM_STDOUT: _ClassVar[ExecStreamOutput.Stream]
        STREAM_STDERR: _ClassVar[ExecStreamOutput.Stream]
    STREAM_UNSPECIFIED: ExecStreamOutput.Stream
    STREAM_STDOUT: ExecStreamOutput.Stream
    STREAM_STDERR: ExecStreamOutput.Stream
    STREAM_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    stream: ExecStreamOutput.Stream
    data: bytes
    def __init__(self, stream: _Optional[_Union[ExecStreamOutput.Stream, str]] = ..., data: _Optional[bytes] = ...) -> None: ...

class ExecStreamExit(_message.Message):
    __slots__ = ("exit_code", "completed_time")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_TIME_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    completed_time: _timestamp_pb2.Timestamp
    def __init__(self, exit_code: _Optional[int] = ..., completed_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExecStreamError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class StreamLogsRequest(_message.Message):
    __slots__ = ("sandbox_id", "follow", "tail_lines", "since_time", "container", "timestamps", "resume_log_session_id", "resume_log_offset")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    FOLLOW_FIELD_NUMBER: _ClassVar[int]
    TAIL_LINES_FIELD_NUMBER: _ClassVar[int]
    SINCE_TIME_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    RESUME_LOG_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    RESUME_LOG_OFFSET_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    follow: bool
    tail_lines: int
    since_time: _timestamp_pb2.Timestamp
    container: str
    timestamps: bool
    resume_log_session_id: str
    resume_log_offset: int
    def __init__(self, sandbox_id: _Optional[str] = ..., follow: bool = ..., tail_lines: _Optional[int] = ..., since_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., container: _Optional[str] = ..., timestamps: bool = ..., resume_log_session_id: _Optional[str] = ..., resume_log_offset: _Optional[int] = ...) -> None: ...

class LogEntry(_message.Message):
    __slots__ = ("data", "timestamp", "log_session_id", "next_log_offset", "error")
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LOG_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NEXT_LOG_OFFSET_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    timestamp: _timestamp_pb2.Timestamp
    log_session_id: str
    next_log_offset: int
    error: LogStreamError
    def __init__(self, data: _Optional[bytes] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., log_session_id: _Optional[str] = ..., next_log_offset: _Optional[int] = ..., error: _Optional[_Union[LogStreamError, _Mapping]] = ...) -> None: ...

class LogStreamError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class WriteFileRequest(_message.Message):
    __slots__ = ("sandbox_id", "path", "content", "container")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    path: str
    content: bytes
    container: str
    def __init__(self, sandbox_id: _Optional[str] = ..., path: _Optional[str] = ..., content: _Optional[bytes] = ..., container: _Optional[str] = ...) -> None: ...

class WriteFileResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReadFileRequest(_message.Message):
    __slots__ = ("sandbox_id", "path", "container")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    path: str
    container: str
    def __init__(self, sandbox_id: _Optional[str] = ..., path: _Optional[str] = ..., container: _Optional[str] = ...) -> None: ...

class ReadFileResponse(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    def __init__(self, content: _Optional[bytes] = ...) -> None: ...

class FileSystemSnapshot(_message.Message):
    __slots__ = ("file_system_snapshot_id", "state", "state_reason", "size_bytes", "source_sandbox_id", "trigger", "create_time", "complete_time", "source_volume_name", "object_bucket", "updated_at", "request_id")
    FILE_SYSTEM_SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATE_REASON_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_TIME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_VOLUME_NAME_FIELD_NUMBER: _ClassVar[int]
    OBJECT_BUCKET_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    file_system_snapshot_id: str
    state: SnapshotState
    state_reason: str
    size_bytes: int
    source_sandbox_id: str
    trigger: SnapshotTrigger
    create_time: _timestamp_pb2.Timestamp
    complete_time: _timestamp_pb2.Timestamp
    source_volume_name: str
    object_bucket: str
    updated_at: _timestamp_pb2.Timestamp
    request_id: str
    def __init__(self, file_system_snapshot_id: _Optional[str] = ..., state: _Optional[_Union[SnapshotState, str]] = ..., state_reason: _Optional[str] = ..., size_bytes: _Optional[int] = ..., source_sandbox_id: _Optional[str] = ..., trigger: _Optional[_Union[SnapshotTrigger, str]] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., complete_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., source_volume_name: _Optional[str] = ..., object_bucket: _Optional[str] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class CreateFileSystemSnapshotRequest(_message.Message):
    __slots__ = ("sandbox_id", "request_id", "scratch_volume_name")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SCRATCH_VOLUME_NAME_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    request_id: str
    scratch_volume_name: str
    def __init__(self, sandbox_id: _Optional[str] = ..., request_id: _Optional[str] = ..., scratch_volume_name: _Optional[str] = ...) -> None: ...

class GetFileSystemSnapshotRequest(_message.Message):
    __slots__ = ("file_system_snapshot_id",)
    FILE_SYSTEM_SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    file_system_snapshot_id: str
    def __init__(self, file_system_snapshot_id: _Optional[str] = ...) -> None: ...

class ListFileSystemSnapshotsRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "source_sandbox_id")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    source_sandbox_id: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., source_sandbox_id: _Optional[str] = ...) -> None: ...

class ListFileSystemSnapshotsResponse(_message.Message):
    __slots__ = ("file_system_snapshots", "next_page_token")
    FILE_SYSTEM_SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    file_system_snapshots: _containers.RepeatedCompositeFieldContainer[FileSystemSnapshot]
    next_page_token: str
    def __init__(self, file_system_snapshots: _Optional[_Iterable[_Union[FileSystemSnapshot, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class DeleteFileSystemSnapshotRequest(_message.Message):
    __slots__ = ("file_system_snapshot_id", "allow_missing")
    FILE_SYSTEM_SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    file_system_snapshot_id: str
    allow_missing: bool
    def __init__(self, file_system_snapshot_id: _Optional[str] = ..., allow_missing: bool = ...) -> None: ...
