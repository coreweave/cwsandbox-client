# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from cwsandbox._proto import secrets_pb2 as _secrets_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SandboxStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SANDBOX_STATUS_UNSPECIFIED: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_CREATING: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_RUNNING: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_COMPLETED: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_FAILED: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_TERMINATED: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_PENDING: _ClassVar[SandboxStatus]
    SANDBOX_STATUS_PAUSED: _ClassVar[SandboxStatus]

class ActionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_TYPE_UNSPECIFIED: _ClassVar[ActionType]
    ACTION_TYPE_EXEC: _ClassVar[ActionType]
    ACTION_TYPE_ADD_FILE: _ClassVar[ActionType]
    ACTION_TYPE_RETRIEVE_FILE: _ClassVar[ActionType]
    ACTION_TYPE_GET_LOGS: _ClassVar[ActionType]
    ACTION_TYPE_SNAPSHOT: _ClassVar[ActionType]
    ACTION_TYPE_RESTORE: _ClassVar[ActionType]
    ACTION_TYPE_STATUS: _ClassVar[ActionType]
    ACTION_TYPE_STOP: _ClassVar[ActionType]

class EgressType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EGRESS_TYPE_UNSPECIFIED: _ClassVar[EgressType]
    EGRESS_TYPE_NONE: _ClassVar[EgressType]
    EGRESS_TYPE_INTERNET: _ClassVar[EgressType]
    EGRESS_TYPE_USER: _ClassVar[EgressType]
    EGRESS_TYPE_ORG: _ClassVar[EgressType]
    EGRESS_TYPE_RUNWAY: _ClassVar[EgressType]
    EGRESS_TYPE_ALLOWLIST: _ClassVar[EgressType]

class ObjectStoragePermission(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECT_STORAGE_PERMISSION_UNSPECIFIED: _ClassVar[ObjectStoragePermission]
    OBJECT_STORAGE_PERMISSION_READ: _ClassVar[ObjectStoragePermission]
    OBJECT_STORAGE_PERMISSION_READ_WRITE: _ClassVar[ObjectStoragePermission]
SANDBOX_STATUS_UNSPECIFIED: SandboxStatus
SANDBOX_STATUS_CREATING: SandboxStatus
SANDBOX_STATUS_RUNNING: SandboxStatus
SANDBOX_STATUS_COMPLETED: SandboxStatus
SANDBOX_STATUS_FAILED: SandboxStatus
SANDBOX_STATUS_TERMINATED: SandboxStatus
SANDBOX_STATUS_PENDING: SandboxStatus
SANDBOX_STATUS_PAUSED: SandboxStatus
ACTION_TYPE_UNSPECIFIED: ActionType
ACTION_TYPE_EXEC: ActionType
ACTION_TYPE_ADD_FILE: ActionType
ACTION_TYPE_RETRIEVE_FILE: ActionType
ACTION_TYPE_GET_LOGS: ActionType
ACTION_TYPE_SNAPSHOT: ActionType
ACTION_TYPE_RESTORE: ActionType
ACTION_TYPE_STATUS: ActionType
ACTION_TYPE_STOP: ActionType
EGRESS_TYPE_UNSPECIFIED: EgressType
EGRESS_TYPE_NONE: EgressType
EGRESS_TYPE_INTERNET: EgressType
EGRESS_TYPE_USER: EgressType
EGRESS_TYPE_ORG: EgressType
EGRESS_TYPE_RUNWAY: EgressType
EGRESS_TYPE_ALLOWLIST: EgressType
OBJECT_STORAGE_PERMISSION_UNSPECIFIED: ObjectStoragePermission
OBJECT_STORAGE_PERMISSION_READ: ObjectStoragePermission
OBJECT_STORAGE_PERMISSION_READ_WRITE: ObjectStoragePermission

class MountedFile(_message.Message):
    __slots__ = ("mount_path", "file_content")
    MOUNT_PATH_FIELD_NUMBER: _ClassVar[int]
    FILE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    mount_path: str
    file_content: bytes
    def __init__(self, mount_path: _Optional[str] = ..., file_content: _Optional[bytes] = ...) -> None: ...

class GpuRequest(_message.Message):
    __slots__ = ("gpu_count", "gpu_type", "gpu_memory_gb")
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    GPU_TYPE_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    gpu_count: int
    gpu_type: str
    gpu_memory_gb: int
    def __init__(self, gpu_count: _Optional[int] = ..., gpu_type: _Optional[str] = ..., gpu_memory_gb: _Optional[int] = ...) -> None: ...

class ResourceRequest(_message.Message):
    __slots__ = ("cpu", "memory", "gpu")
    CPU_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    cpu: str
    memory: str
    gpu: GpuRequest
    def __init__(self, cpu: _Optional[str] = ..., memory: _Optional[str] = ..., gpu: _Optional[_Union[GpuRequest, _Mapping]] = ...) -> None: ...

class Port(_message.Message):
    __slots__ = ("container_port", "name", "protocol")
    CONTAINER_PORT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    container_port: int
    name: str
    protocol: str
    def __init__(self, container_port: _Optional[int] = ..., name: _Optional[str] = ..., protocol: _Optional[str] = ...) -> None: ...

class ServiceConfig(_message.Message):
    __slots__ = ("exposed_ports",)
    EXPOSED_PORTS_FIELD_NUMBER: _ClassVar[int]
    exposed_ports: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, exposed_ports: _Optional[_Iterable[int]] = ...) -> None: ...

class NetworkOptions(_message.Message):
    __slots__ = ("ingress_mode", "exposed_ports", "egress_mode")
    INGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    EXPOSED_PORTS_FIELD_NUMBER: _ClassVar[int]
    EGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    ingress_mode: str
    exposed_ports: _containers.RepeatedScalarFieldContainer[int]
    egress_mode: str
    def __init__(self, ingress_mode: _Optional[str] = ..., exposed_ports: _Optional[_Iterable[int]] = ..., egress_mode: _Optional[str] = ...) -> None: ...

class KubernetesSecretSource(_message.Message):
    __slots__ = ("secret_name", "secret_key")
    SECRET_NAME_FIELD_NUMBER: _ClassVar[int]
    SECRET_KEY_FIELD_NUMBER: _ClassVar[int]
    secret_name: str
    secret_key: str
    def __init__(self, secret_name: _Optional[str] = ..., secret_key: _Optional[str] = ...) -> None: ...

class TowerClusterSecretReference(_message.Message):
    __slots__ = ("kubernetes", "env_var")
    KUBERNETES_FIELD_NUMBER: _ClassVar[int]
    ENV_VAR_FIELD_NUMBER: _ClassVar[int]
    kubernetes: KubernetesSecretSource
    env_var: str
    def __init__(self, kubernetes: _Optional[_Union[KubernetesSecretSource, _Mapping]] = ..., env_var: _Optional[str] = ...) -> None: ...

class S3Mount(_message.Message):
    __slots__ = ("bucket", "directory", "mount_path")
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    MOUNT_PATH_FIELD_NUMBER: _ClassVar[int]
    bucket: str
    directory: str
    mount_path: str
    def __init__(self, bucket: _Optional[str] = ..., directory: _Optional[str] = ..., mount_path: _Optional[str] = ...) -> None: ...

class ExecPayload(_message.Message):
    __slots__ = ("command", "args")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    command: str
    args: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, command: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ...) -> None: ...

class ExecResponse(_message.Message):
    __slots__ = ("stdout", "stderr", "exit_code")
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    stdout: bytes
    stderr: bytes
    exit_code: int
    def __init__(self, stdout: _Optional[bytes] = ..., stderr: _Optional[bytes] = ..., exit_code: _Optional[int] = ...) -> None: ...

class ResourceUsage(_message.Message):
    __slots__ = ("cpu_millicores_used", "memory_mb_used", "gpu_count_used")
    CPU_MILLICORES_USED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MB_USED_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_USED_FIELD_NUMBER: _ClassVar[int]
    cpu_millicores_used: int
    memory_mb_used: int
    gpu_count_used: int
    def __init__(self, cpu_millicores_used: _Optional[int] = ..., memory_mb_used: _Optional[int] = ..., gpu_count_used: _Optional[int] = ...) -> None: ...

class ObjectStorageAccess(_message.Message):
    __slots__ = ("buckets", "permission")
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    PERMISSION_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedScalarFieldContainer[str]
    permission: ObjectStoragePermission
    def __init__(self, buckets: _Optional[_Iterable[str]] = ..., permission: _Optional[_Union[ObjectStoragePermission, str]] = ...) -> None: ...

class StartSandboxRequest(_message.Message):
    __slots__ = ("command", "args", "tags", "resources", "container_image", "environment_variables", "ports", "mounted_files", "s3_mount", "network", "runway_ids", "tower_ids", "max_lifetime_seconds", "max_timeout_seconds", "tower_cluster_secrets", "object_storage_access", "pod_annotations", "secret_stores", "resource_limits", "resource_requests")
    class EnvironmentVariablesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class PodAnnotationsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_IMAGE_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    MOUNTED_FILES_FIELD_NUMBER: _ClassVar[int]
    S3_MOUNT_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_IDS_FIELD_NUMBER: _ClassVar[int]
    TOWER_IDS_FIELD_NUMBER: _ClassVar[int]
    MAX_LIFETIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOWER_CLUSTER_SECRETS_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STORAGE_ACCESS_FIELD_NUMBER: _ClassVar[int]
    POD_ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    SECRET_STORES_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_LIMITS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    command: str
    args: _containers.RepeatedScalarFieldContainer[str]
    tags: _containers.RepeatedScalarFieldContainer[str]
    resources: ResourceRequest
    container_image: str
    environment_variables: _containers.ScalarMap[str, str]
    ports: _containers.RepeatedCompositeFieldContainer[Port]
    mounted_files: _containers.RepeatedCompositeFieldContainer[MountedFile]
    s3_mount: S3Mount
    network: NetworkOptions
    runway_ids: _containers.RepeatedScalarFieldContainer[str]
    tower_ids: _containers.RepeatedScalarFieldContainer[str]
    max_lifetime_seconds: int
    max_timeout_seconds: int
    tower_cluster_secrets: _containers.RepeatedCompositeFieldContainer[TowerClusterSecretReference]
    object_storage_access: ObjectStorageAccess
    pod_annotations: _containers.ScalarMap[str, str]
    secret_stores: _containers.RepeatedCompositeFieldContainer[_secrets_pb2.SecretStoreReference]
    resource_limits: ResourceRequest
    resource_requests: ResourceRequest
    def __init__(self, command: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ..., tags: _Optional[_Iterable[str]] = ..., resources: _Optional[_Union[ResourceRequest, _Mapping]] = ..., container_image: _Optional[str] = ..., environment_variables: _Optional[_Mapping[str, str]] = ..., ports: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., mounted_files: _Optional[_Iterable[_Union[MountedFile, _Mapping]]] = ..., s3_mount: _Optional[_Union[S3Mount, _Mapping]] = ..., network: _Optional[_Union[NetworkOptions, _Mapping]] = ..., runway_ids: _Optional[_Iterable[str]] = ..., tower_ids: _Optional[_Iterable[str]] = ..., max_lifetime_seconds: _Optional[int] = ..., max_timeout_seconds: _Optional[int] = ..., tower_cluster_secrets: _Optional[_Iterable[_Union[TowerClusterSecretReference, _Mapping]]] = ..., object_storage_access: _Optional[_Union[ObjectStorageAccess, _Mapping]] = ..., pod_annotations: _Optional[_Mapping[str, str]] = ..., secret_stores: _Optional[_Iterable[_Union[_secrets_pb2.SecretStoreReference, _Mapping]]] = ..., resource_limits: _Optional[_Union[ResourceRequest, _Mapping]] = ..., resource_requests: _Optional[_Union[ResourceRequest, _Mapping]] = ...) -> None: ...

class StartSandboxResponse(_message.Message):
    __slots__ = ("sandbox_id", "started_at_time", "service_address", "exposed_ports", "requested_resources", "runway_id", "tower_id", "sandbox_status", "applied_ingress_mode", "applied_egress_mode", "requested_resource_limits", "requested_resource_requests")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_TIME_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    EXPOSED_PORTS_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_RESOURCES_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_ID_FIELD_NUMBER: _ClassVar[int]
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_STATUS_FIELD_NUMBER: _ClassVar[int]
    APPLIED_INGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    APPLIED_EGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_RESOURCE_LIMITS_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_RESOURCE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    started_at_time: _timestamp_pb2.Timestamp
    service_address: str
    exposed_ports: _containers.RepeatedCompositeFieldContainer[Port]
    requested_resources: ResourceRequest
    runway_id: str
    tower_id: str
    sandbox_status: SandboxStatus
    applied_ingress_mode: str
    applied_egress_mode: str
    requested_resource_limits: ResourceRequest
    requested_resource_requests: ResourceRequest
    def __init__(self, sandbox_id: _Optional[str] = ..., started_at_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., service_address: _Optional[str] = ..., exposed_ports: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., requested_resources: _Optional[_Union[ResourceRequest, _Mapping]] = ..., runway_id: _Optional[str] = ..., tower_id: _Optional[str] = ..., sandbox_status: _Optional[_Union[SandboxStatus, str]] = ..., applied_ingress_mode: _Optional[str] = ..., applied_egress_mode: _Optional[str] = ..., requested_resource_limits: _Optional[_Union[ResourceRequest, _Mapping]] = ..., requested_resource_requests: _Optional[_Union[ResourceRequest, _Mapping]] = ...) -> None: ...

class StopSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "graceful_shutdown_seconds", "snapshot_on_stop", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    GRACEFUL_SHUTDOWN_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ON_STOP_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    graceful_shutdown_seconds: int
    snapshot_on_stop: bool
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., graceful_shutdown_seconds: _Optional[int] = ..., snapshot_on_stop: bool = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class StopSandboxResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class GetSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class GetSandboxResponse(_message.Message):
    __slots__ = ("sandbox_id", "started_at_time", "sandbox_status", "current_resource_usage", "tower_id", "tower_group_id", "runway_id", "service_address", "exposed_ports", "applied_ingress_mode", "applied_egress_mode")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_TIME_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    TOWER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_ID_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    EXPOSED_PORTS_FIELD_NUMBER: _ClassVar[int]
    APPLIED_INGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    APPLIED_EGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    started_at_time: _timestamp_pb2.Timestamp
    sandbox_status: SandboxStatus
    current_resource_usage: ResourceUsage
    tower_id: str
    tower_group_id: str
    runway_id: str
    service_address: str
    exposed_ports: _containers.RepeatedCompositeFieldContainer[Port]
    applied_ingress_mode: str
    applied_egress_mode: str
    def __init__(self, sandbox_id: _Optional[str] = ..., started_at_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., sandbox_status: _Optional[_Union[SandboxStatus, str]] = ..., current_resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., tower_id: _Optional[str] = ..., tower_group_id: _Optional[str] = ..., runway_id: _Optional[str] = ..., service_address: _Optional[str] = ..., exposed_ports: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., applied_ingress_mode: _Optional[str] = ..., applied_egress_mode: _Optional[str] = ...) -> None: ...

class ListSandboxesRequest(_message.Message):
    __slots__ = ("tags", "status", "runway_ids", "tower_ids", "max_timeout_seconds", "include_stopped")
    TAGS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_IDS_FIELD_NUMBER: _ClassVar[int]
    TOWER_IDS_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_STOPPED_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.RepeatedScalarFieldContainer[str]
    status: SandboxStatus
    runway_ids: _containers.RepeatedScalarFieldContainer[str]
    tower_ids: _containers.RepeatedScalarFieldContainer[str]
    max_timeout_seconds: int
    include_stopped: bool
    def __init__(self, tags: _Optional[_Iterable[str]] = ..., status: _Optional[_Union[SandboxStatus, str]] = ..., runway_ids: _Optional[_Iterable[str]] = ..., tower_ids: _Optional[_Iterable[str]] = ..., max_timeout_seconds: _Optional[int] = ..., include_stopped: bool = ...) -> None: ...

class ListSandboxesResponse(_message.Message):
    __slots__ = ("sandboxes",)
    SANDBOXES_FIELD_NUMBER: _ClassVar[int]
    sandboxes: _containers.RepeatedCompositeFieldContainer[SandboxInfo]
    def __init__(self, sandboxes: _Optional[_Iterable[_Union[SandboxInfo, _Mapping]]] = ...) -> None: ...

class SandboxInfo(_message.Message):
    __slots__ = ("sandbox_id", "started_at_time", "sandbox_status", "current_resource_usage", "tower_id", "tower_group_id", "runway_id", "service_address", "exposed_ports", "applied_ingress_mode", "applied_egress_mode")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_TIME_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    TOWER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_ID_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    EXPOSED_PORTS_FIELD_NUMBER: _ClassVar[int]
    APPLIED_INGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    APPLIED_EGRESS_MODE_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    started_at_time: _timestamp_pb2.Timestamp
    sandbox_status: SandboxStatus
    current_resource_usage: ResourceUsage
    tower_id: str
    tower_group_id: str
    runway_id: str
    service_address: str
    exposed_ports: _containers.RepeatedCompositeFieldContainer[Port]
    applied_ingress_mode: str
    applied_egress_mode: str
    def __init__(self, sandbox_id: _Optional[str] = ..., started_at_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., sandbox_status: _Optional[_Union[SandboxStatus, str]] = ..., current_resource_usage: _Optional[_Union[ResourceUsage, _Mapping]] = ..., tower_id: _Optional[str] = ..., tower_group_id: _Optional[str] = ..., runway_id: _Optional[str] = ..., service_address: _Optional[str] = ..., exposed_ports: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., applied_ingress_mode: _Optional[str] = ..., applied_egress_mode: _Optional[str] = ...) -> None: ...

class DeleteSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class DeleteSandboxResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class ExecSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "command", "args", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    command: _containers.RepeatedScalarFieldContainer[str]
    args: _containers.RepeatedScalarFieldContainer[str]
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., args: _Optional[_Iterable[str]] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class ExecSandboxResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ExecResponse
    def __init__(self, result: _Optional[_Union[ExecResponse, _Mapping]] = ...) -> None: ...

class AddFileSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "file_contents", "filepath", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    file_contents: bytes
    filepath: str
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., file_contents: _Optional[bytes] = ..., filepath: _Optional[str] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class AddFileSandboxResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class RetrieveFileSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "filepath", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    FILEPATH_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    filepath: str
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., filepath: _Optional[str] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class RetrieveFileSandboxResponse(_message.Message):
    __slots__ = ("file_contents", "success", "error_message")
    FILE_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    file_contents: bytes
    success: bool
    error_message: str
    def __init__(self, file_contents: _Optional[bytes] = ..., success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class PauseSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class PauseSandboxResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class ResumeSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class ResumeSandboxResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class RawSandboxRequest(_message.Message):
    __slots__ = ("sandbox_id", "action_type", "exec_payload", "add_file_payload", "retrieve_file_payload", "max_timeout_seconds")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    EXEC_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    ADD_FILE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    RETRIEVE_FILE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    action_type: ActionType
    exec_payload: ExecPayload
    add_file_payload: AddFileSandboxRequest
    retrieve_file_payload: RetrieveFileSandboxRequest
    max_timeout_seconds: int
    def __init__(self, sandbox_id: _Optional[str] = ..., action_type: _Optional[_Union[ActionType, str]] = ..., exec_payload: _Optional[_Union[ExecPayload, _Mapping]] = ..., add_file_payload: _Optional[_Union[AddFileSandboxRequest, _Mapping]] = ..., retrieve_file_payload: _Optional[_Union[RetrieveFileSandboxRequest, _Mapping]] = ..., max_timeout_seconds: _Optional[int] = ...) -> None: ...

class RawSandboxResponse(_message.Message):
    __slots__ = ("action_type", "exec_response", "add_file_response", "retrieve_file_response")
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    EXEC_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ADD_FILE_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    RETRIEVE_FILE_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    action_type: ActionType
    exec_response: ExecSandboxResponse
    add_file_response: AddFileSandboxResponse
    retrieve_file_response: RetrieveFileSandboxResponse
    def __init__(self, action_type: _Optional[_Union[ActionType, str]] = ..., exec_response: _Optional[_Union[ExecSandboxResponse, _Mapping]] = ..., add_file_response: _Optional[_Union[AddFileSandboxResponse, _Mapping]] = ..., retrieve_file_response: _Optional[_Union[RetrieveFileSandboxResponse, _Mapping]] = ...) -> None: ...

class ObjectStorageWIFConfig(_message.Message):
    __slots__ = ("id", "wif_config_id", "enabled", "allowed_buckets", "max_permission", "created_at", "updated_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    WIF_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_BUCKETS_FIELD_NUMBER: _ClassVar[int]
    MAX_PERMISSION_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    wif_config_id: str
    enabled: bool
    allowed_buckets: _containers.RepeatedScalarFieldContainer[str]
    max_permission: ObjectStoragePermission
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., wif_config_id: _Optional[str] = ..., enabled: bool = ..., allowed_buckets: _Optional[_Iterable[str]] = ..., max_permission: _Optional[_Union[ObjectStoragePermission, str]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetObjectStorageWIFConfigRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetObjectStorageWIFConfigRequest(_message.Message):
    __slots__ = ("wif_config_id", "enabled", "allowed_buckets", "max_permission")
    WIF_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_BUCKETS_FIELD_NUMBER: _ClassVar[int]
    MAX_PERMISSION_FIELD_NUMBER: _ClassVar[int]
    wif_config_id: str
    enabled: bool
    allowed_buckets: _containers.RepeatedScalarFieldContainer[str]
    max_permission: ObjectStoragePermission
    def __init__(self, wif_config_id: _Optional[str] = ..., enabled: bool = ..., allowed_buckets: _Optional[_Iterable[str]] = ..., max_permission: _Optional[_Union[ObjectStoragePermission, str]] = ...) -> None: ...

class DeleteObjectStorageWIFConfigRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteObjectStorageWIFConfigResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
