# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

from cwsandbox._types import NetworkOptions, ResourceOptions, Secret

DEFAULT_CONTAINER_IMAGE: str = "python:3.11"
DEFAULT_COMMAND: str = "tail"
DEFAULT_ARGS: tuple[str, ...] = ("-f", "/dev/null")
DEFAULT_BASE_URL: str = "https://atc.cw-sandbox.com"
DEFAULT_GRACEFUL_SHUTDOWN_SECONDS: float = 10.0
DEFAULT_POLL_INTERVAL_SECONDS: float = 0.2
DEFAULT_MAX_POLL_INTERVAL_SECONDS: float = 2.0
DEFAULT_POLL_BACKOFF_FACTOR: float = 1.5

# Default timeout for HTTP requests and API operations (seconds)
# This controls how long to wait for API responses, not sandbox lifetime.
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 300.0

# Timeout for lightweight discovery RPCs (list/get towers and runways).
DEFAULT_DISCOVERY_TIMEOUT_SECONDS: float = 30.0

# If not set, the backend controls the default lifetime of the sandboxes
DEFAULT_MAX_LIFETIME_SECONDS: float | None = None

# Buffer to add to client-side timeout for exec and stop requests
DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS: float = 5.0

# Default temp directory used within Sandboxes
DEFAULT_TEMP_DIR: str = "/tmp"

# Max bytes per stdin chunk sent to the process
STDIN_CHUNK_SIZE: int = 64 * 1024  # 64KB

# Max protobuf messages buffered between the gRPC reader and the processing
# loop.  When the bounded queue is full the gRPC reader blocks, preventing
# unbounded memory growth in long-lived streams (follow-mode logs).
STREAMING_RESPONSE_QUEUE_SIZE: int = 256

# Max items buffered in the output queue between the gRPC processing loop
# and the consumer (StreamReader).  Larger than STREAMING_RESPONSE_QUEUE_SIZE
# because each item here is a single line/chunk rather than a protobuf
# message that may decode into many lines.
STREAMING_OUTPUT_QUEUE_SIZE: int = 4096

# Max bytes accumulated in the line buffer before flushing a partial line.
# Protects against memory exhaustion from newline-free log output (e.g. binary
# data or extremely long lines in follow mode).
MAX_LINE_BUFFER_BYTES: int = 1024 * 1024  # 1 MB


def _merge_dicts(base: dict[str, str], additional: dict[str, str] | None) -> dict[str, str]:
    """Merge two string dicts, with additional values overriding base on collision."""
    merged = dict(base)
    if additional:
        merged.update(additional)
    return merged


@dataclass(frozen=True)
class SandboxDefaults:
    """Immutable configuration defaults for sandbox creation.

    All fields have sensible defaults. Override only what you need.

    There are two separate timeout concepts:
    - request_timeout_seconds: How long to wait for API responses (client-side)
    - max_lifetime_seconds: How long the sandbox runs before auto-termination (server-side)
      If not set, the backend controls the default lifetime.

    Tags enable filtering and organizing sandboxes. They are propagated to
    the backend and can be used to query sandboxes by tag.

    Attributes:
        container_image: Docker image for the sandbox container.
        command: Entrypoint command to run.
        args: Arguments passed to the command.
        base_url: CWSandbox API endpoint URL.
        request_timeout_seconds: Client-side HTTP timeout in seconds.
        max_lifetime_seconds: Server-side sandbox lifetime limit in seconds.
            None lets the backend control the default.
        temp_dir: Temp directory path inside the sandbox.
        tags: Tags for filtering and organizing sandboxes.
        runway_ids: Restrict to specific runway IDs.
        tower_ids: Restrict to specific tower IDs.
        resources: Resource configuration. Accepts ``ResourceOptions`` for separate
            requests/limits, or a flat dict for backward-compatible Guaranteed QoS.
        network: Network configuration via ``NetworkOptions``.
        secrets: Secrets to inject as environment variables.
        environment_variables: Environment variables injected into the sandbox.
        annotations: Kubernetes pod annotations (key-value string pairs).
            Merged with per-sandbox annotations; explicit values override defaults.
            Use for non-sensitive metadata only.

    Examples:
        ```python
        defaults = SandboxDefaults(
            container_image="python:3.12",
            command="tail",
            args=("-f", "/dev/null"),
            request_timeout_seconds=60,
            max_lifetime_seconds=3600,  # 1 hour sandbox lifetime
            tags=("my-workload", "experiment-42"),
            environment_variables={"LOG_LEVEL": "info", "REGION": "us-west"},
            annotations={"team": "ml-infra"},
        )
        ```
    """

    container_image: str = DEFAULT_CONTAINER_IMAGE
    command: str = DEFAULT_COMMAND
    args: tuple[str, ...] = DEFAULT_ARGS
    base_url: str = DEFAULT_BASE_URL
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_lifetime_seconds: float | None = DEFAULT_MAX_LIFETIME_SECONDS
    temp_dir: str = DEFAULT_TEMP_DIR
    tags: tuple[str, ...] = field(default_factory=tuple)
    runway_ids: tuple[str, ...] | None = None
    tower_ids: tuple[str, ...] | None = None
    resources: ResourceOptions | dict[str, Any] | None = None
    network: NetworkOptions | None = None
    secrets: tuple[Secret, ...] | None = None
    environment_variables: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def merge_tags(self, additional: list[str] | None) -> list[str]:
        """Combine default tags with additional tags.

        Tags from both sources are included. Order is: defaults first,
        then additional tags appended.
        """
        base = list(self.tags)
        if additional:
            base.extend(additional)
        return base

    def merge_environment_variables(self, additional: dict[str, str] | None) -> dict[str, str]:
        """Combine default environment variables with additional ones.

        Additional environment variables override defaults with the same key.
        """
        return _merge_dicts(self.environment_variables, additional)

    def merge_annotations(self, additional: dict[str, str] | None) -> dict[str, str]:
        """Combine default annotations with additional ones.

        Additional annotations override defaults with the same key.
        """
        return _merge_dicts(self.annotations, additional)

    def with_overrides(self, **kwargs: Any) -> SandboxDefaults:
        """Create new defaults with some values overridden."""
        return replace(self, **kwargs)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> SandboxDefaults:
        """Build ``SandboxDefaults`` from a mapping, coercing nested fields.

        Accepts plain dicts or OmegaConf ``DictConfig`` objects.  Unknown
        keys are silently ignored so callers can pass a config section
        that may contain extra fields.

        Coercions applied:
        - ``network`` dict -> ``NetworkOptions``
        - ``secrets`` list of dicts -> tuple of ``Secret``
        - ``args``, ``tags``, ``runway_ids``, ``tower_ids`` lists -> tuples
        - ``resources``, ``environment_variables`` -> plain ``dict``
        """
        if d is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in valid}
        # Drop None values for non-optional fields so they fall back to
        # dataclass defaults rather than creating invalid instances.
        _non_optional = (
            "container_image",
            "command",
            "args",
            "base_url",
            "request_timeout_seconds",
            "temp_dir",
            "tags",
            "environment_variables",
        )
        for key in _non_optional:
            if key in kwargs and kwargs[key] is None:
                del kwargs[key]
        # Coerce network dict -> NetworkOptions (preserve None)
        net = kwargs.get("network")
        if net is not None and not isinstance(net, NetworkOptions):
            kwargs["network"] = NetworkOptions(**net)
        # Coerce secrets list of dicts -> tuple of Secret (preserve None)
        secrets = kwargs.get("secrets")
        if secrets is not None:
            kwargs["secrets"] = tuple(
                Secret(**s) if not isinstance(s, Secret) else s for s in secrets
            )
        # Coerce sequences -> tuples for tuple fields (reject bare strings)
        for key in ("args", "tags", "runway_ids", "tower_ids"):
            val = kwargs.get(key)
            if val is None or isinstance(val, tuple):
                continue
            if isinstance(val, str):
                raise TypeError(f"{key} must be a sequence of strings, not a bare string")
            kwargs[key] = tuple(val)
        # Coerce resources: preserve ResourceOptions, convert mappings to dicts
        res = kwargs.get("resources")
        if res is not None and not isinstance(res, ResourceOptions):
            kwargs["resources"] = dict(res)
        # Coerce mapping types -> plain dicts for protobuf compat
        for key in ("environment_variables",):
            if key in kwargs and kwargs[key] is not None and not isinstance(kwargs[key], dict):
                kwargs[key] = dict(kwargs[key])
        return cls(**kwargs)
