# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cwsandbox._types import NetworkOptions

DEFAULT_CONTAINER_IMAGE: str = "python:3.11"
DEFAULT_COMMAND: str = "tail"
DEFAULT_ARGS: tuple[str, ...] = ("-f", "/dev/null")
DEFAULT_BASE_URL: str = "https://atc.cw-sandbox.com"
DEFAULT_GRACEFUL_SHUTDOWN_SECONDS: float = 10.0
DEFAULT_POLL_INTERVAL_SECONDS: float = 0.2
DEFAULT_MAX_POLL_INTERVAL_SECONDS: float = 2.0
DEFAULT_POLL_BACKOFF_FACTOR: float = 1.5
WANDB_NETRC_HOST: str = "api.wandb.ai"

# Default timeout for HTTP requests and API operations (seconds)
# This controls how long to wait for API responses, not sandbox lifetime.
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 300.0

# If not set, the backend controls the default lifetime of the sandboxes
DEFAULT_MAX_LIFETIME_SECONDS: float | None = None

# Buffer to add to client-side timeout for exec and stop requests
DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS: float = 5.0

# Default temp directory used within Sandboxes
DEFAULT_TEMP_DIR: str = "/tmp"

# Default W&B project name when not specified
DEFAULT_PROJECT_NAME: str = "uncategorized"

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
        resources: Resource requests (CPU, memory, GPU) as a dict.
        network: Network configuration via ``NetworkOptions``.
        environment_variables: Environment variables injected into the sandbox.

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
    resources: dict[str, Any] | None = None
    network: NetworkOptions | None = None
    environment_variables: dict[str, str] = field(default_factory=dict)

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
        merged = dict(self.environment_variables)
        if additional:
            merged.update(additional)
        return merged

    def with_overrides(self, **kwargs: Any) -> SandboxDefaults:
        """Create new defaults with some values overridden."""
        return replace(self, **kwargs)
