from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_CONTAINER_IMAGE: str = "python:3.11"
DEFAULT_COMMAND: str = "tail"
DEFAULT_ARGS: tuple[str, ...] = ("-f", "/dev/null")
DEFAULT_BASE_URL: str = "https://atc.cwaviato.com"
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

# Buffer to add to client-side timeout in addition to the supplied exec command's timeout
DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS: float = 5.0

# Buffer to add to stop request timeout (graceful_shutdown_seconds + buffer)
DEFAULT_STOP_TIMEOUT_BUFFER_SECONDS: float = 5.0

# Default temp directory used within Sandboxes
DEFAULT_TEMP_DIR: str = "/tmp"

# Default W&B project name when not specified
DEFAULT_PROJECT_NAME: str = "uncategorized"


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

    Example:
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
