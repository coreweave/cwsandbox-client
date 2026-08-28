# AGENTS.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Project Overview

Python client library for CoreWeave Sandbox - a remote code execution platform. The SDK provides a sync/async hybrid API for creating, managing, and executing code in containerized sandbox environments.

**API dialect:** package **1.0.x+** speaks Sandbox **v1** only (prefer `cwsandbox>=1.0.0`; `0.27.0` is the same cutover under the prior 0.x tag). Callers that still need v1beta2 must pin **`cwsandbox==0.26.x`**. There is no hybrid / dual-dialect mode in one install.

## Public API and Documentation

When adding, removing, or renaming public exports in `src/cwsandbox/__init__.py`, the API reference generator in the `coreweave/docs` repo needs its `MANIFEST_GROUPS` updated (in `scripts/cwsandbox-api-ref/generate.py`). Docstrings use Google style with `Examples:` and `Attributes:` sections for structured parsing by Griffe.

## Development Setup

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup, workflow, and all development tasks.

## Architecture

### Core Classes

**`Sandbox`** (`_sandbox.py`): Main entry point with sync/async hybrid API. All methods return immediately; operations execute in background via `_LoopManager`.

Construction patterns:
```python
# Factory method (recommended)
sb = Sandbox.run("echo", "hello")  # Returns immediately
result = sb.exec(["echo", "more"]).result()  # Block for result
sb.stop().result()  # Block for completion

# Context manager (recommended for most use cases)
with Sandbox.run() as sb:  # Default command keeps sandbox alive
    result = sb.exec(["echo", "hello"]).result()
# Automatically stopped on exit

# Streaming output before getting result
with Sandbox.run() as sb:
    process = sb.exec(["echo", "hello"])
    for line in process.stdout:  # Stream lines as they arrive
        print(line, end="")
    result = process.result()  # Get final ProcessResult

# Async context manager
async with Sandbox.run() as sb:
    result = await sb.exec(["echo", "hello"])
```

Key methods:
- `run(*args, **kwargs)`: Create and start sandbox (CreateSandbox), return immediately. Accepts advanced configuration kwargs (see below). Rejects removed kwargs (`profile_ids`, `profile_names`, `s3_mount`, `ports`, `max_timeout_seconds`) with `TypeError`.
- `run_from_template(template_id, /, *args, command=None, defaults=None, **kwargs)`: Create from an org template (CreateSandboxFromTemplate) with replace-on-presence overlays.
- `start()`: Send create request, return `OperationRef[None]`. Call `.result()` to block until backend accepts. Freezes one create `request_id` for idempotent retries.
- `wait()`: Block until RUNNING status, returns self for chaining
- `wait_until_complete(timeout=None, raise_on_termination=True)`: Wait until terminal state (COMPLETED, FAILED, TERMINATED), return `OperationRef[Sandbox]`. Polls through TERMINATING automatically. Call `.result()` to block or `await` in async contexts. Set `raise_on_termination=False` to handle externally-terminated sandboxes without raising `SandboxTerminatedError`.
- `exec(command, cwd=None, check=False, timeout_seconds=None, stdin=False, container=None)`: Execute command, return `Process`. Call `.result()` to block for `ProcessResult`. Iterate `process.stdout` before `.result()` for real-time streaming. Set `check=True` to raise `SandboxExecutionError` on non-zero returncode. Set `cwd` to an absolute path to run the command in a specific working directory (implemented via shell wrapping, requires /bin/sh in container). Set `stdin=True` to enable stdin streaming via `process.stdin`. Empty/`None` `container` targets the primary.
- `shell(command=None, *, width=None, height=None, container=None)`: Start an interactive TTY session, return `TerminalSession`. Always allocates a TTY and enables stdin. Output is raw bytes (merged stdout/stderr) with no buffering — safe for long-running interactive sessions. Defaults to `["/bin/bash"]`. Empty/`None` `container` targets the primary.
- `stream_logs(*, follow=False, tail_lines=None, since_time=None, timestamps=False, container=None)`: Stream logs from the sandbox's main process (PID 1), return `StreamReader[str]`. Only captures stdout/stderr from the command passed to `Sandbox.run()` — output from `exec()` commands is **not** included. Set `follow=True` for continuous streaming (like `tail -f`). Uses bounded queues for backpressure in follow mode. Empty/`None` `container` targets the primary.
- `read_file(path, *, container=None)`: Return `OperationRef[bytes]`. Empty/`None` `container` targets the primary.
- `write_file(path, content, *, container=None)`: Return `OperationRef[None]`. Empty/`None` `container` targets the primary.
- `stop(snapshot_on_stop=False, graceful_shutdown_seconds=10.0, missing_ok=False, wait_for_ready=True, request_id=None)`: Stop sandbox via DeleteSandbox and return `OperationRef[None]`. The sandbox transitions through TERMINATING (grace period) before reaching a terminal state (COMPLETED or FAILED). The returned OperationRef resolves when the backend confirms a terminal state, not just when the delete RPC succeeds. Multiple callers share the same stop task. Raises `SandboxError` on failure. Set `snapshot_on_stop=True` to capture a file-system snapshot of the configured scratch volume before shutdown — the resulting ID is then available via the `file_system_snapshot_id` property. Because `stop()` coalesces concurrent callers onto one shared stop task, a `snapshot_on_stop=True` request that would join (or observe) a stop not capturing a snapshot — sandbox already stopping/stopped, or a plain `stop()` already in flight — raises `SnapshotOnStopConflictError` rather than completing with no archive; plain stops always coalesce. `wait_for_ready`/`request_id` apply only when `snapshot_on_stop=True` (the client uses a larger timeout when snapshotting, since the stop blocks on the archive). Set `missing_ok=True` to suppress `SandboxNotFoundError`.
- `snapshot(wait_for_ready=True, request_id=None)`: Capture a file-system snapshot (FSS) of the configured scratch volume without stopping, return `OperationRef[str]` (the new snapshot's ID). Call `Sandbox.get_snapshot(id)` for the full record. Requires a scratch / `file_system_snapshot` mount and the org to be enabled for FSS. Auto-starts the sandbox first if needed. With `wait_for_ready=True` (default) blocks until the snapshot is READY/FAILED. To fork a sandbox, `snapshot()` then `Sandbox.run(file_system_snapshot=FileSystemSnapshotOptions(..., file_system_snapshot_id=<id>))` (or equivalent `volumes=`).
- `get_status()`: Fetch fresh status from API (sync). Returns cached status for terminal sandboxes (COMPLETED, FAILED, TERMINATED) since terminal states are immutable. TERMINATING is non-terminal and always fetches fresh status.

Properties:
- `status`: Cached status from last API call (use `get_status()` for fresh)
- `status_updated_at`: When status was last fetched
- `sandbox_id`, `runner_id`, `runner_group_id`, `returncode`, `started_at`
- `service_urls`: Tuple of `(port, name, url)` from typed services once assigned (CREATING or RUNNING; not serving)
- `service_endpoints`: Tuple of `HttpsEndpointStatus` (port, name, kind, auth, url, applied `request_timeout_seconds`) for HTTPS product endpoints; timeout remains after URL suppression
- `exposed_ports`: `(port, name)` pairs derived from status services when present
- `dns_egress_names`: Hostnames granted at create, echoed from status.effective_egress
- `effective_egress` / `effective_ingress`: Full echoed rule sets from status
- `effective_runtime_class`: Runtime class applied by the backend
- `attached_volume_ids`: Registered Volume IDs attached to the sandbox
- `containers`: Echoed create-time `Container` spec. `primary=True` is filled on the inferred primary so a clone of this list is a valid create.
- `container_statuses`: Per-container observed state. Sandbox `status` / `returncode` stay primary-owned.
- `resource_requests`, `resource_limits` - Confirmed resources from start response (None for discovered sandboxes)
- `file_system_snapshot_id` - Snapshot ID produced by `stop(snapshot_on_stop=True)` once the stop resolves (None otherwise)

Advanced configuration kwargs (for `run()`, `run_from_template()`, `Session.sandbox()`, and `@session.function()`):
- `placement_mode` - `PlacementMode` (`serverless` / `cks`) or string; first-attempt mode when using spillover
- `placement_spillover` - `PlacementSpillover` (`strict` default | `cks_then_serverless` | `serverless_then_cks`). On CreateSandbox failure when the primary mode cannot place the request (`CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED`, `CWSANDBOX_PLACEMENT_REJECTED`, `CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED`, `CWSANDBOX_NO_SUITABLE_RUNNER`, `CWSANDBOX_RUNNER_OVERLOADED`, `CWSANDBOX_RUNNER_UNAVAILABLE`), retries once with the alternate mode and a new `request_id`. CKS→serverless clears `runner_ids`. `serverless_then_cks` rejects `runner_ids` at construction. Does not spill on serverless product gates, auth, or `INVALID_ARGUMENT`. Template creates (`template_id` / `run_from_template`) require `strict`.
- `runner_ids` - CKS runner pin (rejected with serverless and with `serverless_then_cks`)
- `services` - Typed ports via `Service` / `ServiceVisibility` / `ServiceProtocol` / `Endpoint`
- `network` - `NetworkOptions` deny flags (`deny_egress` / `deny_ingress`) plus create-time `egress` / `ingress` grants (`EgressRule` / `IngressRule`), or dict
- `volumes` - Scratch (`ScratchVolumeOptions`) or registered (`RegisteredVolumeOptions`) volumes. `mount_path` is optional on scratch (omit to declare without mounting). A set `mount_path` is a convenience mount on the primary.
- `runtime_class` - Optional runtime-class pin (e.g. `"gvisor"`), clamped by policy
- `security_context` - In-guest privilege for the primary container (`SecurityContext` or dict). Mutually exclusive with `containers=`.
- `working_dir` - Working directory for the primary container command. Mutually exclusive with `containers=` (set it on `Container` instead).
- `object_storage_access` - Temporary object-storage credentials (`ObjectStorageAccess` or dict)
- `file_system_snapshot` - Convenience single-mount FSS via `FileSystemSnapshotOptions` or dict (optional `mount_path`, optional `size`, optional `file_system_snapshot_id`, optional `name` default `"workspace"`). Omit `mount_path` to declare without mounting.
- `containers` - List of `Container`. Mutually exclusive with `container_image`, `command`/`args`, `resources`, `mounted_files`, `secrets`, `image_pull_credentials`, `environment_variables`, `security_context`, and `working_dir`. Not used by `@session.function()`. One container: names/`primary` may be omitted. More than one: every row needs a name and resources, and exactly one `primary=True`. GPU is allowed only on the primary.
- `resources` - Resource configuration via `ResourceOptions`, nested dict, or legacy flat dict (CPU, memory, GPU)
- `mounted_files` - Files to mount into the sandbox at startup (read-only at runtime; use `write_file()` for writable files)
- `image_pull_credentials` - Private registry pull credentials. On `run_from_template()`, requires `container_image` (whole-container replace) unless `containers=` replaces the list; omit to keep a credential stored on the template
- `secrets` - Create-time secret inject from secret stores as env vars, via `Secret` or dict
- `environment_variables` - Environment variables to inject (merges with defaults)
- `annotations` - Kubernetes pod annotations (merges with defaults, explicit keys win)
- Removed (loud `TypeError`): `profile_ids`, `profile_names`, `s3_mount`, `ports`, `max_timeout_seconds` — use `request_timeout_seconds` for client deadlines

Class methods:
- `Sandbox.session(defaults)`: Create a `Session` for managing multiple sandboxes (sync)
- `Sandbox.list(tags=None, status=None, runner_ids=None, volume_ids=None, show_terminated=False, ...)`: Query existing sandboxes, return `OperationRef[list[Sandbox]]`. Use `.result()` to block or `await` in async contexts. By default, terminal sandboxes (completed, failed, terminated) are excluded. Set `show_terminated=True` to include them. `volume_ids` filters to sandboxes attached to those registered Volumes. `profile_ids` / `profile_names` raise `TypeError`.
- `Volume.create` / `Volume.get` / `Volume.list` / `volume.update` / `volume.delete` / `volume.validate` / `volume.wait_until_ready`: Registered Volume CRUD (`_volume.py`). Classmethods and instance methods accept `auth=` like `Sandbox.list` / `from_id`; returned handles keep that selection. Create returns immediately in `VALIDATING`; poll with `wait_until_ready()` (the wait budget bounds each Get and its retries). Mount with `RegisteredVolumeOptions` on `Sandbox.run(volumes=...)`.
- `Sandbox.from_id(sandbox_id)`: Attach to existing sandbox by ID, return `OperationRef[Sandbox]`. Works for both active and stopped sandboxes.
- `Sandbox.delete(sandbox_id, missing_ok=False)`: Delete sandbox by ID, return `OperationRef[None]`. Raises `SandboxError` on failure. Set `missing_ok=True` to suppress `SandboxNotFoundError` for already-deleted sandboxes.
- `Sandbox.get_snapshot(file_system_snapshot_id)`: Fetch a `FileSystemSnapshot` record by ID, return `OperationRef[FileSystemSnapshot]`. Snapshots are org-scoped. Raises `SnapshotNotFoundError` if absent.
- `Sandbox.list_snapshots(source_sandbox_id=None, status=None, ...)`: List FSS records for the org, return `OperationRef[list[FileSystemSnapshot]]` (auto-paginated). `source_sandbox_id` and `status` are applied client-side.
- `Sandbox.delete_snapshot(file_system_snapshot_id, missing_ok=False)`: Delete an FSS by ID, return `OperationRef[None]`. Does not affect sandboxes already restored from it. Set `missing_ok=True` to suppress `SnapshotNotFoundError`.
- `Sandbox.get_snapshot_bucket_config()` / `Sandbox.set_snapshot_bucket_config(*, bucket_name, region="")`: Get/set the org's FSS object-storage bucket (admin), return `OperationRef[FileSystemSnapshotBucketConfig]`. Pass `bucket_name=""` to revert to the CoreWeave-managed bucket.

**`Session`** (`_session.py`): Manages multiple sandboxes with shared defaults. Supports both sync and async context managers for the hybrid API.

Key methods:
- `session.sandbox(command, args, **kwargs)` - create an unstarted sandbox with session defaults. Auto-starts on first operation (exec, read_file, write_file, wait). Accepts advanced configuration kwargs.
- `session.function()` - decorator for remote function execution
- `session.adopt(sandbox)` - register an existing Sandbox (from `Sandbox.list()` or `Sandbox.from_id()`) for cleanup when session closes
- `session.close()` - return `OperationRef[None]` for cleanup
- `session.list(tags=None, status=None, runner_ids=None, volume_ids=None, show_terminated=False, adopt=False)` - find sandboxes matching session tags, return `OperationRef[list[Sandbox]]`. Use `.result()` to block or `await` in async contexts. Set `show_terminated=True` to include terminal sandboxes.
- `session.from_id(sandbox_id, adopt=True)` - attach to existing sandbox by ID, return `OperationRef[Sandbox]`

Properties:
- `sandbox_count`: Number of sandboxes currently tracked by this session

Usage pattern:
```python
with Session(defaults) as session:
    sb = session.sandbox()  # Default command keeps sandbox alive
    result = sb.exec(["echo", "hello"]).result()
# Automatically cleans up all sandboxes on exit
```

**`SandboxDefaults`** (`_defaults.py`): Immutable configuration dataclass. Tags propagate to backend for filtering.

Fields (all optional with sensible defaults):
- `container_image`, `command`, `args` - Container configuration
- `base_url` - API endpoint (default: `https://api.cwsandbox.com`)
- `auth` - Optional `AuthStrategy`, `AuthHeaders`, or `AuthProvider`; omitted defaults to CoreWeave API-key auth
- `request_timeout_seconds` - Client-side HTTP timeout (default: 300.0)
- `max_lifetime_seconds` - Server-side sandbox lifetime limit (default: None, backend controls)
- `temp_dir` - Sandbox temp directory (default: `/tmp`)
- `tags` - Tuple of tags for filtering
- `runner_ids` - Optional CKS runner pin (tuple). Empty list clears a default; `None` inherits
- `placement_mode` - `PlacementMode` or string (`serverless` / `cks`)
- `placement_spillover` - `PlacementSpillover` (default `strict`); see advanced kwargs above
- `resources` - Resource configuration (`ResourceOptions | dict[str, Any] | None`)
- `network` - Deny-flag `NetworkOptions` plus optional `egress` / `ingress` grants
- `services` - Tuple of typed `Service` ports
- `volumes` - Tuple of `ScratchVolumeOptions` / `RegisteredVolumeOptions` (`mount_path` optional on scratch)
- `runtime_class`, `security_context`, `working_dir`, `object_storage_access` - Create-spec fields shared across sandboxes
- `file_system_snapshot` - Convenience single-mount FSS via `FileSystemSnapshotOptions` (shareable mount_path/size; explicit `run()` value replaces it wholesale; `mount_path` optional)
- `containers` - Optional `tuple[Container, ...]`. Mutually exclusive with single-container fields on `Sandbox.run()` / `session.sandbox()`. Do not set this on defaults used with `@session.function()`.
- `secrets` - Create-time secret inject (tuple of `Secret`)
- `environment_variables` - Environment variables to inject
- `annotations` - Kubernetes pod annotations (`dict[str, str]`, default: empty)

Utility methods:
- `merge_tags(additional)` - Combine default tags with additional tags list
- `merge_annotations(additional)` - Combine default annotations with additional dict (explicit keys win)
- `merge_environment_variables(additional)` - Combine default env vars with additional dict (explicit keys win)
- `with_overrides(**kwargs)` - Create new defaults with some values overridden

Key constants (from `_defaults.py`):
- `DEFAULT_CONTAINER_IMAGE = "python:3.11"`
- `DEFAULT_COMMAND = "/bin/sh"`, `DEFAULT_ARGS = ("-c", 'trap "exit 0" TERM INT; sleep infinity & wait')` - shell-trapped keep-alive so PID 1 responds to SIGTERM on stop
- `DEFAULT_BASE_URL = "https://api.cwsandbox.com"`
- `DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0` - Client-side HTTP timeout
- `DEFAULT_MAX_LIFETIME_SECONDS = None` - Server controls sandbox lifetime
- `DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 10.0`
- `DEFAULT_TEMP_DIR = "/tmp"`
- Polling: `DEFAULT_POLL_INTERVAL_SECONDS = 0.2`, `DEFAULT_POLL_BACKOFF_FACTOR = 1.5`, `DEFAULT_MAX_POLL_INTERVAL_SECONDS = 2.0`
- `DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS = 5.0` - Buffer added to exec timeout

**`OperationRef[T]`** (`_types.py`): Generic wrapper for async operations with lazy result retrieval. Bridges `concurrent.futures.Future` to asyncio for the sync/async hybrid API.

Key methods:
- `result(timeout=None)` - Block until complete and return result
- `__await__` - Awaitable in async contexts

Usage pattern:
```python
ref = sandbox.read_file("/path")  # Returns immediately
data = ref.result()               # Block when result needed
# Or in async context:
data = await ref
```

**`SandboxStatus`** (`_sandbox.py`): StrEnum for sandbox lifecycle states. Lifecycle: `CREATING` -> `RUNNING` -> `TERMINATING` -> `COMPLETED` | `FAILED`. Values: `PENDING`, `CREATING`, `RUNNING`, `PAUSED`, `TERMINATING`, `COMPLETED`, `FAILED`, `TERMINATED` (deprecated), `UNSPECIFIED`. `TERMINATING` is non-terminal: the sandbox is draining through its grace period. `TERMINATED` is deprecated in favor of the `TERMINATING` -> `COMPLETED`/`FAILED` flow but still emitted by older backends. Terminal statuses (used for caching and polling): `COMPLETED`, `FAILED`, `TERMINATED`. Methods `from_proto()` and `to_proto()` for protobuf conversion.

**Exec Types** (`_types.py`): Types for command execution, returned by `Sandbox.exec()`:

- `Process`: Handle for running process with `stdout`/`stderr` StreamReaders and optional `stdin` StreamWriter. Properties: `returncode` (exit code or None), `command` (list executed), `stdin` (StreamWriter when `stdin=True`, or None). Methods: `poll()`, `wait(timeout)`, `result(timeout)`, `cancel()`. Awaitable in async contexts.
- `StreamReader`: Dual sync/async iterable wrapping asyncio.Queue. Supports both `for line in reader` and `async for line in reader`. Parameterized: `StreamReader[str]` for text (exec output, logs), `StreamReader[bytes]` for raw bytes (TTY output). Call `close()` to stop the underlying producer and end iteration.
- `StreamWriter`: Writable stream for stdin. Methods: `write(data: bytes)`, `writeline(text: str)`, `close()`. All return `OperationRef[None]`. Property: `closed` (bool). Uses bounded queue (16 items, ~1MB with 64KB chunks) for backpressure.
- `ProcessResult`: Dataclass with `stdout`, `stderr`, `returncode`, `command`, plus raw byte variants (`stdout_bytes`, `stderr_bytes`).

**Terminal Types** (`_types.py`): Types for interactive TTY sessions, returned by `Sandbox.shell()`:

- `TerminalSession`: Handle for an interactive TTY session. Extends `OperationRef[TerminalResult]`. Properties: `output` (StreamReader[bytes] — merged stdout/stderr as raw bytes), `stdin` (StreamWriter — always present), `command` (list executed). Methods: `resize(width, height)` (fire-and-forget), `wait(timeout)` (blocks until session ends, returns exit code), `result(timeout)` (returns TerminalResult). Awaitable in async contexts.
- `TerminalResult`: Frozen dataclass with `returncode` and `command`. Unlike `ProcessResult`, does not contain captured stdout/stderr because TTY sessions do not buffer output.

**`PlacementMode`** (`_types.py`): `UNSPECIFIED` | `SERVERLESS` | `CKS`. Use with `runner_ids` only for CKS.

**`PlacementSpillover`** (`_types.py`): `STRICT` (default) | `CKS_THEN_SERVERLESS` | `SERVERLESS_THEN_CKS`. Client-side one-shot CreateSandbox retry onto the alternate mode on spillable capacity/placement failures. Templates require `STRICT`.

**`Service` / `ServiceVisibility` / `ServiceProtocol` / `Endpoint` / `HttpsEndpointStatus`** (`_types.py`): Typed service ports replace beta string ingress/egress modes. Pass as `services=` on `run()` / defaults. HTTPS is create-time only: `endpoint=Endpoint(kind=HTTPS, auth=OPEN)` on PUBLIC. Optional `Endpoint.request_timeout_seconds` is the server-side HTTPS request clock (504 while the sandbox stays alive). The SDK only checks that the value is an `int`. It is not `Sandbox.run(request_timeout_seconds=...)` (client RPC deadline). Omit/`0` is the platform default (15s on serverless). The server accepts `0` or `[15, 900]`. On create-from-template, `0` is replace-on-presence and does not clear a template timeout back to the platform default. Applied timeout is echoed on `Sandbox.service_endpoints`. Listen-only PUBLIC or PRIVATE plus a product endpoint is `CWSANDBOX_NOT_IMPLEMENTED`.

```python
from cwsandbox import PlacementMode, Service, ServiceVisibility, Sandbox

sandbox = Sandbox.run(
    services=[Service(port=8080, visibility=ServiceVisibility.PUBLIC)],
)

# CKS pin
sandbox = Sandbox.run(
    placement_mode=PlacementMode.CKS,
    runner_ids=["runner-123"],
)
```

**`NetworkOptions`** / **`EgressRule`** / **`IngressRule`** (`_types.py`): Deny flags (`deny_egress`, `deny_ingress`) plus create-time grants via `egress` and `ingress`. `EgressRule` requires exactly one destination (`dns_name`, `cidr`, `tenant`, `any`, or `selector`). DNS names are HTTPS (TCP 443) grants: exact names (`pypi.org`) or a single leftmost wildcard (`*.pypi.org`). `"*"` is a policy ceiling, not a sandbox grant. `IngressRule` requires exactly one source (`cidr`, `tenant`, or `any`) and applies to CUSTOM-visibility ports. Port exposure is via `services=`, not this type.

```python
from cwsandbox import EgressRule, NetworkOptions, Sandbox

sandbox = Sandbox.run(
    network=NetworkOptions(
        egress=[
            EgressRule(dns_name="pypi.org"),
            EgressRule(dns_name="*.pypi.org"),
        ],
    ),
)
```

**`Secret`** (`_types.py`): Frozen dataclass for injecting secrets from secret stores into sandbox environment variables. The `secrets` parameter accepts `Secret` instances or plain dicts (which are automatically converted via `Secret(**d)`).

Fields:
- `store: str` - Name of the secret store (e.g. `"wandb"`).
- `name: str` - Name of the secret in the store.
- `field: str` - Specific field within a structured secret (optional, defaults to `""`).
- `env_var: str | None` - Environment variable name the secret is injected as (defaults to `name`).

Duplicate `env_var` targets across secrets raise `ValueError` at merge time.

Usage:
```python
from cwsandbox import Secret

# Minimal: env_var defaults to name
sandbox = Sandbox.run(
    secrets=[Secret(store="wandb", name="HF_TOKEN")],
)

# Extracting a field from a structured secret
sandbox = Sandbox.run(
    secrets=[
        Secret(store="wandb", name="db-credentials", field="password", env_var="DB_PASS"),
    ],
)

# Using dicts (convenient for config files)
sandbox = Sandbox.run(
    secrets=[{"store": "wandb", "name": "HF_TOKEN"}],
)
```

**`Container`** / **`VolumeMount`** / **`ContainerStatus`**: Multi-container create and echo. Pass `containers=[Container(...), ...]` to `Sandbox.run()` / `session.sandbox()`. Volumes stay sandbox-level; sharing is two containers listing the same volume name in `volume_mounts`. The kwargs path (`Sandbox.run("echo", "hello")`) still sends one container named `"main"` with `primary` unset.

```python
from cwsandbox import Container, ResourceOptions, Sandbox, ScratchVolumeOptions, VolumeMount

with Sandbox.run(
    containers=[
        Container(
            image="python:3.11",
            name="main",
            primary=True,
            resources=ResourceOptions(requests={"cpu": "1"}, limits={"cpu": "1"}),
            volume_mounts=[VolumeMount(volume="workspace", mount_path="/workspace")],
        ),
        Container(
            image="redis:7",
            name="cache",
            resources=ResourceOptions(requests={"cpu": "1"}, limits={"cpu": "1"}),
        ),
    ],
    volumes=[ScratchVolumeOptions(name="workspace")],
) as sb:
    sb.exec(["echo", "hello"], container="cache").result()
```

**`ResourceOptions`** (`_types.py`): Frozen dataclass for typed resource configuration. Supports separate requests and limits for Burstable QoS pods. GPU is a separate top-level field because GPU overcommit is not supported by the backend. The `resources` parameter accepts a `ResourceOptions` instance, a nested dict, or a legacy flat dict (which is automatically coerced).

Fields:
- `requests: dict[str, str] | None` - CPU/memory resource requests (e.g. `{"cpu": "1", "memory": "256Mi"}`)
- `limits: dict[str, str] | None` - CPU/memory resource limits (e.g. `{"cpu": "8", "memory": "2Gi"}`)
- `gpu: dict[str, Any] | None` - GPU configuration (e.g. `{"count": 1, "type": "A100"}`)

Usage:
```python
from cwsandbox import ResourceOptions

# Using ResourceOptions (recommended for overcommit)
sandbox = Sandbox.run(
    resources=ResourceOptions(
        requests={"cpu": "1", "memory": "256Mi"},
        limits={"cpu": "8", "memory": "2Gi"},
    ),
)

# Using nested dict
sandbox = Sandbox.run(
    resources={"requests": {"cpu": "1"}, "limits": {"cpu": "8"}},
)

# Legacy flat dict (Guaranteed QoS - requests == limits)
sandbox = Sandbox.run(
    resources={"cpu": "8", "memory": "2Gi"},
)
```

**File System Snapshots (FSS)** (`_types.py`): A configured scratch volume can be snapshotted (on request or on stop) and restored into new sandboxes. FSS is gated per-organization on the backend; orgs that are not enabled get `SnapshotNotSupportedError`.

- **`ScratchVolumeOptions`**: Named scratch volume. Fields: `name`, optional `mount_path` (absolute convenience mount on the primary; omit to declare without mounting), optional `size`, `restore_from_snapshot_id`, `medium` (`disk`/`memory`), `sub_path`, `read_only`. Prefer `volumes=` for multi-volume setups. Attach per-container with `Container.volume_mounts`.
- **`RegisteredVolumeOptions`**: Mount a registered Volume. Fields: `name`, `volume_id`, `mount_path`, optional `sub_path`, `read_only`.
- **`SecurityContext`**: In-guest privilege (run-as, privileged, capabilities, seccomp). Host-reaching knobs are policy-only.
- **`ObjectStorageAccess`**: Temporary object-storage credentials (`buckets`, optional `permission`, `object_prefix`).
- **`FileSystemSnapshotOptions`**: Convenience single-mount wrapper (optional `mount_path`, optional `size`, optional `file_system_snapshot_id`, optional `name` default `"workspace"`). Maps to a scratch volume via `to_scratch_volume()`.
- **`FileSystemSnapshot`**: Frozen record from `get_snapshot()` / `list_snapshots()` (`snapshot()` returns only the ID). Fields include `file_system_snapshot_id`, `status`, `status_reason`, `size_bytes`, `source_sandbox_id`, `trigger`, `request_id`, `object_bucket`, `source_volume_name`, timestamps.
- **`FileSystemSnapshotStatus`**: StrEnum — `UNSPECIFIED`, `CREATING`, `READY`, `FAILED`, `DELETING`.
- **`FileSystemSnapshotTrigger`**: StrEnum — `UNSPECIFIED`, `ON_DELETE` (from `stop(snapshot_on_stop=True)`), `MANUAL` (from `snapshot()`).
- **`FileSystemSnapshotBucketConfig`** / **`FileSystemSnapshotBucketMode`**: Org bucket config (`mode`, `bucket_name`, `region`, `effective_bucket_name`); mode is `UNSPECIFIED`/`CW_MANAGED`/`BRING_YOUR_OWN`.

Usage:
```python
from cwsandbox import Sandbox, FileSystemSnapshotOptions, ScratchVolumeOptions

# Convenience single mount
with Sandbox.run(
    file_system_snapshot=FileSystemSnapshotOptions(mount_path="/workspace", size="10Gi"),
) as sb:
    sb.exec(["sh", "-c", "echo seed > /workspace/data.txt"]).result()
    snapshot_id = sb.snapshot().result()

# Explicit named scratch volume
with Sandbox.run(
    volumes=[ScratchVolumeOptions(name="workspace", mount_path="/workspace", size="10Gi")],
) as sb:
    ...

# Fork = restore into a fresh sandbox
with Sandbox.run(
    file_system_snapshot=FileSystemSnapshotOptions(
        mount_path="/workspace", file_system_snapshot_id=snapshot_id
    ),
) as restored:
    ...
```

### Authentication Flow

`_auth.py` resolves auth per Sandbox, Session, or class-level operation:
1. Omitted auth selection defaults to CoreWeave: `CWSANDBOX_API_KEY` is sent as a Bearer token, or requests are unauthenticated when it is absent.
2. `AuthStrategy.WANDB` explicitly delegates credential discovery to the optional W&B SDK (session, `WANDB_API_KEY`, or host-scoped `.netrc`) and sends `x-wandb-api-key`.
3. `AuthHeaders` and `AuthProvider` support explicit custom per-instance auth.

The legacy process-global `set_auth_mode()` hook remains for compatibility. New integrations should pass `auth=` and must not send a W&B API key as a CoreWeave Bearer token.

### Function Execution (`_function.py`)

**`RemoteFunction[P, R]`**: Wrapper class returned by `@session.function()` decorator. Provides sync/async hybrid API for remote function execution.

Usage pattern:
```python
with Session(defaults) as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y

    # Call .remote() to execute in sandbox
    ref = compute.remote(2, 3)  # Returns OperationRef immediately
    result = ref.result()       # Block for result: 5

    # Parallel execution across inputs
    refs = compute.map([(1, 2), (3, 4), (5, 6)])
    results = [r.result() for r in refs]  # [3, 7, 11]

    # Local testing without sandbox
    result = compute.local(2, 3)  # Runs in current process
```

Key methods:
- `__call__(*args, **kwargs)` - Execute in sandbox via `.remote()`, enabling natural `func(args)` syntax
- `remote(*args, **kwargs)` - Execute in sandbox, return `OperationRef[R]` immediately
- `map(items)` - Execute for each item tuple in parallel, return list of `OperationRef[R]`
- `local(*args, **kwargs)` - Execute locally without sandbox (for testing)

Configuration options (passed to decorator):
- `container_image` - Override image for this function
- Plus advanced configuration kwargs (see Sandbox section above)

Internals:
1. Extracts function source via AST, removes the `@session.function` decorator
2. Captures closure variables from `__closure__` and `co_freevars`
3. Walks bytecode (`LOAD_GLOBAL`, `STORE_GLOBAL`, `DELETE_GLOBAL`) to find referenced globals
4. Serializes payload as JSON, creates ephemeral sandbox, executes, reads JSON result

Arguments, closures, referenced globals, and return values must be
JSON-serializable (str, int, float, dict, list, bool, None). Non-JSON
values surface as a `SandboxExecutionError` from inside the sandbox.

`@session.function()` stays a single-container workflow. Do not put `containers` on session defaults used with the decorator.

### Event Loop Management (`_loop_manager.py`)

**`_LoopManager`**: Singleton managing a background daemon thread with asyncio event loop. Enables sync code to execute async operations without user-managed event loops.

Key methods:
- `_LoopManager.get()` - Get singleton instance (thread-safe, double-checked locking)
- `run_sync(coro)` - Execute coroutine and block until complete
- `run_async(coro)` - Execute coroutine and return Future immediately
- `register_session(session)` - Track session in WeakSet for cleanup
- `cleanup_all()` - Stop all sandboxes in registered sessions

The daemon thread approach:
- Works in Jupyter notebooks without nest_asyncio
- Independent of user-managed event loops
- Allows cleanup via atexit and signal handlers

### Cleanup Handlers (`_cleanup.py`)

Auto-installed handlers for graceful sandbox shutdown on process exit. Installed automatically on module import.

- `_cleanup()`: Calls `_LoopManager.cleanup_all()` with re-entrancy guard
- `_signal_handler()`: Handles SIGINT/SIGTERM, chains to original handlers
- `_install_handlers()`: Registers atexit handler and signal handlers
- `_reset_for_testing()`: Resets module state for test isolation

On first signal, performs cleanup then chains to original handler. On second signal during cleanup, forces immediate exit.

### Module-Level Utilities

**`cwsandbox.results()`**: Block for one or more OperationRefs and return results.

```python
# Single ref
data = cwsandbox.results(sandbox.read_file("/path"))

# Multiple refs
all_data = cwsandbox.results([sb.read_file(f) for f in files])
```

**`cwsandbox.wait()`**: Wait for Sandbox, OperationRef, or Process objects to complete. Returns `(done, pending)` tuple.

```python
# Wait for all sandboxes to be running
sandboxes = [Sandbox.run(...) for _ in range(5)]
done, pending = cwsandbox.wait(sandboxes)

# Wait for first N to complete
done, pending = cwsandbox.wait(refs, num_returns=2)

# Wait with timeout
done, pending = cwsandbox.wait(procs, timeout=30.0)
```

**`Waitable`**: Type alias for objects that can be waited on: `Sandbox | OperationRef[Any] | Process | TerminalSession`.

### Discovery API

Module-level sync functions (`_discovery.py`) for querying available runners and capabilities. These are simple read-only queries that return results directly (no `OperationRef`/`await` needed). **Profiles are removed** — place with `placement_mode` + runner capabilities / `runner_ids`.

**Functions:**
- `list_runners(*, runner_group_id=None, gpu_type=None, architecture=None, healthy_only=False, include_resources=False, min_available_cpu_millicores=None, min_available_memory_bytes=None, min_available_gpu_count=None, service_visibility=None)` -> `list[Runner]`: List available runners with optional filtering. Set `include_resources=True` for live resource availability. Auto-paginates.
- `get_runner(runner_id, *, organization_id)` -> `Runner`: Get a single runner by `(organization_id, runner_id)`. Raises `RunnerNotFoundError` if not found.

**Types:**
- `Runner`: Frozen dataclass with capacity, `supported_gpu_types`, `supported_architectures`, `supported_service_visibilities`, optional `RunnerResources`. Has human-readable `__repr__`.
- `RunnerResources`: Live resource availability (`available_cpu_millicores`, `available_memory_bytes`, `available_gpu_count`, `running_sandboxes`).

**Utilities:**
- `format_bytes(value)`: Format bytes as human-readable string (e.g., `17179869184` -> `'16.0 GiB'`).
- `format_cpu(millicores)`: Format CPU millicores (e.g., `4000` -> `'4.0 vCPU'`).

Usage:
```python
import cwsandbox

runners = cwsandbox.list_runners(include_resources=True, service_visibility="public")
for r in runners:
    print(r)

runner = cwsandbox.get_runner("runner-123", organization_id=runners[0].organization_id)
print(f"CPU: {cwsandbox.format_cpu(runner.max_cpu_millicores)}")
```

The API reference generator in `coreweave/docs` needs `MANIFEST_GROUPS` updated for v1 types.

### Unsupported on 1.0 (not a hybrid fallback)

These were never public 0.26 SDK surfaces and remain unsupported until v1 backends implement them: Settings **WIF admin**, **SecretStore admin** CRUD, **NetworkService** / `network_ids`, **TOKEN** / TLS_PASSTHROUGH product endpoints, mixed PRIVATE+PUBLIC on one sandbox. Create-time `Secret` inject and FSS bucket get/set still work.

### Backend Communication

Uses gRPC via `grpcio` with vendored v1 proto stubs in `src/cwsandbox/_proto/` (`sandbox_*`, `discovery_*`, `settings_*`, `sandbox_template_*`, `volume_*`). Refresh via `scripts/update-protos.sh` (protobuf runtime v26.1; see `scripts/buf.gen.python.yaml`).

**Channel management** (`_network.py`): Provides `parse_grpc_target()` for URL-to-target conversion and `create_channel()` for secure/insecure async channel creation. Auth headers are passed directly to streaming calls via metadata (interceptors don't work with request iterators).

**Streaming exec**: Uses native gRPC bidirectional streaming with request iterator pattern for proper half-close semantics via iterator completion.

### Related Repositories

- **Backend**: [github.com/coreweave/sandbox](https://github.com/coreweave/sandbox) - Server-side implementation (Go). Use `/repo-explore` to investigate backend behavior, API contracts, or debug client-server issues.

## Test Structure

- `tests/unit/` - Mock-based tests, no network calls. Default pytest path.
- `tests/integration/` - Real sandbox operations, requires auth. Run explicitly.

Unit test conftest clears all auth env vars before each test (`autouse=True` fixture).

### Integration Test Timing

Integration tests create real sandboxes and take significant time:
- **Individual test**: 5-15 seconds (sandbox startup + operation)
- **Full suite**: ~3 minutes total
- **Sandbox startup**: 30-60 seconds (mostly backend scheduling)

When running integration tests:
```bash
mise run test:e2e                         # Full suite (~2.5 minutes)
mise run test:e2e:parallel                # Parallel execution (faster)

# Individual test with timeout
timeout 120 uv run pytest tests/integration/cwsandbox/test_sandbox.py::test_sandbox_lifecycle -v
```

**Important**: If integration tests hang beyond expected times, check:
1. API patterns match current sync/async hybrid design (use `.result()`, not `await`)
2. Sandbox reaches RUNNING status before file operations

### Integration Test Patterns

Tests should use the sync/async hybrid API:
```python
# Correct pattern
def test_sandbox_example(sandbox_defaults: SandboxDefaults) -> None:
    with Sandbox.run(defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["echo", "hello"]).result()
        assert result.returncode == 0
```

## Exception Hierarchy

```
CWSandboxError
├── CWSandboxAuthenticationError
├── SandboxError
│   ├── SandboxNotRunningError
│   │   └── SandboxUnavailableError      # transient service unavailability (gRPC UNAVAILABLE / AIP-193 UNAVAILABLE_REASONS)
│   │       └── SnapshotBackendThrottledError  # transient FSS throttle/inflight cap (CWSANDBOX_FSS_BACKEND_THROTTLED / _INFLIGHT_LIMIT); retryable
│   │   # raw SandboxNotRunningError is also emitted for CANCELLED and local-stop paths
│   ├── SandboxTimeoutError
│   │   ├── SandboxRequestTimeoutError   # gRPC request deadline (DEADLINE_EXCEEDED)
│   │   ├── SandboxCommandTimeoutError   # user command exceeded its timeout (AIP-193 CWSANDBOX_COMMAND_TIMEOUT)
│   │   ├── SnapshotWaitTimeoutError     # wait_for_ready budget exceeded (CWSANDBOX_FSS_WAIT_TIMEOUT)
│   │   └── VolumeWaitTimeoutError       # Volume.wait_until_ready budget exceeded
│   ├── SandboxResourceExhaustedError    # backend resource pressure (gRPC RESOURCE_EXHAUSTED)
│   ├── SandboxTerminalStateUnavailableError  # post-stop NOT_FOUND past retry budget (backend did not report terminal state)
│   ├── SandboxTerminatedError
│   ├── SandboxFailedError
│   ├── SandboxNotFoundError             # .sandbox_id attribute
│   ├── SandboxExecutionError            # .exec_result, .exception_type, .exception_message attributes
│   ├── SandboxFileError                 # .filepath attribute
│   └── SandboxSnapshotError             # FSS failures; .file_system_snapshot_id attribute
│       ├── SnapshotNotFoundError        # CWSANDBOX_FSS_NOT_FOUND
│       ├── SnapshotNotReadyError        # CWSANDBOX_FSS_NOT_READY
│       ├── SnapshotNotSupportedError    # CWSANDBOX_FSS_NOT_SUPPORTED (org not enabled for FSS)
│       ├── SnapshotSizeExceededError    # CWSANDBOX_FSS_SIZE_EXCEEDED
│       ├── SnapshotQuotaExceededError   # CWSANDBOX_FSS_QUOTA_EXCEEDED
│       └── SnapshotBucketMismatchError  # CWSANDBOX_FSS_BUCKET_MISMATCH (reversible)
│   └── VolumeError                      # registered Volume failures; .volume_id attribute
│       ├── VolumeNotFoundError
│       ├── VolumeNotReadyError
│       ├── VolumePlacementConflictError
│       ├── VolumeTypeNotSupportedError
│       ├── VolumeNotSnapshottableError
│       ├── VolumeRunnerIneligibleError
│       ├── VolumeBackendNotFoundError
│       ├── VolumeInUseError
│       └── VolumeQuotaExceededError
│       # VolumeRunnerUnavailableError also subclasses SandboxUnavailableError (retryable)
├── DiscoveryError
│   └── RunnerNotFoundError              # .runner_id attribute
└── FunctionError
    └── AsyncFunctionError
```

(`ProfileNotFoundError` may still exist in `exceptions.py` for import compatibility but is not part of the public 1.0 surface.)

**Poll retry classification**: The sandbox-status poll loop splits exception
classes into retryable and fatal, dispatched purely by ``isinstance`` against a
registry tuple. See ``_classify_poll_error`` and ``_RETRYABLE_POLL_EXCEPTIONS``
in ``src/cwsandbox/_sandbox.py`` for the current membership. Retryable classes
are subclasses of the existing umbrella exceptions, so callers catching the
parent classes continue to work unchanged.

**FSS RPC retries**: The file-system snapshot RPCs (`snapshot()`/create,
`get_snapshot`, `list_snapshots`, `delete_snapshot`, bucket config) retry
*transient* errors with a bounded wall-clock budget
(`DEFAULT_FSS_RETRY_BUDGET_SECONDS`, default 30s; set to 0 to disable). They
reuse the poll loop's `_classify_poll_error`, so only the same retryable classes
are retried — transient unavailability, request-deadline, resource-exhaustion,
and FSS backend-throttling. Everything else (NOT_FOUND, FAILED_PRECONDITION,
quota/size, NOT_SUPPORTED) is fatal on the first attempt. Backoff is decorrelated
jitter honoring AIP-193 `RetryInfo` hints, via the shared `_retry_transient_rpc`
helper. `snapshot()` auto-generates a `request_id` when the caller omits one,
so a retried create dedups instead of producing a duplicate snapshot.

## Examples

The `examples/` directory contains runnable scripts demonstrating common patterns:
- `discover_infrastructure.py` - Runner discovery, visibilities, capacity filters, placement tips
- `quick_start.py`, `basic_execution.py`, `streaming_exec.py`, `stdin_streaming.py` - Sandbox creation and execution
- `resource_configuration.py` - ResourceOptions, flat dict, nested dict, GPU, and response properties
- `function_decorator.py` - Remote function execution with `@session.function()`
- `multiple_sandboxes.py` - Session-based parallel execution
- `interactive_streaming_sandbox.py` - Log streaming with `stream_logs()` and CLI interaction (`exec`, `sh`, `logs`)
- `reconnect_to_sandbox.py`, `async_patterns.py` - Reconnection and async patterns
- `delete_sandboxes.py` - Deletion patterns with `Sandbox.delete()`
- `list_stopped_sandboxes.py` - `Sandbox.list(show_terminated=True)`
- `file_system_snapshots.py` - Snapshot/restore/fork with `file_system_snapshot` / scratch volumes
- `error_handling.py` - Exception hierarchy and error recovery patterns
- `session_adopt_orphans.py`, `cleanup_by_tag.py`, `cleanup_old_sandboxes.py` - Orphan management and cleanup
- `parallel_batch_job.py` - Parallel batch processing with progress tracking

See `examples/README.md` and `examples/AGENTS.md` for full documentation. For detailed guides, see [docs.coreweave.com](https://docs.coreweave.com/products/coreweave-sandbox/client).

### Key Design Decisions

**One dialect per release**: 1.0.x is Sandbox v1 only; freeze 0.26.x for v1beta2. No dual stubs or `api_version` switch.

**Thread Safety**: The sync API is designed for **single-threaded use**. Calling `.result()` from multiple threads simultaneously is not supported without external synchronization. Users wanting multi-threaded access should use one sandbox per thread or add their own locking. This is intentional to keep the implementation simple.

**Lazy-Start Model**: `Sandbox.run()` returns immediately once the backend accepts the request - it does NOT wait for RUNNING status. Blocking happens explicitly via `.result()` or `.wait()`.

**Single Internal Implementation**: There is one async implementation internally. The sync/async flexibility comes from how users consume results (`.result()` vs `await`), not from duplicate code paths.

**ResourceOptions and Overcommit**: GPU is a separate field from requests/limits because the backend does not support GPU overcommit. Flat dict backward compatibility is maintained for CPU/memory (legacy form sets requests == limits for Guaranteed QoS). The flat dict form for GPU (`{"gpu_count": N}`) is a breaking change - use `ResourceOptions(gpu={"count": N})` instead. The `resource_requests` and `resource_limits` properties are populated only from start-response data, so they are None for sandboxes discovered via `Sandbox.from_id()` or `Sandbox.list()`.

## License Headers

All new files MUST include an SPDX license header. See [CONTRIBUTING.md](CONTRIBUTING.md) for full policy.

**License by directory:**
- Everything: `Apache-2.0`
- `examples/`: `BSD-3-Clause`

**Python files** (`.py`):
```python
# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
```

**Markdown files** (`.md`):
```html
<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: cwsandbox-client
-->
```

Use `BSD-3-Clause` instead for files under `examples/`. Validate with `reuse lint`.

## Temporary File Conventions

When creating temporary analysis or planning documents, use these filename suffixes to ensure they are gitignored:

| Suffix | Use Case |
|--------|----------|
| `-OLD.md` | Superseded or archived versions of documents |
| `-draft.md` | Work in progress, not ready for review |
| `-tmp.md` | Temporary files for single-session analysis |
| `-notes.md` | Personal analysis notes |

Example: `docs/api-redesign-draft.md` or `docs/spec-sync-api-OLD.md`

Files with these suffixes are excluded from git via `.gitignore`. For permanent documentation, use clear names without temporary markers.
