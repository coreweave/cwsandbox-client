<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: cwsandbox-client
-->

# CLI Quickstart

The `cwsandbox` CLI provides terminal access to sandbox management. All commands wrap the Python SDK.

## Installation

```bash
pip install cwsandbox[cli]
```

Or with uv in a development checkout:

```bash
uv sync --extra cli
```

Verify the installation:

```bash
cwsandbox --version
cwsandbox --help
```

## Common workflows

### List and inspect sandboxes

```bash
cwsandbox ls                                      # All sandboxes
cwsandbox ls --status running                     # Filter by status
cwsandbox ls --tag my-project                     # Filter by tag
cwsandbox ls --runway-id default --tower-id t1    # Filter by infrastructure
```

### Run a command

```bash
cwsandbox exec <sandbox-id> echo hello
cwsandbox exec <sandbox-id> --cwd /app python main.py
cwsandbox exec <sandbox-id> --timeout 30 make test
```

The exit code matches the remote command's exit code.

### Stream logs

```bash
cwsandbox logs <sandbox-id>                       # Recent logs
cwsandbox logs <sandbox-id> --follow              # Continuous (like tail -f)
cwsandbox logs <sandbox-id> --tail 50 --timestamps
```

## JSON output for scripting

`cwsandbox ls -o json` returns a JSON array for use with `jq`:

```bash
# List as JSON
cwsandbox ls -o json

# Get sandbox IDs
cwsandbox ls -o json | jq -r '.[].sandbox_id'

# Exec into each running sandbox
cwsandbox ls -o json | jq -r '.[].sandbox_id' | xargs -I{} cwsandbox exec {} echo hello

# Filter by field
cwsandbox ls -o json | jq '.[] | select(.tower_id == "my-tower")'
```

Each object contains: `sandbox_id`, `status`, `tower_id`, `runway_id`, `tower_group_id`, `started_at` (ISO 8601 or null).

## See also

- [Command Execution](execution.md) — `exec()` SDK method details
- [Sandbox Logging](logging.md) — `stream_logs()` SDK method details
