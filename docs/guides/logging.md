<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: cwsandbox-client
-->

# Sandbox Logging

Stream logs from a sandbox's **main process** with `stream_logs()`. This captures stdout/stderr from the command passed to `Sandbox.run()` — output from `exec()` commands is **not** included (see the [execution guide](execution.md) for `Process.stdout`/`Process.stderr`). Returns a `StreamReader` that yields log lines — iterate synchronously or asynchronously.

> **Note:** Sandboxes created with the default command (`tail -f /dev/null`) produce no log output. To use `stream_logs()`, pass a command that writes to stdout/stderr when calling `Sandbox.run()`.

## Retrieve recent logs

```python
for line in sandbox.stream_logs(tail_lines=100):
    print(line, end="")
```

## Follow mode

Stream logs continuously, like `tail -f`. The iterator blocks until new data arrives.

```python
for line in sandbox.stream_logs(follow=True):
    print(line, end="")
```

Press Ctrl-C to stop when iterating in follow mode.

## Filter by time

Only retrieve logs after a specific timestamp.

```python
from datetime import datetime, timezone

since = datetime(2026, 2, 20, 14, 0, 0, tzinfo=timezone.utc)
for line in sandbox.stream_logs(since_time=since):
    print(line, end="")
```

## Timestamps

Prefix each line with an ISO 8601 timestamp from the server.

```python
for line in sandbox.stream_logs(tail_lines=10, timestamps=True):
    print(line, end="")
# Output: 2026-02-20T14:30:00Z some log line
```

## Async iteration

```python
async for line in sandbox.stream_logs(follow=True):
    print(line, end="")
```

## Retrieving logs from stopped sandboxes

You can retrieve historical logs from sandboxes that have already completed, failed, or been terminated:

```python
sb = Sandbox.from_id("sbx-abc123").result()
for line in sb.stream_logs(tail_lines=50):
    print(line, end="")
```

Only `follow=False` (the default) is supported for stopped sandboxes.

## What are container logs?

Container logs capture stdout and stderr from the sandbox's main process (PID 1). Commands run via `exec()` produce output on their own streams (`Process.stdout`/`Process.stderr`), not container logs. To generate logs visible to `stream_logs()`, your sandbox command must write to stdout or stderr.

> **Looking for exec output?** See the [Command Execution](execution.md) guide for streaming stdout/stderr from commands run via `exec()`.

## See also

- [Command Execution](execution.md) — running commands and streaming their output with `exec()`
- [Sync vs Async](sync-vs-async.md) — iteration patterns
