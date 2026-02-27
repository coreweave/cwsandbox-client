<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: cwsandbox-client
-->

# Sandbox Logging

Stream logs from a running sandbox with `stream_logs()`. Returns a `StreamReader` that yields log lines — iterate synchronously or asynchronously.

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

## What are container logs?

Container logs capture stdout and stderr from the sandbox's main process (PID 1). Commands run via `exec()` produce output on the exec stream, not container logs. To generate logs visible to `stream_logs()`, your sandbox command must write to stdout or stderr.

## See also

- [Command Execution](execution.md) — running commands with `exec()`
- [Sync vs Async](sync-vs-async.md) — iteration patterns
