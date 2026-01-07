# Aviato SDK Documentation

Python client library for Aviato sandboxes - a remote code execution platform.

## Installation

```bash
pip install aviato
```

## Authentication

The SDK checks for credentials in this order:

### Option 1: Aviato API Key (Recommended)

```bash
export AVIATO_API_KEY="your-api-key"
```

### Option 2: W&B Credentials

If you have W&B credentials configured, the SDK can use them:

```bash
export WANDB_API_KEY="your-wandb-key"
export WANDB_ENTITY_NAME="your-entity"
```

The SDK also reads W&B credentials from `~/.netrc` if `WANDB_API_KEY` isn't set:

```
machine api.wandb.ai
  login user
  password your-wandb-key
```

`WANDB_ENTITY_NAME` is still required when using netrc.

## Quick Start

```python
from aviato import Sandbox

with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "Hello, Aviato!"]).result()
    print(result.stdout)  # "Hello, Aviato!\n"
```

## Core Concepts

### Non-Blocking by Default

Operations return immediately. Blocking happens explicitly when you need results:

| Operation | Returns | Block with |
|-----------|---------|------------|
| `Sandbox.run()` | `Sandbox` | `.wait()` |
| `sandbox.exec()` | `Process` | `.result()` |
| `sandbox.read_file()` | `OperationRef` | `.get()` |
| `sandbox.write_file()` | `OperationRef` | `.get()` |
| `sandbox.stop()` | `OperationRef` | `.get()` |

This enables natural parallelism - start multiple operations, then collect results:

```python
refs = [sandbox.read_file(f"/app/file{i}.txt") for i in range(10)]
contents = [ref.get() for ref in refs]
```

### Sandbox Lifecycle

Sandboxes progress through these states:

```
PENDING -> CREATING -> RUNNING -> (COMPLETED | FAILED | TERMINATED)
```

Most operations handle state transitions automatically. For example, `exec()` waits for RUNNING before executing.

## Guides

| Guide | Topic |
|-------|-------|
| [Execution](guides/execution.md) | Commands, streaming, timeouts, working directories |
| [File Operations](guides/file-operations.md) | Reading, writing, parallel transfers |
| [Sessions](guides/sessions.md) | Managing multiple sandboxes with shared config |
| [Remote Functions](guides/remote-functions.md) | The `@session.function()` decorator |
| [Sandbox Configuration](guides/sandbox-config.md) | Resources, mounted files, ports |
| [Cleanup Patterns](guides/cleanup-patterns.md) | Resource management, orphan handling |
| [Sync vs Async](guides/sync-vs-async.md) | When to use sync vs async patterns |
| [Troubleshooting](guides/troubleshooting.md) | Common issues and solutions |

## Examples

See the [examples directory](../examples/README.md) for runnable Python scripts.

## Architecture

For class internals, authentication flow, and implementation details, see [CLAUDE.md](../CLAUDE.md).
