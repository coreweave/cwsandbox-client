# Aviato Examples

This directory contains example scripts demonstrating how to use the `aviato` package.

## Prerequisites

1. Install the package:
   ```bash
   pip install aviato
   ```

2. Set your API key:
   ```bash
   export AVIATO_API_KEY="your-api-key"
   ```

## Examples

### Quick Start (`quick_start.py`)

The simplest way to get started with sandboxes:

```bash
python examples/quick_start.py
```

Demonstrates:
- Using `Sandbox.create()` factory method for quick sandbox creation
- Auto-starting sandboxes with positional command arguments
- Using `wait()` to wait for completion

### Basic Execution (`basic_execution.py`)

More detailed sandbox usage with full control:

```bash
python examples/basic_execution.py
```

Demonstrates:
- Creating sandboxes with explicit constructor
- Using `SandboxDefaults` for configuration
- Executing commands with `exec()`
- Reading and writing files

### Function Decorator (`function_decorator.py`)

Execute Python functions in sandboxes:

```bash
python examples/function_decorator.py
```

Demonstrates:
- Using `Sandbox.session()` for session management
- The `@session.function()` decorator
- Pickle and JSON serialization modes
- Functions with closure variables

### Multiple Sandboxes (`multiple_sandboxes.py`)

Managing multiple sandboxes concurrently:

```bash
python examples/multiple_sandboxes.py
```

Demonstrates:
- Creating multiple sandboxes via session
- Running parallel commands with `asyncio.gather()`
- Automatic cleanup on session exit

### List Sandboxes (`list_sandboxes.py`)

Query and filter existing sandboxes:

```bash
python examples/list_sandboxes.py
```

Demonstrates:
- Using `Sandbox.list()` to query existing sandboxes
- Filtering by status, tags, tower IDs, etc.
- Performing operations on discovered sandboxes

### Delete Sandboxes (`delete_sandboxes.py`)

Multiple ways to delete sandboxes:

```bash
python examples/delete_sandboxes.py
```

Demonstrates:
- Using `Sandbox.delete(sandbox_id)` class method for deletion by ID
- Using `sandbox.stop()` instance method
- Using `Sandbox.from_id()` to attach and then stop
- Handling `SandboxNotFoundError` with `missing_ok=True`

### Reconnect to Sandbox (`reconnect_to_sandbox.py`)

Attach to existing sandboxes by ID:

```bash
# Create a long-running sandbox
python examples/reconnect_to_sandbox.py --create

# Reconnect to it later
python examples/reconnect_to_sandbox.py --sandbox-id <id>

# Stop it after reconnecting
python examples/reconnect_to_sandbox.py --sandbox-id <id> --stop
```

Demonstrates:
- Using `Sandbox.from_id()` to attach to existing sandboxes
- Executing commands on reconnected sandboxes
- Managing long-running sandboxes across script invocations

### Session Adopt Orphans (`session_adopt_orphans.py`)

Use sessions to adopt and clean up orphaned sandboxes:

```bash
# Create orphaned sandboxes
python examples/session_adopt_orphans.py --create-orphans

# Adopt and clean them up
python examples/session_adopt_orphans.py --cleanup

# Demo session.adopt()
python examples/session_adopt_orphans.py --demo-adopt

# Demo session.from_id()
python examples/session_adopt_orphans.py --demo-from-id
```

Demonstrates:
- Using `session.list(adopt=True)` to adopt orphaned sandboxes
- Using `session.adopt()` to manually adopt sandboxes
- Using `session.from_id()` to attach and adopt by ID
- Automatic cleanup of adopted sandboxes on session exit

### Batch Job with Cleanup (`batch_job_with_cleanup.py`)

Robust batch job pattern with cleanup:

```bash
python examples/batch_job_with_cleanup.py
```

Demonstrates:
- Using unique tags for batch job identification
- Cleanup in `finally` block to handle interruptions
- Using `Sandbox.list()` to find orphaned sandboxes from the batch

### Cleanup by Tag (`cleanup_by_tag.py`)

Clean up sandboxes identified by tags:

```bash
# Create tagged sandboxes
python examples/cleanup_by_tag.py --create

# Clean them up
python examples/cleanup_by_tag.py --cleanup
```

Demonstrates:
- Tagging sandboxes for easy identification
- Using `Sandbox.list(tags=...)` to find sandboxes
- Parallel cleanup with `asyncio.gather()`

### Cleanup Old Sandboxes (`cleanup_old_sandboxes.py`)

Clean up sandboxes older than a threshold:

```bash
# Dry run
python examples/cleanup_old_sandboxes.py --dry-run

# Actually clean up sandboxes older than 2 hours
python examples/cleanup_old_sandboxes.py --max-age-hours 2
```

Demonstrates:
- Filtering sandboxes by age using `started_at` timestamp
- Client-side filtering after `Sandbox.list()`
- Dry run mode for safe testing

## API Patterns

### Quick Usage (Factory Method)

```python
# One-liner creation - auto-starts
sandbox = await Sandbox.create("echo", "hello")
await sandbox.stop()
```

### Full Control (Constructor)

```python
# Explicit construction with context manager
async with Sandbox(
    command="sleep",
    args=["infinity"],
    container_image="python:3.11",
) as sandbox:
    result = await sandbox.exec(["echo", "hello"])
```

### With Defaults

```python
defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=60,
    tags=("my-app",),
)

async with Sandbox(command="...", defaults=defaults) as sandbox:
    ...
```

### Session for Multiple Sandboxes

```python
async with Sandbox.session(defaults) as session:
    sb1 = session.create(command="sleep", args=["infinity"])
    sb2 = session.create(command="sleep", args=["infinity"])
    
    async with sb1, sb2:
        ...
```

### Function Execution

```python
async with Sandbox.session(defaults) as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y
    
    result = await compute(2, 3)
    print(result.value)  # 5
```
