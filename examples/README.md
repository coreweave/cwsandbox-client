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
