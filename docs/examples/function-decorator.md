# Function Decorator

The `@session.function()` decorator allows you to execute Python functions in isolated sandboxes.

## Basic Usage

```python
import asyncio
from aviato import Sandbox, SandboxDefaults

async def main():
    defaults = SandboxDefaults(container_image="python:3.11")

    async with Sandbox.session(defaults) as session:
        @session.function()
        def add(x: int, y: int) -> int:
            return x + y

        result = await add(2, 3)
        print(f"Result: {result}")  # 5

asyncio.run(main())
```

## Important Constraints

### Synchronous Functions Only

The `@session.function()` decorator only supports synchronous functions. Async functions
are not supported because the sandbox executes Python synchronously.

```python
# CORRECT: Synchronous function
@session.function()
def compute(x: int) -> int:
    return x * 2

# ERROR: Async functions are not supported
@session.function()
async def compute_async(x: int) -> int:  # Raises AsyncFunctionError
    return x * 2
```

If you need async behavior inside your sandbox function, run it explicitly:

```python
@session.function()
def fetch_urls(urls: list[str]) -> list[str]:
    import asyncio
    import aiohttp

    async def fetch_all():
        async with aiohttp.ClientSession() as session:
            tasks = [session.get(url) for url in urls]
            responses = await asyncio.gather(*tasks)
            return [await r.text() for r in responses]

    return asyncio.run(fetch_all())
```

### Decorator Order

`@session.function()` can be placed anywhere in the decorator stack. It will be
automatically removed when sending the function to the sandbox, while all other
decorators are preserved:

```python
# Both of these work correctly:
@session.function()
@retry(max_attempts=3)
def fetch_data(url: str) -> dict:
    return requests.get(url).json()

@retry(max_attempts=3)
@session.function()
def fetch_data(url: str) -> dict:
    return requests.get(url).json()
```

## Serialization Modes

### JSON (Default)

JSON serialization is the default because it is safe and cannot execute code during
deserialization. It supports basic Python types: `dict`, `list`, `str`, `int`, `float`,
`bool`, and `None`.

```python
@session.function()  # Uses Serialization.JSON by default
def create_config(name: str, value: int) -> dict[str, object]:
    return {"name": name, "value": value}
```

### Pickle (Complex Types)

Pickle serialization supports complex Python types but should only be used in trusted
environments. Use it when you need to pass custom classes, dataclasses, or other
complex objects.

```python
from dataclasses import dataclass
from aviato import Serialization

@dataclass
class ModelConfig:
    layers: int
    learning_rate: float

@session.function(serialization=Serialization.PICKLE)
def process_config(config: ModelConfig) -> dict:
    return {"layers": config.layers, "lr": config.learning_rate}
```

### Security: JSON vs Pickle

**Why does serialization mode matter for security?**

Python's `pickle` module can execute arbitrary code during deserialization. When using
`@session.function()` with pickle:

1. Your arguments are pickled and sent to the sandbox
2. The sandbox runs your function and pickles the result
3. Your client deserializes the result from the sandbox

The risk is in step 3: if a sandbox is compromised or running malicious code, it could
return a crafted pickle payload that executes arbitrary code on your machine when
deserialized.

JSON deserialization cannot execute code - it only produces basic Python types.

```python
# Safe for any environment
@session.function()
def process_data(data: dict) -> dict:
    return {"result": data["value"] * 2}

# Only use with trusted code in the sandbox
@session.function(serialization=Serialization.PICKLE)
def train_model(config: TrainingConfig) -> TrainedModel:
    return model.train(config)
```

## Functions with Closures and Globals

Functions can access variables from their enclosing scope (closures) and module-level
globals. These are automatically captured and sent to the sandbox.

```python
# Module-level global
MULTIPLIER = 10

async with Sandbox.session(defaults) as session:
    # Local closure variable
    offset = 5

    @session.function()
    def compute(x: int) -> int:
        return x * MULTIPLIER + offset  # Both captured automatically

    result = await compute(3)
    print(result)  # 35
```

### Serialization Validation

Variables must be serializable with the chosen serialization mode. The decorator
validates this upfront and provides clear error messages:

```python
import threading

lock = threading.Lock()  # Not serializable

@session.function()
def broken(x: int) -> int:
    return x * lock  # ValueError: Variable 'lock' cannot be serialized
```

## Custom Container Image

Override the session's default container for specific functions:

```python
@session.function(container_image="pytorch/pytorch:latest")
def train_model(epochs: int) -> float:
    import torch
    # ... training logic
    return final_loss
```

## Custom Temp Directory

By default, payload and result files are stored in `/tmp` inside the sandbox.
Override this if your container image has a different writable directory:

```python
@session.function(temp_dir="/var/sandbox/temp")
def process(data: dict) -> dict:
    return {"processed": True}
```

## FunctionResult

The decorator returns the function's return value directly:

```python
result = await my_function(args)
print(result)  # Function return value
```

## Error Handling

When function execution fails, a `SandboxExecutionError` is raised with detailed
information accessible via `exec_result`:

```python
from aviato import SandboxExecutionError

@session.function()
def might_fail(x: int) -> int:
    if x < 0:
        raise ValueError("x must be non-negative")
    return x * 2

try:
    result = await might_fail(-1)
except SandboxExecutionError as e:
    print(f"Exception type: {e.exception_type}")  # "ValueError"
    print(f"Exception message: {e.exception_message}")  # "x must be non-negative"
    print(f"Return code: {e.exec_result.returncode}")  # 1
    print(f"Stderr: {e.exec_result.stderr_text}")  # Full traceback (decoded)
```
