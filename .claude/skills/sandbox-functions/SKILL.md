---
name: sandbox-functions
description: "Use when executing Python functions remotely in CoreWeave Sandbox, using @session.function() decorator, remote function execution, .map() for parallel, or .local() for testing. Covers serialization modes and closure capture."
disable-model-invocation: false
---

# Remote Functions

Run Python functions inside sandboxes without writing command strings.

## Basic Usage

```python
from cwsandbox import Sandbox, SandboxDefaults

with Sandbox.session(SandboxDefaults()) as session:
    @session.function()
    def add(x: int, y: int) -> int:
        return x + y

    result = add.remote(2, 3).result()  # 5
```

## How It Works

1. Extracts function source via AST
2. Captures closure variables
3. Serializes payload (JSON or PICKLE)
4. Creates ephemeral sandbox
5. Executes and returns result

## Parallel with .map()

```python
@session.function()
def square(x: int) -> int:
    return x * x

refs = square.map([(1,), (2,), (3,)])
results = [r.result() for r in refs]  # [1, 4, 9]
```

## Local Testing

Test without sandbox overhead:

```python
@session.function()
def slow_computation(x: int) -> int:
    return x * 2

# Test locally
result = slow_computation.local(5)  # 10 - runs in-process
```

## Serialization Modes

```python
from cwsandbox import Serialization

@session.function(serialization=Serialization.JSON)  # Default
def json_func(x: list) -> dict:
    return {"sum": sum(x)}

@session.function(serialization=Serialization.PICKLE)  # Supports complex objects
def pickle_func(x: numpy.ndarray) -> numpy.ndarray:
    return x * 2
```

| Mode | Description |
|------|-------------|
| `JSON` | Safe, human-readable, JSON-serializable types only |
| `PICKLE` | Complex Python objects, numpy arrays, requires trust |

## Closure Capture

Functions can reference external variables:

```python
model_path = "/tmp/model.pt"

@session.function()
def predict(x: int) -> float:
    import torch
    model = torch.load(model_path)
    return model.predict(x)
```

## Error Handling

```python
from cwsandbox import FunctionError, FunctionSerializationError

try:
    result = fragile_function.remote("arg").result()
except FunctionSerializationError:
    print("Could not serialize arguments")
except Exception as e:
    print(f"Execution failed: {e}")
```

## References

- [Remote Functions Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/remote-functions)
