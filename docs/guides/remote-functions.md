# Remote Functions Guide

This guide covers the `@session.function()` decorator for running Python functions in sandboxes.

## Overview

The function decorator API lets you execute Python functions in isolated sandbox containers:

```python
import aviato

with aviato.Session(container_image="python:3.11") as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y

    # Execute in sandbox
    result = compute.remote(2, 3).result()
    print(result)  # 5
```

## Basic Usage

### Defining Functions

Decorate functions with `@session.function()`:

```python
@session.function()
def process_data(data: dict) -> dict:
    # Code runs inside the sandbox
    import pandas as pd
    df = pd.DataFrame(data)
    return {"mean": df["value"].mean()}
```

### Calling Functions

Call `.remote()` on the decorated function to execute in the sandbox:

```python
# Returns OperationRef immediately
ref = compute.remote(2, 3)

# Block for result
result = ref.result()

# One-liner
result = compute.remote(2, 3).result()
```

## Execution Methods

### map() - Parallel Execution

Execute across multiple inputs:

```python
@session.function()
def square(x: int) -> int:
    return x * x

# Execute for each input
refs = square.map((x,) for x in range(10))

# Collect all results
from aviato import get
results = get(refs)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

With tuples for multiple arguments:

```python
@session.function()
def add(x: int, y: int) -> int:
    return x + y

# Each tuple is unpacked as arguments
refs = add.map([(1, 2), (3, 4), (5, 6)])
results = get(refs)  # [3, 7, 11]
```

### local() - Local Execution

Run locally without a sandbox (useful for testing):

```python
# No sandbox created - runs in current process
result = compute.local(2, 3)
print(result)  # 5
```

## Serialization Modes

### JSON (Default)

Safe and human-readable, but limited to JSON-serializable types:

```python
from aviato import Serialization

@session.function(serialization=Serialization.JSON)
def process(data: dict) -> dict:
    return {"result": data["value"] * 2}
```

### Pickle

Supports complex Python objects:

```python
import numpy as np

@session.function(serialization=Serialization.PICKLE)
def compute_numpy(arr: np.ndarray) -> np.ndarray:
    return arr * 2

arr = np.array([1, 2, 3])
result = compute_numpy.remote(arr).result()  # array([2, 4, 6])
```

Use pickle when you need:
- NumPy arrays
- Pandas DataFrames
- Custom class instances
- Complex nested objects

## Closures and Globals

### Closure Variables

Functions can capture variables from their enclosing scope:

```python
multiplier = 10

@session.function()
def multiply(x: int) -> int:
    return x * multiplier  # Captures 'multiplier'

result = multiply.remote(5).result()  # 50
```

### Global Variables

Referenced globals are serialized with the function:

```python
CONFIG = {"threshold": 0.5}

@session.function()
def check_value(x: float) -> bool:
    return x > CONFIG["threshold"]

result = check_value.remote(0.7).result()  # True
```

## Container Image

Override the container image for specific functions:

```python
@session.function(container_image="pytorch/pytorch:latest")
def train_model(data: dict) -> dict:
    import torch
    # GPU training code
    return {"loss": 0.01}
```

## Error Handling

Function exceptions propagate to the caller:

```python
@session.function()
def failing_function() -> None:
    raise ValueError("Something went wrong")

try:
    failing_function.remote().result()
except Exception as e:
    print(f"Function failed: {e}")
```

## When to Use Functions vs Sandboxes

| Use Case | Recommended API |
|----------|-----------------|
| Simple Python computation | Function decorator |
| Map/reduce over data | Function decorator |
| Interactive workflow, multiple commands | Sandbox |
| Streaming output | Sandbox |
| File manipulation | Sandbox |
| Long-running processes | Sandbox |

## Limitations

The function API is intentionally simple. For complex workflows:

- **Retries/backoff**: Implement in calling code
- **Task dependencies/DAGs**: Use Airflow, Prefect, etc.
- **Complex scheduling**: Use the sandbox API directly

## Complete Example

```python
import aviato
from aviato import SandboxDefaults, Serialization, get
import numpy as np

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("remote-functions-demo",),
)

with aviato.Session(defaults=defaults) as session:
    # JSON serialization for simple types
    @session.function()
    def square(x: int) -> int:
        return x * x

    # Pickle for complex types
    @session.function(serialization=Serialization.PICKLE)
    def process_array(arr: np.ndarray) -> float:
        return float(arr.mean())

    # Single execution
    result = square.remote(7).result()
    print(f"7 squared: {result}")

    # Parallel execution
    refs = square.map((x,) for x in range(5))
    results = get(refs)
    print(f"Squares: {results}")

    # NumPy example
    arr = np.array([1, 2, 3, 4, 5])
    mean = process_array.remote(arr).result()
    print(f"Array mean: {mean}")
```
