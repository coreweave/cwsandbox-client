# Command Execution Guide

This guide covers running commands in sandboxes using the `exec()` method.

## Basic Execution

The `exec()` method runs commands in a sandbox and returns a `Process` handle:

```python
from aviato import Sandbox

with Sandbox.run() as sandbox:
    # Run a command and get the result
    result = sandbox.exec(["echo", "Hello, World!"]).result()

    print(result.stdout)      # "Hello, World!\n"
    print(result.returncode)  # 0
```

## Getting Results

The `exec()` method returns a `Process` object. Call `.result()` to block for the output:

```python
# Returns Process immediately
process = sandbox.exec(["python", "-c", "print('hello')"])

# Block for result
result = process.result()
print(result.stdout)      # "hello\n"
print(result.stderr)      # ""
print(result.returncode)  # 0

# One-liner pattern
result = sandbox.exec(["ls", "-la"]).result()
```

## Output Handling

### Default: Silent

By default, `exec()` captures output without printing:

```python
result = sandbox.exec(["echo", "Hello!"]).result()
print(result.stdout)  # "Hello!\n"
```

### Streaming Output

For real-time output, iterate over `process.stdout` before calling `.result()`:

```python
# Returns Process immediately
process = sandbox.exec(["python", "long_script.py"])

# Stream stdout line by line
for line in process.stdout:
    print(f"[stdout] {line}", end="")

# Get final result
result = process.result()
print(f"Exit code: {result.returncode}")
```

Use streaming when you need to:
- Monitor long-running processes
- Process output as it arrives
- Implement progress indicators

### Auto-print (Convenience)

For quick debugging or when you just want to watch output without processing it, use `print_output=True`:

```python
result = sandbox.exec(["python", "long_script.py"], print_output=True).result()
```

Both stdout and stderr are printed to stdout. Set `AVIATO_EXEC_PRINT=1` to enable globally.

## Working Directory

Set the working directory with `cwd`:

```python
result = sandbox.exec(
    ["ls", "-la"],
    cwd="/app/data",
).result()
```

The path must be absolute.

## Timeouts

Set command timeout with `timeout_seconds`:

```python
from aviato import SandboxTimeoutError

try:
    result = sandbox.exec(
        ["sleep", "60"],
        timeout_seconds=5.0,
    ).result()
except SandboxTimeoutError:
    print("Command timed out")
```

## Error Handling with check

The `check` parameter controls error behavior for non-zero exit codes:

### check=False (Default)

Returns the result regardless of exit code:

```python
result = sandbox.exec(["false"]).result()
print(result.returncode)  # 1 (no exception)
```

### check=True

Raises `SandboxExecutionError` on non-zero exit:

```python
from aviato import SandboxExecutionError

try:
    result = sandbox.exec(
        ["python", "-c", "raise ValueError('oops')"],
        check=True,
    ).result()
except SandboxExecutionError as e:
    print(f"Command failed: {e.exec_result.returncode}")
    print(f"stderr: {e.exec_result.stderr}")
```

## Running Python Code

Execute Python scripts or one-liners:

```python
# One-liner
result = sandbox.exec(
    ["python", "-c", "import sys; print(sys.version)"],
).result()

# Script from string
code = '''
import json
data = {"result": 42}
print(json.dumps(data))
'''
result = sandbox.exec(["python", "-c", code]).result()
output = json.loads(result.stdout)
```

## Sequential vs Parallel Execution

### Sequential (Order Matters)

```python
# Dependencies require sequential execution
sandbox.exec(["pip", "install", "requests"]).result()
sandbox.exec(["python", "script_using_requests.py"]).result()
```

### Parallel (Independent Commands)

```python
# Start multiple sandboxes
sandboxes = [Sandbox.run() for _ in range(3)]

# Start commands on each
processes = [
    sb.exec(["python", "-c", f"print({i})"])
    for i, sb in enumerate(sandboxes)
]

# Collect all results
results = [p.result() for p in processes]
for r in results:
    print(r.stdout)
```

### Waiting for N of M to Complete

Use `aviato.wait()` to wait for a subset of processes:

```python
import aviato

processes = [sb.exec(["python", "task.py"]) for sb in sandboxes]

# Wait for first 2 to complete
done, pending = aviato.wait(processes, num_returns=2)

# Process completed ones immediately
for p in done:
    print(p.result().stdout)

# Wait for remaining
for p in pending:
    print(p.result().stdout)
```

## Process Control

The `Process` object provides methods for monitoring and control:

```python
process = sandbox.exec(["python", "server.py"])

# Check if running (non-blocking)
if process.poll() is None:
    print("Still running")

# Wait for completion
exit_code = process.wait()
print(f"Exited with code: {exit_code}")
```
