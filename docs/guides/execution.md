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

## Streaming Output

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

## Stdin Streaming

Send input to running commands by enabling stdin with `stdin=True`:

```python
with Sandbox.run() as sandbox:
    process = sandbox.exec(["cat"], stdin=True)

    process.stdin.write(b"hello world\n").result()
    process.stdin.close().result()

    result = process.result()
    print(result.stdout)  # "hello world\n"
```

### StreamWriter Methods

When `stdin=True`, `process.stdin` is a `StreamWriter` with three methods:

- **`write(data: bytes)`** - Write raw bytes. Returns `OperationRef[None]`.
- **`writeline(text: str)`** - Write text with a trailing newline (encodes to UTF-8). Returns `OperationRef[None]`.
- **`close()`** - Signal EOF. Pending writes complete first. Returns `OperationRef[None]`.

When `stdin=False` (the default), `process.stdin` is `None`.

### Multiple Writes

Send data incrementally before closing:

```python
process = sandbox.exec(["cat"], stdin=True)

process.stdin.writeline("line 1").result()
process.stdin.writeline("line 2").result()
process.stdin.writeline("line 3").result()
process.stdin.close().result()

result = process.result()
print(result.stdout)  # "line 1\nline 2\nline 3\n"
```

### Interactive Python via Stdin

Feed Python code to an interactive interpreter:

```python
process = sandbox.exec(["python3"], stdin=True)

process.stdin.writeline("x = 40 + 2").result()
process.stdin.writeline("print(f'answer: {x}')").result()
process.stdin.close().result()

result = process.result()
print(result.stdout)  # "answer: 42\n"
```

### Combining Stdin and Stdout Streaming

Stream output while sending input:

```python
process = sandbox.exec(["cat"], stdin=True)

# Send input
process.stdin.writeline("hello").result()
process.stdin.writeline("world").result()
process.stdin.close().result()

# Stream output as it arrives
for line in process.stdout:
    print(f"[out] {line}", end="")

result = process.result()
```

### EOF-Dependent Commands

Some commands (like `sort`) read all input before producing output. Close stdin to signal EOF:

```python
process = sandbox.exec(["sort"], stdin=True)

process.stdin.writeline("banana").result()
process.stdin.writeline("apple").result()
process.stdin.writeline("cherry").result()
process.stdin.close().result()  # sort needs EOF to begin

result = process.result()
print(result.stdout)  # "apple\nbanana\ncherry\n"
```

### Async Usage

In async contexts, `await` the OperationRefs directly:

```python
async with Sandbox.run() as sandbox:
    process = sandbox.exec(["cat"], stdin=True)

    await process.stdin.write(b"async hello\n")
    await process.stdin.close()

    result = await process
    print(result.stdout)  # "async hello\n"
```

### When to Use stdin=True vs stdin=False

| Scenario | stdin | Reason |
|----------|-------|--------|
| Run a command with arguments | `False` | Input comes from args, not stdin |
| Pipe data into a command | `True` | Command reads from stdin |
| Interactive interpreter | `True` | Interpreter reads commands from stdin |
| Process that reads until EOF | `True` | Need `close()` to signal EOF |
| Fire-and-forget command | `False` | No input needed |

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

## CLI

The `aviato exec` command runs a one-off command in a sandbox from the terminal:

```bash
aviato exec <sandbox-id> echo hello
aviato exec <sandbox-id> --cwd /app ls -la
aviato exec <sandbox-id> --timeout 30 python script.py
```

Exits with the command's return code.

!!! note "exec stdout vs container logs"
    `exec()` streams output from a specific command. For container-level logs (PID 1 stdout/stderr),
    use [`stream_logs()`](logging.md) instead.

## Interactive Shells

For interactive TTY sessions, use `sandbox.shell()` instead of `exec()`. See the [Interactive Shells guide](interactive-shells.md) for details.

```python
session = sandbox.shell(["/bin/bash"], width=80, height=24)
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
