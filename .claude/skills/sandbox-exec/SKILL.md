---
name: sandbox-exec
description: "Use when executing commands in CoreWeave Sandbox, capturing output, handling errors, timeouts, or running commands in specific directories. Covers exec(), parallel execution, Process, and ProcessResult."
disable-model-invocation: false
---

# Running Commands in Sandboxes

Execute commands inside a sandbox and capture output.

## Basic exec()

```python
sandbox.exec(["echo", "Hello"]).result()
```

## Options

```python
# Raise SandboxExecutionError on non-zero exit
sandbox.exec(["ls", "/nonexistent"], check=True).result()

# Raise SandboxTimeoutError if exceeded
sandbox.exec(["sleep", "60"], timeout_seconds=5).result()

# Run in specific directory
sandbox.exec(["ls"], cwd="/app").result()

# Enable stdin streaming
process = sandbox.exec(["python"], stdin=True)
process.stdin.write(b"print('hello')\n")
process.stdin.close()
result = process.result()
```

## Parallel Execution

```python
import cwsandbox

# exec() returns immediately
p1 = sandbox.exec(["sleep", "1"])
p2 = sandbox.exec(["sleep", "1"])
p3 = sandbox.exec(["sleep", "1"])

# .result() blocks ~1 second total, not 3
cwsandbox.result([p1, p2, p3])
```

## Process Result

```python
result = sandbox.exec(["echo", "hello"]).result()

print(result.stdout)        # "hello\n"
print(result.stderr)        # ""
print(result.returncode)    # 0
print(result.command)       # ["echo", "hello"]
```

## Streaming Output

```python
process = sandbox.exec(["echo", "hello"])
for line in process.stdout:
    print(line, end="")
result = process.result()
```

## Error Handling

```python
from cwsandbox import (
    SandboxExecutionError,
    SandboxTimeoutError,
)

try:
    sandbox.exec(["false"], check=True).result()
except SandboxExecutionError as e:
    print(f"Command failed: {e.exec_result.returncode}")

try:
    sandbox.exec(["sleep", "10"], timeout_seconds=1).result()
except SandboxTimeoutError:
    print("Command timed out")
```

## References

- [Execution Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/execution)
