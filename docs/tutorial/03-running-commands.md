# 3. Running Commands

```python
# Basic command
sandbox.exec(["echo", "Hello"]).result()

# Raise SandboxExecutionError if command returns non-zero exit code
sandbox.exec(["ls", "/nonexistent"], check=True).result()

# Raise SandboxTimeoutError if command exceeds timeout
sandbox.exec(["sleep", "60"], timeout_seconds=5).result()

# Run in a specific directory
sandbox.exec(["ls"], cwd="/app").result()
```

## Parallel Execution

```python
import cwsandbox

# exec() returns a Process immediately without waiting for the command to finish
p1 = sandbox.exec(["sleep", "1"])
p2 = sandbox.exec(["sleep", "1"])
p3 = sandbox.exec(["sleep", "1"])

# Call .result() when you need the output - ~1 second total, not 3
cwsandbox.result([p1, p2, p3])
```

For streaming, error handling, and process control, see the [Execution Guide](../guides/execution.md).

