# Troubleshooting Guide

This guide covers common issues and their solutions when working with the CWSandbox SDK.

## Authentication Issues

**Symptom**: `CWSandboxAuthenticationError` or `WandbAuthError` raised on sandbox operations.

The SDK resolves authentication in this order:

1. `CWSANDBOX_API_KEY` env var (takes priority)
2. `WANDB_API_KEY` + `WANDB_ENTITY_NAME` env vars
3. `~/.netrc` (api.wandb.ai) + `WANDB_ENTITY_NAME`

You need **one** of these configured. Check which method you're using:

```bash
# Option 1: CWSandbox API key
echo $CWSANDBOX_API_KEY

# Option 2: W&B credentials
echo $WANDB_API_KEY
echo $WANDB_ENTITY_NAME
```

### Common Issues

| Issue | Solution |
|-------|----------|
| No credentials configured | Set `CWSANDBOX_API_KEY` or W&B credentials |
| Invalid or expired API key | Contact your administrator for a new key |
| W&B API key set but entity missing | Set `WANDB_ENTITY_NAME` to your W&B entity/team |
| Using netrc but entity missing | Set `WANDB_ENTITY_NAME` - it's always required for W&B |
| Netrc parse errors | Check `~/.netrc` file syntax and permissions |

---

## Command Execution Issues

### Timeout Tuning

The SDK has two types of timeouts:

| Timeout | Scope | Default | Where Set |
|---------|-------|---------|-----------|
| `timeout_seconds` | Per-exec | None | `exec()` parameter |
| `max_lifetime_seconds` | Sandbox lifetime | Backend-controlled | `SandboxDefaults` |

**Client-side timeout** (`timeout_seconds`):

```python
from cwsandbox import SandboxTimeoutError

try:
    result = sandbox.exec(
        ["python", "slow_script.py"],
        timeout_seconds=60.0,  # 60 second timeout
    ).result()
except SandboxTimeoutError:
    print("Command timed out")
```

**Server-side lifetime** (`max_lifetime_seconds`):

```python
from cwsandbox import SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=3600,  # Sandbox terminates after 1 hour
)
```

### Long-Running Commands

**Issue**: Command takes longer than expected.

**Solutions**:

1. Set appropriate timeout:

```python
process = sandbox.exec(
    ["pip", "install", "tensorflow"],
    timeout_seconds=300.0,  # 5 minutes for large packages
)
result = process.result()
```

2. Use streaming to monitor progress:

```python
process = sandbox.exec(["pip", "install", "tensorflow"])

for line in process.stdout:
    print(line, end="")

result = process.result()
```

### Exit Code Interpretation

**Issue**: Understanding command failures.

Exit codes follow Unix conventions:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Misuse of command |
| 126 | Command not executable |
| 127 | Command not found |
| 128+N | Killed by signal N |

Use `check=True` to raise on non-zero exit:

```python
from cwsandbox import SandboxExecutionError

try:
    result = sandbox.exec(
        ["python", "-c", "import nonexistent"],
        check=True,
    ).result()
except SandboxExecutionError as e:
    print(f"Exit code: {e.exec_result.returncode}")
    print(f"stderr: {e.exec_result.stderr}")
```

---

## Streaming Output Issues

### Line Buffering Behavior

**Issue**: Output appears delayed or all at once when streaming.

Python buffers stdout when not connected to a TTY. Force unbuffered output:

```python
# Option 1: Use -u flag
process = sandbox.exec(["python", "-u", "script.py"])
for line in process.stdout:
    print(line, end="")
result = process.result()

# Option 2: Set PYTHONUNBUFFERED
process = sandbox.exec(["sh", "-c", "PYTHONUNBUFFERED=1 python script.py"])
for line in process.stdout:
    print(line, end="")
result = process.result()
```

For your own scripts, flush explicitly:

```python
print("Progress...", flush=True)
```

---

## Cleanup Problems

### Orphaned Sandboxes

**Issue**: Sandboxes remain running after script exits or crashes.

**Prevention**: Use context managers:

```python
# Recommended: automatic cleanup
with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "hello"]).result()
# Sandbox stopped automatically
```

See [Cleanup Patterns - Orphan Management](cleanup-patterns.md#orphan-management) for how to find and clean up orphaned sandboxes by tag.

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CWSandboxAuthenticationError` | Missing or invalid credentials | Check `CWSANDBOX_API_KEY` is set |
| `WandbAuthError: WANDB_ENTITY_NAME is not set` | W&B API key found but entity missing | Set `WANDB_ENTITY_NAME` env var |
| `SandboxNotRunningError` | Operation on stopped sandbox | Check `sandbox.status` before operations |
| `SandboxTimeoutError` | Operation exceeded timeout | Increase `timeout_seconds` or optimize command |
| `SandboxTerminatedError` | Sandbox killed externally | Check `max_lifetime_seconds` or external termination |
| `SandboxFailedError` | Sandbox failed to start | Check container image and resources |
| `SandboxNotFoundError` | Sandbox deleted or never existed | Verify sandbox ID is correct |
| `SandboxExecutionError` | Command returned non-zero (with `check=True`) | Check `e.exec_result.stderr` for details |
| `SandboxFileError` | File operation failed | Check file path and permissions |
| `FunctionSerializationError` | Can't serialize function args | Use JSON-serializable types or `Serialization.PICKLE` |
| `AsyncFunctionError` | Async function passed to `@session.function()` | Use sync functions only |

---
