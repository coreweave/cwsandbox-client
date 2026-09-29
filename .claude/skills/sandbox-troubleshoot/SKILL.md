---
name: sandbox-troubleshoot
description: "Use when debugging CoreWeave Sandbox issues, authentication errors, timeouts, streaming problems, or orphaned sandboxes. Covers error messages, exception handling, and common fixes."
disable-model-invocation: false
---

# Troubleshooting CoreWeave Sandbox

Common errors and solutions.

## Authentication Issues

**Error**: `CWSandboxAuthenticationError` or `WandbAuthError`

Auth resolution order:
1. `CWSANDBOX_API_KEY` env var (recommended)
2. `WANDB_API_KEY` + `WANDB_ENTITY_NAME` env vars
3. `~/.netrc` (api.wandb.ai) + `WANDB_ENTITY_NAME`

```bash
# Check what's configured
echo $CWSANDBOX_API_KEY
echo $WANDB_API_KEY
echo $WANDB_ENTITY_NAME
```

| Issue | Solution |
|-------|----------|
| No credentials | Set `CWSANDBOX_API_KEY` |
| Invalid/expired token | [Create new token](https://console.coreweave.com/tokens) |
| W&B key but no entity | Set `WANDB_ENTITY_NAME` |
| Netrc parse errors | Check `~/.netrc` syntax and permissions |

## Timeout Issues

Two timeout types:

| Timeout | Scope | Where Set |
|---------|-------|-----------|
| `timeout_seconds` | Per-exec | `exec()` parameter |
| `max_lifetime_seconds` | Sandbox lifetime | `SandboxDefaults` |

```python
# Per-exec timeout
sandbox.exec(["slow_script.py"], timeout_seconds=60.0).result()

# Sandbox lifetime
defaults = SandboxDefaults(max_lifetime_seconds=3600)
```

## Exit Code Interpretation

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Misuse of command |
| 126 | Not executable |
| 127 | Command not found |
| 128+N | Killed by signal N |

```python
# Raise on non-zero
sandbox.exec(["false"], check=True).result()

# Or catch
result = sandbox.exec(["false"]).result()
if result.returncode != 0:
    print(f"Failed: {result.stderr}")
```

## Streaming Output Issues

Output appears delayed or all at once? Python buffers stdout:

```python
# Fix: Use -u flag
process = sandbox.exec(["python", "-u", "script.py"])

# Or set env var
process = sandbox.exec(["sh", "-c", "PYTHONUNBUFFERED=1 python script.py"])
```

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CWSandboxAuthenticationError` | Missing/invalid credentials | Check `CWSANDBOX_API_KEY` |
| `WandbAuthError: WANDB_ENTITY_NAME not set` | W&B key without entity | Set `WANDB_ENTITY_NAME` |
| `SandboxNotRunningError` | Operation on stopped sandbox | Check `sandbox.status` |
| `SandboxTimeoutError` | Command exceeded timeout | Increase `timeout_seconds` |
| `SandboxTerminatedError` | External kill / lifetime exceeded | Check `max_lifetime_seconds` |
| `SandboxFailedError` | Startup failed | Check container image, resources |
| `SandboxNotFoundError` | Sandbox deleted/never existed | Verify sandbox ID |
| `SandboxExecutionError` | Non-zero exit with `check=True` | Check `e.exec_result.stderr` |
| `SandboxFileError` | File operation failed | Check path, permissions |
| `FunctionSerializationError` | Can't serialize args | Use JSON types or PICKLE |
| `AsyncFunctionError` | Async func with `@session.function()` | Use sync functions only |

## Exception Handling

```python
from cwsandbox import (
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxFileError,
    SandboxNotRunningError,
)

try:
    result = sandbox.exec(["python", "-c", "import nonexistent"], check=True).result()
except SandboxExecutionError as e:
    print(f"Exit code: {e.exec_result.returncode}")
    print(f"stderr: {e.exec_result.stderr}")
except SandboxTimeoutError:
    print("Command timed out")
except SandboxFileError as e:
    print(f"File error at {e.filepath}")
except SandboxNotRunningError:
    print("Sandbox not running")
```

## Orphaned Sandboxes

Sandboxes left running after script crashes. Prevention:

```python
# Recommended: always use context managers
with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "hello"]).result()
# Always stopped
```

Cleanup existing orphans:

```python
from cwsandbox import Sandbox, SandboxStatus

# Find by tag
sandboxes = Sandbox.list(tags=["my-job"], include_stopped=True).result()

for sb in sandboxes:
    if sb.status in (SandboxStatus.RUNNING, SandboxStatus.PENDING):
        Sandbox.delete(sb.sandbox_id).result()
```

## Startup Failures

`SandboxFailedError` on creation:

1. Check container image exists
2. Verify resources are valid
3. Check network configuration

```python
try:
    sandbox = Sandbox.run("nonexistent-image").wait()
except SandboxFailedError:
    print("Failed to start - check image name")
```

## Integration Test Hangs

If tests hang:
1. Use `.result()`, not `await` in sync tests
2. Wait for RUNNING status before file ops
3. Check sandbox reaches RUNNING before operations

```python
# Correct
sandbox.wait()  # Wait until RUNNING
result = sandbox.exec(["echo", "hello"]).result()
```

## Getting Help

- Docs: https://docs.coreweave.com/products/coreweave-sandbox
- Issues: https://github.com/coreweave/cwsandbox-client

## References

- [Troubleshooting Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/troubleshooting)
