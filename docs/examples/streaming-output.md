# Streaming Output

This guide demonstrates how to stream stdout and stderr in real-time from sandbox commands, useful for long-running processes like training scripts or package installations.

## Quick Start: Simple Streaming

The simplest way to get real-time output is to add `stream_output=True` to `exec()`:

```python
import asyncio
from aviato import Sandbox

async def main():
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        # Stream output directly to console
        result = await sandbox.exec(["python", "train.py"], stream_output=True)
        
        print(f"Training complete! Exit code: {result.returncode}")

asyncio.run(main())
```

This gives you:
- **Real-time output** printed directly to stdout/stderr
- **Complete result** returned with all output collected

## Custom Callbacks

For more control over how output is handled, use `on_stdout` and `on_stderr` callbacks:

```python
import asyncio
from aviato import Sandbox

async def main():
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        # Custom handling of output
        result = await sandbox.exec(
            ["python", "train.py"],
            on_stdout=lambda data: my_logger.info(data.decode()),
            on_stderr=lambda data: my_logger.error(data.decode()),
        )

asyncio.run(main())
```

## Practical Examples

### Progress Tracking

```python
async def install_with_progress(sandbox):
    lines_received = 0
    
    def track_progress(data: bytes):
        nonlocal lines_received
        lines_received += data.count(b'\n')
        print(f"\rLines received: {lines_received}", end="")
    
    result = await sandbox.exec(
        ["pip", "install", "-r", "requirements.txt"],
        on_stdout=track_progress,
        on_stderr=track_progress,
    )
    
    print(f"\nInstallation complete: {result.returncode}")
    return result
```

### Separate Output Files

```python
async def exec_with_logging(sandbox, command):
    with open("stdout.log", "wb") as stdout_file, \
         open("stderr.log", "wb") as stderr_file:
        
        result = await sandbox.exec(
            command,
            on_stdout=lambda data: stdout_file.write(data),
            on_stderr=lambda data: stderr_file.write(data),
        )
    
    return result
```

### Real-Time Training Metrics

```python
import json

async def monitor_training(sandbox):
    metrics = []
    
    def parse_metrics(data: bytes):
        text = data.decode()
        for line in text.strip().split('\n'):
            if line.startswith('{"epoch":'):
                metrics.append(json.loads(line))
                print(f"Epoch {metrics[-1]['epoch']}: loss={metrics[-1]['loss']:.4f}")
    
    result = await sandbox.exec(
        ["python", "train.py", "--json-metrics"],
        on_stdout=parse_metrics,
    )
    
    return metrics
```

### Timeout with Partial Output

```python
from aviato import SandboxTimeoutError

async def exec_with_timeout(sandbox, command, timeout=30):
    output_so_far = []
    
    try:
        result = await sandbox.exec(
            command,
            timeout_seconds=timeout,
            on_stdout=lambda data: output_so_far.append(data),
        )
        return result
    except SandboxTimeoutError:
        print(f"Command timed out. Partial output received:")
        print(b"".join(output_so_far).decode())
        raise
```

## Error Handling

Streaming methods raise the same exceptions as regular `exec()`:

```python
from aviato import SandboxExecutionError, SandboxTimeoutError

try:
    result = await sandbox.exec(
        ["failing-command"],
        check=True,  # Raise on non-zero exit
        on_stdout=lambda data: print(data.decode(), end=""),
    )
except SandboxExecutionError as e:
    print(f"Command failed: {e.exec_result.stderr}")
except SandboxTimeoutError:
    print("Command timed out")
```

## Best Practices

1. **Flush output in your scripts**: Python buffers stdout by default. Use `sys.stdout.flush()` or run with `python -u` for unbuffered output.

2. **Handle partial data**: Callbacks may receive partial lines. Buffer if you need complete lines:

   ```python
   buffer = ""
   
   def on_stdout(data: bytes):
       nonlocal buffer
       buffer += data.decode()
       while '\n' in buffer:
           line, buffer = buffer.split('\n', 1)
           process_complete_line(line)
   ```

3. **Use appropriate timeouts**: Streaming connections can hang. Always set reasonable timeouts.

4. **Prefer callbacks for simple cases**: Use `on_stdout`/`on_stderr` with `exec()` unless you need fine-grained chunk metadata.

