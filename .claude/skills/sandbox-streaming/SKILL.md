---
name: sandbox-streaming
description: "Use when streaming command output in real-time from CoreWeave Sandbox, handling stdin streaming, line buffering, or continuous log following. Covers StreamReader, StreamWriter, process.stdout iteration, and follow mode."
disable-model-invocation: false
---

# Streaming in Sandboxes

Stream command output in real-time and handle stdin.

## Streaming stdout

Iterate over output as lines arrive (not all at once after completion):

```python
process = sandbox.exec(["python", "-c", "import time; [print(i) for i in range(5)]"])

for line in process.stdout:
    print(line, end="")

result = process.result()
```

## Async Streaming

```python
process = sandbox.exec(["python", "-c", "import time; [print(i) for i in range(5)]"])

async for line in process.stdout:
    print(line, end="")

result = await process
```

## stdin Streaming

Enable stdin with `stdin=True` on `exec()`:

```python
process = sandbox.exec(["cat"], stdin=True)

# Write bytes
process.stdin.write(b"hello ").result()

# Write text line (adds newline)
process.stdin.writeline("world").result()

# Close to signal EOF
process.stdin.close().result()

result = process.result()
print(result.stdout)  # "hello world\n"
```

## Streaming Logs

Stream the sandbox main process logs (PID 1):

```python
# Get last 10 lines
reader = sandbox.stream_logs(tail_lines=10)

for line in reader:
    print(line, end="")

# Follow mode (like tail -f)
reader = sandbox.stream_logs(follow=True)

for line in reader:
    print(line, end="")
    # Don't forget to close when done
    if some_condition:
        reader.close()
```

## Line Buffering

Python buffers stdout when not connected to a TTY. Force unbuffered:

```python
# Option 1: Use -u flag
process = sandbox.exec(["python", "-u", "script.py"])

# Option 2: Set PYTHONUNBUFFERED env
process = sandbox.exec(["sh", "-c", "PYTHONUNBUFFERED=1 python script.py"])
```

## StreamReader Methods

```python
reader = sandbox.stream_logs(follow=True)

# Sync iteration
for line in reader:
    print(line)

# Async iteration
async for line in reader:
    print(line)

# Close the stream
reader.close()
```

## StreamWriter Methods

```python
process = sandbox.exec(["python"], stdin=True)

process.stdin.write(b"print('hello')")     # Write bytes
process.stdin.writeline("exit()")           # Write line with newline
process.stdin.close()                        # Signal EOF - returns OperationRef[None]
```

## Important Notes

- `stream_logs()` only captures stdout/stderr from the main command passed to `Sandbox.run()` — not from `exec()` commands
- Always close `StreamReader` in follow mode to stop the background producer
- stdin streaming requires `stdin=True` on `exec()`

## References

- [Streaming Output Tutorial](https://docs.coreweave.com/products/coreweave-sandbox/client/tutorial/streaming-output)
