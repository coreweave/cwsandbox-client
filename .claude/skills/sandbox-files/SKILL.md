---
name: sandbox-files
description: "Use when reading or writing files in CoreWeave Sandbox, transferring files between host and sandbox, or working with mounted files. Covers read_file(), write_file(), mounted_files, and binary vs text content."
disable-model-invocation: false
---

# File Operations in Sandboxes

Read and write files in sandbox environments.

## read_file()

Returns `OperationRef[bytes]` — call `.result()` to get content:

```python
data = sandbox.read_file("/path/to/file.txt").result()
print(data.decode())  # Convert bytes to string

# Or work with binary directly
data = sandbox.read_file("/path/to/image.png").result()
```

## write_file()

Write bytes or string content:

```python
# Write string (encoded to bytes automatically)
sandbox.write_file("/path/to/file.txt", "hello world").result()

# Write bytes
sandbox.write_file("/path/to/data.bin", b"\x00\x01\x02").result()
```

## Binary vs Text

```python
# Text files - decode after reading
text = sandbox.read_file("/path/to/file.txt").result().decode("utf-8")

# Binary files - work with raw bytes
png_data = sandbox.read_file("/path/to/image.png").result()

# Write binary
sandbox.write_file("/path/to/output.bin", b"\x00\x01\x02").result()
```

## Async Patterns

```python
# Async: await the OperationRef
data = await sandbox.read_file("/path/to/file.txt")

# Batch read multiple files
files = ["/data/1.txt", "/data/2.txt", "/data/3.txt"]
refs = [sandbox.read_file(f) for f in files]
data_list = await cwsandbox.results(refs)
```

## Directory Operations

Sandbox has no built-in directory listing. Use `exec()` for that:

```python
result = sandbox.exec(["ls", "-la", "/path"]).result()
print(result.stdout)

result = sandbox.exec(["find", "/path", "-name", "*.py"]).result()
```

## Path Handling

Paths are absolute within the sandbox filesystem:

```python
# Read from sandbox temp dir
data = sandbox.read_file("/tmp/data.txt").result()

# Write to sandbox working dir
sandbox.write_file("/tmp/results.csv", "a,b,c\n1,2,3").result()
```

## mounted_files

Pre-populate read-only files at sandbox startup:

```python
with Sandbox.run(
    mounted_files=[
        {"path": "/app/config.json", "content": '{"debug": true}'},
        {"path": "/app/script.py", "content": "print('hello')"},
    ]
) as sandbox:
    # Files already exist when sandbox starts
    result = sandbox.exec(["python", "/app/script.py"]).result()
```

## Error Handling

```python
from cwsandbox import SandboxFileError

try:
    data = sandbox.read_file("/nonexistent/file.txt").result()
except SandboxFileError as e:
    print(f"File error: {e.filepath}")
```

## Use Cases

| Task | Method |
|------|--------|
| Read config file | `read_file("/app/config.json").result()` |
| Write output data | `write_file("/tmp/results.csv", csv_data).result()` |
| Load model weights | `read_file("/models/model.pt").result()` |
| Save checkpoint | `write_file("/checkpoints/step-100.pt", state).result()` |
| Read log file | `read_file("/var/log/app.log").result()` |

## References

- [File Operations Tutorial](https://docs.coreweave.com/products/coreweave-sandbox/client/tutorial/file-operations)
