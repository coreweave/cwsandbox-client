# File Operations Guide

This guide covers reading and writing files in sandboxes.

## Basic Operations

File operations return `OperationRef` objects. Use `.result()` to block for completion.

### Writing Files

```python
from aviato import Sandbox

with Sandbox.run() as sandbox:
    # Write bytes to a file
    sandbox.write_file("/app/data.txt", b"Hello, World!").result()

    # Write JSON
    import json
    config = {"key": "value", "count": 42}
    sandbox.write_file(
        "/app/config.json",
        json.dumps(config).encode()
    ).result()
```

### Reading Files

```python
# Read file contents as bytes
content = sandbox.read_file("/app/data.txt").result()
print(content.decode())  # "Hello, World!"

# Read JSON
config_bytes = sandbox.read_file("/app/config.json").result()
config = json.loads(config_bytes.decode())
```

## Parallel Operations

File operations return immediately, enabling natural parallelism.

### Parallel Uploads

```python
from aviato import get

# Start all uploads simultaneously
write_refs = [
    sandbox.write_file("/app/config.json", config_bytes),
    sandbox.write_file("/app/data.csv", data_bytes),
    sandbox.write_file("/app/model.pkl", model_bytes),
]

# Wait for all to complete
get(write_refs)
```

### Parallel Downloads

```python
# Start all downloads simultaneously
read_refs = [
    sandbox.read_file("/app/output.json"),
    sandbox.read_file("/app/metrics.json"),
    sandbox.read_file("/app/logs.txt"),
]

# Get all results
output, metrics, logs = get(read_refs)
```

## Upload-Process-Download Pattern

A common workflow: upload input files, run processing, download results.

```python
from aviato import Sandbox, get

with Sandbox.run() as sandbox:
    # 1. Parallel uploads
    get([
        sandbox.write_file("/app/config.json", config_bytes),
        sandbox.write_file("/app/input.csv", input_bytes),
    ])

    # 2. Sequential processing
    sandbox.exec(["pip", "install", "-r", "requirements.txt"]).result()
    sandbox.exec(["python", "/app/process.py"]).result()

    # 3. Parallel downloads
    output, metrics = get([
        sandbox.read_file("/app/output.json"),
        sandbox.read_file("/app/metrics.json"),
    ])
```

## Error Handling

### File Not Found

```python
from aviato import SandboxFileError

try:
    content = sandbox.read_file("/nonexistent/file.txt").result()
except SandboxFileError as e:
    print(f"File error: {e.filepath}")
```

### Write Errors

```python
try:
    sandbox.write_file("/readonly/path.txt", b"data").result()
except SandboxFileError as e:
    print(f"Cannot write to: {e.filepath}")
```

## Binary Files

File operations work with any binary content:

```python
# Images
with open("image.png", "rb") as f:
    sandbox.write_file("/app/image.png", f.read()).result()

# Pickle files
import pickle
model_bytes = pickle.dumps(my_model)
sandbox.write_file("/app/model.pkl", model_bytes).result()

# Download and unpickle
model_bytes = sandbox.read_file("/app/trained_model.pkl").result()
trained_model = pickle.loads(model_bytes)
```

## Text Encoding

Files are transferred as bytes. Handle encoding explicitly:

```python
# Write text
text = "Hello, Unicode! "
sandbox.write_file("/app/text.txt", text.encode("utf-8")).result()

# Read text
content = sandbox.read_file("/app/text.txt").result()
text = content.decode("utf-8")
```

## Large File Considerations

For very large files:

1. Files are transferred through the API - consider bandwidth
2. Use streaming for large datasets when possible
3. Consider mounting S3/object storage for large data

```python
# For large data, use S3 mount instead
sandbox = Sandbox(
    s3_mount={
        "bucket": "my-data-bucket",
        "mount_path": "/data",
        "read_only": False,
    }
)
```
