# 5. Reading & Writing Files

```python
# Write (content must be bytes)
sandbox.write_file("/tmp/hello.txt", b"Hello!").result()

# Read (returns bytes)
content = sandbox.read_file("/tmp/hello.txt").result()
print(content.decode())
```

Like `exec()`, file operations return an `OperationRef` immediately without waiting for completion. Call `.result()` when you need the data. This lets you start multiple operations and wait for them together:

```python
refs = [sandbox.write_file(f"/tmp/{i}.txt", b"data") for i in range(5)]
for ref in refs:
    ref.result()
```

For parallel upload/download patterns and S3 mounts, see the [File Operations Guide](../guides/file-operations.md).

