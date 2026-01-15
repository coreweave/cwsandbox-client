# 4. Streaming Output

Get output as it happens instead of waiting for completion:

```python
process = sandbox.exec(["bash", "-c", "for i in 1 2 3; do echo $i; sleep 1; done"])

for line in process.stdout:
    print(line, end="")

result = process.result()
```

**Note:** To stream output in real-time, iterate over `process.stdout` before calling `.result()`. If you only need the final output, you can skip iteration and access it via `result.stdout` after calling `.result()`.

For more streaming patterns, see the [Execution Guide](../guides/execution.md).

