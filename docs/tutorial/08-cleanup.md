# 8. Cleanup

Context managers handle cleanup automatically:

```python
with Sandbox.run() as sandbox:
    sandbox.exec(["echo", "hello"]).result()
# Stopped automatically
```

For sandboxes created without a context manager, call `.stop()` explicitly:

```python
sandbox = Sandbox.run()
sandbox.exec(["echo", "hello"]).result()
sandbox.stop().result()
```

The SDK also registers global cleanup handlers for `atexit` and signals (Ctrl+C, SIGTERM), so sandboxes are stopped even on unexpected exits.

For batch cleanup, tagging strategies, and orphan recovery, see the [Cleanup Patterns Guide](../guides/cleanup-patterns.md).
