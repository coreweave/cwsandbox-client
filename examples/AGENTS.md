# Examples

Runnable example scripts (14 total) demonstrating aviato SDK usage patterns.

## API Patterns

**All aviato operations support sync/async hybrid usage.** Use `.result()` in sync code, or `await` in async code. Sync patterns are simpler for most use cases.

| File | Pattern | Entry Point | Description |
|------|---------|-------------|-------------|
| `quick_start.py` | Sync | `def main()` | Context manager with exec |
| `basic_execution.py` | Sync | `def main()` | Context manager pattern with exec, file ops |
| `streaming_exec.py` | Sync | `def main()` | Real-time stdout iteration |
| `function_decorator.py` | Sync | `def main()` | Remote function execution with `@session.function()` |
| `error_handling.py` | Sync | `def main()` | Exception hierarchy: SandboxExecutionError, TimeoutError, NotFoundError |
| `multiple_sandboxes.py` | Sync | `def main()` | Session-based multi-sandbox management |
| `delete_sandboxes.py` | Sync | `def main()` | Deletion patterns with `Sandbox.delete()` |
| `reconnect_to_sandbox.py` | Sync | `def main()` | Attach via `Sandbox.from_id()` |
| `async_patterns.py` | Async | `async def main()` | Using await with OperationRef and Process |
| `session_adopt_orphans.py` | Sync | `def main()` | Orphan management with `session.list()` |
| `parallel_batch_job.py` | Sync | `def main()` | Parallel batch processing with aviato.wait() |
| `cleanup_by_tag.py` | Sync | `def main()` | Tag-based cleanup |
| `cleanup_old_sandboxes.py` | Sync | `def main()` | Age-based cleanup |
| `swebench/run_evaluation.py` | Sync | `def main()` | SWE-bench evaluation with parallel sandboxes |

The `exec()` method returns a `Process` object. Call `.result()` to block for the final `ProcessResult`. Iterate over `process.stdout` before calling `.result()` if you need real-time streaming output.

For detailed guides, see `docs/guides/`. See [Sync vs Async](../docs/guides/sync-vs-async.md) for when to use each pattern.

## Running Examples

```bash
# Set API key
export AVIATO_API_KEY="your-api-key"

# Run any example
python examples/quick_start.py
python examples/basic_execution.py
```

## Structure

Each example follows this pattern:

1. Docstring explaining what it demonstrates
2. Imports from `aviato` package
3. Self-contained `main()` function
4. Clear output showing results

## Writing New Examples

Guidelines:

- **Single focus**: Each example demonstrates one concept
- **Self-contained**: No dependencies on other examples
- **Prefer hybrid API**: Use sync/async hybrid unless demonstrating async-specific patterns
- **Use streaming default**: Call `.result()` for the final result
- **Clean output**: Print clear labels with results
- **Error handling**: Show realistic error handling where appropriate

### Sync Template (Recommended)

Use this pattern for most examples. Simple and works in Jupyter notebooks.

```python
"""Short description of what this example demonstrates.

Demonstrates:
- Bullet point of key concept
- Another concept shown
"""

from aviato import Sandbox, SandboxDefaults


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "your-example-name"),
    )

    with Sandbox.run(defaults=defaults) as sb:
        result = sb.exec(["echo", "hello"]).result()
        print(f"Output: {result.stdout}")


if __name__ == "__main__":
    main()
```

### Async Template (Optional)

Use this pattern when you prefer async/await syntax or need `asyncio.gather()` for parallel operations.

```python
"""Example using async patterns.

Demonstrates:
- Awaiting OperationRef and Process objects
"""

import asyncio
from aviato import Sandbox


async def main() -> None:
    # All methods return awaitable objects
    sandboxes = await Sandbox.list(tags=["example"])
    for sb in sandboxes:
        await Sandbox.delete(sb.sandbox_id, missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Examples

Examples are not covered by the test suite. To verify:

```bash
# Quick smoke test
python examples/quick_start.py

# Full validation (creates real sandboxes)
for f in examples/*.py; do
    echo "Testing $f..."
    python "$f" || echo "FAILED: $f"
done
```
