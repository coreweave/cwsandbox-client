# How-To Guides

Task-oriented guides for common operations. Each guide answers "How do I...?"

## Files

| Guide | Task |
|-------|------|
| `execution.md` | Run commands with `exec()` - buffered and streaming modes |
| `file-operations.md` | Read and write files in sandboxes |
| `sessions.md` | Manage multiple sandboxes with shared configuration |
| `remote-functions.md` | Execute Python functions remotely with `@session.function()` |
| `cleanup-patterns.md` | Resource management and graceful shutdown |
| `sync-vs-async.md` | When to use sync vs async patterns, calling async from sync |
| `sandbox-config.md` | Configure resources, mounted files, ports, timeouts |
| `troubleshooting.md` | Debug common issues - sandbox failures, auth errors, timeouts |
| `environment-variables.md` | Use environment variables in sandboxes |
| `swebench.md` | Run SWE-bench evaluations with parallel Aviato sandboxes |

## Key Patterns

All guides follow these conventions:

- **Process for exec**: `exec()` returns a `Process` (inherits from OperationRef) - call `.result()` to block for the result
- **OperationRef for other ops**: `read_file()`, `write_file()`, `stop()` return `OperationRef` - call `.result()` to block
- **Context managers**: Use `with` statements for automatic cleanup

## Writing New Guides

1. Focus on a single task ("How do I X?")
2. Start with the simplest working example
3. Add variations for common needs
4. Include error handling where relevant
5. Link to reference docs for parameter details
6. Use `.result()` consistently for both Process and OperationRef

## Relationship to Other Docs

- **Examples** (`examples/`): Runnable Python scripts
