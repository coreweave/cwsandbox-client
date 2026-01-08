# Source Code

This directory contains the aviato SDK implementation.

## Module Structure

```
aviato/
├── __init__.py       # Public API exports, get(), wait() utilities
├── _sandbox.py       # Sandbox class, SandboxStatus enum (largest file)
├── _session.py       # Session class for multi-sandbox management
├── _function.py      # RemoteFunction for @session.function() decorator
├── _types.py         # OperationRef, ProcessResult, Process, StreamReader
├── _defaults.py      # SandboxDefaults configuration dataclass
├── _auth.py          # Authentication resolution (Aviato, W&B, netrc)
├── _loop_manager.py  # Background asyncio event loop singleton
├── _cleanup.py       # atexit/signal handlers for graceful shutdown
├── exceptions.py     # Exception hierarchy
└── py.typed          # PEP 561 type information marker
```

## Naming Conventions

- `_` prefix (e.g., `_sandbox.py`): Internal modules, not intended for direct import
- Public API: Exported via `__init__.py`, documented in root CLAUDE.md
- Internal methods: `_async` suffix for async implementations (e.g., `_start_async`)

## Key Patterns

### Sync/Async Hybrid

All public methods return immediately. Blocking happens via:
- `OperationRef.result()` for operations returning data
- `Process.result()` for streaming exec (Process inherits from OperationRef)
- Context manager exit for cleanup

The `_LoopManager` runs async code in a background daemon thread, enabling sync usage without user-managed event loops.

**Thread Safety**: The API is designed for **single-threaded use**. No internal locking is provided - users wanting multi-threaded access must add their own synchronization. This is intentional to keep the implementation simple.

### Adding New Sandbox Methods

1. Implement async version as `_method_async()` in `_sandbox.py`
2. Add sync wrapper that calls `_LoopManager.get().run_async()`
3. Return `OperationRef[T]` for deferred results
4. Update `__init__.py` exports if adding new types
5. Add tests in `tests/unit/aviato/test_sandbox.py`

### Modifying Types

When adding to `_types.py`:
- Use `@dataclass(frozen=True)` for immutable result types
- Prefer simple dataclass fields for result types (see `ProcessResult`)
- Export new types via `__init__.py` and `__all__`

## Testing Changes

```bash
mise run typecheck     # Verify type annotations
mise run test          # Run unit tests
mise run lint          # Check style
```

Run integration tests only for changes affecting backend communication:
```bash
mise run test:e2e      # Requires auth credentials
```

## Notes

- **Exec API**: `exec()` returns a `Process` object. Call `.result()` to block for the final `ProcessResult`. Iterate `process.stdout` before `.result()` for real-time streaming output.
