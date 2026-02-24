<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: aviato-client
-->

# Source Code

This directory contains the aviato SDK implementation.

## Module Structure

```
aviato/
├── __init__.py       # Public API exports, get(), wait() utilities
├── __main__.py       # Entry point for `python -m aviato` and console script
├── _sandbox.py       # Sandbox class, SandboxStatus enum (largest file)
├── _session.py       # Session class for multi-sandbox management
├── _function.py      # RemoteFunction for @session.function() decorator
├── _types.py         # OperationRef, ProcessResult, Process, StreamReader
├── _defaults.py      # SandboxDefaults configuration dataclass
├── _auth.py          # Authentication resolution (Aviato, W&B, netrc)
├── _loop_manager.py  # Background asyncio event loop singleton
├── _cleanup.py       # atexit/signal handlers for graceful shutdown
├── exceptions.py     # Exception hierarchy
├── py.typed          # PEP 561 type information marker
└── cli/              # CLI subpackage (only loaded when `aviato` command is invoked)
    ├── __init__.py   # Click group, registers commands
    └── logs.py       # aviato logs command
```

## Naming Conventions

- `_` prefix (e.g., `_sandbox.py`): Internal modules, not intended for direct import
- Public API: Exported via `__init__.py`, documented in root AGENTS.md
- Internal methods: `_async` suffix for async implementations (e.g., `_start_async`)

## Key Patterns

### Sync/Async Hybrid

All public methods return immediately. Blocking happens via:
- `OperationRef.result()` for operations returning data
- `Process.result()` for streaming exec (Process inherits from OperationRef)
- Context manager exit for cleanup

The `_LoopManager` runs async code in a background daemon thread, enabling sync usage without user-managed event loops.

**Thread Safety**: The public API is designed for **single-threaded use**. Users wanting multi-threaded access to the same sandbox must add their own synchronization.

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

### Adding New Configuration Options

When adding configuration options that users pass to `Sandbox.run()`, `Session.sandbox()`, or `@session.function()`:

1. **Add to `SandboxDefaults`** in `_defaults.py` if the option should be shareable across sandboxes. This allows users to set a default value once and have it apply to all sandboxes in a session.

2. **Update `Sandbox.__init__`** to use the default when the explicit parameter is None:
   ```python
   effective_value = param if param is not None else self._defaults.param
   if effective_value is not None:
       self._start_kwargs["param"] = effective_value
   ```

3. **Update documentation**:
   - `AGENTS.md` (root) - SandboxDefaults fields list
   - `docs/guides/sandbox-configuration.md` - usage examples

4. **Add tests** for:
   - Default value in `test_defaults.py`
   - Sandbox uses defaults when param is None in `test_sandbox.py`
   - Explicit param overrides defaults in `test_sandbox.py`

**Design principle**: If a configuration option makes sense to share across multiple sandboxes (e.g., resources, network, environment variables), it belongs in `SandboxDefaults`. Per-sandbox-only options (e.g., mounted_files with sandbox-specific content) do not need to be in defaults.

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
