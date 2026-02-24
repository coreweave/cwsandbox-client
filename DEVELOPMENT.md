# Development

Development setup and workflows for aviato-client contributors.

## Prerequisites

- Python 3.10-3.13
- [uv](https://github.com/astral-sh/uv) - Python package manager
- [buf](https://buf.build/docs/installation/) - Protocol buffer tooling (for dependency authentication)
- [mise](https://mise.jdx.dev/) - Optional task runner (convenience wrapper for `uv` commands)

---

## Setup

```bash
git clone https://github.com/coreweave/aviato-client.git
cd aviato-client

# Install buf (if not already installed)
brew install bufbuild/buf/buf # macOS (or see https://buf.build/docs/installation/)

# Authenticate with buf.build (required for protobuf dependencies)
buf registry login            # Opens browser for authentication

# Install dependencies
uv sync --extra dev           # Install package + dev dependencies
pre-commit install            # Install git hooks
```

**Optional:** Install `mise` for task shortcuts:
```bash
brew install mise             # macOS (or see https://mise.jdx.dev)
mise trust                    # Trust project config
```

---

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit to format and lint your code ([.pre-commit-config.yaml](.pre-commit-config.yaml)). This ensures code quality and consistency.

The `pre-commit install` command in the setup above configures these hooks. You can also run them manually:

```bash
pre-commit run --all-files     # Or: mise run precommit
```

---

## Integration Test Authentication

Integration tests require authentication credentials. Copy and configure:

```bash
cp .env.example .env
# Edit .env with your credentials (see .env.example for options)
```

Authentication options:
- **Aviato**: `AVIATO_API_KEY` (takes priority)
- **Weights & Biases**: `WANDB_API_KEY` + `WANDB_ENTITY_NAME` (or `~/.netrc`)

---

## Workflow

### Common Commands

**Using `uv run`** (recommended, no venv activation needed):
```bash
uv run ruff format .                      # Format code
uv run ruff check --fix .                 # Lint with auto-fix
uv run mypy src/                          # Type check
uv run pytest                             # Unit tests
uv run pytest tests/integration/ -v      # Integration tests
uv run pytest --cov=src                   # Coverage report
pre-commit run --all-files                # Run pre-commit hooks
```

**After activating venv** (`source .venv/bin/activate`):
```bash
ruff format .                             # Format code
ruff check --fix .                        # Lint with auto-fix
mypy src/                                 # Type check
pytest                                    # Unit tests
```

### Optional: mise shortcuts

If you installed `mise`, you can use shorter commands:

| Task | mise shortcut | Equivalent uv command |
|------|---------------|----------------------|
| Format code | `mise run format` | `uv run ruff format .` |
| Lint (auto-fix) | `mise run lint` | `uv run ruff check --fix .` |
| Type check | `mise run typecheck` | `uv run mypy src/` |
| Unit tests | `mise run test` | `uv run pytest` |
| Integration tests | `mise run test:e2e` | `uv run pytest tests/integration/ -v` |
| Coverage | `mise run test:cov` | `uv run pytest --cov=src` |
| **All quality checks** | `mise run check` | Runs format:check, lint:check, typecheck, test in parallel |

See [mise.toml](mise.toml) for all available tasks.

---

## Testing

### Unit Tests

Fast, mock-based tests with no external dependencies (default pytest path: `tests/unit/`):

```bash
uv run pytest                                    # All unit tests
uv run pytest tests/unit/aviato/test_sandbox.py  # Single file
uv run pytest -k "test_create"                   # By name pattern
```

Or with `mise`: `mise run test`

### Integration Tests

Real sandbox operations requiring authentication. **Authentication required**: See [.env.example](.env.example) for setup.

These tests provision real sandboxes and will take time to complete. Use `-n auto` for parallel execution.

```bash
uv run pytest tests/integration/ -v             # Sequential
uv run pytest tests/integration/ -n auto -v     # Parallel
```

Or with `mise`: `mise run test:e2e` or `mise run test:e2e:parallel`

### Coverage

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Or with `mise`: `mise run test:cov`

---

## CLI

The `aviato` CLI requires the `cli` extra. After `uv sync --extra cli`, the `aviato` command is available:

```bash
aviato --help                              # Show available commands
aviato --version                           # Show SDK version
aviato list                                # List sandboxes
aviato list --status running --tag dev     # Filter sandboxes
```

You can also invoke it as a Python module:

```bash
uv run python -m aviato --help
```

---

## Troubleshooting

### buf.build Authentication

If you encounter errors about buf.build authentication when installing:

```
hint: An index URL (https://buf.build/gen/python) could not be queried
due to a lack of valid authentication credentials (401 Unauthorized).
```

Run the buf login command:

```bash
buf registry login            # Opens browser for authentication
uv sync --extra dev           # Retry installation
```

The token is saved to `~/.netrc` and will be used automatically for future installs.

### Import Errors

If `import aviato` fails after installation:
1. Ensure you've activated the virtual environment: `source .venv/bin/activate`
2. Check the package is installed: `pip list | grep aviato`
3. Reinstall: `uv sync --extra dev`

### Integration Tests Hanging

Integration tests provision real sandboxes and may take time to complete depending on backend availability and network conditions. If tests appear stuck:
1. Check your auth credentials in `.env`
2. Verify network connectivity to the Aviato backend
3. Try running a single test with `-v` flag for detailed output

### Pre-commit Hook Failures

If pre-commit hooks fail:
1. Run the specific check manually to see details: `ruff check .` or `mypy src/`
2. Auto-fix formatting issues: `ruff format .`
3. Check pre-commit installation: `pre-commit --version`