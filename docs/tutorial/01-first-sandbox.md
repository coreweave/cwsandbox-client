# 1. Your First Sandbox

## Setup

Clone and install the SDK locally:

```bash
git clone https://github.com/coreweave/aviato-client.git
cd aviato-client
uv sync
```

Set your credentials:

```bash
# Option 1: Aviato API key
export AVIATO_API_KEY="your-api-key"

# Option 2: W&B credentials
export WANDB_API_KEY="your-wandb-key"
export WANDB_ENTITY_NAME="your-entity"
```

## Run Your First Sandbox

```python
from aviato import Sandbox

with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "Hello!"]).result()
    print(result.stdout)
```

**What's happening:**

- `Sandbox.run()` creates a sandbox and returns it inside a context manager
- `exec()` runs a command and returns a `Process` object
- `.result()` waits for the command to complete and returns the output

