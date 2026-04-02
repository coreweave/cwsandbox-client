# 1. Your First Sandbox

## Setup

Clone and install the SDK locally:

```bash
git clone https://github.com/coreweave/cwsandbox-client.git
cd cwsandbox-client
uv sync
```

Set your credentials:

```bash
export CWSANDBOX_API_KEY="your-api-key"
```

If you are using a provider integration such as `wandb.sandbox`, import that
integration instead of bare `cwsandbox` so it can install its own auth mode for
the current process.

## Run Your First Sandbox

```python
from cwsandbox import Sandbox

with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "Hello!"]).result()
    print(result.stdout)
```

**What's happening:**

- `Sandbox.run()` creates a sandbox and returns it inside a context manager
- `exec()` runs a command and returns a `Process` object
- `.result()` waits for the command to complete and returns the output
