# Environment Variables

This example demonstrates how to use environment variables in sandboxes.

> **Security Note:** Environment variables should **not** be used for sensitive information like API keys, passwords, or other secrets.

## Basic Usage

```python
from aviato import Sandbox

async with Sandbox(
    command="python", args=["app.py"],
    env_vars={"LOG_LEVEL": "info", "REGION": "us-west"},
) as sandbox:
    pass
```

## Session-Level Defaults

```python
from aviato import Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    env_vars={"PROJECT_ID": "my-project", "LOG_LEVEL": "info"},
)

async with Sandbox.session(defaults) as session:
    # Both sandboxes inherit session env vars
    sb1 = session.create(command="python", args=["task1.py"])
    sb2 = session.create(command="python", args=["task2.py"])
```

## Sandbox-Level Override

```python
async with Sandbox.session(defaults) as session:
    # This sandbox adds new vars and overrides LOG_LEVEL
    async with session.create(
        command="python", args=["debug.py"],
        env_vars={"LOG_LEVEL": "debug", "MODEL": "gpt2"},
    ) as sandbox:
        pass
    # Result: {"PROJECT_ID": "my-project", "LOG_LEVEL": "debug", "MODEL": "gpt2"}
```

## With Function Decorator

```python
async with Sandbox.session(defaults) as session:
    @session.function(env_vars={"MODEL_VERSION": "v2.0"})
    def process(task_id: int) -> dict:
        import os
        return {"task": task_id, "version": os.environ["MODEL_VERSION"]}
    
    result = await process(42)
```

## Loading environment variables from .env file

You can use the `python-dotenv` package to load environment variables from a `.env` file:

```python
from dotenv import dotenv_values
from aviato import SandboxDefaults

env_vars = dict(dotenv_values(".env"))
defaults = SandboxDefaults(env_vars=env_vars)
```