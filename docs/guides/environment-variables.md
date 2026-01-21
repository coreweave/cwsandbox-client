# Environment Variables

This guide covers how to use environment variables in sandboxes.

> **Security Note:** Environment variables should **not** be used for sensitive information like API keys, passwords, or other secrets.

## Basic Usage

Set environment variables when creating a sandbox:

```python
from aviato import Sandbox

with Sandbox.run(
    environment_variables={"LOG_LEVEL": "info"},
) as sandbox:
    result = sandbox.exec([
        "python",
        "-c",
        "import os; print(os.environ.get('LOG_LEVEL'));",
    ]).result()
    print(result.stdout.strip())  # "info"
```

## Session-Level Defaults

Use sessions to share environment variables across multiple sandboxes:

```python
from aviato import SandboxDefaults, Session

defaults = SandboxDefaults(
    environment_variables={
        "PROJECT_ID": "my-project",
        "LOG_LEVEL": "info",
    },
)

with Session(defaults) as session:
    with session.sandbox() as sb1:
        result = sb1.exec([
            "python",
            "-c",
            "import os; print(os.environ.get('LOG_LEVEL'));",
        ]).result()
        print(result.stdout.strip())  # "info"

    # Override LOG_LEVEL and add new variable
    with session.sandbox(
        environment_variables={
            "LOG_LEVEL": "debug",  # Override session default
            "MODEL_NAME": "gpt-4",  # Add new variable
        }
    ) as sb2:
        result = sb2.exec([
            "python",
            "-c",
            "import os; "
            "print(os.environ.get('PROJECT_ID')); "
            "print(os.environ.get('LOG_LEVEL')); "
            "print(os.environ.get('MODEL_NAME'));",
        ]).result()
        lines = result.stdout.strip().split("\n")
        print(lines)  # ["my-project", "debug", "gpt-4"]
```

## Remote Functions

Environment variables work with remote functions:

```python
with Session(defaults) as session:
    @session.function(environment_variables={"MODEL_VERSION": "v2.0"})
    def process(task_id: int) -> dict:
        import os
        return {
            "task": task_id,
            "project": os.environ.get("PROJECT_ID"),  # From session defaults
            "version": os.environ.get("MODEL_VERSION"),  # From function decorator
        }
    
    result = process.remote(42).result()
    print(result)  # {"task": 42, "project": "my-project", "version": "v2.0"}
```

> **⚠️ Caution:** Environment variables are passed by reference. Mutations will be reflected in subsequent function calls:

```python
env_vars = {"MODEL_VERSION": "v2.0"}

@session.function(environment_variables=env_vars)
def process(task_id: int) -> dict:
    import os
    return {"version": os.environ.get("MODEL_VERSION")}

result = process.remote(42).result()  # version: "v2.0"

env_vars["MODEL_VERSION"] = "v3.0"  # Mutate the dictionary

result = process.remote(42).result()  # version: "v3.0" (changed)
```
