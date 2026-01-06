"""Environment variables example using sessions.

This example demonstrates:
- Basic usage of environment variables in sandboxes
- Using environment variables with function decorators
- Loading environment variables from a .env file
"""

import asyncio
import tempfile

from aviato import Sandbox, SandboxDefaults


async def basic_usage() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "env-vars"),
        environment_variables={
            "PROJECT_ID": "ml-project-123",
            "LOG_LEVEL": "info",
            "ENVIRONMENT": "production",
        },
    )

    async with Sandbox.session(defaults) as session:
        # Sandbox 1: Inherits session env vars
        async with session.create(
            command="sleep",
            args=["infinity"],
        ) as sb1:
            result = await sb1.exec([
                "python",
                "-c",
                "import os; "
                "print(f'PROJECT_ID={os.environ.get(\"PROJECT_ID\")}'); "
                "print(f'LOG_LEVEL={os.environ.get(\"LOG_LEVEL\")}'); "
                "print(f'ENVIRONMENT={os.environ.get(\"ENVIRONMENT\")}');",
            ])
             # Prints: PROJECT_ID=ml-project-123 LOG_LEVEL=info ENVIRONMENT=production
            print("Basic usage example 1 - Sandbox inherits session defaults:")
            print(result.stdout.strip())
            print()

        # Sandbox 2: Adds task-specific env vars
        async with session.create(
            command="sleep",
            args=["infinity"],
            environment_variables={
                "MODEL_NAME": "resnet50",
                "BATCH_SIZE": "32",
                "EPOCHS": "100",
            },
        ) as sb2:
            result = await sb2.exec([
                "python",
                "-c",
                "import os; "
                "print(f'PROJECT_ID={os.environ.get(\"PROJECT_ID\")}'); "
                "print(f'LOG_LEVEL={os.environ.get(\"LOG_LEVEL\")}'); "
                "print(f'ENVIRONMENT={os.environ.get(\"ENVIRONMENT\")}'); "
                "print(f'MODEL_NAME={os.environ.get(\"MODEL_NAME\")}'); "
                "print(f'BATCH_SIZE={os.environ.get(\"BATCH_SIZE\")}'); "
                "print(f'EPOCHS={os.environ.get(\"EPOCHS\")}');",
            ])
            # Prints:
            # PROJECT_ID=ml-project-123 LOG_LEVEL=info ENVIRONMENT=production
            # MODEL_NAME=resnet50 BATCH_SIZE=32 EPOCHS=100
            print("Basic usage example 2 - Sandbox adds task-specific environment variables:")
            print(result.stdout.strip())
            print()

        # Sandbox 3: Override session env vars
        async with session.create(
            command="sleep",
            args=["infinity"],
            environment_variables={"LOG_LEVEL": "debug"},  # Override session default
        ) as sb3:
            result = await sb3.exec([
                "python",
                "-c",
                "import os; "
                "print(f'PROJECT_ID={os.environ.get(\"PROJECT_ID\")}'); "
                "print(f'LOG_LEVEL={os.environ.get(\"LOG_LEVEL\")}'); "
                "print(f'ENVIRONMENT={os.environ.get(\"ENVIRONMENT\")}');",
            ])
            # Prints: PROJECT_ID=ml-project-123 LOG_LEVEL=debug ENVIRONMENT=production
            print("Basic usage example 3 - Sandbox overrides session default LOG_LEVEL:")
            print(result.stdout.strip())
            print()

async def environment_variables_with_functions() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "env-vars-functions"),
        environment_variables={
            "PROJECT_ID": "ml-project-123",
            "LOG_LEVEL": "info",
        },
    )

    async with Sandbox.session(defaults) as session:
        additional_environment_variables = {
            "MODEL_VERSION": "v2.0",
            "LOG_LEVEL": "debug",
        }
        @session.function(environment_variables=additional_environment_variables)
        def process_request(task_id: int) -> dict:
            import os

            return {
                "task_id": task_id,
                "project": os.environ.get("PROJECT_ID"),  # From session defaults
                "version": os.environ.get("MODEL_VERSION"),  # From function decorator
                "log_level": os.environ.get("LOG_LEVEL"),  # Overridden by function decorator
            }

        # Example 1:
        # * PROJECT_ID is inherited from the session default
        # * LOG_LEVEL is overridden with "debug"
        # * MODEL_VERSION is added as a new env var
        result = await process_request(42)

        # Prints:
        # {"task_id": 42, "project": "ml-project-123", "version": "v2.0", "log_level": "debug"}
        print("Environment variables with functions example 1:")
        print(result)
        print()


        additional_environment_variables["MODEL_VERSION"] = "v3.0"

        # Example 2: Note that environment variables are passed by reference
        # so mutations will be reflected in subsequent function calls.
        result = await process_request(42)

        # Prints:
        # {"task_id": 42, "project": "ml-project-123", "version": "v3.0", "log_level": "debug"}
        print("Environment variables with functions example 2:")
        print(result)
        print()

async def loading_environment_variables_from_dotenv() -> None:
    """
    Here we demonstrate loading environment variables from a .env file using dotenv.

    Note that this requires installing the python-dotenv package:
    ```
    pip install python-dotenv
    ```
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as env_file:
        env_file.write("PROJECT_ID=ml-project-123\n")
        env_file.write("LOG_LEVEL=info\n")
        env_file.flush()

        from dotenv import dotenv_values

        env_vars = dict(dotenv_values(env_file.name))
        defaults = SandboxDefaults(environment_variables=env_vars)

        async with Sandbox.session(defaults) as session:
            async with session.create(command="sleep", args=["infinity"]) as sandbox:
                result = await sandbox.exec([
                    "python",
                    "-c",
                    "import os; "
                    "print(f'PROJECT_ID={os.environ.get(\"PROJECT_ID\")}'); "
                    "print(f'LOG_LEVEL={os.environ.get(\"LOG_LEVEL\")}');",
                ])
                print("Loading environment variables from .env file example:")
                print(result.stdout.strip())
                print()

async def main() -> None:
    await basic_usage()
    await environment_variables_with_functions()
    await loading_environment_variables_from_dotenv()

if __name__ == "__main__":
    asyncio.run(main())

