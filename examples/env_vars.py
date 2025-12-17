"""Environment variables example using sessions.

This example demonstrates:
- Session-level env vars (shared across sandboxes)
- Sandbox-level env vars (task-specific overrides)
- Loading env vars from a .env file
- Using env vars with @session.function() decorator
"""

import asyncio
import os

from aviato import Sandbox, SandboxDefaults, load_dotenv


async def main() -> None:
    if not os.environ.get("AVIATO_API_KEY"):
        raise RuntimeError(
            "Missing AVIATO_API_KEY. Set it in your environment before running this example."
        )

    # Session-level env vars (shared across all sandboxes)
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "env-vars"),
        env_vars={
            "PROJECT_ID": "ml-project-123",
            "LOG_LEVEL": "info",
            "ENVIRONMENT": "production",
        },
    )

    async with Sandbox.session(defaults) as session:
        # Sandbox 1: Inherits session env vars
        async with session.create(
            command="python",
            args=["-c", "import os; print(os.environ.get('PROJECT_ID'))"],
        ) as sb1:
            result = await sb1.exec([
                "python",
                "-c",
                "import os; print(f'Project: {os.environ.get(\"PROJECT_ID\")}')",
            ])
            print(f"Sandbox 1 (session defaults): {result.stdout.strip()}")

        # Sandbox 2: Adds task-specific env vars
        async with session.create(
            command="python",
            args=["-c", "print('Training model...')"],
            env_vars={
                "MODEL_NAME": "resnet50",
                "BATCH_SIZE": "32",
                "EPOCHS": "100",
            },
        ) as sb2:
            result = await sb2.exec([
                "python",
                "-c",
                "import os; print(f'Model: {os.environ.get(\"MODEL_NAME\")}, "
                "Project: {os.environ.get(\"PROJECT_ID\")}')",
            ])
            print(f"Sandbox 2 (with overrides): {result.stdout.strip()}")

        # Sandbox 3: Override session env vars
        async with session.create(
            command="python",
            args=["-c", "import os; print(os.environ.get('LOG_LEVEL'))"],
            env_vars={"LOG_LEVEL": "debug"},  # Override session default
        ) as sb3:
            result = await sb3.exec([
                "python",
                "-c",
                "import os; print(f'Log level: {os.environ.get(\"LOG_LEVEL\")}')",
            ])
            print(f"Sandbox 3 (override session): {result.stdout.strip()}")

        # Use with @session.function() decorator
        @session.function(env_vars={"MODEL_VERSION": "v2.0"})
        def process_request(task_id: int) -> dict:
            import os

            return {
                "task_id": task_id,
                "project": os.environ.get("PROJECT_ID"),
                "version": os.environ.get("MODEL_VERSION"),
            }

        result = await process_request(42)
        print(f"Function result: {result}")

    # Load env vars from .env file
    print("\nLoading from .env file:")
    # Create example .env file
    with open(".env.example", "w") as f:
        f.write("DB_HOST=localhost\nDB_PORT=5432\n")

    env_vars = load_dotenv(".env.example")
    print(f"Loaded: {env_vars}")

    # Use with SandboxDefaults
    defaults_from_file = SandboxDefaults(
        env_vars=env_vars,
        container_image="python:3.11",
    )
    print(f"Defaults env_vars: {defaults_from_file.env_vars}")

    # Clean up
    os.remove(".env.example")


if __name__ == "__main__":
    asyncio.run(main())

