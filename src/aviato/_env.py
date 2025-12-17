"""Environment variable utilities."""

from __future__ import annotations


def load_dotenv(filepath: str = ".env") -> dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        filepath: Path to .env file (default: ".env")

    Returns:
        Dictionary of environment variables from the file

    Raises:
        FileNotFoundError: If the .env file doesn't exist

    Example:
        # Simple usage
        env_vars = load_dotenv(".env")
        defaults = SandboxDefaults(env_vars=env_vars)

        # Merge multiple sources
        env_vars = {
            **load_dotenv(".env"),
            **load_dotenv(".env.local"),
            "OVERRIDE": "value",
        }
        defaults = SandboxDefaults(env_vars=env_vars)
    """
    from dotenv import dotenv_values

    return dict(dotenv_values(filepath))

