"""Tests for aviato._env module (load_dotenv function)."""

import pytest

from aviato import load_dotenv


class TestLoadDotenv:
    def test_load_dotenv(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test loading env vars from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("LOG_LEVEL=info\nREGION=us-west\nDEBUG=true\n")

        result = load_dotenv(str(env_file))
        assert result == {
            "LOG_LEVEL": "info",
            "REGION": "us-west",
            "DEBUG": "true",
        }