"""Integration test configuration."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root, but don't override existing env vars
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env", override=False)
