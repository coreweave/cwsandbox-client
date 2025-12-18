from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property


class Serialization(str, Enum):
    """Serialization modes for sandbox function execution."""

    PICKLE = "pickle"
    JSON = "json"


@dataclass
class ExecResult:
    """Result from a completed sandbox exec operation.

    Attributes:
        stdout_bytes: Raw stdout bytes from the command
        stderr_bytes: Raw stderr bytes from the command
        returncode: Exit code from the command
        command: The command that was executed (for debugging)

    Properties:
        stdout: Lazily decoded stdout as UTF-8 string
        stderr: Lazily decoded stderr as UTF-8 string
    """

    stdout_bytes: bytes
    stderr_bytes: bytes
    returncode: int
    command: list[str] = field(default_factory=list)

    @cached_property
    def stdout(self) -> str:
        """Decode stdout as UTF-8 (lazy, cached)."""
        return self.stdout_bytes.decode("utf-8", errors="replace")

    @cached_property
    def stderr(self) -> str:
        """Decode stderr as UTF-8 (lazy, cached)."""
        return self.stderr_bytes.decode("utf-8", errors="replace")
