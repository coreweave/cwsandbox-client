"""Unit tests for aviato._types module."""

from aviato import ExecResult, Serialization


class TestExecResult:
    """Tests for ExecResult dataclass."""

    def test_exec_result_creation(self) -> None:
        """Test ExecResult can be created with all fields."""
        result = ExecResult(stdout_bytes=b"hello", stderr_bytes=b"", returncode=0)

        assert result.stdout_bytes == b"hello"
        assert result.stderr_bytes == b""
        assert result.returncode == 0
        assert result.command == []

    def test_exec_result_with_command(self) -> None:
        """Test ExecResult stores the command that was executed."""
        result = ExecResult(
            stdout_bytes=b"hello",
            stderr_bytes=b"",
            returncode=0,
            command=["echo", "hello"],
        )

        assert result.command == ["echo", "hello"]

    def test_exec_result_with_binary_data(self) -> None:
        """Test ExecResult handles binary stdout/stderr."""
        binary_data = bytes([0, 1, 2, 255])
        result = ExecResult(stdout_bytes=binary_data, stderr_bytes=b"", returncode=0)

        assert result.stdout_bytes == binary_data

    def test_exec_result_text_properties(self) -> None:
        """Test stdout/stderr are decoded strings (common case)."""
        result = ExecResult(
            stdout_bytes=b"hello world",
            stderr_bytes=b"error occurred",
            returncode=1,
        )

        assert result.stdout == "hello world"
        assert result.stderr == "error occurred"

    def test_exec_result_handles_invalid_utf8(self) -> None:
        """Test text properties handle invalid UTF-8 gracefully."""
        result = ExecResult(
            stdout_bytes=b"hello \xff world",
            stderr_bytes=b"",
            returncode=0,
        )

        # Should replace invalid bytes, not raise
        assert "hello" in result.stdout
        assert "world" in result.stdout


class TestSerialization:
    """Tests for Serialization enum."""

    def test_serialization_values(self) -> None:
        """Test Serialization enum has expected values."""
        assert Serialization.PICKLE.value == "pickle"
        assert Serialization.JSON.value == "json"

    def test_serialization_is_string(self) -> None:
        """Test Serialization inherits from str."""
        assert isinstance(Serialization.PICKLE, str)
        assert isinstance(Serialization.JSON, str)
