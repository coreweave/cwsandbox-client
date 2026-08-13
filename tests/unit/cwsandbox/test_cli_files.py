# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox files CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox.cli import cli
from cwsandbox.exceptions import SandboxFileError, SandboxNotFoundError
from tests.unit.cwsandbox.conftest import make_operation_ref


def _patch_sandbox(mock_sandbox: MagicMock):
    """Patch Sandbox.from_id for files CLI tests."""
    return patch(
        "cwsandbox.cli.files.Sandbox",
        **{"from_id.return_value": make_operation_ref(mock_sandbox)},
    )


class TestFilesCommand:
    """Tests for the cwsandbox files command group."""

    def test_files_registered(self) -> None:
        """Files command group is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["files", "--help"])
        assert result.exit_code == 0
        assert "read" in result.output
        assert "write" in result.output

    def test_files_read_prints_stdout(self) -> None:
        """cwsandbox files read writes file bytes to stdout."""
        mock_sandbox = MagicMock()
        mock_sandbox.read_file.return_value = make_operation_ref(b"hello\n")

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["files", "read", "sb-1", "/tmp/data.txt"])

        assert result.exit_code == 0
        assert result.output == "hello\n"
        mock_sandbox.read_file.assert_called_once_with(
            "/tmp/data.txt",
            timeout_seconds=None,
        )

    def test_files_read_writes_output_file(self, tmp_path: Path) -> None:
        """cwsandbox files read --output writes file bytes to a local file."""
        output_path = tmp_path / "data.bin"
        mock_sandbox = MagicMock()
        mock_sandbox.read_file.return_value = make_operation_ref(b"\x00hello")

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["files", "read", "sb-1", "/tmp/data.bin", "--output", str(output_path)]
            )

        assert result.exit_code == 0
        assert result.output == ""
        assert output_path.read_bytes() == b"\x00hello"

    def test_files_read_with_timeout(self) -> None:
        """cwsandbox files read --timeout passes timeout_seconds."""
        mock_sandbox = MagicMock()
        mock_sandbox.read_file.return_value = make_operation_ref(b"")

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["files", "read", "sb-1", "/tmp/data.txt", "--timeout", "5"]
            )

        assert result.exit_code == 0
        mock_sandbox.read_file.assert_called_once_with(
            "/tmp/data.txt",
            timeout_seconds=5.0,
        )

    def test_files_write_uploads_local_file(self, tmp_path: Path) -> None:
        """cwsandbox files write reads a local file and writes it to the sandbox."""
        local_path = tmp_path / "data.txt"
        local_path.write_bytes(b"hello\n")
        mock_sandbox = MagicMock()
        mock_sandbox.write_file.return_value = make_operation_ref(None)

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["files", "write", "sb-1", "/tmp/data.txt", str(local_path)]
            )

        assert result.exit_code == 0
        assert "Wrote 6 bytes to /tmp/data.txt." in result.output
        mock_sandbox.write_file.assert_called_once_with(
            "/tmp/data.txt",
            b"hello\n",
            timeout_seconds=None,
        )

    def test_files_write_with_options(self, tmp_path: Path) -> None:
        """cwsandbox files write passes --timeout and honors --quiet."""
        local_path = tmp_path / "data.txt"
        local_path.write_bytes(b"hello")
        mock_sandbox = MagicMock()
        mock_sandbox.write_file.return_value = make_operation_ref(None)

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "files",
                    "write",
                    "sb-1",
                    "/tmp/data.txt",
                    str(local_path),
                    "--timeout",
                    "7",
                    "--quiet",
                ],
            )

        assert result.exit_code == 0
        assert result.output == ""
        mock_sandbox.write_file.assert_called_once_with(
            "/tmp/data.txt",
            b"hello",
            timeout_seconds=7.0,
        )

    def test_files_read_sandbox_not_found(self) -> None:
        """cwsandbox files read shows clean errors for SandboxNotFoundError."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("cwsandbox.cli.files.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["files", "read", "bad-id", "/tmp/data.txt"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_files_write_file_error(self, tmp_path: Path) -> None:
        """cwsandbox files write shows clean errors for SandboxFileError."""
        local_path = tmp_path / "data.txt"
        local_path.write_bytes(b"hello")
        mock_sandbox = MagicMock()
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxFileError("write failed", filepath="/tmp/data.txt")
        mock_sandbox.write_file.return_value = mock_op_ref

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["files", "write", "sb-1", "/tmp/data.txt", str(local_path)]
            )

        assert result.exit_code == 1
        assert "write failed" in result.output
