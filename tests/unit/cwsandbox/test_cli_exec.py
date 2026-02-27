# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox exec CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox.cli import cli
from cwsandbox.exceptions import SandboxNotFoundError
from tests.unit.cwsandbox.conftest import make_operation_ref, make_process


def _patch_sandbox(process):
    """Patch cwsandbox.cli.exec.Sandbox.from_id to return a sandbox whose exec() returns *process*.

    Returns a context manager that patches the Sandbox class.
    """
    mock_sandbox = MagicMock()
    mock_sandbox.exec.return_value = process
    return patch(
        "cwsandbox.cli.exec.Sandbox",
        **{"from_id.return_value": make_operation_ref(mock_sandbox)},
    )


class TestExecCommand:
    """Tests for the cwsandbox exec CLI command."""

    def test_exec_registered(self) -> None:
        """Exec command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["exec", "--help"])
        assert result.exit_code == 0
        assert "SANDBOX_ID" in result.output

    def test_exec_prints_stdout(self) -> None:
        """cwsandbox exec prints command stdout and exits with returncode."""
        process = make_process(stdout="hello world\n", command=["echo", "hello"])

        with _patch_sandbox(process) as mock_cls:
            mock_sandbox = mock_cls.from_id.return_value.result()
            runner = CliRunner()
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "echo", "hello"])

        assert result.exit_code == 0
        assert "hello world\n" in result.output
        mock_sandbox.exec.assert_called_once_with(
            ["echo", "hello"],
            cwd=None,
            timeout_seconds=None,
        )

    def test_exec_nonzero_returncode(self) -> None:
        """cwsandbox exec exits with the process returncode."""
        process = make_process(returncode=1, command=["false"])

        with _patch_sandbox(process):
            runner = CliRunner()
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "false"])

        assert result.exit_code == 1

    def test_exec_with_cwd(self) -> None:
        """cwsandbox exec --cwd passes cwd to sandbox.exec."""
        process = make_process(command=["ls"])

        with _patch_sandbox(process) as mock_cls:
            mock_sandbox = mock_cls.from_id.return_value.result()
            runner = CliRunner()
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "--cwd", "/app", "ls"])

        assert result.exit_code == 0
        mock_sandbox.exec.assert_called_once_with(
            ["ls"],
            cwd="/app",
            timeout_seconds=None,
        )

    def test_exec_with_timeout(self) -> None:
        """cwsandbox exec --timeout passes timeout_seconds to sandbox.exec."""
        process = make_process(command=["sleep", "10"])

        with _patch_sandbox(process) as mock_cls:
            mock_sandbox = mock_cls.from_id.return_value.result()
            runner = CliRunner()
            result = runner.invoke(
                cli, ["exec", "test-sandbox-id", "--timeout", "30", "sleep", "10"]
            )

        assert result.exit_code == 0
        mock_sandbox.exec.assert_called_once_with(
            ["sleep", "10"],
            cwd=None,
            timeout_seconds=30.0,
        )

    def test_exec_concurrent_stdout_stderr(self) -> None:
        """cwsandbox exec drains stdout and stderr concurrently on separate threads."""
        process = make_process(
            stdout="out line 1\nout line 2\n",
            stderr="err line 1\nerr line 2\n",
            returncode=0,
            command=["some-cmd"],
        )

        with _patch_sandbox(process):
            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "some-cmd"])

        assert result.exit_code == 0
        assert "out line 1" in result.output
        assert "out line 2" in result.output
        assert "err line 1" in result.stderr
        assert "err line 2" in result.stderr

    def test_exec_prints_stderr(self) -> None:
        """cwsandbox exec prints command stderr to stderr."""
        process = make_process(
            stderr="error: something went wrong\n",
            returncode=1,
            command=["bad-cmd"],
        )

        with _patch_sandbox(process):
            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "bad-cmd"])

        assert result.exit_code == 1
        assert "error: something went wrong" in result.stderr

    def test_exec_sandbox_not_found(self) -> None:
        """cwsandbox exec shows clean error for SandboxNotFoundError."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("cwsandbox.cli.exec.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["exec", "bad-id", "echo", "hello"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_exec_keyboard_interrupt(self) -> None:
        """cwsandbox exec exits 130 on KeyboardInterrupt."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.result.side_effect = KeyboardInterrupt

        with _patch_sandbox(mock_process):
            runner = CliRunner()
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "sleep", "100"])

        assert result.exit_code == 130

    def test_exec_broken_pipe(self) -> None:
        """cwsandbox exec exits 0 on BrokenPipeError (piped to head/etc)."""
        mock_process = MagicMock()
        mock_process.stdout.__iter__ = MagicMock(side_effect=BrokenPipeError)

        with _patch_sandbox(mock_process):
            runner = CliRunner()
            result = runner.invoke(cli, ["exec", "test-sandbox-id", "echo", "hello"])

        assert result.exit_code == 0
