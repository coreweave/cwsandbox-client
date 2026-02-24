# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Tests for aviato logs CLI command."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aviato.cli import cli
from tests.unit.aviato.conftest import make_operation_ref


@contextmanager
def _patch_sandbox(mock_sandbox: MagicMock) -> Iterator[MagicMock]:
    """Patch Sandbox.from_id to return the given mock sandbox."""
    op_ref = make_operation_ref(mock_sandbox)
    with patch("aviato.cli.logs.Sandbox") as cls:
        cls.from_id.return_value = op_ref
        yield cls


class TestLogsCommand:
    """Tests for the aviato logs CLI command."""

    def test_logs_streams_output(self) -> None:
        """aviato logs prints log lines to stdout."""
        mock_sandbox = MagicMock()
        mock_sandbox.stream_logs.return_value = iter(["line one\n", "line two\n"])

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "test-sandbox-id"])

        assert result.exit_code == 0
        assert "line one\n" in result.output
        assert "line two\n" in result.output
        mock_sandbox.stream_logs.assert_called_once_with(
            follow=False,
            tail_lines=None,
            since_time=None,
            timestamps=False,
        )

    def test_logs_follow_flag(self) -> None:
        """aviato logs --follow passes follow=True to stream_logs."""
        mock_sandbox = MagicMock()
        mock_sandbox.stream_logs.return_value = iter([])

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "test-sandbox-id", "--follow"])

        assert result.exit_code == 0
        mock_sandbox.stream_logs.assert_called_once_with(
            follow=True,
            tail_lines=None,
            since_time=None,
            timestamps=False,
        )

    def test_logs_with_options(self) -> None:
        """aviato logs passes --tail and --timestamps options correctly."""
        mock_sandbox = MagicMock()
        mock_sandbox.stream_logs.return_value = iter([])

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "test-sandbox-id", "--tail", "50", "--timestamps"])

        assert result.exit_code == 0
        mock_sandbox.stream_logs.assert_called_once_with(
            follow=False,
            tail_lines=50,
            since_time=None,
            timestamps=True,
        )

    def test_logs_since_option(self) -> None:
        """aviato logs --since passes UTC-aware datetime to stream_logs."""
        mock_sandbox = MagicMock()
        mock_sandbox.stream_logs.return_value = iter([])

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["logs", "test-sandbox-id", "--since", "2026-01-15 10:30:00"]
            )

        assert result.exit_code == 0
        call_kwargs = mock_sandbox.stream_logs.call_args[1]
        assert call_kwargs["since_time"].tzinfo is not None  # UTC-aware
        assert call_kwargs["since_time"].year == 2026
        assert call_kwargs["since_time"].hour == 10

    def test_logs_aviato_error_during_iteration(self) -> None:
        """aviato logs shows error when stream_logs raises during iteration."""
        from aviato.exceptions import SandboxError

        def _exploding_reader():
            yield "line one\n"
            raise SandboxError("stream broke")

        mock_sandbox = MagicMock()
        mock_sandbox.stream_logs.return_value = _exploding_reader()

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "test-sandbox-id"])

        assert result.exit_code != 0
        assert "stream broke" in result.output

    def test_logs_sandbox_not_found(self) -> None:
        """aviato logs shows error when sandbox is not found."""
        from aviato.exceptions import SandboxNotFoundError

        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("aviato.cli.logs.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "bad-id"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_logs_keyboard_interrupt(self) -> None:
        """aviato logs exits cleanly on KeyboardInterrupt (Ctrl+C during --follow)."""
        mock_sandbox = MagicMock()

        def _interrupted_reader():
            yield "line one\n"
            raise KeyboardInterrupt

        mock_sandbox.stream_logs.return_value = _interrupted_reader()

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "test-sandbox-id", "--follow"])

        assert result.exit_code == 0

    def test_logs_broken_pipe(self) -> None:
        """aviato logs exits cleanly on BrokenPipeError (piped to head/etc)."""
        mock_sandbox = MagicMock()

        def _piped_reader():
            yield "line one\n"
            raise BrokenPipeError

        mock_sandbox.stream_logs.return_value = _piped_reader()

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["logs", "test-sandbox-id"])

        assert result.exit_code == 0
