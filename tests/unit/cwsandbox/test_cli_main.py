# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox.__main__ entry point."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner, Result

from cwsandbox._auth import AuthHeaders, _reset_auth_mode_for_testing, resolve_auth
from cwsandbox.cli import cli


class TestCliMain:
    """Tests for the CLI entry point and group."""

    def test_main_cli_help(self) -> None:
        """--help flag shows usage and exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CWSandbox CLI" in result.output

    def test_main_cli_version(self) -> None:
        """--version flag shows version and exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_main_import_error_missing_cli_module(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() prints install hint and exits 1 when cwsandbox.cli is missing."""
        with patch.dict(sys.modules, {"cwsandbox.cli": None}):
            mod = importlib.import_module("cwsandbox.__main__")
            importlib.reload(mod)
            with pytest.raises(SystemExit, match="1"):
                mod.main()

        captured = capsys.readouterr()
        assert "pip install cwsandbox[cli]" in captured.err

    def test_main_import_error_missing_click(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() prints install hint and exits 1 when click import fails."""
        mod = importlib.import_module("cwsandbox.__main__")
        importlib.reload(mod)

        def _raise_click_missing(*_args: object, **_kwargs: object) -> None:
            raise ImportError("cwsandbox CLI requires the 'cli' extra.", name="click")

        with (
            patch.object(mod, "__import__", side_effect=_raise_click_missing, create=True),
            patch("builtins.__import__", side_effect=_raise_click_missing),
            pytest.raises(SystemExit, match="1"),
        ):
            mod.main()

        captured = capsys.readouterr()
        assert "pip install cwsandbox[cli]" in captured.err

    def test_main_unrelated_import_error_reraises(self) -> None:
        """main() re-raises ImportError when it is not about missing click/CLI."""
        mod = importlib.import_module("cwsandbox.__main__")
        importlib.reload(mod)

        def _raise_unrelated(*_args: object, **_kwargs: object) -> None:
            raise ImportError("No module named 'some_other_lib'", name="some_other_lib")

        with (
            patch.object(mod, "__import__", side_effect=_raise_unrelated, create=True),
            patch("builtins.__import__", side_effect=_raise_unrelated),
            pytest.raises(ImportError, match="some_other_lib"),
        ):
            mod.main()


class TestCliAuthOption:
    """Tests for the global --auth option."""

    @pytest.fixture(autouse=True)
    def reset_auth_mode(self) -> Iterator[None]:
        _reset_auth_mode_for_testing()
        yield
        _reset_auth_mode_for_testing()

    @staticmethod
    def _invoke_ls(
        args: list[str], env: dict[str, str] | None = None
    ) -> tuple[Result, list[AuthHeaders]]:
        """Run ``ls`` and capture the auth the command would send."""
        resolved: list[AuthHeaders] = []

        def _list(**_kwargs: object) -> MagicMock:
            resolved.append(resolve_auth())
            op_ref = MagicMock()
            op_ref.result.return_value = []
            return op_ref

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.side_effect = _list
            result = CliRunner().invoke(cli, [*args, "ls"], env=env)
        return result, resolved

    def test_auth_wandb_flag_uses_wandb_credentials(self) -> None:
        """--auth wandb resolves W&B credentials for subcommands."""
        wandb_headers = AuthHeaders(headers={"x-wandb-api-key": "k"}, strategy="wandb_api_key")
        with patch("cwsandbox._auth._resolve_wandb_auth", return_value=wandb_headers):
            result, resolved = self._invoke_ls(["--auth", "wandb"])

        assert result.exit_code == 0, result.output
        assert resolved == [wandb_headers]

    def test_auth_wandb_env_var_uses_wandb_credentials(self) -> None:
        """CWSANDBOX_AUTH=wandb is equivalent to --auth wandb."""
        wandb_headers = AuthHeaders(headers={"x-wandb-api-key": "k"}, strategy="wandb_api_key")
        with patch("cwsandbox._auth._resolve_wandb_auth", return_value=wandb_headers):
            result, resolved = self._invoke_ls([], env={"CWSANDBOX_AUTH": "wandb"})

        assert result.exit_code == 0, result.output
        assert resolved == [wandb_headers]

    def test_auth_omitted_keeps_default_coreweave_auth(self) -> None:
        """Without --auth, CWSANDBOX_API_KEY is used as before."""
        result, resolved = self._invoke_ls([], env={"CWSANDBOX_API_KEY": "cw-key"})

        assert result.exit_code == 0, result.output
        assert resolved[0].headers == {"Authorization": "Bearer cw-key"}

    def test_auth_coreweave_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--auth coreweave_api_key fails cleanly when CWSANDBOX_API_KEY is unset."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        result, _ = self._invoke_ls(["--auth", "coreweave_api_key"])

        assert result.exit_code == 1
        assert "CWSANDBOX_API_KEY" in result.output

    def test_auth_rejects_unknown_strategy(self) -> None:
        """Unknown strategies are rejected as usage errors."""
        result = CliRunner().invoke(cli, ["--auth", "nope", "ls"])

        assert result.exit_code == 2
        assert "nope" in result.output

    def test_auth_help_mentions_wandb_entity_and_project(self) -> None:
        """--help documents how to choose the W&B entity and project."""
        result = CliRunner().invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "WANDB_ENTITY" in result.output
        assert "WANDB_PROJECT" in result.output
