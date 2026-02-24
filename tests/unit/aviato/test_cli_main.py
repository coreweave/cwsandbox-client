# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Tests for aviato.__main__ entry point."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from aviato.cli import cli


class TestCliMain:
    """Tests for the CLI entry point and group."""

    def test_main_cli_help(self) -> None:
        """--help flag shows usage and exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Aviato sandbox CLI" in result.output

    def test_main_cli_version(self) -> None:
        """--version flag shows version and exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_main_import_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() prints install hint and exits 1 when click is missing."""
        with patch.dict(sys.modules, {"aviato.cli": None}):
            mod = importlib.import_module("aviato.__main__")
            importlib.reload(mod)
            with pytest.raises(SystemExit, match="1"):
                mod.main()

        captured = capsys.readouterr()
        assert "pip install aviato[cli]" in captured.err

    def test_main_unrelated_import_error_reraises(self) -> None:
        """main() re-raises ImportError when it is not about missing click/CLI."""
        mod = importlib.import_module("aviato.__main__")
        importlib.reload(mod)

        def _raise_unrelated(*_args: object, **_kwargs: object) -> None:
            raise ImportError("No module named 'some_other_lib'", name="some_other_lib")

        with (
            patch.object(mod, "__import__", side_effect=_raise_unrelated, create=True),
            patch("builtins.__import__", side_effect=_raise_unrelated),
            pytest.raises(ImportError, match="some_other_lib"),
        ):
            mod.main()
