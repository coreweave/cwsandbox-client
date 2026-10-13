# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Transport selection for every CLI data-plane command."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cwsandbox.cli import cli
from cwsandbox.exceptions import CWSandboxError


@pytest.mark.parametrize("command", ["exec", "logs", "sh", "files read", "files write"])
@pytest.mark.parametrize("mode", [None, "auto", "direct", "GaTeWaY", "invalid"])
def test_data_plane_mode_reaches_sandbox(command: str, mode: str | None, tmp_path: Path) -> None:
    local_file = tmp_path / "input.txt"
    local_file.write_text("upload")
    args = command.split()
    if mode is not None:
        args += ["--data-plane-mode", mode]
    args += ["sandbox-1"]
    if command == "exec":
        args += ["--", "echo", "hello"]
    elif command == "files read":
        args += ["/tmp/input.txt"]
    elif command == "files write":
        args += ["/tmp/input.txt", str(local_file)]

    with (
        patch("cwsandbox.Sandbox.from_id", side_effect=CWSandboxError("lookup reached")) as lookup,
        patch("cwsandbox.cli.shell.platform.system", return_value="Linux"),
        patch("cwsandbox.cli.shell.sys") as shell_sys,
    ):
        shell_sys.stdin.isatty.return_value = True
        shell_sys.stdout.isatty.return_value = True
        result = CliRunner().invoke(cli, args)

    if mode == "invalid":
        assert result.exit_code == 2
        assert "Invalid value for '--data-plane-mode'" in result.output
        lookup.assert_not_called()
    else:
        assert "lookup reached" in result.output
        lookup.assert_called_once_with("sandbox-1", data_plane_mode=(mode or "auto").lower())
