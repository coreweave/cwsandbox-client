# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""CWSandbox CLI - terminal interface for CoreWeave sandboxes.

The functions in this package are intended to be called via the CLI,
not from Python code. No backwards compatibility guarantees are made
for Python calling patterns.
"""

from __future__ import annotations

from typing import Any

try:
    import click
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "click":
        raise ImportError(
            "cwsandbox CLI requires the 'cli' extra. Install it with: pip install cwsandbox[cli]",
            name="click",
        ) from e
    raise

from cwsandbox.cli.delete import delete_sandbox
from cwsandbox.cli.exec import exec_command
from cwsandbox.cli.files import files
from cwsandbox.cli.get import get_sandbox
from cwsandbox.cli.list import list_sandboxes
from cwsandbox.cli.logs import logs
from cwsandbox.cli.shell import shell
from cwsandbox.cli.snapshots import snapshots
from cwsandbox.exceptions import CWSandboxError


class _CWSandboxCLI(click.Group):
    """Click group with top-level CWSandboxError handling.

    SDK errors (SandboxNotFoundError, auth failures, etc.) are caught and
    printed as clean "Error: <message>" output instead of raw tracebacks.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except CWSandboxError as exc:
            raise click.ClickException(str(exc)) from None


@click.group(cls=_CWSandboxCLI)
@click.version_option(package_name="cwsandbox")
def cli() -> None:
    """CWSandbox CLI."""


cli.add_command(list_sandboxes, "ls")
cli.add_command(get_sandbox, "get")
cli.add_command(delete_sandbox, "delete")
cli.add_command(exec_command, "exec")
cli.add_command(files, "files")
cli.add_command(logs, "logs")
cli.add_command(shell, "sh")
cli.add_command(snapshots, "snapshots")
