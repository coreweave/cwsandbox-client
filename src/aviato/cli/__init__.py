# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Aviato CLI — terminal interface for Aviato sandboxes.

The functions in this package are intended to be called via the CLI,
not from Python code. No backwards compatibility guarantees are made
for Python calling patterns.
"""

from __future__ import annotations

try:
    import click
except ImportError as e:
    raise ImportError(
        "aviato CLI requires the 'cli' extra. Install it with:  pip install aviato[cli]"
    ) from e

from aviato.cli.exec import exec_command
from aviato.cli.list import list_sandboxes


@click.group()
@click.version_option(package_name="aviato")
def cli() -> None:
    """Aviato sandbox CLI."""


cli.add_command(exec_command, "exec")
cli.add_command(list_sandboxes)
