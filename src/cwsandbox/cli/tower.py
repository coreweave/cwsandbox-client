# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox tower — tower management commands."""

from __future__ import annotations

import click

from cwsandbox.cli.tower_create_join_token import create_join_token


@click.group("tower")
def tower() -> None:
    """Tower management commands."""


tower.add_command(create_join_token, "create-join-token")
