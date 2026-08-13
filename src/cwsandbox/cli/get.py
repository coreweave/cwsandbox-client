# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox get - show sandbox details."""

from __future__ import annotations

import json
from typing import Any

import click

from cwsandbox import Sandbox


def _sandbox_details(sandbox: Sandbox) -> dict[str, Any]:
    """Return the CLI detail payload for a sandbox."""
    return {
        "sandbox_id": sandbox.sandbox_id,
        "status": sandbox.status.value if sandbox.status else None,
        "runner_id": sandbox.runner_id,
        "runner_group_id": sandbox.runner_group_id,
        "started_at": sandbox.started_at.isoformat() if sandbox.started_at else None,
        "returncode": sandbox.returncode,
    }


def _display_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


@click.command("get")
@click.argument("sandbox_id")
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format.",
)
def get_sandbox(sandbox_id: str, output_format: str) -> None:
    """Show details for a sandbox.

    SANDBOX_ID is the ID of the sandbox to inspect.
    """
    sandbox = Sandbox.from_id(sandbox_id).result()
    details = _sandbox_details(sandbox)

    if output_format == "json":
        click.echo(json.dumps(details, indent=2))
        return

    started = sandbox.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if sandbox.started_at else None
    rows = [
        ("SANDBOX ID", details["sandbox_id"]),
        ("STATUS", details["status"]),
        ("RUNNER ID", details["runner_id"]),
        ("RUNNER GROUP ID", details["runner_group_id"]),
        ("STARTED AT", started),
        ("RETURNCODE", details["returncode"]),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        click.echo(f"{label:<{width}}  {_display_value(value)}")
