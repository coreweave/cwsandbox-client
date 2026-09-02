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
    details: dict[str, Any] = {
        "sandbox_id": sandbox.sandbox_id,
        "status": sandbox.status.value if sandbox.status else None,
        "runner_id": sandbox.runner_id,
        "runner_group_id": sandbox.runner_group_id,
        "started_at": sandbox.started_at.isoformat() if sandbox.started_at else None,
        "returncode": sandbox.returncode,
    }
    containers = getattr(sandbox, "containers", ())
    if isinstance(containers, (list, tuple)) and containers:
        details["containers"] = [
            {
                "name": row.name,
                "image": row.image,
                "primary": row.primary,
            }
            for row in containers
        ]
    statuses = getattr(sandbox, "container_statuses", ())
    if isinstance(statuses, (list, tuple)) and statuses:
        details["container_statuses"] = [
            {
                "name": row.name,
                "state": row.state.value if hasattr(row.state, "value") else str(row.state),
                "exit_code": row.exit_code,
                "restart_count": row.restart_count,
            }
            for row in statuses
        ]
    return details


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

    containers = details.get("containers") or ()
    if containers:
        click.echo("")
        click.echo("CONTAINERS")
        for row in containers:
            marker = "primary" if row.get("primary") else "helper"
            name = row.get("name") or "-"
            image = row.get("image") or "-"
            click.echo(f"  {name} ({marker})  {image}")

    statuses = details.get("container_statuses") or ()
    if statuses:
        click.echo("")
        click.echo("CONTAINER STATUS")
        for row in statuses:
            name = row.get("name") or "-"
            state = row.get("state") or "-"
            exit_code = row.get("exit_code")
            restarts = row.get("restart_count")
            extra = f"  exit={exit_code}" if exit_code is not None else ""
            click.echo(f"  {name}  {state}{extra}  restarts={restarts}")
