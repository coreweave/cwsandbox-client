# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox ls — list sandboxes."""

from __future__ import annotations

import json

import click

from cwsandbox import Sandbox, SandboxStatus

_STATUS_CHOICES = [s.value for s in SandboxStatus if s != SandboxStatus.UNSPECIFIED]


@click.command("ls")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(_STATUS_CHOICES, case_sensitive=False),
    help="Filter by status.",
)
@click.option("--tag", "-t", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option(
    "--profile-id", "-r", "profile_ids", multiple=True, help="Filter by profile ID (repeatable)."
)
@click.option(
    "--runner-id", "-T", "runner_ids", multiple=True, help="Filter by runner ID (repeatable)."
)
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "wide", "json"], case_sensitive=False),
    help="Output format (table, wide, json).",
)
def list_sandboxes(
    status: str | None,
    tags: tuple[str, ...],
    profile_ids: tuple[str, ...],
    runner_ids: tuple[str, ...],
    output_format: str,
) -> None:
    """List sandboxes.

    Default table shows ID, status, image, and started time. Use -o wide
    for runner, runner group, profile, and tags. Use -o json for all fields.
    """
    sandboxes = Sandbox.list(
        tags=list(tags) if tags else None,
        status=status,
        profile_ids=list(profile_ids) if profile_ids else None,
        runner_ids=list(runner_ids) if runner_ids else None,
    ).result()

    if output_format == "json":
        data = [
            {
                "sandbox_id": sb.sandbox_id,
                "status": sb.status.value if sb.status else None,
                "runner_id": sb.runner_id,
                "profile_id": sb.profile_id,
                "runner_group_id": sb.runner_group_id,
                "started_at": sb.started_at.isoformat() if sb.started_at else None,
                "container_image": sb.container_image,
                "tags": list(sb.tags) if sb.tags else [],
            }
            for sb in sandboxes
        ]
        click.echo(json.dumps(data, indent=2))
        return

    if not sandboxes:
        click.echo("No sandboxes found.")
        return

    wide = output_format == "wide"

    header = f"{'SANDBOX ID':<40} {'STATUS':<14} {'IMAGE':<25} {'STARTED AT':<24}"
    sep = f"{'-' * 40} {'-' * 14} {'-' * 25} {'-' * 24}"
    if wide:
        header += f" {'RUNNER':<20} {'RUNNER GROUP':<20} {'PROFILE':<20} {'TAGS'}"
        sep += f" {'-' * 20} {'-' * 20} {'-' * 20} {'-' * 40}"
    click.echo(header)
    click.echo(sep)

    for sb in sandboxes:
        sid = sb.sandbox_id or "-"
        st = sb.status.value if sb.status else "-"
        image = sb.container_image or "-"
        started = sb.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if sb.started_at else "-"
        line = f"{sid:<40} {st:<14} {image:<25} {started:<24}"
        if wide:
            runner = sb.runner_id or "-"
            runner_group = sb.runner_group_id or "-"
            profile = sb.profile_id or "-"
            line += f" {runner:<20} {runner_group:<20} {profile:<20} {','.join(sb.tags) if sb.tags else '-'}"
        click.echo(line)
