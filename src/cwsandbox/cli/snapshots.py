# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox snapshots - manage file-system snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

import click

from cwsandbox import FileSystemSnapshot, FileSystemSnapshotStatus, Sandbox

_STATUS_CHOICES = [
    s.value for s in FileSystemSnapshotStatus if s != FileSystemSnapshotStatus.UNSPECIFIED
]


@click.group()
def snapshots() -> None:
    """Create, inspect, list, and delete file-system snapshots."""


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _table_time(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S UTC") if value is not None else "-"


def _snapshot_to_dict(snapshot: FileSystemSnapshot) -> dict[str, Any]:
    return {
        "file_system_snapshot_id": snapshot.file_system_snapshot_id,
        "status": _enum_value(snapshot.status),
        "status_reason": snapshot.status_reason,
        "size_bytes": snapshot.size_bytes,
        "source_sandbox_id": snapshot.source_sandbox_id,
        "trigger": _enum_value(snapshot.trigger),
        "request_id": snapshot.request_id,
        "object_bucket": snapshot.object_bucket,
        "source_volume_name": snapshot.source_volume_name,
        "created_at": _isoformat(snapshot.created_at),
        "updated_at": _isoformat(snapshot.updated_at),
        "completed_at": _isoformat(snapshot.completed_at),
    }


def _echo_snapshots_table(snapshot_rows: list[FileSystemSnapshot]) -> None:
    click.echo(
        f"{'SNAPSHOT ID':<40} {'STATUS':<10} {'SOURCE SANDBOX':<40} "
        f"{'TRIGGER':<10} {'SIZE':>12} {'CREATED AT'}"
    )
    click.echo(f"{'-' * 40} {'-' * 10} {'-' * 40} {'-' * 10} {'-' * 12} {'-' * 24}")

    for snapshot in snapshot_rows:
        snapshot_id = snapshot.file_system_snapshot_id or "-"
        status = _enum_value(snapshot.status) or "-"
        source = snapshot.source_sandbox_id or "-"
        trigger = _enum_value(snapshot.trigger) or "-"
        created = _table_time(snapshot.created_at)
        click.echo(
            f"{snapshot_id:<40} {status:<10} {source:<40} "
            f"{trigger:<10} {snapshot.size_bytes:>12} {created}"
        )


@snapshots.command("create")
@click.argument("sandbox_id")
@click.option(
    "--wait/--no-wait",
    "wait_for_ready",
    default=True,
    help="Wait for the snapshot to reach READY before returning.",
)
@click.option(
    "--request-id",
    default=None,
    help="Client-supplied request ID to deduplicate snapshot creation retries.",
)
@click.option(
    "--timeout",
    "-t",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Timeout in seconds for attaching to the sandbox.",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format.",
)
def create_snapshot(
    sandbox_id: str,
    wait_for_ready: bool,
    request_id: str | None,
    timeout_seconds: float | None,
    output_format: str,
) -> None:
    """Create a file-system snapshot from a running sandbox.

    SANDBOX_ID is the ID of the sandbox to snapshot.
    """
    sandbox = Sandbox.from_id(sandbox_id, timeout_seconds=timeout_seconds).result()
    snapshot_id = sandbox.snapshot(
        wait_for_ready=wait_for_ready,
        request_id=request_id,
    ).result()

    if output_format == "json":
        click.echo(json.dumps({"file_system_snapshot_id": snapshot_id}, indent=2))
        return

    click.echo(snapshot_id)


@snapshots.command("get")
@click.argument("file_system_snapshot_id")
@click.option(
    "--timeout",
    "-t",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Timeout in seconds.",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format.",
)
def get_snapshot(
    file_system_snapshot_id: str,
    timeout_seconds: float | None,
    output_format: str,
) -> None:
    """Get a file-system snapshot by ID."""
    snapshot = Sandbox.get_snapshot(
        file_system_snapshot_id,
        timeout_seconds=timeout_seconds,
    ).result()

    if output_format == "json":
        click.echo(json.dumps(_snapshot_to_dict(snapshot), indent=2))
        return

    _echo_snapshots_table([snapshot])


@snapshots.command("list")
@click.option(
    "--source-sandbox-id",
    default=None,
    help="Only show snapshots captured from this sandbox.",
)
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(_STATUS_CHOICES, case_sensitive=False),
    help="Filter by snapshot status.",
)
@click.option(
    "--timeout",
    "-t",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Timeout in seconds.",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format.",
)
def list_snapshots(
    source_sandbox_id: str | None,
    status: str | None,
    timeout_seconds: float | None,
    output_format: str,
) -> None:
    """List file-system snapshots."""
    snapshot_rows = Sandbox.list_snapshots(
        source_sandbox_id=source_sandbox_id,
        status=status,
        timeout_seconds=timeout_seconds,
    ).result()

    if output_format == "json":
        click.echo(
            json.dumps([_snapshot_to_dict(snapshot) for snapshot in snapshot_rows], indent=2)
        )
        return

    if not snapshot_rows:
        click.echo("No snapshots found.")
        return

    _echo_snapshots_table(snapshot_rows)


@snapshots.command("delete")
@click.argument("file_system_snapshot_id")
@click.option(
    "--missing-ok",
    is_flag=True,
    default=False,
    help="Succeed if the snapshot is already missing.",
)
@click.option(
    "--timeout",
    "-t",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Timeout in seconds.",
)
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress success output.")
def delete_snapshot(
    file_system_snapshot_id: str,
    missing_ok: bool,
    timeout_seconds: float | None,
    quiet: bool,
) -> None:
    """Delete a file-system snapshot by ID."""
    Sandbox.delete_snapshot(
        file_system_snapshot_id,
        timeout_seconds=timeout_seconds,
        missing_ok=missing_ok,
    ).result()

    if not quiet:
        click.echo(f"Deleted {file_system_snapshot_id}.")
