"""Sandbox subcommand group."""

from __future__ import annotations

import asyncio
import sys

import click

from aviato import Sandbox
from aviato.cli.formatters import (
    format_sandbox_json,
    format_sandbox_quiet,
    format_sandbox_table,
)


@click.group()
def sandbox() -> None:
    """Manage Aviato sandboxes."""
    pass


@sandbox.command(name="list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["running", "completed", "failed", "stopped"], case_sensitive=False),
    help="Filter by sandbox status.",
)
@click.option(
    "--tag",
    "-t",
    "tags",
    multiple=True,
    help="Filter by tag (can be specified multiple times).",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["table", "json", "quiet"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show additional columns in table output.",
)
def list_sandboxes(
    status: str | None,
    tags: tuple[str, ...],
    output: str,
    verbose: bool,
) -> None:
    """List sandboxes.

    Examples:

        # List all sandboxes
        aviato sandbox list

        # Filter by status
        aviato sandbox list --status running

        # Filter by tag
        aviato sandbox list --tag my-project

        # Output as JSON
        aviato sandbox list -o json

        # Show only IDs (useful for scripting)
        aviato sandbox list -o quiet
    """
    try:
        sandboxes = asyncio.run(
            Sandbox.list(
                tags=list(tags) if tags else None,
                status=status,
            )
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output == "json":
        result = format_sandbox_json(sandboxes)
    elif output == "quiet":
        result = format_sandbox_quiet(sandboxes)
    else:
        result = format_sandbox_table(sandboxes, verbose=verbose)

    if result:
        click.echo(result)
