"""Sandbox commands for the Aviato CLI."""

from __future__ import annotations

import sys

import click

from aviato import Sandbox, SandboxStatus
from aviato.cli.formatters import (
    format_sandbox_json,
    format_sandbox_quiet,
    format_sandbox_table,
)


@click.command(name="list")
@click.option(
    "--status",
    "-s",
    type=click.Choice([s.value for s in SandboxStatus], case_sensitive=False),
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
        aviato list

        # Filter by status
        aviato list --status running

        # Filter by tag
        aviato list --tag my-project

        # Output as JSON
        aviato list -o json

        # Show only IDs (useful for scripting)
        aviato list -o quiet
    """
    try:
        sandboxes = Sandbox.list(
            tags=list(tags) if tags else None,
            status=status,
        ).result()
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
