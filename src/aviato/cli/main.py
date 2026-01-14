"""Aviato CLI entry point."""

import click

from aviato.cli._sandbox import list_sandboxes


@click.group()
@click.version_option(package_name="aviato")
def cli() -> None:
    """Aviato CLI - manage sandboxes from the command line."""
    pass


cli.add_command(list_sandboxes)


if __name__ == "__main__":
    cli()
