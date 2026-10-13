# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox files - read and write sandbox files."""

from __future__ import annotations

from pathlib import Path

import click

from cwsandbox import DataPlaneMode, Sandbox


@click.group()
def files() -> None:
    """Read and write files in a sandbox.

    Examples:

        cwsandbox files read <sandbox-id> /tmp/data.json

        cwsandbox files write <sandbox-id> /tmp/data.json data.json
    """


@files.command("read")
@click.argument("sandbox_id")
@click.argument("remote_path")
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write output to a local file instead of stdout.",
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
    "--container",
    default=None,
    help="Container name. Omit to target the primary.",
)
@click.option(
    "--data-plane-mode",
    type=click.Choice([mode.value for mode in DataPlaneMode], case_sensitive=False),
    default=DataPlaneMode.AUTO.value,
    show_default=True,
    help="Data connection: auto falls back to the gateway; direct requires direct access.",
)
def read_file(
    sandbox_id: str,
    remote_path: str,
    output_path: Path | None,
    timeout_seconds: float | None,
    container: str | None,
    data_plane_mode: str,
) -> None:
    """Read a file from a sandbox.

    SANDBOX_ID is the ID of the sandbox to read from.
    REMOTE_PATH is the file path inside the sandbox.

    Examples:

        cwsandbox files read <sandbox-id> /tmp/data.json

        cwsandbox files read <sandbox-id> /tmp/data.json -o data.json
    """
    sandbox = Sandbox.from_id(sandbox_id, data_plane_mode=data_plane_mode).result()
    data = sandbox.read_file(
        remote_path,
        timeout_seconds=timeout_seconds,
        container=container,
    ).result()

    if output_path is not None:
        output_path.write_bytes(data)
        return

    try:
        click.get_binary_stream("stdout").write(data)
    except BrokenPipeError:
        pass


@files.command("write")
@click.argument("sandbox_id")
@click.argument("remote_path")
@click.argument(
    "local_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
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
@click.option(
    "--container",
    default=None,
    help="Container name. Omit to target the primary.",
)
@click.option(
    "--data-plane-mode",
    type=click.Choice([mode.value for mode in DataPlaneMode], case_sensitive=False),
    default=DataPlaneMode.AUTO.value,
    show_default=True,
    help="Data connection: auto falls back to the gateway; direct requires direct access.",
)
def write_file(
    sandbox_id: str,
    remote_path: str,
    local_path: Path,
    timeout_seconds: float | None,
    quiet: bool,
    container: str | None,
    data_plane_mode: str,
) -> None:
    """Write a local file into a sandbox.

    SANDBOX_ID is the ID of the sandbox to write to.
    REMOTE_PATH is the destination path inside the sandbox.
    LOCAL_PATH is the local file to upload.

    Examples:

        cwsandbox files write <sandbox-id> /tmp/data.json data.json

        cwsandbox files write <sandbox-id> /tmp/config.yaml config.yaml --quiet
    """
    data = local_path.read_bytes()
    sandbox = Sandbox.from_id(sandbox_id, data_plane_mode=data_plane_mode).result()
    sandbox.write_file(
        remote_path,
        data,
        timeout_seconds=timeout_seconds,
        container=container,
    ).result()

    if not quiet:
        click.echo(f"Wrote {len(data)} bytes to {remote_path}.")
