#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""Example: List sandboxes including stopped ones.

By default, Sandbox.list() only returns active sandboxes. Use
``include_stopped=True`` to also retrieve terminal sandboxes
(completed, failed, terminated) from persistent storage.

This is useful for:
- Auditing past sandbox runs
- Cleaning up or deleting old sandboxes
- Verifying that sandboxes completed successfully

Usage:
    # List only active sandboxes (default)
    uv run examples/list_stopped_sandboxes.py

    # Include stopped sandboxes
    uv run examples/list_stopped_sandboxes.py --include-stopped

    # Filter by tag
    uv run examples/list_stopped_sandboxes.py --include-stopped --tag my-batch-job
"""

import argparse

from aviato import Sandbox, SandboxDefaults, Session


def list_with_sandbox_api(tags: list[str] | None, include_stopped: bool) -> None:
    """List sandboxes using the Sandbox class method."""
    print("=== Sandbox.list() ===")
    sandboxes = Sandbox.list(tags=tags, include_stopped=include_stopped).result()

    if not sandboxes:
        print("  No sandboxes found.")
        return

    for sb in sandboxes:
        print(f"  {sb.sandbox_id}  status={sb.status}  tower={sb.tower_id}")

    print(f"\n  Total: {len(sandboxes)} sandbox(es)")


def list_with_session_api(tags: list[str] | None, include_stopped: bool) -> None:
    """List sandboxes using the Session API (auto-filters by session tags)."""
    session_tags = tags or ["list-stopped-example"]
    defaults = SandboxDefaults(tags=tuple(session_tags))

    print(f"\n=== Session.list(tags={session_tags}) ===")
    with Session(defaults) as session:
        sandboxes = session.list(include_stopped=include_stopped).result()

        if not sandboxes:
            print("  No sandboxes found for session tags.")
            return

        for sb in sandboxes:
            print(f"  {sb.sandbox_id}  status={sb.status}  tower={sb.tower_id}")

        print(f"\n  Total: {len(sandboxes)} sandbox(es)")


def main() -> None:
    parser = argparse.ArgumentParser(description="List sandboxes including stopped ones")
    parser.add_argument(
        "--include-stopped",
        action="store_true",
        help="Include terminal sandboxes (completed, failed, terminated)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Filter by tag (can be repeated)",
    )
    args = parser.parse_args()

    label = "active + stopped" if args.include_stopped else "active only"
    print(f"Listing sandboxes ({label})\n")

    list_with_sandbox_api(args.tags, args.include_stopped)
    list_with_session_api(args.tags, args.include_stopped)


if __name__ == "__main__":
    main()
