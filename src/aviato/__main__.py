# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Entry point for `python -m aviato` and `aviato` console script."""

from __future__ import annotations

import sys


def main() -> None:
    """Run the Aviato CLI."""
    try:
        from aviato.cli import cli
    except ImportError as e:
        if getattr(e, "name", None) in ("aviato.cli", "click"):
            print(
                "aviato CLI requires the 'cli' extra.\n"
                "Install it with:  pip install aviato[cli]",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    cli()


if __name__ == "__main__":
    main()
