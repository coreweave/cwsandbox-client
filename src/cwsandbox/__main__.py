# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Entry point for `python -m cwsandbox` and `cwsandbox` console script."""

from __future__ import annotations

import sys


def main() -> None:
    """Run the CWSandbox CLI."""
    try:
        from cwsandbox.cli import cli
    except ImportError as e:
        if getattr(e, "name", None) in ("cwsandbox.cli", "click"):
            print(
                "cwsandbox CLI requires the 'cli' extra.\n"
                "Install it with: pip install cwsandbox[cli]",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    cli()


if __name__ == "__main__":
    main()
