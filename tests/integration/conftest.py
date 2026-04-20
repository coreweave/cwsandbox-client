# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Root conftest for integration tests.

Registers integration-wide command-line options. Defined at this level (not
inside ``cwsandbox/``) so pytest picks them up when collecting any subtree of
``tests/integration/``.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CWSandbox integration test options.

    The default for ``--cwsandbox-runner-ids`` is ``None`` (the sentinel for
    "flag not passed"), which is distinct from ``""`` ("flag passed empty,
    clear the env var").
    """
    group = parser.getgroup("cwsandbox", "CWSandbox integration tests")
    group.addoption(
        "--cwsandbox-runner-ids",
        action="store",
        default=None,
        metavar="RUNNER_IDS",
        help=(
            "Comma-separated runner IDs to pin e2e sandboxes to. Overrides "
            "CWSANDBOX_TEST_RUNNER_IDS. Pass empty (--cwsandbox-runner-ids=) "
            "to clear the env var and auto-schedule."
        ),
    )
