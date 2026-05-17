# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Smoke test for the cwsandbox.harbor adapter.

Skipped automatically when harbor is not installed in the dev environment
(i.e., when only the base `dev` extra is in play, not `harbor`).
"""

import pytest

pytest.importorskip("harbor")

from cwsandbox.harbor import CWSandboxEnvironment


def test_adapter_type() -> None:
    assert CWSandboxEnvironment.type() == "cwsandbox"
