# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for sandbox display names.

Set CWSANDBOX_API_KEY (and optionally CWSANDBOX_BASE_URL) before running.
"""

from __future__ import annotations

import uuid

import pytest

from cwsandbox import Sandbox, SandboxDefaults, SandboxNameInUseError


def test_display_name_lookup_uniqueness_and_reuse(sandbox_defaults: SandboxDefaults) -> None:
    """A name reads back, resolves by lookup, is unique while active, then reusable."""
    defaults = sandbox_defaults.with_overrides(max_lifetime_seconds=300)
    name = f"cwsandbox-it-{uuid.uuid4().hex[:12]}"
    created: list[Sandbox] = []
    try:
        first = Sandbox.run(defaults=defaults, display_name=name)
        created.append(first)
        assert first.display_name == name

        found = Sandbox.from_id(name).result()
        assert found.sandbox_id == first.sandbox_id
        assert found.display_name == name

        with pytest.raises(SandboxNameInUseError):
            created.append(Sandbox.run(defaults=defaults, display_name=name))

        first.stop().result()
        second = Sandbox.run(defaults=defaults, display_name=name)
        created.append(second)
        assert second.display_name == name
        assert second.sandbox_id != first.sandbox_id
        assert Sandbox.from_id(name).result().sandbox_id == second.sandbox_id
    finally:
        for sandbox in created:
            sandbox.stop(missing_ok=True).result()
