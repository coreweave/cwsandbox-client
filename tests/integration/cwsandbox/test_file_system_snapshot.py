# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for file-system snapshots (FSS).

These tests require a running CWSandbox backend AND an organization that is
enabled for FSS (the feature is gated by a per-org allowlist). When the org is
not enabled, the backend returns ``CWSANDBOX_FSS_NOT_SUPPORTED`` and these tests
skip rather than fail — so they only error when the FSS logic is actually
exercised.

Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY before running.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import pytest

from cwsandbox import (
    FileSystemSnapshotOptions,
    FileSystemSnapshotStatus,
    Sandbox,
    SandboxDefaults,
)
from cwsandbox.exceptions import SnapshotNotFoundError, SnapshotNotSupportedError

MOUNT_PATH = "/workspace"


@contextmanager
def skip_if_fss_unsupported() -> Generator[None, None, None]:
    """Skip the test if the org is not enabled for FSS.

    FSS is gated per-organization on the backend; an org not on the allowlist
    gets ``SnapshotNotSupportedError``. Wrapping the FSS calls in this context
    means the test only fails when FSS is actually enabled and the logic runs.
    """
    try:
        yield
    except SnapshotNotSupportedError:
        pytest.skip("Organization is not enabled for file-system snapshots (FSS)")


def _fss_options(**overrides: object) -> FileSystemSnapshotOptions:
    kwargs: dict[str, object] = {"mount_path": MOUNT_PATH, "size": "1Gi"}
    kwargs.update(overrides)
    return FileSystemSnapshotOptions(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def fss_defaults(sandbox_defaults: SandboxDefaults) -> SandboxDefaults:
    """Defaults with a longer lifetime for snapshot workflows.

    The shared ``sandbox_defaults`` uses a 60s lifetime for fast cleanup, but a
    snapshot needs the source sandbox to stay RUNNING through start + exec +
    archive, which exceeds 60s. Each test still stops its sandboxes promptly
    (via the context manager / finally), so the larger lifetime is only a
    ceiling, not added runtime. The runner pin and tags are preserved.
    """
    return sandbox_defaults.with_overrides(max_lifetime_seconds=600)


def test_snapshot_and_restore(fss_defaults: SandboxDefaults) -> None:
    """Snapshot a seeded mount, then restore it into a fresh sandbox."""
    created_snapshot_id: str | None = None
    with skip_if_fss_unsupported():
        with Sandbox.run(
            "sleep",
            "infinity",
            defaults=fss_defaults,
            file_system_snapshot=_fss_options(),
        ) as source:
            source.exec(["sh", "-c", f"echo restored-content > {MOUNT_PATH}/marker.txt"]).result()
            created_snapshot_id = source.snapshot().result()
            record = Sandbox.get_snapshot(created_snapshot_id).result()
            assert record.status == FileSystemSnapshotStatus.READY
            assert record.source_sandbox_id == source.sandbox_id

        try:
            with Sandbox.run(
                "sleep",
                "infinity",
                defaults=fss_defaults,
                file_system_snapshot=_fss_options(file_system_snapshot_id=created_snapshot_id),
            ) as restored:
                result = restored.exec(["cat", f"{MOUNT_PATH}/marker.txt"]).result()
                assert result.returncode == 0
                assert result.stdout.strip() == "restored-content"
        finally:
            if created_snapshot_id:
                Sandbox.delete_snapshot(created_snapshot_id, missing_ok=True).result()


def test_snapshot_on_stop(fss_defaults: SandboxDefaults) -> None:
    """stop(snapshot_on_stop=True) captures a snapshot and exposes its ID."""
    created_snapshot_id: str | None = None
    with skip_if_fss_unsupported():
        sandbox = Sandbox.run(
            "sleep",
            "infinity",
            defaults=fss_defaults,
            file_system_snapshot=_fss_options(),
        )
        try:
            sandbox.exec(["sh", "-c", f"echo bye > {MOUNT_PATH}/farewell.txt"]).result()
            sandbox.stop(snapshot_on_stop=True).result()
            created_snapshot_id = sandbox.file_system_snapshot_id
            assert created_snapshot_id, "expected a snapshot ID after snapshot-on-stop"

            fetched = Sandbox.get_snapshot(created_snapshot_id).result()
            assert fetched.file_system_snapshot_id == created_snapshot_id
            assert fetched.status == FileSystemSnapshotStatus.READY
        finally:
            sandbox.stop(missing_ok=True).result()
            if created_snapshot_id:
                Sandbox.delete_snapshot(created_snapshot_id, missing_ok=True).result()


def test_list_get_delete_snapshot(fss_defaults: SandboxDefaults) -> None:
    """Snapshot management: list (filtered), get, and delete."""
    with skip_if_fss_unsupported():
        with Sandbox.run(
            "sleep",
            "infinity",
            defaults=fss_defaults,
            file_system_snapshot=_fss_options(),
        ) as source:
            source.exec(["sh", "-c", f"echo x > {MOUNT_PATH}/x.txt"]).result()
            snapshot_id = source.snapshot().result()

        try:
            # list_snapshots (client-side filter by source sandbox) finds it.
            listed = Sandbox.list_snapshots(source_sandbox_id=source.sandbox_id).result()
            assert any(s.file_system_snapshot_id == snapshot_id for s in listed)

            # get_snapshot returns the same record.
            fetched = Sandbox.get_snapshot(snapshot_id).result()
            assert fetched.file_system_snapshot_id == snapshot_id
        finally:
            Sandbox.delete_snapshot(snapshot_id, missing_ok=True).result()

        # After deletion, get raises SnapshotNotFoundError.
        with pytest.raises(SnapshotNotFoundError):
            Sandbox.get_snapshot(snapshot_id).result()
