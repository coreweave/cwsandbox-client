#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Example: File-system snapshots (FSS) — snapshot, restore, and fork.

A configured working directory (the ``mount_path``) can be snapshotted on
request or on stop. The resulting snapshot can be restored into new sandboxes —
restoring is how you "fork" a sandbox's filesystem.

Demonstrates:
- Starting a sandbox with a snapshot-capable mount (``file_system_snapshot``)
- Taking a mid-life snapshot with ``sandbox.snapshot()`` (returns the ID)
- Forking = snapshot + restore into a fresh sandbox via ``Sandbox.run(...)``
- Capturing a snapshot on stop and reading ``file_system_snapshot_id``
- Managing snapshots: ``list_snapshots()`` / ``get_snapshot()`` / ``delete_snapshot()``

FSS is gated per-organization on the backend. If your org is not enabled, the
snapshot calls raise ``SnapshotNotSupportedError``.

Usage:
    uv run examples/file_system_snapshots.py
"""

from cwsandbox import (
    FileSystemSnapshotOptions,
    Sandbox,
    SandboxDefaults,
)
from cwsandbox.exceptions import SnapshotNotSupportedError

MOUNT_PATH = "/workspace"


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "file-system-snapshots"),
    )

    try:
        # 1. Start a sandbox with a snapshot-capable mount and seed some data.
        with Sandbox.run(
            defaults=defaults,
            file_system_snapshot=FileSystemSnapshotOptions(mount_path=MOUNT_PATH, size="1Gi"),
        ) as source:
            source.exec(["sh", "-c", f"echo 'hello from source' > {MOUNT_PATH}/data.txt"]).result()
            print(f"Seeded {MOUNT_PATH}/data.txt in source sandbox {source.sandbox_id}")

            # 2. Take a mid-life snapshot (waits until READY by default).
            #    snapshot() returns just the ID; use get_snapshot() for details.
            snapshot_id = source.snapshot().result()
            print(f"Created snapshot {snapshot_id}")

        # 3. Fork = restore the snapshot into a brand-new sandbox.
        with Sandbox.run(
            defaults=defaults,
            file_system_snapshot=FileSystemSnapshotOptions(
                mount_path=MOUNT_PATH, file_system_snapshot_id=snapshot_id
            ),
        ) as restored:
            contents = restored.exec(["cat", f"{MOUNT_PATH}/data.txt"]).result()
            print(f"Restored sandbox sees: {contents.stdout.strip()!r}")

        # 4. Snapshot on stop: the ID is available after the stop resolves.
        on_stop = Sandbox.run(
            defaults=defaults,
            file_system_snapshot=FileSystemSnapshotOptions(mount_path=MOUNT_PATH),
        )
        on_stop.exec(["sh", "-c", f"echo bye > {MOUNT_PATH}/farewell.txt"]).result()
        on_stop.stop(snapshot_on_stop=True).result()
        print(f"Snapshot-on-stop produced: {on_stop.file_system_snapshot_id}")

        # 5. Manage snapshots: list, fetch, and clean up.
        snapshots = Sandbox.list_snapshots().result()
        print(f"Org has {len(snapshots)} snapshot(s)")

        fetched = Sandbox.get_snapshot(snapshot_id).result()
        print(f"Fetched {fetched.file_system_snapshot_id}: {fetched.size_bytes} bytes")

        # Delete the snapshots we created (idempotent with missing_ok).
        for snap_id in {snapshot_id, on_stop.file_system_snapshot_id}:
            if snap_id:
                Sandbox.delete_snapshot(snap_id, missing_ok=True).result()
                print(f"Deleted snapshot {snap_id}")

    except SnapshotNotSupportedError:
        print(
            "File-system snapshots are not enabled for this organization. "
            "Contact CoreWeave to enable FSS."
        )


if __name__ == "__main__":
    main()
