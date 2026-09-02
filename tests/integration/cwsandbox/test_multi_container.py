# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for v1 multi-container sandboxes.

These tests require a running CWSandbox backend that admits more than one
user container. When the fleet rejects N>1 (not implemented, or a
single-container limit), they skip rather than fail.

Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY before running.
"""

from __future__ import annotations

import uuid

import pytest

from cwsandbox import (
    Container,
    Sandbox,
    SandboxDefaults,
    ScratchVolumeOptions,
    VolumeMount,
)
from cwsandbox._sandbox import SandboxStatus
from cwsandbox.exceptions import SandboxError, SandboxFileError, SnapshotNotSupportedError

_IMAGE = "python:3.11"
_REDIS_IMAGE = "redis:7"
_RESOURCES: dict[str, str] = {"cpu": "500m", "memory": "256Mi"}
_PRIMARY = "main"
_HELPER = "helper"
_CACHE = "cache"

# RESP SET from the primary. One connection; a refused or timed-out connect
# is a failed loopback, not a race to retry.
_REDIS_SET = (
    "import socket, sys\n"
    "key, value = sys.argv[1], sys.argv[2]\n"
    "def bulk(s: str) -> bytes:\n"
    "    b = s.encode()\n"
    "    return b'$%d\\r\\n' % len(b) + b + b'\\r\\n'\n"
    "payload = b'*3\\r\\n$3\\r\\nSET\\r\\n' + bulk(key) + bulk(value)\n"
    "with socket.create_connection(('127.0.0.1', 6379), timeout=3) as sock:\n"
    "    sock.sendall(payload)\n"
    "    print(sock.recv(64).decode(), end='')\n"
)

_SKIP_HINTS = (
    "not implemented",
    "unimplemented",
    "at most 1 container",
    "at most one container",
    "only one container",
    "single container",
)


def _skip_if_multi_container_unavailable(exc: BaseException) -> None:
    """Skip when the backend does not admit more than one user container."""
    reason = getattr(exc, "reason", None) or ""
    text = str(exc).lower()
    if "NOT_IMPLEMENTED" in reason or any(hint in text for hint in _SKIP_HINTS):
        pytest.skip(f"backend does not admit multi-container: {exc}")


def _keepalive(**overrides: object) -> Container:
    kwargs: dict[str, object] = {
        "image": _IMAGE,
        "command": "sleep",
        "args": ("infinity",),
        "resources": _RESOURCES,
    }
    kwargs.update(overrides)
    return Container(**kwargs)  # type: ignore[arg-type]


def _pair(*, volume_mounts: list[VolumeMount] | None = None) -> list[Container]:
    mounts = volume_mounts
    return [
        _keepalive(name=_PRIMARY, primary=True, volume_mounts=mounts),
        _keepalive(name=_HELPER, volume_mounts=mounts),
    ]


@pytest.fixture
def multi_container_defaults(sandbox_defaults: SandboxDefaults) -> SandboxDefaults:
    """Longer lifetime: N>1 start plus two execs can exceed the 60s fixture."""
    return sandbox_defaults.with_overrides(max_lifetime_seconds=300)


def test_multi_container_exec_files_and_echo(
    multi_container_defaults: SandboxDefaults,
) -> None:
    """N=2 create reaches RUNNING; default ops hit primary; container= isolates."""
    try:
        with Sandbox.run(
            containers=_pair(),
            defaults=multi_container_defaults,
        ) as sandbox:
            sandbox.wait()
            assert sandbox.status == SandboxStatus.RUNNING

            by_name = {row.name: row for row in sandbox.containers}
            assert set(by_name) == {_PRIMARY, _HELPER}
            assert by_name[_PRIMARY].primary is True
            assert by_name[_HELPER].primary is False
            assert by_name[_PRIMARY].image == _IMAGE
            assert by_name[_HELPER].image == _IMAGE

            status_by_name = {row.name: row for row in sandbox.container_statuses}
            assert set(status_by_name) == {_PRIMARY, _HELPER}

            primary_marker = uuid.uuid4().hex
            primary = sandbox.exec(
                ["sh", "-c", f"echo {primary_marker} > /tmp/primary.txt && cat /tmp/primary.txt"]
            ).result()
            assert primary.returncode == 0
            assert primary_marker in primary.stdout

            helper_marker = uuid.uuid4().hex
            helper = sandbox.exec(
                [
                    "sh",
                    "-c",
                    f"echo {helper_marker} > /tmp/helper.txt && cat /tmp/helper.txt",
                ],
                container=_HELPER,
            ).result()
            assert helper.returncode == 0
            assert helper_marker in helper.stdout

            miss = sandbox.exec(["cat", "/tmp/helper.txt"]).result()
            assert miss.returncode != 0

            path = f"/tmp/file_{uuid.uuid4().hex}.txt"
            payload = b"from-helper"
            sandbox.write_file(path, payload, container=_HELPER).result()
            assert sandbox.read_file(path, container=_HELPER).result() == payload
            with pytest.raises(SandboxFileError):
                sandbox.read_file(path).result()
    except SandboxError as exc:
        _skip_if_multi_container_unavailable(exc)
        raise


def test_multi_container_shared_volume(
    multi_container_defaults: SandboxDefaults,
) -> None:
    """A declare-only scratch volume is visible to both containers when mounted."""
    mount = VolumeMount(volume="workspace", mount_path="/shared")
    try:
        with Sandbox.run(
            containers=_pair(volume_mounts=[mount]),
            volumes=[ScratchVolumeOptions(name="workspace", size="1Gi")],
            defaults=multi_container_defaults,
        ) as sandbox:
            sandbox.wait()
            marker = uuid.uuid4().hex
            write = sandbox.exec(["sh", "-c", f"echo {marker} > /shared/marker.txt"]).result()
            assert write.returncode == 0
            read = sandbox.exec(["cat", "/shared/marker.txt"], container=_HELPER).result()
            assert read.returncode == 0
            assert marker in read.stdout
    except SnapshotNotSupportedError:
        pytest.skip("Organization is not enabled for scratch / file-system snapshots")
    except SandboxError as exc:
        _skip_if_multi_container_unavailable(exc)
        raise


def test_multi_container_localhost_helper(
    multi_container_defaults: SandboxDefaults,
) -> None:
    """Primary talks to a different-image helper on loopback.

    RUNNING means the helper process has Started, not that it is listening.
    Wait for redis-cli PING inside the cache container, then SET from the
    primary exactly once. Retrying that SET would hide a missing shared
    network namespace.
    """
    containers = [
        _keepalive(name=_PRIMARY, primary=True),
        Container(
            image=_REDIS_IMAGE,
            name=_CACHE,
            resources=_RESOURCES,
        ),
    ]
    try:
        with Sandbox.run(
            containers=containers,
            defaults=multi_container_defaults,
        ) as sandbox:
            sandbox.wait()
            assert sandbox.status == SandboxStatus.RUNNING

            by_name = {row.name: row for row in sandbox.containers}
            assert set(by_name) == {_PRIMARY, _CACHE}
            assert by_name[_PRIMARY].primary is True
            assert by_name[_CACHE].primary is False
            assert by_name[_PRIMARY].image == _IMAGE
            assert by_name[_CACHE].image == _REDIS_IMAGE

            ready = sandbox.exec(
                [
                    "sh",
                    "-c",
                    'i=0; while [ "$i" -lt 40 ]; do '
                    "redis-cli PING && exit 0; "
                    "i=$((i+1)); sleep 0.25; "
                    "done; echo 'redis never answered PING' >&2; exit 1",
                ],
                container=_CACHE,
                timeout_seconds=20,
            ).result()
            assert ready.returncode == 0, ready.stderr or ready.stdout
            assert ready.stdout.strip().splitlines()[-1] == "PONG"

            key = uuid.uuid4().hex
            value = uuid.uuid4().hex
            written = sandbox.exec(["python", "-c", _REDIS_SET, key, value]).result()
            assert written.returncode == 0, written.stderr or written.stdout
            assert written.stdout.strip() == "+OK"

            got = sandbox.exec(["redis-cli", "GET", key], container=_CACHE).result()
            assert got.returncode == 0, got.stderr or got.stdout
            assert got.stdout.strip() == value
    except SandboxError as exc:
        _skip_if_multi_container_unavailable(exc)
        raise
