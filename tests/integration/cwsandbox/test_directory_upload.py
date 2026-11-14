# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for Sandbox.upload_directory.

Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY before running.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from cwsandbox import Container, Sandbox, SandboxDefaults
from cwsandbox.exceptions import SandboxError, SandboxFileError

_DEST = "/tmp/upload-dest"
_BIG = 50 * 1024 * 1024
_LONG = "d" * 60 + "/" + "f" * 80 + ".txt"


class _StopAfterFirstPart(Exception):
    pass


@pytest.fixture
def upload_defaults(sandbox_defaults: SandboxDefaults) -> SandboxDefaults:
    """A 50 MB upload outlives the 60s fixture lifetime."""
    return sandbox_defaults.with_overrides(max_lifetime_seconds=600)


def _make_tree(root: Path) -> dict[str, str]:
    """Build the test tree and return {relative path: sha256} for regular files."""
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "deep.txt").write_text("deep")
    (root / "a" / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "a" / "run.sh").chmod(0o755)
    (root / "a" / "link").symlink_to("b/deep.txt")
    (root / "empty").mkdir()
    (root / "café.txt").write_text("unicode")
    (root / _LONG).parent.mkdir()
    (root / _LONG).write_text("long")
    (root / "skip.log").write_text("excluded")
    (root / "cache").mkdir()
    (root / "cache" / "x").write_text("excluded")
    (root / "big.bin").write_bytes(os.urandom(_BIG))
    return {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in ("a/b/deep.txt", "a/run.sh", "café.txt", _LONG, "big.bin")
    }


def _remote_listing(sandbox: Sandbox, container: str | None = None) -> dict[str, str]:
    script = (
        f"cd {_DEST} && find . -mindepth 1 | sort | while read -r p; do "
        'if [ -L "$p" ]; then echo "l $(readlink "$p") $p"; '
        'elif [ -d "$p" ]; then echo "d - $p"; '
        'else echo "f $(sha256sum < "$p" | cut -d" " -f1) $p"; fi; done; '
        "stat -c '%a' a/run.sh"
    )
    result = sandbox.exec(["sh", "-c", script], timeout_seconds=120, container=container).result()
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    listing = {}
    for line in lines[:-1]:
        kind, value, path = line.split(" ", 2)
        listing[path.removeprefix("./")] = f"{kind} {value}"
    listing["mode:a/run.sh"] = lines[-1]
    return listing


@pytest.mark.parametrize("image", ["python:3.11", "alpine:3.20"])
def test_upload_directory_tree_and_resume(
    upload_defaults: SandboxDefaults, tmp_path: Path, image: str
) -> None:
    """Nested tree with symlink, excludes and a 50 MB file; interrupted then resumed."""
    sums = _make_tree(tmp_path)
    exclude = ["*.log", "cache"]

    def stop_after_first_part(done: int, total: int) -> None:
        raise _StopAfterFirstPart

    with Sandbox.run(defaults=upload_defaults, container_image=image) as sandbox:
        with pytest.raises(_StopAfterFirstPart):
            sandbox.upload_directory(
                tmp_path, _DEST, exclude=exclude, on_progress=stop_after_first_part
            ).result()

        progress: list[tuple[int, int]] = []
        result = sandbox.upload_directory(
            tmp_path,
            _DEST,
            exclude=exclude,
            on_progress=lambda done, total: progress.append((done, total)),
        ).result()

        assert result.resumed_bytes == 32 * 1024 * 1024
        assert result.files == 5
        assert result.symlinks == 1
        assert result.directories == 4
        assert result.archive_bytes > _BIG
        assert progress[-1] == (result.bytes, result.bytes)

        listing = _remote_listing(sandbox)
        for rel, digest in sums.items():
            assert listing.pop(rel) == f"f {digest}"
        assert listing.pop("a/link") == "l b/deep.txt"
        assert listing.pop("mode:a/run.sh") == "755"
        assert listing == {name: "d -" for name in ("a", "a/b", "empty", _LONG.split("/")[0])}


def test_upload_directory_rejects_external_symlink(
    sandbox_defaults: SandboxDefaults, tmp_path: Path
) -> None:
    (tmp_path / "ok.txt").write_text("ok")
    (tmp_path / "escape").symlink_to("../../etc/passwd")
    with Sandbox.run(defaults=sandbox_defaults) as sandbox:
        with pytest.raises(SandboxFileError) as exc_info:
            sandbox.upload_directory(tmp_path, _DEST).result()
        assert exc_info.value.metadata["stage"] == "archive"
        assert exc_info.value.metadata["path"] == "escape"
        assert sandbox.exec(["test", "-e", _DEST]).result().returncode != 0


def test_upload_directory_missing_tools(sandbox_defaults: SandboxDefaults, tmp_path: Path) -> None:
    """ubi-micro has sh and sha256sum but no tar or gzip."""
    (tmp_path / "a.txt").write_text("a")
    with Sandbox.run(
        "sleep",
        "infinity",
        defaults=sandbox_defaults,
        container_image="registry.access.redhat.com/ubi9/ubi-micro:latest",
    ) as sandbox:
        with pytest.raises(SandboxFileError) as exc_info:
            sandbox.upload_directory(tmp_path, _DEST).result()
        assert exc_info.value.metadata["stage"] == "preflight"
        assert exc_info.value.metadata["missing_tools"] == "tar gzip"


def test_upload_directory_to_sidecar(upload_defaults: SandboxDefaults, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_text("sidecar")
    containers = [
        Container(
            name=name,
            image="python:3.11",
            command="sleep",
            args=("infinity",),
            primary=name == "main",
            resources={"cpu": "500m", "memory": "256Mi"},
        )
        for name in ("main", "helper")
    ]
    try:
        sandbox = Sandbox.run(containers=containers, defaults=upload_defaults)
    except SandboxError as exc:
        pytest.skip(f"backend does not admit multi-container: {exc}")
    with sandbox:
        result = sandbox.upload_directory(tmp_path, _DEST, container="helper").result()
        assert result.files == 1
        assert sandbox.read_file(f"{_DEST}/sub/x.txt", container="helper").result() == b"sidecar"
        assert sandbox.exec(["test", "-e", _DEST]).result().returncode != 0
