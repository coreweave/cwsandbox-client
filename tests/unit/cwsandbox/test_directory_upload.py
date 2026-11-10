# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for Sandbox.upload_directory."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cwsandbox import DirectoryUploadResult, Sandbox
from cwsandbox import _directory_upload as dirup
from cwsandbox._sandbox import _Running
from cwsandbox.exceptions import (
    SandboxFileError,
    SandboxNotRunningError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)

TOOLS_PRESENT = all(shutil.which(tool) for tool in (*dirup.BASE_TOOLS, "gzip"))
needs_tools = pytest.mark.skipif(not TOOLS_PRESENT, reason="needs sh, tar, sha256sum and gzip")


# Shared with the JavaScript SDK: same inputs must give the same key.
KEY_VECTOR_ENTRIES = (
    dirup.UploadEntry("d", "src", 0o755, 0, 1700000000, "", ""),
    dirup.UploadEntry("f", "src/a.txt", 0o644, 5, 1700000001, "", ""),
    dirup.UploadEntry("l", "src/link", 0o777, 0, 1700000002, "a.txt", ""),
)
KEY_VECTOR_MANIFEST = (
    b"cwsandbox-directory-upload/1\n"
    b"gzip\n"
    b"d 755 0 1700000000 3:src 0:\n"
    b"f 644 5 1700000001 9:src/a.txt 0:\n"
    b"l 777 0 1700000002 8:src/link 5:a.txt\n"
)
KEY_VECTOR_KEY = "03894960432b2c9b57ee9b554541eda2"

# Shared with the JavaScript SDK: (pattern, relative path, excluded).
EXCLUDE_VECTORS = [
    (".git", ".git", True),
    (".git", "sub/.git", True),
    ("*.pyc", "a/b/c.pyc", True),
    ("build/*.o", "build/x.o", True),
    ("build/*.o", "src/build/x.o", False),
    ("/docs/private/", "docs/private", True),
    ("node_modules", "node_modules_old", False),
    ("a?c", "abc", True),
    ("[!x]y", "zy", True),
    ("src/*", "src/a/b", True),
]


def make_tree(root: Path) -> None:
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "sub" / "data.txt").write_text("data")
    (root / "pkg" / "run.sh").write_text("#!/bin/sh\n")
    (root / "pkg" / "run.sh").chmod(0o755)
    (root / "pkg" / "link").symlink_to("sub/data.txt")
    (root / "empty").mkdir()
    (root / "skip.pyc").write_text("x")
    (root / ("n" * 120)).write_text("long")
    (root / "café.txt").write_text("u")


def running_sandbox() -> Sandbox:
    sandbox = Sandbox(command="sleep", args=["infinity"])
    sandbox._sandbox_id = "sb-upload"
    sandbox._state = _Running(sandbox_id="sb-upload")
    sandbox._channel = MagicMock()
    sandbox._stub = MagicMock()
    return sandbox


class LocalRemote:
    """Runs the real remote scripts with the local /bin/sh in place of an exec."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str | None, bytes | None]] = []
        self.inject: Callable[[str, list[str], int], Any] | None = None

    async def __call__(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: float | None = None,
        operation: str,
        filepath: str | None = None,
        container: str | None = None,
    ) -> tuple[int, bytes, bytes]:
        argv0, args = command[3], list(command[4:])
        self.calls.append((argv0, args, container, stdin))
        if self.inject is not None:
            injected = self.inject(argv0, args, len(self.calls))
            if isinstance(injected, BaseException):
                raise injected
            if injected is not None:
                return injected
        proc = subprocess.run(list(command), input=stdin or b"", capture_output=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


def upload(
    sandbox: Sandbox, remote: LocalRemote, local: Path, dest: Path, **kwargs: Any
) -> DirectoryUploadResult:
    with (
        patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
        patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
        patch.object(sandbox, "_exec_streaming_binary_async", new=remote),
        patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock),
    ):
        return sandbox.upload_directory(str(local), str(dest), **kwargs).result()


def listing(root: Path) -> dict[str, tuple[str, int, str]]:
    out: dict[str, tuple[str, int, str]] = {}
    for path in sorted(root.rglob("*")):
        # macOS tar may store names decomposed; compare in NFC.
        rel = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        if path.is_symlink():
            out[rel] = ("l", 0, os.readlink(path))
        elif path.is_dir():
            out[rel] = ("d", path.stat().st_mode & 0o777, "")
        else:
            out[rel] = ("f", path.stat().st_mode & 0o777, path.read_text())
    return out


# Shared with the JavaScript SDK: the remote scripts must stay byte-identical.
SCRIPT_SHA256 = {
    "PREFLIGHT_SCRIPT": "029e4958f0051615488abfdb50a27f79d488ad6301da5663248a6ac2b4fcd767",
    "PART_SCRIPT": "50d726dee5281f5dd5a86d373d729a68b64f3346f31f3538c673c1d0e2de879a",
    "FINAL_SCRIPT": "21239579a8378024a1dd0603fe9b595f060a99878158f490a0da28fc46f97623",
}


class TestManifestAndExcludes:
    @pytest.mark.parametrize(("name", "digest"), sorted(SCRIPT_SHA256.items()))
    def test_scripts_match_the_javascript_sdk(self, name: str, digest: str) -> None:
        assert hashlib.sha256(getattr(dirup, name).encode()).hexdigest() == digest

    def test_key_vector(self) -> None:
        assert dirup.manifest_bytes(KEY_VECTOR_ENTRIES, True) == KEY_VECTOR_MANIFEST
        assert dirup.content_key(KEY_VECTOR_ENTRIES, True) == KEY_VECTOR_KEY
        assert dirup.content_key(KEY_VECTOR_ENTRIES, False) != KEY_VECTOR_KEY

    @pytest.mark.parametrize(("pattern", "relative", "excluded"), EXCLUDE_VECTORS)
    def test_exclude_vectors(self, pattern: str, relative: str, excluded: bool) -> None:
        patterns = dirup.normalize_excludes([pattern])
        name = relative.rsplit("/", 1)[-1]
        assert dirup.is_excluded(name, relative, patterns) is excluded

    @pytest.mark.parametrize(
        ("relative", "target", "escapes"),
        [
            ("a/link", "b.txt", False),
            ("a/link", "../top.txt", False),
            ("link", ".", False),
            ("link", "/etc/passwd", True),
            ("link", "..", True),
            ("a/link", "../../x", True),
            ("a/link", "b/../../../x", True),
            ("link", "", True),
        ],
    )
    def test_symlink_escape(self, relative: str, target: str, escapes: bool) -> None:
        assert dirup.symlink_escapes(relative, target) is escapes


class TestScanAndArchive:
    def test_walk_order_excludes_and_skips(self, tmp_path: Path) -> None:
        make_tree(tmp_path)
        (tmp_path / "pkg" / "node_modules").mkdir()
        (tmp_path / "pkg" / "node_modules" / "dep.js").write_text("x")
        os.mkfifo(tmp_path / "pipe")
        plan = dirup.scan_directory(
            str(tmp_path), dirup.normalize_excludes(["*.pyc", "node_modules"]), True, "/w"
        )
        names = [entry.name for entry in plan.entries]
        assert names == [
            "café.txt",
            "empty",
            "n" * 120,
            "pkg",
            "pkg/link",
            "pkg/run.sh",
            "pkg/sub",
            "pkg/sub/data.txt",
        ]
        assert "skip.pyc" not in names
        assert not any("node_modules" in name for name in names)
        assert plan.skipped == ("pipe",)
        assert plan.total_bytes == len("data") + len("#!/bin/sh\n") + len("long") + len("u")

    def test_external_symlink_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "out").symlink_to("/etc/passwd")
        with pytest.raises(SandboxFileError) as exc_info:
            dirup.scan_directory(str(tmp_path), (), True, "/w")
        assert exc_info.value.metadata["stage"] == "archive"
        assert exc_info.value.metadata["path"] == "out"
        assert "exclude" in str(exc_info.value)

    def test_excluded_external_symlink_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "out").symlink_to("../elsewhere")
        plan = dirup.scan_directory(str(tmp_path), ("out",), True, "/w")
        assert plan.entries == ()

    @pytest.mark.parametrize("compress", [True, False])
    def test_archive_round_trip(self, tmp_path: Path, compress: bool) -> None:
        make_tree(tmp_path)
        (tmp_path / "pkg" / "run.sh").chmod(0o4755)
        plan = dirup.scan_directory(str(tmp_path), (), compress, "/w")
        data = b"".join(part.data for part in dirup.iter_parts(plan, compress, "/w"))
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz" if compress else "r:") as tar:
            members = {member.name: member for member in tar.getmembers()}
        assert set(members) == {entry.name for entry in plan.entries}
        assert all(not name.startswith("/") and ".." not in name.split("/") for name in members)
        assert members["pkg/run.sh"].mode == 0o755
        assert members["pkg/link"].issym() and members["pkg/link"].linkname == "sub/data.txt"
        assert members["n" * 120].isfile()
        assert all(m.uid == 0 and m.gid == 0 and m.uname == "" for m in members.values())
        if not compress:
            assert len(data) % dirup.TAR_RECORD_SIZE == 0

    def test_archive_is_deterministic_and_split_into_parts(self, tmp_path: Path) -> None:
        make_tree(tmp_path)
        (tmp_path / "big.bin").write_bytes(os.urandom(5000))
        plan = dirup.scan_directory(str(tmp_path), (), True, "/w")
        first = list(dirup.iter_parts(plan, True, "/w", part_size=1024))
        second = list(dirup.iter_parts(plan, True, "/w", part_size=1024))
        assert [p.data for p in first] == [p.data for p in second]
        assert all(len(p.data) == 1024 for p in first[:-1]) and len(first) > 2
        assert [p.index for p in first] == list(range(len(first)))
        assert first[-1].source_bytes_done == plan.total_bytes
        assert first[0].name == f"000000-{hashlib.sha256(first[0].data).hexdigest()}.part"

    def test_streams_large_file_without_buffering(self, tmp_path: Path) -> None:
        big = tmp_path / "sparse.bin"
        with big.open("wb") as handle:
            handle.truncate(64 * 1024 * 1024)
        plan = dirup.scan_directory(str(tmp_path), (), False, "/w")
        largest = max(len(chunk) for chunk, _ in dirup.iter_tar(plan, "/w"))
        assert largest <= dirup.READ_CHUNK_SIZE
        parts = list(dirup.iter_parts(plan, False, "/w"))
        assert [len(p.data) for p in parts[:-1]] == [dirup.PART_SIZE, dirup.PART_SIZE]

    def test_file_changed_during_upload(self, tmp_path: Path) -> None:
        (tmp_path / "grow.txt").write_text("abc")
        plan = dirup.scan_directory(str(tmp_path), (), False, "/w")
        (tmp_path / "grow.txt").write_text("abcdef")
        with pytest.raises(SandboxFileError) as exc_info:
            list(dirup.iter_parts(plan, False, "/w"))
        assert exc_info.value.metadata == {
            "operation": "upload_directory",
            "stage": "archive",
            "path": "grow.txt",
        }


class TestValidation:
    @pytest.mark.parametrize(
        ("remote", "error"),
        [("relative/x", ValueError), ("/", ValueError), ("/a/../b", ValueError), (3, TypeError)],
    )
    def test_bad_remote_path(self, tmp_path: Path, remote: Any, error: type[Exception]) -> None:
        sandbox = running_sandbox()
        with patch.object(sandbox, "_exec_streaming_binary_async") as exec_mock:
            with pytest.raises(error):
                sandbox.upload_directory(str(tmp_path), remote)
        exec_mock.assert_not_called()

    def test_bad_local_path_and_excludes(self, tmp_path: Path) -> None:
        sandbox = running_sandbox()
        (tmp_path / "file").write_text("x")
        with pytest.raises(FileNotFoundError):
            sandbox.upload_directory(str(tmp_path / "missing"), "/w")
        with pytest.raises(NotADirectoryError):
            sandbox.upload_directory(str(tmp_path / "file"), "/w")
        with pytest.raises(TypeError):
            sandbox.upload_directory(str(tmp_path), "/w", exclude="*.pyc")
        with pytest.raises(ValueError):
            sandbox.upload_directory(str(tmp_path), "/w", exclude=[""])

    def test_remote_path_normalized(self) -> None:
        assert dirup.normalize_remote_path("//w//x/") == "/w/x"


@needs_tools
class TestUploadEndToEndWithLocalShell:
    def test_upload_command_sequence_and_result(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        make_tree(local)
        dest.mkdir()
        (dest / "keep.txt").write_text("remote")
        progress: list[tuple[int, int]] = []
        remote = LocalRemote()

        result = upload(
            running_sandbox(),
            remote,
            local,
            dest,
            exclude=["*.pyc"],
            on_progress=lambda done, total: progress.append((done, total)),
            container="sidecar",
        )

        assert remote.names() == [
            "cwsandbox-upload-preflight",
            "cwsandbox-upload-part",
            "cwsandbox-upload-final",
        ]
        preflight, part, final = remote.calls
        plan = dirup.scan_directory(str(local), ("*.pyc",), True, str(dest))
        stage = f"{dest}/.cwsandbox-upload-{plan.key}"
        assert preflight[1] == [str(dest), stage, *dirup.BASE_TOOLS, "gzip"]
        assert preflight[3] is None and final[3] is None and part[3] is not None
        assert part[1][:3] == [stage, "000000", str(len(part[3] or b""))]
        assert final[1][:4] == [str(dest), stage, result.sha256, "z"]
        assert {call[2] for call in remote.calls} == {"sidecar"}

        expected = listing(local)
        del expected["skip.pyc"]
        got = listing(dest)
        assert got.pop("keep.txt") == ("f", 0o644, "remote")
        assert got == expected
        assert not any(name.startswith(".cwsandbox-upload-") for name in os.listdir(dest))
        assert result == DirectoryUploadResult(
            files=4,
            directories=3,
            symlinks=1,
            bytes=plan.total_bytes,
            archive_bytes=len(part[3] or b""),
            sha256=hashlib.sha256(part[3] or b"").hexdigest(),
            resumed_bytes=0,
            skipped=(),
        )
        assert progress[-1] == (plan.total_bytes, plan.total_bytes)

    def test_uncompressed_skips_gzip(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        make_tree(local)
        remote = LocalRemote()
        upload(running_sandbox(), remote, local, dest, compress=False)
        assert "gzip" not in remote.calls[0][1]
        assert remote.calls[-1][1][3] == "-"
        assert (dest / "pkg" / "sub" / "data.txt").read_text() == "data"

    def test_resume_after_interruption(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dirup, "PART_SIZE", 4096)
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "big.bin").write_bytes(os.urandom(20000))
        remote = LocalRemote()
        sandbox = running_sandbox()

        def interrupt(argv0: str, args: list[str], n: int) -> Any:
            if argv0 == "cwsandbox-upload-part" and args[1] == "000002":
                return SandboxNotRunningError("Sandbox sb-upload has been stopped")
            return None

        remote.inject = interrupt
        with pytest.raises(SandboxNotRunningError):
            upload(sandbox, remote, local, dest, compress=False)
        staged = [p for p in dest.iterdir() if p.name.startswith(".cwsandbox-upload-")]
        assert len(staged) == 1
        assert sorted(p.name[:6] for p in staged[0].glob("*.part")) == ["000000", "000001"]

        remote = LocalRemote()
        result = upload(sandbox, remote, local, dest, compress=False)
        sent = [call[1][1] for call in remote.calls if call[0] == "cwsandbox-upload-part"]
        assert "000000" not in sent and "000001" not in sent and sent[0] == "000002"
        assert result.resumed_bytes == 2 * 4096
        assert (dest / "big.bin").read_bytes() == (local / "big.bin").read_bytes()
        assert not staged[0].exists()

    def test_stale_part_with_other_checksum_is_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "a.txt").write_text("hello")
        plan = dirup.scan_directory(str(local), (), True, str(dest))
        stage = dest / f".cwsandbox-upload-{plan.key}"
        stage.mkdir(parents=True)
        (stage / f"000000-{'0' * 64}.part").write_bytes(b"stale")
        result = upload(running_sandbox(), LocalRemote(), local, dest)
        assert result.resumed_bytes == 0
        assert (dest / "a.txt").read_text() == "hello"

    def test_part_retried_after_transient_error(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "a.txt").write_text("hello")
        remote = LocalRemote()
        failures = [SandboxUnavailableError("runner unavailable"), (42, b"", b"checksum")]

        def flaky(argv0: str, args: list[str], n: int) -> Any:
            if argv0 == "cwsandbox-upload-part" and failures:
                return failures.pop(0)
            return None

        remote.inject = flaky
        upload(running_sandbox(), remote, local, dest)
        assert remote.names().count("cwsandbox-upload-part") == 3
        assert (dest / "a.txt").read_text() == "hello"

    def test_part_gives_up_after_three_attempts(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "a.txt").write_text("hello")
        remote = LocalRemote()
        remote.inject = lambda argv0, args, n: (
            SandboxUnavailableError("down") if argv0 == "cwsandbox-upload-part" else None
        )
        with pytest.raises(SandboxUnavailableError):
            upload(running_sandbox(), remote, local, dest)
        assert remote.names().count("cwsandbox-upload-part") == 3

    def test_part_write_failure_not_retried(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "a.txt").write_text("hello")
        remote = LocalRemote()
        remote.inject = lambda argv0, args, n: (
            (40, b"", b"failed to write part 000000") if argv0 == "cwsandbox-upload-part" else None
        )
        with pytest.raises(SandboxFileError) as exc_info:
            upload(running_sandbox(), remote, local, dest)
        assert exc_info.value.metadata["stage"] == "transfer"
        assert exc_info.value.metadata["exit_code"] == "40"
        assert remote.names().count("cwsandbox-upload-part") == 1

    def test_missing_tools(self, tmp_path: Path) -> None:
        local = tmp_path / "local"
        local.mkdir()
        remote = LocalRemote()
        remote.inject = lambda argv0, args, n: (20, b"", b"missing tools: tar gzip\n")
        with pytest.raises(SandboxFileError) as exc_info:
            upload(running_sandbox(), remote, local, tmp_path / "dest")
        assert exc_info.value.metadata["stage"] == "preflight"
        assert exc_info.value.metadata["missing_tools"] == "tar gzip"
        assert "tar gzip" in str(exc_info.value)
        assert remote.names() == ["cwsandbox-upload-preflight"]

    def test_not_writable_destination(self, tmp_path: Path) -> None:
        local = tmp_path / "local"
        local.mkdir()
        (tmp_path / "file").write_text("x")
        with pytest.raises(SandboxFileError) as exc_info:
            upload(running_sandbox(), LocalRemote(), local, tmp_path / "file" / "sub")
        assert exc_info.value.metadata["stage"] == "preflight"
        assert exc_info.value.metadata["exit_code"] == "21"

    def test_archive_checksum_mismatch_is_verify_error(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "a.txt").write_text("hello")
        remote = LocalRemote()

        def corrupt(argv0: str, args: list[str], n: int) -> Any:
            if argv0 == "cwsandbox-upload-final":
                args[2] = "0" * 64
                cmd = ["/bin/sh", "-c", dirup.FINAL_SCRIPT, argv0, *args]
                proc = subprocess.run(cmd, capture_output=True, check=False)
                return proc.returncode, proc.stdout, proc.stderr
            return None

        remote.inject = corrupt
        with pytest.raises(SandboxFileError) as exc_info:
            upload(running_sandbox(), remote, local, dest)
        assert exc_info.value.metadata["stage"] == "verify"
        assert not (dest / "a.txt").exists()
        assert not any(name.startswith(".cwsandbox-upload-") for name in os.listdir(dest))

    def test_extract_failure_keeps_staging(self, tmp_path: Path) -> None:
        local, dest = tmp_path / "local", tmp_path / "dest"
        local.mkdir()
        (local / "clash").write_text("a file where the sandbox has a non-empty directory")
        (dest / "clash").mkdir(parents=True)
        (dest / "clash" / "x").write_text("x")
        with pytest.raises(SandboxFileError) as exc_info:
            upload(running_sandbox(), LocalRemote(), local, dest)
        assert exc_info.value.metadata["stage"] == "extract"
        assert "may be partly updated" in str(exc_info.value)
        assert any(name.startswith(".cwsandbox-upload-") for name in os.listdir(dest))

    def test_overall_timeout(self, tmp_path: Path) -> None:
        local = tmp_path / "local"
        local.mkdir()
        remote = LocalRemote()

        def slow(argv0: str, args: list[str], n: int) -> Any:
            if argv0 == "cwsandbox-upload-preflight":
                import time

                time.sleep(0.05)
            return None

        remote.inject = slow
        with pytest.raises(SandboxTimeoutError):
            upload(running_sandbox(), remote, local, tmp_path / "dest", timeout_seconds=0.01)
        assert "cwsandbox-upload-part" not in remote.names()
