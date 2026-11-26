# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Local half of ``Sandbox.upload_directory``: walk, archive, parts, scripts.

The archive must be byte-identical across calls on an unchanged tree so that a
later call can resume from parts an earlier call already staged. Everything
here is deterministic: sorted walk order, integer mtimes, zeroed ownership and
a gzip stream with no name and mtime 0. The remote scripts and exit codes are
shared verbatim with the JavaScript SDK.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import re
import stat
import tarfile
import zlib
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass

from cwsandbox.exceptions import (
    CWSandboxValidationError,
    SandboxError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotFoundError,
    SandboxNotRunningError,
    SandboxTerminatedError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)

PART_SIZE = 32 * 1024 * 1024
STAGING_PREFIX = ".cwsandbox-upload-"
MANIFEST_HEADER = "cwsandbox-directory-upload/1"
READ_CHUNK_SIZE = 1024 * 1024
GZIP_LEVEL = 6
TAR_RECORD_SIZE = 10240

PREFLIGHT_TIMEOUT_SECONDS = 60.0
PART_ATTEMPT_TIMEOUT_SECONDS = 300.0
FINAL_TIMEOUT_SECONDS = 1800.0
PART_MAX_ATTEMPTS = 3
PART_RETRY_DELAYS_SECONDS = (1.0, 2.0)

BASE_TOOLS = ("tar", "sha256sum", "head", "wc", "cat", "mv", "rm", "mkdir")

EXIT_MISSING_TOOLS = 20
EXIT_NOT_WRITABLE = 21
EXIT_MISSING_PART = 30
EXIT_ARCHIVE_CHECKSUM = 31
EXIT_TAR_FAILED = 32
EXIT_PART_WRITE = 40
EXIT_PART_SIZE = 41
EXIT_PART_CHECKSUM = 42

PART_NAME_RE = re.compile(r"^[0-9]{6}-[0-9a-f]{64}\.part$")

PREFLIGHT_SCRIPT = """dest=$1
stage=$2
shift 2
missing=
for tool in "$@"; do
  command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
  printf 'missing tools:%s\\n' "$missing" >&2
  exit 20
fi
if ! mkdir -p -- "$dest" "$stage" 2>/dev/null || [ ! -d "$stage" ] || [ ! -w "$stage" ]; then
  printf 'not a writable directory: %s\\n' "$dest" >&2
  exit 21
fi
if command -v find >/dev/null 2>&1; then
  find "$dest" -maxdepth 1 -type d -name '.cwsandbox-upload-*' ! -name "${stage##*/}" \\
    -mmin +1440 -exec rm -rf -- {} + 2>/dev/null
fi
touch -- "$stage" 2>/dev/null
for part in "$stage"/*.part; do
  [ -f "$part" ] && printf '%s\\n' "${part##*/}"
done
exit 0
"""

PART_SCRIPT = """stage=$1
index=$2
size=$3
sum=$4
tmp="$stage/.tmp-$index-$$"
trap 'rm -f -- "$tmp"' EXIT
for old in "$stage/$index"-*.part; do
  [ "$old" = "$stage/$index-$sum.part" ] || rm -f -- "$old"
done
if ! head -c "$size" > "$tmp"; then
  printf 'failed to write part %s\\n' "$index" >&2
  exit 40
fi
actual=$(wc -c < "$tmp") || exit 40
set -- $actual
if [ "$1" != "$size" ]; then
  printf 'part %s: expected %s bytes, received %s\\n' "$index" "$size" "$1" >&2
  exit 41
fi
got=$(sha256sum < "$tmp") || exit 42
got=${got%% *}
if [ "$got" != "$sum" ]; then
  printf 'part %s: checksum mismatch\\n' "$index" >&2
  exit 42
fi
if ! mv -f -- "$tmp" "$stage/$index-$sum.part"; then
  printf 'failed to commit part %s\\n' "$index" >&2
  exit 40
fi
exit 0
"""

FINAL_SCRIPT = """dest=$1
stage=$2
sum=$3
z=$4
shift 4
cd -- "$stage" || exit 30
for part in "$@"; do
  if [ ! -f "$part" ]; then
    printf 'missing staged part: %s\\n' "$part" >&2
    exit 30
  fi
done
got=$(cat -- "$@" | sha256sum) || exit 31
got=${got%% *}
if [ "$got" != "$sum" ]; then
  printf 'archive checksum mismatch: expected %s, got %s\\n' "$sum" "$got" >&2
  cd / && rm -rf -- "$stage"
  exit 31
fi
if [ "$z" = z ]; then flags=-xzof; else flags=-xof; fi
if ! cat -- "$@" | tar "$flags" - -C "$dest"; then
  printf 'tar failed extracting into %s\\n' "$dest" >&2
  exit 32
fi
cd / && rm -rf -- "$stage"
exit 0
"""


@dataclass(frozen=True)
class UploadEntry:
    """One archived path: ``kind`` is ``d`` (directory), ``f`` (file) or ``l`` (symlink)."""

    kind: str
    name: str
    mode: int
    size: int
    mtime: int
    target: str
    path: str


@dataclass(frozen=True)
class UploadPlan:
    """Result of the local walk; ``key`` names the remote staging directory."""

    entries: tuple[UploadEntry, ...]
    skipped: tuple[str, ...]
    total_bytes: int
    key: str


def upload_error(message: str, stage: str, remote_path: str, **metadata: str) -> SandboxFileError:
    """Build the single error type ``upload_directory`` raises for its own failures."""
    return SandboxFileError(
        message,
        filepath=remote_path,
        metadata={"operation": "upload_directory", "stage": stage, **metadata},
    )


def timeout_error(remote_path: str) -> SandboxTimeoutError:
    return SandboxTimeoutError(
        f"Upload directory to '{remote_path}' timed out; staged parts are kept and the "
        "next call resumes"
    )


def normalize_remote_path(remote_path: object) -> str:
    """Validate and normalize an absolute sandbox directory path."""
    if not isinstance(remote_path, str):
        raise TypeError("remote_path must be a string")
    if not remote_path.startswith("/") or "\x00" in remote_path:
        raise ValueError(f"remote_path must be an absolute path, got {remote_path!r}")
    parts = [part for part in remote_path.split("/") if part]
    if ".." in parts:
        raise ValueError(f"remote_path must not contain '..', got {remote_path!r}")
    if not parts:
        raise ValueError("remote_path must not be the filesystem root")
    return "/" + "/".join(parts)


def normalize_excludes(exclude: Sequence[str] | None) -> tuple[str, ...]:
    """Validate ``exclude`` and strip leading/trailing slashes from each pattern."""
    if exclude is None:
        return ()
    if isinstance(exclude, (str, bytes)):
        raise TypeError("exclude must be a list of glob patterns, not a single string")
    patterns: list[str] = []
    for pattern in exclude:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("exclude patterns must be non-empty strings")
        stripped = pattern.strip("/")
        if not stripped:
            raise ValueError(f"exclude pattern {pattern!r} matches nothing")
        patterns.append(stripped)
    return tuple(patterns)


def is_excluded(name: str, relative: str, patterns: Sequence[str]) -> bool:
    """Match a basename pattern against ``name`` and a path pattern against ``relative``."""
    for pattern in patterns:
        candidate = relative if "/" in pattern else name
        if fnmatch.fnmatchcase(candidate, pattern):
            return True
    return False


def symlink_escapes(relative: str, target: str) -> bool:
    """Whether a link target is absolute or resolves (lexically) outside the tree."""
    if not target or target.startswith("/"):
        return True
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(relative), target))
    return resolved == ".." or resolved.startswith("../")


def manifest_bytes(entries: Sequence[UploadEntry], compress: bool) -> bytes:
    """Canonical walk manifest; its hash is the resume key (shared with the JS SDK)."""
    lines = [MANIFEST_HEADER, "gzip" if compress else "tar"]
    for entry in entries:
        name = entry.name.encode("utf-8")
        target = entry.target.encode("utf-8")
        lines.append(
            f"{entry.kind} {entry.mode:o} {entry.size} {entry.mtime} "
            f"{len(name)}:{entry.name} {len(target)}:{entry.target}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def content_key(entries: Sequence[UploadEntry], compress: bool) -> str:
    return hashlib.sha256(manifest_bytes(entries, compress)).hexdigest()[:32]


def _utf8_key(name: str) -> bytes:
    return name.encode("utf-8", "surrogateescape")


def _entry_mode(st: os.stat_result, kind: str) -> int:
    if os.name == "nt":
        return 0o755 if kind == "d" else (0o777 if kind == "l" else 0o644)
    return stat.S_IMODE(st.st_mode) & 0o777


def scan_directory(
    local_path: str, patterns: Sequence[str], compress: bool, remote_path: str
) -> UploadPlan:
    """Walk ``local_path`` into a deterministic, pre-ordered entry list."""
    root = os.path.realpath(local_path)
    entries: list[UploadEntry] = []
    skipped: list[str] = []
    total = 0
    stack: list[tuple[str, str]] = [(root, "")]
    while stack:
        directory, prefix = stack.pop()
        try:
            names = os.listdir(directory)
        except OSError as e:
            raise upload_error(
                f"Cannot list local directory '{directory}': {e}",
                "archive",
                remote_path,
                path=prefix or ".",
            ) from e
        children: list[tuple[str, str]] = []
        for name in sorted(names, key=_utf8_key):
            relative = f"{prefix}/{name}" if prefix else name
            try:
                name.encode("utf-8")
            except UnicodeEncodeError:
                raise upload_error(
                    f"Local path name is not valid UTF-8: {relative!r}",
                    "archive",
                    remote_path,
                    path=relative.encode("utf-8", "surrogateescape").decode("utf-8", "replace"),
                ) from None
            if is_excluded(name, relative, patterns):
                continue
            full = os.path.join(directory, name)
            try:
                st = os.lstat(full)
            except OSError as e:
                raise upload_error(
                    f"Cannot read local path '{relative}': {e}",
                    "archive",
                    remote_path,
                    path=relative,
                ) from e
            mtime = int(st.st_mtime)
            if stat.S_ISDIR(st.st_mode):
                entries.append(UploadEntry("d", relative, _entry_mode(st, "d"), 0, mtime, "", full))
                children.append((full, relative))
            elif stat.S_ISREG(st.st_mode):
                entries.append(
                    UploadEntry("f", relative, _entry_mode(st, "f"), st.st_size, mtime, "", full)
                )
                total += st.st_size
            elif stat.S_ISLNK(st.st_mode):
                target = os.readlink(full)
                if symlink_escapes(relative, target):
                    raise upload_error(
                        f"Symbolic link '{relative}' -> '{target}' points outside the uploaded "
                        "directory; remove it or add it to exclude",
                        "archive",
                        remote_path,
                        path=relative,
                    )
                entries.append(
                    UploadEntry("l", relative, _entry_mode(st, "l"), 0, mtime, target, full)
                )
            else:
                skipped.append(relative)
        stack.extend(children)
    # Sorting by path components gives pre-order (a directory before its
    # children) with siblings in UTF-8 byte order, whatever order we walked in.
    entries.sort(key=lambda entry: [_utf8_key(part) for part in entry.name.split("/")])
    skipped.sort(key=lambda name: [_utf8_key(part) for part in name.split("/")])
    return UploadPlan(
        entries=tuple(entries),
        skipped=tuple(skipped),
        total_bytes=total,
        key=content_key(entries, compress),
    )


def _tar_header(entry: UploadEntry) -> bytes:
    info = tarfile.TarInfo(entry.name)
    info.mode = entry.mode
    info.mtime = entry.mtime
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if entry.kind == "d":
        info.type = tarfile.DIRTYPE
    elif entry.kind == "l":
        info.type = tarfile.SYMTYPE
        info.linkname = entry.target
    else:
        info.type = tarfile.REGTYPE
        info.size = entry.size
    return info.tobuf(tarfile.PAX_FORMAT, "utf-8", "strict")


def _changed(entry: UploadEntry, remote_path: str) -> SandboxFileError:
    return upload_error(
        f"Local file '{entry.name}' changed during upload; retry when it is stable",
        "archive",
        remote_path,
        path=entry.name,
    )


def iter_tar(plan: UploadPlan, remote_path: str) -> Iterator[tuple[bytes, int]]:
    """Yield ``(tar bytes, source file bytes they carry)`` without buffering whole files."""
    written = 0
    for entry in plan.entries:
        header = _tar_header(entry)
        written += len(header)
        yield header, 0
        if entry.kind != "f":
            continue
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(entry.path, flags)
        except OSError as e:
            raise _changed(entry, remote_path) from e
        with os.fdopen(fd, "rb") as stream:
            st = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(st.st_mode)
                or st.st_size != entry.size
                or int(st.st_mtime) != entry.mtime
            ):
                raise _changed(entry, remote_path)
            remaining = entry.size
            while remaining:
                data = stream.read(min(READ_CHUNK_SIZE, remaining))
                if not data:
                    raise _changed(entry, remote_path)
                remaining -= len(data)
                written += len(data)
                yield data, len(data)
        padding = -entry.size % tarfile.BLOCKSIZE
        if padding:
            written += padding
            yield b"\0" * padding, 0
    trailer = 2 * tarfile.BLOCKSIZE
    trailer += -(written + trailer) % TAR_RECORD_SIZE
    yield b"\0" * trailer, 0


def iter_archive(plan: UploadPlan, compress: bool, remote_path: str) -> Iterator[tuple[bytes, int]]:
    """Tar stream, gzip-compressed (level 6, no name, mtime 0) when ``compress``."""
    if not compress:
        yield from iter_tar(plan, remote_path)
        return
    compressor = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 31)
    for data, consumed in iter_tar(plan, remote_path):
        out = compressor.compress(data)
        if out or consumed:
            yield out, consumed
    yield compressor.flush(), 0


@dataclass(frozen=True)
class ArchivePart:
    index: int
    data: bytes
    sha256: str
    source_bytes_done: int

    @property
    def name(self) -> str:
        return part_name(self.index, self.sha256)


def part_name(index: int, sha256: str) -> str:
    return f"{index:06d}-{sha256}.part"


def iter_parts(
    plan: UploadPlan, compress: bool, remote_path: str, part_size: int | None = None
) -> Generator[ArchivePart, None, None]:
    """Cut the archive into ``PART_SIZE`` slices; always yields at least one part."""
    part_size = part_size or PART_SIZE
    buffer = bytearray()
    consumed = 0
    index = 0
    for data, source_bytes in iter_archive(plan, compress, remote_path):
        buffer += data
        consumed += source_bytes
        while len(buffer) >= part_size:
            chunk = bytes(buffer[:part_size])
            del buffer[:part_size]
            yield ArchivePart(index, chunk, hashlib.sha256(chunk).hexdigest(), consumed)
            index += 1
    if buffer or index == 0:
        chunk = bytes(buffer)
        yield ArchivePart(index, chunk, hashlib.sha256(chunk).hexdigest(), consumed)


def required_tools(compress: bool) -> list[str]:
    return [*BASE_TOOLS, "gzip"] if compress else list(BASE_TOOLS)


def is_retryable_part_error(error: BaseException) -> bool:
    """Whether a failed part attempt is worth sending again."""
    if not isinstance(error, SandboxError):
        return False
    if isinstance(
        error,
        (
            SandboxNotFoundError,
            SandboxTerminatedError,
            SandboxFailedError,
            CWSandboxValidationError,
        ),
    ):
        return False
    # SandboxUnavailableError subclasses SandboxNotRunningError but is transient.
    return not (
        isinstance(error, SandboxNotRunningError) and not isinstance(error, SandboxUnavailableError)
    )
