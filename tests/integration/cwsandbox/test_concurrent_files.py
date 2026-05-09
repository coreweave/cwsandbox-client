# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for concurrent file operations.

These tests exercise parallel file uploads to detect race conditions in the
backend's command ID handling. Related to:
https://coreweave.slack.com/archives/C0AE27R91B7/p1777510544430799

The backend bug: concurrent AddFile requests on the same sandbox can collide
because command_id is derived from container_id, causing overwrites in the
pending commands map. This manifests as timeouts (DeadlineExceeded) for
~N-1 of N concurrent requests.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pytest

from cwsandbox import Sandbox, SandboxDefaults

if TYPE_CHECKING:
    from cwsandbox._types import OperationRef


def test_concurrent_write_file_basic(sandbox_defaults: SandboxDefaults) -> None:
    """Test multiple concurrent write_file operations on the same sandbox.

    Issues 8 concurrent AddFile requests to detect command ID collision bugs.
    All files should be written successfully without timeouts.
    """
    num_files = 8

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        sandbox.wait()

        files = {
            f"/tmp/concurrent_test_{i}_{uuid.uuid4().hex[:8]}.txt": f"content_{i}".encode()
            for i in range(num_files)
        }

        refs = [sandbox.write_file(path, content) for path, content in files.items()]

        for ref in refs:
            ref.result(timeout=60.0)

        for path, expected_content in files.items():
            actual = sandbox.read_file(path).result(timeout=30.0)
            assert actual == expected_content, f"Content mismatch for {path}"


def test_concurrent_write_file_threaded(sandbox_defaults: SandboxDefaults) -> None:
    """Test concurrent write_file from multiple threads.

    Uses ThreadPoolExecutor to issue parallel write requests, simulating
    real-world concurrent usage patterns more closely than sequential dispatch.
    """
    num_files = 8

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        sandbox.wait()

        files = {
            f"/tmp/threaded_test_{i}_{uuid.uuid4().hex[:8]}.txt": f"threaded_content_{i}".encode()
            for i in range(num_files)
        }

        def write_and_verify(path: str, content: bytes) -> tuple[str, bool, str]:
            """Write file and return (path, success, error_msg)."""
            try:
                sandbox.write_file(path, content).result(timeout=60.0)
                actual = sandbox.read_file(path).result(timeout=30.0)
                if actual != content:
                    return path, False, f"Content mismatch: expected {content!r}, got {actual!r}"
                return path, True, ""
            except Exception as e:
                return path, False, str(e)

        with ThreadPoolExecutor(max_workers=num_files) as executor:
            futures = {
                executor.submit(write_and_verify, path, content): path
                for path, content in files.items()
            }

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        failures = [(path, msg) for path, success, msg in results if not success]
        assert not failures, f"File operations failed: {failures}"


def test_concurrent_read_write_interleaved(sandbox_defaults: SandboxDefaults) -> None:
    """Test interleaved read and write operations.

    First writes all files concurrently, then reads them all concurrently.
    This tests both AddFile and GetFile command ID handling.
    """
    num_files = 8

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        sandbox.wait()

        files = {
            f"/tmp/interleaved_test_{i}_{uuid.uuid4().hex[:8]}.txt": f"interleaved_{i}".encode()
            for i in range(num_files)
        }

        write_refs = [sandbox.write_file(path, content) for path, content in files.items()]
        for write_ref in write_refs:
            write_ref.result(timeout=60.0)

        read_refs: dict[str, OperationRef[bytes]] = {
            path: sandbox.read_file(path) for path in files
        }
        for path, read_ref in read_refs.items():
            actual = read_ref.result(timeout=30.0)
            expected = files[path]
            assert actual == expected, f"Content mismatch for {path}"


def test_concurrent_write_large_files(sandbox_defaults: SandboxDefaults) -> None:
    """Test concurrent writes with larger file sizes.

    Uses 64KB files to exercise chunked transfer paths under concurrency.
    """
    num_files = 4
    file_size = 64 * 1024  # 64KB each

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        sandbox.wait()

        files = {
            f"/tmp/large_file_{i}_{uuid.uuid4().hex[:8]}.bin": bytes([i % 256] * file_size)
            for i in range(num_files)
        }

        refs = [sandbox.write_file(path, content) for path, content in files.items()]

        for ref in refs:
            ref.result(timeout=120.0)

        for path, expected_content in files.items():
            actual = sandbox.read_file(path).result(timeout=60.0)
            assert len(actual) == len(expected_content), f"Size mismatch for {path}"
            assert actual == expected_content, f"Content mismatch for {path}"


@pytest.mark.asyncio
async def test_concurrent_write_file_async(sandbox_defaults: SandboxDefaults) -> None:
    """Test concurrent write_file using async/await pattern.

    Verifies the async API handles concurrent operations correctly.
    """
    import asyncio

    num_files = 8

    async with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        sandbox.wait()

        files = {
            f"/tmp/async_test_{i}_{uuid.uuid4().hex[:8]}.txt": f"async_content_{i}".encode()
            for i in range(num_files)
        }

        write_tasks = [sandbox.write_file(path, content) for path, content in files.items()]
        await asyncio.gather(*write_tasks)

        read_tasks = [sandbox.read_file(path) for path in files]
        results = await asyncio.gather(*read_tasks)

        for (path, expected), actual in zip(files.items(), results, strict=True):
            assert actual == expected, f"Content mismatch for {path}"
