# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox.results() and cwsandbox.wait() utility functions."""

from __future__ import annotations

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cwsandbox
from cwsandbox import OperationRef, Process, Sandbox, results, wait
from cwsandbox._types import ProcessResult, StreamReader


class TestResults:
    """Tests for cwsandbox.results() function."""

    def test_results_single_ref_returns_result(self) -> None:
        """results() with single OperationRef returns the result."""
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        future.set_result("test_value")
        ref: OperationRef[str] = OperationRef(future)

        value = results(ref)

        assert value == "test_value"

    def test_results_single_ref_propagates_exception(self) -> None:
        """results() with single OperationRef propagates exceptions."""
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        future.set_exception(ValueError("test error"))
        ref: OperationRef[str] = OperationRef(future)

        with pytest.raises(ValueError, match="test error"):
            results(ref)

    def test_results_list_returns_all_results(self) -> None:
        """results() with list of OperationRefs returns list of results."""
        futures = []
        refs = []
        for i in range(3):
            future: concurrent.futures.Future[int] = concurrent.futures.Future()
            future.set_result(i * 10)
            futures.append(future)
            refs.append(OperationRef(future))

        all_results = results(refs)

        assert all_results == [0, 10, 20]

    def test_results_empty_list_returns_empty_list(self) -> None:
        """results() with empty list returns empty list."""
        value = results([])

        assert value == []

    def test_results_list_propagates_first_exception(self) -> None:
        """results() with list propagates exception from first failing ref."""
        future1: concurrent.futures.Future[int] = concurrent.futures.Future()
        future1.set_result(1)
        future2: concurrent.futures.Future[int] = concurrent.futures.Future()
        future2.set_exception(RuntimeError("failed"))
        refs = [OperationRef(future1), OperationRef(future2)]

        with pytest.raises(RuntimeError, match="failed"):
            results(refs)


class TestWait:
    """Tests for cwsandbox.wait() function."""

    def test_wait_empty_list_returns_empty_tuples(self) -> None:
        """wait() with empty list returns ([], [])."""
        done, pending = wait([])

        assert done == []
        assert pending == []

    @patch("cwsandbox._LoopManager.get")
    def test_wait_operation_refs_all_complete(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() for OperationRefs waits until all complete."""
        # Create completed futures
        future1: concurrent.futures.Future[str] = concurrent.futures.Future()
        future1.set_result("a")
        future2: concurrent.futures.Future[str] = concurrent.futures.Future()
        future2.set_result("b")
        refs = [OperationRef(future1), OperationRef(future2)]

        # Mock the loop manager
        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait(refs)

        assert len(done) == 2
        assert len(pending) == 0
        assert set(done) == set(refs)

    @patch("cwsandbox._LoopManager.get")
    def test_wait_processes_all_complete(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() for Processes waits until all complete."""
        # Create completed process futures
        result1 = ProcessResult(stdout="out1", stderr="", returncode=0)
        result2 = ProcessResult(stdout="out2", stderr="", returncode=0)

        future1: concurrent.futures.Future[ProcessResult] = concurrent.futures.Future()
        future1.set_result(result1)
        future2: concurrent.futures.Future[ProcessResult] = concurrent.futures.Future()
        future2.set_result(result2)

        # Create mock StreamReaders
        mock_loop_manager = MagicMock()
        stdout1 = StreamReader(asyncio.Queue(), mock_loop_manager)
        stderr1 = StreamReader(asyncio.Queue(), mock_loop_manager)
        stdout2 = StreamReader(asyncio.Queue(), mock_loop_manager)
        stderr2 = StreamReader(asyncio.Queue(), mock_loop_manager)

        proc1 = Process(future1, ["cmd1"], stdout1, stderr1)
        proc2 = Process(future2, ["cmd2"], stdout2, stderr2)
        procs = [proc1, proc2]

        # Mock the loop manager
        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait(procs)

        assert len(done) == 2
        assert len(pending) == 0
        assert set(done) == set(procs)

    @patch("cwsandbox._LoopManager.get")
    def test_wait_sandboxes_waits_for_running(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() for Sandboxes waits until RUNNING status."""
        # Create mock sandboxes
        sandbox1 = MagicMock(spec=Sandbox)
        sandbox2 = MagicMock(spec=Sandbox)

        # Make _wait_until_running_async return immediately
        sandbox1._wait_until_running_async = AsyncMock(return_value=None)
        sandbox2._wait_until_running_async = AsyncMock(return_value=None)

        sandboxes = [sandbox1, sandbox2]

        # Mock the loop manager
        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait(sandboxes)

        assert len(done) == 2
        assert len(pending) == 0
        sandbox1._wait_until_running_async.assert_called_once()
        sandbox2._wait_until_running_async.assert_called_once()

    @patch("cwsandbox._LoopManager.get")
    def test_wait_num_returns_partial(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() with num_returns returns after that many complete."""
        # Create futures with different completion states
        future1: concurrent.futures.Future[int] = concurrent.futures.Future()
        future1.set_result(1)  # Completed
        future2: concurrent.futures.Future[int] = concurrent.futures.Future()
        future2.set_result(2)  # Completed
        future3: concurrent.futures.Future[int] = concurrent.futures.Future()
        future3.set_result(3)  # Completed

        refs = [OperationRef(future1), OperationRef(future2), OperationRef(future3)]

        # Mock the loop manager
        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait(refs, num_returns=2)

        assert len(done) == 2
        assert len(pending) == 1
        # All original refs should be in done or pending
        assert set(done) | set(pending) == set(refs)

    @patch("cwsandbox._LoopManager.get")
    def test_wait_mixed_types(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() handles mixed waitable types."""
        # Create an OperationRef
        future1: concurrent.futures.Future[str] = concurrent.futures.Future()
        future1.set_result("ref_result")
        ref = OperationRef(future1)

        # Create a Process
        result = ProcessResult(stdout="proc_out", stderr="", returncode=0)
        future2: concurrent.futures.Future[ProcessResult] = concurrent.futures.Future()
        future2.set_result(result)
        mock_lm = MagicMock()
        stdout = StreamReader(asyncio.Queue(), mock_lm)
        stderr = StreamReader(asyncio.Queue(), mock_lm)
        proc = Process(future2, ["cmd"], stdout, stderr)

        # Create a Sandbox mock
        sandbox = MagicMock(spec=Sandbox)
        sandbox._wait_until_running_async = AsyncMock(return_value=None)

        waitables: list[cwsandbox.Waitable] = [ref, proc, sandbox]

        # Mock the loop manager
        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait(waitables)

        assert len(done) == 3
        assert len(pending) == 0
        assert set(done) == set(waitables)

    def test_wait_invalid_num_returns_raises_valueerror(self) -> None:
        """wait() raises ValueError for num_returns < 1."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(1)
        ref: OperationRef[int] = OperationRef(future)

        with pytest.raises(ValueError, match="num_returns must be at least 1"):
            wait([ref], num_returns=0)

        with pytest.raises(ValueError, match="num_returns must be at least 1"):
            wait([ref], num_returns=-1)

    def test_wait_invalid_num_returns_with_empty_list_raises_valueerror(self) -> None:
        """wait() raises ValueError for num_returns < 1 even with empty list."""
        with pytest.raises(ValueError, match="num_returns must be at least 1"):
            wait([], num_returns=0)


class TestWaitTimeout:
    """Tests for cwsandbox.wait() timeout behavior."""

    @patch("cwsandbox._LoopManager.get")
    def test_wait_timeout_with_num_returns_respects_deadline(
        self, mock_get_loop_manager: MagicMock
    ) -> None:
        """wait() with num_returns and timeout respects total deadline across rounds.

        This is a regression test for the deadline-tracking fix. Previously, each
        round of waiting would use the full timeout value instead of the remaining
        time until the deadline.
        """
        # Create 3 futures, none of which complete
        futures = [concurrent.futures.Future() for _ in range(3)]
        refs = [OperationRef(f) for f in futures]

        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        import time

        start = time.monotonic()
        done, pending = wait(refs, num_returns=2, timeout=0.1)
        elapsed = time.monotonic() - start

        # Should return within 0.2s (timeout + some overhead), not 0.3s+
        # (which would happen if each round used the full timeout)
        assert elapsed < 0.25, f"Expected <0.25s but took {elapsed:.2f}s"
        assert len(done) == 0
        assert len(pending) == 3

    @patch("cwsandbox._LoopManager.get")
    def test_wait_timeout_returns_pending(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() with timeout returns pending items when timeout expires."""
        # Create a future that never completes
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        ref: OperationRef[str] = OperationRef(future)

        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait([ref], timeout=0.01)

        assert len(done) == 0
        assert len(pending) == 1
        assert ref in pending

    @patch("cwsandbox._LoopManager.get")
    def test_wait_timeout_partial_completion(self, mock_get_loop_manager: MagicMock) -> None:
        """wait() with timeout returns completed items even when some time out."""
        # One completed, one pending
        future1: concurrent.futures.Future[int] = concurrent.futures.Future()
        future1.set_result(1)
        future2: concurrent.futures.Future[int] = concurrent.futures.Future()  # Never completes

        refs = [OperationRef(future1), OperationRef(future2)]

        mock_manager = MagicMock()
        mock_manager.run_sync.side_effect = asyncio.run
        mock_get_loop_manager.return_value = mock_manager

        done, pending = wait(refs, timeout=0.01)

        assert len(done) == 1
        assert len(pending) == 1


class TestExports:
    """Tests for module exports."""

    def test_results_exported(self) -> None:
        """results function is exported from cwsandbox module."""
        assert hasattr(cwsandbox, "results")
        assert cwsandbox.results is results

    def test_wait_exported(self) -> None:
        """wait function is exported from cwsandbox module."""
        assert hasattr(cwsandbox, "wait")
        assert cwsandbox.wait is wait

    def test_waitable_exported(self) -> None:
        """Waitable type alias is exported from cwsandbox module."""
        assert hasattr(cwsandbox, "Waitable")

    def test_results_in_all(self) -> None:
        """results is in __all__."""
        assert "results" in cwsandbox.__all__

    def test_wait_in_all(self) -> None:
        """wait is in __all__."""
        assert "wait" in cwsandbox.__all__

    def test_waitable_in_all(self) -> None:
        """Waitable is in __all__."""
        assert "Waitable" in cwsandbox.__all__
