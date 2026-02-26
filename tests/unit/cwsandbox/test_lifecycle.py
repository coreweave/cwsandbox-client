# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for _LifecycleState types and transition helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from cwsandbox._sandbox import (
    SandboxStatus,
    _lifecycle_state_from_info,
    _LifecycleState,
    _NotStarted,
    _Running,
    _Starting,
    _Terminal,
)

if TYPE_CHECKING:
    from cwsandbox import Sandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proto_info(
    *,
    sandbox_id: str = "sb-test",
    sandbox_status: int = 2,  # SANDBOX_STATUS_RUNNING
    tower_id: str = "",
    runway_id: str = "",
    tower_group_id: str = "",
    started_at_time: object | None = None,
    returncode: int = 0,
) -> SimpleNamespace:
    """Build a lightweight stand-in for a protobuf sandbox info message."""
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        sandbox_status=sandbox_status,
        tower_id=tower_id,
        runway_id=runway_id,
        tower_group_id=tower_group_id,
        started_at_time=started_at_time,
        returncode=returncode,
    )


def _proto_status(status: SandboxStatus) -> int:
    """Convert SandboxStatus to the proto int value."""
    return status.to_proto()


def _make_started_at() -> MagicMock:
    """Build a mock started_at_time with ToDatetime()."""
    ts = MagicMock()
    ts.ToDatetime.return_value = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    return ts


# ---------------------------------------------------------------------------
# State type immutability
# ---------------------------------------------------------------------------


class TestStateImmutability:
    """Verify each state variant is frozen."""

    def test_not_started_is_frozen(self) -> None:
        state = _NotStarted()
        with pytest.raises(AttributeError):
            state.cancelled = True  # type: ignore[misc]

    def test_starting_is_frozen(self) -> None:
        state = _Starting(sandbox_id="sb-1")
        with pytest.raises(AttributeError):
            state.sandbox_id = "sb-2"  # type: ignore[misc]

    def test_running_is_frozen(self) -> None:
        state = _Running(sandbox_id="sb-1")
        with pytest.raises(AttributeError):
            state.status = SandboxStatus.PAUSED  # type: ignore[misc]

    def test_terminal_is_frozen(self) -> None:
        state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)
        with pytest.raises(AttributeError):
            state.returncode = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# State defaults
# ---------------------------------------------------------------------------


class TestStateDefaults:
    """Verify default field values."""

    def test_not_started_defaults(self) -> None:
        state = _NotStarted()
        assert state.cancelled is False

    def test_starting_defaults(self) -> None:
        state = _Starting(sandbox_id="sb-1")
        assert state.status == SandboxStatus.PENDING

    def test_running_defaults(self) -> None:
        state = _Running(sandbox_id="sb-1")
        assert state.status == SandboxStatus.RUNNING
        assert state.tower_id is None
        assert state.runway_id is None
        assert state.tower_group_id is None
        assert state.started_at is None

    def test_terminal_defaults(self) -> None:
        state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.FAILED)
        assert state.returncode is None
        assert state.tower_id is None


# ---------------------------------------------------------------------------
# _is_done property
# ---------------------------------------------------------------------------


class TestIsDone:
    """Test _is_done helper on Sandbox instances."""

    def test_terminal_is_done(self) -> None:
        from cwsandbox import Sandbox

        sb = Sandbox.__new__(Sandbox)
        sb._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)
        assert sb._is_done is True

    def test_cancelled_not_started_is_done(self) -> None:
        from cwsandbox import Sandbox

        sb = Sandbox.__new__(Sandbox)
        sb._state = _NotStarted(cancelled=True)
        assert sb._is_done is True

    def test_not_started_is_not_done(self) -> None:
        from cwsandbox import Sandbox

        sb = Sandbox.__new__(Sandbox)
        sb._state = _NotStarted()
        assert sb._is_done is False

    def test_starting_is_not_done(self) -> None:
        from cwsandbox import Sandbox

        sb = Sandbox.__new__(Sandbox)
        sb._state = _Starting(sandbox_id="sb-1")
        assert sb._is_done is False

    def test_running_is_not_done(self) -> None:
        from cwsandbox import Sandbox

        sb = Sandbox.__new__(Sandbox)
        sb._state = _Running(sandbox_id="sb-1")
        assert sb._is_done is False


# ---------------------------------------------------------------------------
# _lifecycle_state_from_info
# ---------------------------------------------------------------------------


class TestLifecycleStateFromInfo:
    """Test the standalone state builder function."""

    def test_running_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.RUNNING,
            tower_id="tower-1",
        )
        assert isinstance(state, _Running)
        assert state.sandbox_id == "sb-1"
        assert state.tower_id == "tower-1"

    def test_paused_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.PAUSED,
        )
        assert isinstance(state, _Running)
        assert state.status == SandboxStatus.PAUSED

    def test_completed_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.COMPLETED,
            returncode=0,
        )
        assert isinstance(state, _Terminal)
        assert state.returncode == 0

    def test_failed_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.FAILED,
        )
        assert isinstance(state, _Terminal)
        assert state.status == SandboxStatus.FAILED

    def test_terminated_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.TERMINATED,
        )
        assert isinstance(state, _Terminal)
        assert state.status == SandboxStatus.TERMINATED

    def test_pending_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.PENDING,
        )
        assert isinstance(state, _Starting)
        assert state.status == SandboxStatus.PENDING

    def test_creating_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.CREATING,
        )
        assert isinstance(state, _Starting)
        assert state.status == SandboxStatus.CREATING

    def test_unspecified_status(self) -> None:
        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.UNSPECIFIED,
        )
        assert isinstance(state, _Starting)


# ---------------------------------------------------------------------------
# _apply_sandbox_info transitions
# ---------------------------------------------------------------------------


class TestApplySandboxInfo:
    """Test the Sandbox._apply_sandbox_info transition method."""

    def _make_sandbox(self, state: _LifecycleState) -> Sandbox:
        from cwsandbox import Sandbox

        sb = Sandbox.__new__(Sandbox)
        sb._state = state
        return sb

    def test_not_started_to_starting(self) -> None:
        sb = self._make_sandbox(_NotStarted())
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(SandboxStatus.PENDING),
        )
        new_state = sb._apply_sandbox_info(info)
        assert isinstance(new_state, _Starting)
        assert new_state.sandbox_id == "sb-1"

    def test_starting_to_running(self) -> None:
        sb = self._make_sandbox(_Starting(sandbox_id="sb-1"))
        ts = _make_started_at()
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.RUNNING),
            tower_id="tower-1",
            runway_id="runway-1",
            tower_group_id="tg-1",
            started_at_time=ts,
        )
        new_state = sb._apply_sandbox_info(info)
        assert isinstance(new_state, _Running)
        assert new_state.sandbox_id == "sb-1"
        assert new_state.tower_id == "tower-1"
        assert new_state.runway_id == "runway-1"
        assert new_state.tower_group_id == "tg-1"
        assert new_state.started_at is not None

    def test_running_to_completed_with_poll_sets_returncode(self) -> None:
        sb = self._make_sandbox(_Running(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.COMPLETED),
            returncode=0,
        )
        new_state = sb._apply_sandbox_info(info, source="poll")
        assert isinstance(new_state, _Terminal)
        assert new_state.status == SandboxStatus.COMPLETED
        assert new_state.returncode == 0

    def test_running_to_completed_with_query_omits_returncode(self) -> None:
        sb = self._make_sandbox(_Running(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.COMPLETED),
            returncode=42,
        )
        new_state = sb._apply_sandbox_info(info, source="query")
        assert isinstance(new_state, _Terminal)
        assert new_state.returncode is None

    def test_running_to_failed(self) -> None:
        sb = self._make_sandbox(_Running(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.FAILED),
        )
        new_state = sb._apply_sandbox_info(info, source="poll")
        assert isinstance(new_state, _Terminal)
        assert new_state.status == SandboxStatus.FAILED
        assert new_state.returncode is None

    def test_running_to_terminated(self) -> None:
        sb = self._make_sandbox(_Running(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.TERMINATED),
        )
        new_state = sb._apply_sandbox_info(info)
        assert isinstance(new_state, _Terminal)
        assert new_state.status == SandboxStatus.TERMINATED
        assert new_state.returncode is None

    # Guards: terminal and cancelled states never regress

    def test_terminal_stays_terminal(self) -> None:
        terminal = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED, returncode=0)
        sb = self._make_sandbox(terminal)
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.RUNNING),
        )
        new_state = sb._apply_sandbox_info(info)
        assert new_state is terminal

    def test_cancelled_stays_cancelled(self) -> None:
        cancelled = _NotStarted(cancelled=True)
        sb = self._make_sandbox(cancelled)
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(SandboxStatus.RUNNING),
        )
        new_state = sb._apply_sandbox_info(info)
        assert new_state is cancelled

    def test_preserves_sandbox_id_from_starting(self) -> None:
        sb = self._make_sandbox(_Starting(sandbox_id="sb-original"))
        info = _make_proto_info(
            sandbox_id="sb-different",
            sandbox_status=_proto_status(SandboxStatus.RUNNING),
        )
        new_state = sb._apply_sandbox_info(info)
        assert isinstance(new_state, _Running)
        assert new_state.sandbox_id == "sb-original"

    def test_not_started_gets_sandbox_id_from_info(self) -> None:
        sb = self._make_sandbox(_NotStarted())
        info = _make_proto_info(
            sandbox_id="sb-new",
            sandbox_status=_proto_status(SandboxStatus.PENDING),
        )
        new_state = sb._apply_sandbox_info(info)
        assert isinstance(new_state, _Starting)
        assert new_state.sandbox_id == "sb-new"

    def test_query_source_omits_returncode(self) -> None:
        sb = self._make_sandbox(_Running(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.COMPLETED),
            returncode=99,
        )
        new_state = sb._apply_sandbox_info(info, source="query")
        assert isinstance(new_state, _Terminal)
        assert new_state.returncode is None

    def test_poll_unspecified_maps_to_completed(self) -> None:
        sb = self._make_sandbox(_Running(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.UNSPECIFIED),
            returncode=0,
        )
        new_state = sb._apply_sandbox_info(info, source="poll")
        assert isinstance(new_state, _Terminal)
        assert new_state.status == SandboxStatus.COMPLETED
        assert new_state.returncode == 0

    def test_empty_strings_normalize_to_none(self) -> None:
        sb = self._make_sandbox(_Starting(sandbox_id="sb-1"))
        info = _make_proto_info(
            sandbox_status=_proto_status(SandboxStatus.RUNNING),
            tower_id="",
            runway_id="",
            tower_group_id="",
        )
        new_state = sb._apply_sandbox_info(info)
        assert isinstance(new_state, _Running)
        assert new_state.tower_id is None
        assert new_state.runway_id is None
        assert new_state.tower_group_id is None


# ---------------------------------------------------------------------------
# _from_sandbox_info state marking
# ---------------------------------------------------------------------------


class TestFromSandboxInfoState:
    """Verify _from_sandbox_info produces the correct _state."""

    def test_running_sandbox_has_running_state(self) -> None:
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(SandboxStatus.RUNNING),
            tower_id="tower-1",
        )
        from cwsandbox import Sandbox

        sb = Sandbox._from_sandbox_info(
            info, base_url="https://api.example.com", timeout_seconds=300.0
        )
        assert isinstance(sb._state, _Running)
        assert sb._state.sandbox_id == "sb-1"
        assert sb._state.tower_id == "tower-1"

    def test_paused_sandbox_has_running_state(self) -> None:
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(SandboxStatus.PAUSED),
        )
        from cwsandbox import Sandbox

        sb = Sandbox._from_sandbox_info(
            info, base_url="https://api.example.com", timeout_seconds=300.0
        )
        assert isinstance(sb._state, _Running)
        assert sb._state.status == SandboxStatus.PAUSED

    @pytest.mark.parametrize(
        "terminal_status",
        [SandboxStatus.COMPLETED, SandboxStatus.FAILED, SandboxStatus.TERMINATED],
    )
    def test_terminal_sandbox_has_terminal_state(self, terminal_status: SandboxStatus) -> None:
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(terminal_status),
        )
        from cwsandbox import Sandbox

        sb = Sandbox._from_sandbox_info(
            info, base_url="https://api.example.com", timeout_seconds=300.0
        )
        assert isinstance(sb._state, _Terminal)
        assert sb._state.status == terminal_status
        assert sb._is_done is True

    def test_terminal_sandbox_omits_returncode(self) -> None:
        """Non-poll source omits returncode even if info has one."""
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(SandboxStatus.COMPLETED),
            returncode=42,
        )
        from cwsandbox import Sandbox

        sb = Sandbox._from_sandbox_info(
            info, base_url="https://api.example.com", timeout_seconds=300.0
        )
        assert isinstance(sb._state, _Terminal)
        assert sb._state.returncode is None

    def test_pending_sandbox_has_starting_state(self) -> None:
        info = _make_proto_info(
            sandbox_id="sb-1",
            sandbox_status=_proto_status(SandboxStatus.PENDING),
        )
        from cwsandbox import Sandbox

        sb = Sandbox._from_sandbox_info(
            info, base_url="https://api.example.com", timeout_seconds=300.0
        )
        assert isinstance(sb._state, _Starting)
        assert sb._state.sandbox_id == "sb-1"


# ---------------------------------------------------------------------------
# Type union
# ---------------------------------------------------------------------------


class TestLifecycleStateUnion:
    """Verify _LifecycleState is a proper union of the four types."""

    @pytest.mark.parametrize(
        "state",
        [
            _NotStarted(),
            _Starting(sandbox_id="sb-1"),
            _Running(sandbox_id="sb-1"),
            _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED),
        ],
    )
    def test_all_variants_are_lifecycle_states(self, state: _LifecycleState) -> None:
        assert isinstance(state, (_NotStarted, _Starting, _Running, _Terminal))
