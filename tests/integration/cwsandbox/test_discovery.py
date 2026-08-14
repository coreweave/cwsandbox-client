# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for the Discovery API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import cwsandbox
from cwsandbox import Runner, RunnerResources
from cwsandbox.exceptions import RunnerNotFoundError

# ---------------------------------------------------------------------------
# Module-scoped fixtures - fetched once, asserted non-empty
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_runners() -> list[Runner]:
    """Fetch all runners once for the module. Asserts non-empty."""
    runners = cwsandbox.list_runners()
    assert runners, "Backend returned no runners - environment is broken"
    return runners


# ---------------------------------------------------------------------------
# list_runners
# ---------------------------------------------------------------------------


class TestListRunners:
    def test_capacity_filter(self) -> None:
        """Filter runners by a very low CPU threshold to ensure results."""
        runners = cwsandbox.list_runners(min_available_cpu_millicores=1)
        assert runners, "No runners with at least 1 millicore of available CPU"
        for t in runners:
            assert t.resources is not None, f"Runner {t.runner_id} missing resources"
            assert t.resources.available_cpu_millicores >= 1

    def test_returns_runners(self, all_runners: list[Runner]) -> None:
        assert all(isinstance(t, Runner) for t in all_runners)

    def test_runner_fields_populated(self, all_runners: list[Runner]) -> None:
        runner = all_runners[0]
        assert runner.runner_id
        assert isinstance(runner.healthy, bool)
        assert isinstance(runner.connected_at, datetime)
        assert runner.connected_at.tzinfo is not None, "connected_at must be UTC-aware"
        assert runner.connected_at.tzinfo == UTC
        assert isinstance(runner.tags, tuple)
        assert isinstance(runner.supported_gpu_types, tuple)
        assert isinstance(runner.supported_architectures, tuple)
        assert isinstance(runner.available_storage_classes, tuple)
        assert isinstance(runner.supported_service_visibilities, tuple)

    def test_include_resources_true(self, all_runners: list[Runner]) -> None:
        runners = cwsandbox.list_runners(include_resources=True)
        assert runners, "No runners found - cannot validate resources"
        runners_with_resources = [t for t in runners if t.resources is not None]
        assert runners_with_resources, "No runners reported resource availability"
        runner = runners_with_resources[0]
        assert isinstance(runner.resources, RunnerResources)
        assert runner.resources.available_cpu_millicores >= 0
        assert runner.resources.available_memory_bytes >= 0

    def test_include_resources_false_default(self, all_runners: list[Runner]) -> None:
        # all_runners was fetched with default (BASIC view)
        for runner in all_runners:
            assert runner.resources is None

    def test_filter_healthy_only(self) -> None:
        runners = cwsandbox.list_runners(healthy_only=True)
        assert runners, "No healthy runners found"
        assert all(runner.healthy for runner in runners)

    def test_filter_by_service_visibility(self, all_runners: list[Runner]) -> None:
        """Filter by a typed service visibility found in live capabilities."""
        candidate = next(
            (runner for runner in all_runners if runner.supported_service_visibilities),
            None,
        )
        if candidate is None:
            pytest.skip("No runner advertises supported service visibilities")
        visibility = candidate.supported_service_visibilities[0]

        filtered = cwsandbox.list_runners(service_visibility=visibility)
        assert filtered
        assert all(visibility in runner.supported_service_visibilities for runner in filtered)
        assert candidate.runner_id in {runner.runner_id for runner in filtered}

    def test_filter_by_runner_group_id(self, all_runners: list[Runner]) -> None:
        """Filter by a real runner_group_id returns matching runners."""
        target_group = all_runners[0].runner_group_id
        filtered = cwsandbox.list_runners(runner_group_id=target_group)
        assert filtered, f"Filter by runner_group_id={target_group!r} returned nothing"
        assert all(t.runner_group_id == target_group for t in filtered)

    def test_filter_by_architecture(self, all_runners: list[Runner]) -> None:
        """Filter by a real architecture returns only matching runners."""
        # Find a runner with architectures
        candidate = next((t for t in all_runners if t.supported_architectures), None)
        assert candidate, "No runner has supported_architectures"
        target_arch = candidate.supported_architectures[0]

        filtered = cwsandbox.list_runners(architecture=target_arch)
        assert filtered
        assert all(target_arch in t.supported_architectures for t in filtered)

        # Verify filter actually excludes non-matching runners
        non_matching = [t for t in all_runners if target_arch not in t.supported_architectures]
        if non_matching:
            filtered_ids = {t.runner_id for t in filtered}
            for t in non_matching:
                assert t.runner_id not in filtered_ids


# ---------------------------------------------------------------------------
# get_runner
# ---------------------------------------------------------------------------


class TestGetRunner:
    def test_get_existing_runner(self, all_runners: list[Runner]) -> None:
        expected = all_runners[0]
        runner = cwsandbox.get_runner(
            expected.runner_id,
            organization_id=expected.organization_id,
        )
        assert runner.runner_id == expected.runner_id
        assert runner.organization_id == expected.organization_id
        assert runner.runner_group_id == expected.runner_group_id
        assert runner.healthy == expected.healthy

    def test_get_runner_always_has_full_details(self, all_runners: list[Runner]) -> None:
        runners_with_resources = cwsandbox.list_runners(include_resources=True)
        non_shared = [t for t in runners_with_resources if t.resources is not None]
        if not non_shared:
            pytest.skip("No non-shared runners with resources available")
        expected = non_shared[0]
        runner = cwsandbox.get_runner(
            expected.runner_id,
            organization_id=expected.organization_id,
        )
        assert runner.resources is not None

    def test_get_nonexistent_runner(self, all_runners: list[Runner]) -> None:
        with pytest.raises(RunnerNotFoundError) as exc_info:
            cwsandbox.get_runner(
                "nonexistent-runner-id-xyz",
                organization_id=all_runners[0].organization_id,
            )
        assert exc_info.value.runner_id == "nonexistent-runner-id-xyz"
