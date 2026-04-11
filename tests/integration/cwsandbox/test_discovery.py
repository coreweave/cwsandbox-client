# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for the Discovery API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import cwsandbox
from cwsandbox import Profile, Runner, RunnerResources
from cwsandbox.exceptions import ProfileNotFoundError, RunnerNotFoundError

# ---------------------------------------------------------------------------
# Module-scoped fixtures - fetched once, asserted non-empty
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_runners() -> list[Runner]:
    """Fetch all runners once for the module. Asserts non-empty."""
    runners = cwsandbox.list_runners()
    assert runners, "Backend returned no runners - environment is broken"
    return runners


@pytest.fixture(scope="module")
def all_profiles() -> list[Profile]:
    """Fetch all profiles once for the module. Asserts non-empty."""
    profiles = cwsandbox.list_profiles()
    assert profiles, "Backend returned no profiles - environment is broken"
    return profiles


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
        assert isinstance(runner.profile_names, tuple)

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

    def test_filter_nonexistent_profile(self) -> None:
        runners = cwsandbox.list_runners(profile_name="nonexistent-profile-xyz")
        assert runners == []

    def test_filter_by_profile_name(self, all_runners: list[Runner]) -> None:
        """Positive test: filter by a real profile name returns matching runners."""
        # Find a runner with at least one profile
        runner_with_profiles = next((t for t in all_runners if t.profile_names), None)
        assert runner_with_profiles, "No runner has profile_names"
        target_profile = runner_with_profiles.profile_names[0]

        filtered = cwsandbox.list_runners(profile_name=target_profile)
        assert filtered, f"Filter by profile_name={target_profile!r} returned nothing"
        assert all(target_profile in t.profile_names for t in filtered)
        assert runner_with_profiles.runner_id in {t.runner_id for t in filtered}

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
        runner = cwsandbox.get_runner(expected.runner_id)
        assert runner.runner_id == expected.runner_id
        assert runner.runner_group_id == expected.runner_group_id
        assert runner.healthy == expected.healthy

    def test_get_runner_always_has_full_details(self, all_runners: list[Runner]) -> None:
        runners_with_resources = cwsandbox.list_runners(include_resources=True)
        non_shared = [t for t in runners_with_resources if t.resources is not None]
        if not non_shared:
            pytest.skip("No non-shared runners with resources available")
        runner = cwsandbox.get_runner(non_shared[0].runner_id)
        assert runner.resources is not None

    def test_get_nonexistent_runner(self) -> None:
        with pytest.raises(RunnerNotFoundError) as exc_info:
            cwsandbox.get_runner("nonexistent-runner-id-xyz")
        assert exc_info.value.runner_id == "nonexistent-runner-id-xyz"


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------


class TestListProfiles:
    def test_filter_by_egress_mode(self, all_profiles: list[Profile]) -> None:
        """Filter profiles by an egress mode found in live data."""
        candidate = next((r for r in all_profiles if r.egress_modes), None)
        assert candidate, "No profile has egress_modes"
        target_mode = candidate.egress_modes[0].name

        filtered = cwsandbox.list_profiles(egress_mode=target_mode)
        assert filtered, f"Filter by egress_mode={target_mode!r} returned nothing"
        for r in filtered:
            mode_names = {m.name for m in r.egress_modes}
            assert target_mode in mode_names, (
                f"Profile {r.profile_name} missing egress mode {target_mode!r}, has {mode_names}"
            )

    def test_returns_profiles(self, all_profiles: list[Profile]) -> None:
        assert all(isinstance(r, Profile) for r in all_profiles)

    def test_profile_fields_populated(self, all_profiles: list[Profile]) -> None:
        profile = all_profiles[0]
        assert profile.profile_name
        assert profile.runner_id
        assert isinstance(profile.supported_gpu_types, tuple)
        assert isinstance(profile.service_exposure_modes, tuple)
        assert isinstance(profile.egress_modes, tuple)

    def test_filter_by_architecture(self, all_profiles: list[Profile]) -> None:
        # Find a profile with architectures (scan all, not just first)
        candidate = next((r for r in all_profiles if r.supported_architectures), None)
        assert candidate, "No profile has supported_architectures"
        target_arch = candidate.supported_architectures[0]

        filtered = cwsandbox.list_profiles(architecture=target_arch)
        assert filtered
        assert all(target_arch in r.supported_architectures for r in filtered)

        # Verify the candidate appears in filtered results (by identity)
        filtered_pairs = {(r.profile_name, r.runner_id) for r in filtered}
        assert (candidate.profile_name, candidate.runner_id) in filtered_pairs

        # Verify non-matching profiles are excluded
        non_matching = [r for r in all_profiles if target_arch not in r.supported_architectures]
        if non_matching:
            for r in non_matching:
                assert (r.profile_name, r.runner_id) not in filtered_pairs

    def test_filter_by_service_exposure_mode(self, all_profiles: list[Profile]) -> None:
        """Filter profiles by a service exposure mode found in live data."""
        candidate = next((r for r in all_profiles if r.service_exposure_modes), None)
        assert candidate, "No profile has service_exposure_modes"
        target_mode = candidate.service_exposure_modes[0].name

        filtered = cwsandbox.list_profiles(service_exposure_mode=target_mode)
        assert filtered, f"Filter by service_exposure_mode={target_mode!r} returned nothing"
        for r in filtered:
            mode_names = {m.name for m in r.service_exposure_modes}
            assert target_mode in mode_names, (
                f"Profile {r.profile_name} missing service exposure mode "
                f"{target_mode!r}, has {mode_names}"
            )

    def test_filter_by_runner_id(self, all_profiles: list[Profile]) -> None:
        """Filter profiles by a specific runner_id."""
        target_runner_id = all_profiles[0].runner_id
        filtered = cwsandbox.list_profiles(runner_id=target_runner_id)
        assert filtered
        assert all(r.runner_id == target_runner_id for r in filtered)


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_get_existing_profile(self, all_profiles: list[Profile]) -> None:
        expected = all_profiles[0]
        profile = cwsandbox.get_profile(expected.profile_name, runner_id=expected.runner_id)
        assert profile.profile_name == expected.profile_name
        assert profile.runner_id == expected.runner_id

    def test_get_profile_without_runner_id(self, all_profiles: list[Profile]) -> None:
        """Without runner_id, backend returns first match sorted by runner_id."""
        expected = all_profiles[0]
        profile = cwsandbox.get_profile(expected.profile_name)
        assert profile.profile_name == expected.profile_name
        # runner_id should be populated even without specifying it
        assert profile.runner_id

    def test_get_nonexistent_profile(self) -> None:
        with pytest.raises(ProfileNotFoundError):
            cwsandbox.get_profile("nonexistent-profile-xyz")


# ---------------------------------------------------------------------------
# Cross-reference consistency
# ---------------------------------------------------------------------------


class TestCrossReference:
    def test_runner_profiles_match_list_profiles(self, all_runners: list[Runner]) -> None:
        """For each runner, list_profiles(runner_id=...) returns matching names."""
        for runner in all_runners[:3]:
            if not runner.profile_names:
                continue
            runner_profiles = cwsandbox.list_profiles(runner_id=runner.runner_id)
            runner_profile_names = {r.profile_name for r in runner_profiles}
            for rn in runner.profile_names:
                assert rn in runner_profile_names, (
                    f"Runner {runner.runner_id} advertises profile {rn!r} "
                    f"but list_profiles(runner_id=...) returned {runner_profile_names}"
                )
            # Also verify all returned profiles belong to this runner
            for r in runner_profiles:
                assert r.runner_id == runner.runner_id

    def test_profile_runner_ids_exist(
        self, all_runners: list[Runner], all_profiles: list[Profile]
    ) -> None:
        """Every profile's runner_id should appear in list_runners."""
        runner_ids = {t.runner_id for t in all_runners}
        for profile in all_profiles[:10]:
            assert profile.runner_id in runner_ids, (
                f"Profile {profile.profile_name} references runner "
                f"{profile.runner_id} which is not in list_runners"
            )
