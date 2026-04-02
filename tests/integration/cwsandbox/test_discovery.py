# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for the Discovery API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import cwsandbox
from cwsandbox import Runway, Tower, TowerResources
from cwsandbox.exceptions import RunwayNotFoundError, TowerNotFoundError

# ---------------------------------------------------------------------------
# Module-scoped fixtures - fetched once, asserted non-empty
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_towers() -> list[Tower]:
    """Fetch all towers once for the module. Asserts non-empty."""
    towers = cwsandbox.list_towers()
    assert towers, "Backend returned no towers - environment is broken"
    return towers


@pytest.fixture(scope="module")
def all_runways() -> list[Runway]:
    """Fetch all runways once for the module. Asserts non-empty."""
    runways = cwsandbox.list_runways()
    assert runways, "Backend returned no runways - environment is broken"
    return runways


# ---------------------------------------------------------------------------
# list_towers
# ---------------------------------------------------------------------------


class TestListTowers:
    def test_capacity_filter(self) -> None:
        """Filter towers by a very low CPU threshold to ensure results."""
        towers = cwsandbox.list_towers(min_available_cpu_millicores=1)
        assert towers, "No towers with at least 1 millicore of available CPU"
        for t in towers:
            assert t.resources is not None, f"Tower {t.tower_id} missing resources"
            assert t.resources.available_cpu_millicores >= 1

    def test_returns_towers(self, all_towers: list[Tower]) -> None:
        assert all(isinstance(t, Tower) for t in all_towers)

    def test_tower_fields_populated(self, all_towers: list[Tower]) -> None:
        tower = all_towers[0]
        assert tower.tower_id
        assert isinstance(tower.healthy, bool)
        assert isinstance(tower.connected_at, datetime)
        assert tower.connected_at.tzinfo is not None, "connected_at must be UTC-aware"
        assert tower.connected_at.tzinfo == UTC
        assert isinstance(tower.tags, tuple)
        assert isinstance(tower.runway_names, tuple)

    def test_include_resources_true(self, all_towers: list[Tower]) -> None:
        towers = cwsandbox.list_towers(include_resources=True)
        assert towers, "No towers found - cannot validate resources"
        towers_with_resources = [t for t in towers if t.resources is not None]
        assert towers_with_resources, "No towers reported resource availability"
        tower = towers_with_resources[0]
        assert isinstance(tower.resources, TowerResources)
        assert tower.resources.available_cpu_millicores >= 0
        assert tower.resources.available_memory_bytes >= 0

    def test_include_resources_false_default(self, all_towers: list[Tower]) -> None:
        # all_towers was fetched with default (BASIC view)
        for tower in all_towers:
            assert tower.resources is None

    def test_filter_nonexistent_runway(self) -> None:
        towers = cwsandbox.list_towers(runway_name="nonexistent-runway-xyz")
        assert towers == []

    def test_filter_by_runway_name(self, all_towers: list[Tower]) -> None:
        """Positive test: filter by a real runway name returns matching towers."""
        # Find a tower with at least one runway
        tower_with_runways = next((t for t in all_towers if t.runway_names), None)
        assert tower_with_runways, "No tower has runway_names"
        target_runway = tower_with_runways.runway_names[0]

        filtered = cwsandbox.list_towers(runway_name=target_runway)
        assert filtered, f"Filter by runway_name={target_runway!r} returned nothing"
        assert all(target_runway in t.runway_names for t in filtered)
        assert tower_with_runways.tower_id in {t.tower_id for t in filtered}

    def test_filter_by_tower_group_id(self, all_towers: list[Tower]) -> None:
        """Filter by a real tower_group_id returns matching towers."""
        target_group = all_towers[0].tower_group_id
        filtered = cwsandbox.list_towers(tower_group_id=target_group)
        assert filtered, f"Filter by tower_group_id={target_group!r} returned nothing"
        assert all(t.tower_group_id == target_group for t in filtered)

    def test_filter_by_architecture(self, all_towers: list[Tower]) -> None:
        """Filter by a real architecture returns only matching towers."""
        # Find a tower with architectures
        candidate = next((t for t in all_towers if t.supported_architectures), None)
        assert candidate, "No tower has supported_architectures"
        target_arch = candidate.supported_architectures[0]

        filtered = cwsandbox.list_towers(architecture=target_arch)
        assert filtered
        assert all(target_arch in t.supported_architectures for t in filtered)

        # Verify filter actually excludes non-matching towers
        non_matching = [t for t in all_towers if target_arch not in t.supported_architectures]
        if non_matching:
            filtered_ids = {t.tower_id for t in filtered}
            for t in non_matching:
                assert t.tower_id not in filtered_ids


# ---------------------------------------------------------------------------
# get_tower
# ---------------------------------------------------------------------------


class TestGetTower:
    def test_get_existing_tower(self, all_towers: list[Tower]) -> None:
        expected = all_towers[0]
        tower = cwsandbox.get_tower(expected.tower_id)
        assert tower.tower_id == expected.tower_id
        assert tower.tower_group_id == expected.tower_group_id
        assert tower.healthy == expected.healthy

    def test_get_tower_always_has_full_details(self, all_towers: list[Tower]) -> None:
        towers_with_resources = cwsandbox.list_towers(include_resources=True)
        non_shared = [t for t in towers_with_resources if t.resources is not None]
        if not non_shared:
            pytest.skip("No non-shared towers with resources available")
        tower = cwsandbox.get_tower(non_shared[0].tower_id)
        assert tower.resources is not None

    def test_get_nonexistent_tower(self) -> None:
        with pytest.raises(TowerNotFoundError) as exc_info:
            cwsandbox.get_tower("nonexistent-tower-id-xyz")
        assert exc_info.value.tower_id == "nonexistent-tower-id-xyz"


# ---------------------------------------------------------------------------
# list_runways
# ---------------------------------------------------------------------------


class TestListRunways:
    def test_filter_by_egress_mode(self, all_runways: list[Runway]) -> None:
        """Filter runways by an egress mode found in live data."""
        candidate = next((r for r in all_runways if r.egress_modes), None)
        assert candidate, "No runway has egress_modes"
        target_mode = candidate.egress_modes[0].name

        filtered = cwsandbox.list_runways(egress_mode=target_mode)
        assert filtered, f"Filter by egress_mode={target_mode!r} returned nothing"
        for r in filtered:
            mode_names = {m.name for m in r.egress_modes}
            assert target_mode in mode_names, (
                f"Runway {r.runway_name} missing egress mode {target_mode!r}, has {mode_names}"
            )

    def test_returns_runways(self, all_runways: list[Runway]) -> None:
        assert all(isinstance(r, Runway) for r in all_runways)

    def test_runway_fields_populated(self, all_runways: list[Runway]) -> None:
        runway = all_runways[0]
        assert runway.runway_name
        assert runway.tower_id
        assert isinstance(runway.supported_gpu_types, tuple)
        assert isinstance(runway.ingress_modes, tuple)
        assert isinstance(runway.egress_modes, tuple)

    def test_filter_by_architecture(self, all_runways: list[Runway]) -> None:
        # Find a runway with architectures (scan all, not just first)
        candidate = next((r for r in all_runways if r.supported_architectures), None)
        assert candidate, "No runway has supported_architectures"
        target_arch = candidate.supported_architectures[0]

        filtered = cwsandbox.list_runways(architecture=target_arch)
        assert filtered
        assert all(target_arch in r.supported_architectures for r in filtered)

        # Verify the candidate appears in filtered results (by identity)
        filtered_pairs = {(r.runway_name, r.tower_id) for r in filtered}
        assert (candidate.runway_name, candidate.tower_id) in filtered_pairs

        # Verify non-matching runways are excluded
        non_matching = [r for r in all_runways if target_arch not in r.supported_architectures]
        if non_matching:
            for r in non_matching:
                assert (r.runway_name, r.tower_id) not in filtered_pairs

    def test_filter_by_tower_id(self, all_runways: list[Runway]) -> None:
        """Filter runways by a specific tower_id."""
        target_tower_id = all_runways[0].tower_id
        filtered = cwsandbox.list_runways(tower_id=target_tower_id)
        assert filtered
        assert all(r.tower_id == target_tower_id for r in filtered)


# ---------------------------------------------------------------------------
# get_runway
# ---------------------------------------------------------------------------


class TestGetRunway:
    def test_get_existing_runway(self, all_runways: list[Runway]) -> None:
        expected = all_runways[0]
        runway = cwsandbox.get_runway(expected.runway_name, tower_id=expected.tower_id)
        assert runway.runway_name == expected.runway_name
        assert runway.tower_id == expected.tower_id

    def test_get_runway_without_tower_id(self, all_runways: list[Runway]) -> None:
        """Without tower_id, backend returns first match sorted by tower_id."""
        expected = all_runways[0]
        runway = cwsandbox.get_runway(expected.runway_name)
        assert runway.runway_name == expected.runway_name
        # tower_id should be populated even without specifying it
        assert runway.tower_id

    def test_get_nonexistent_runway(self) -> None:
        with pytest.raises(RunwayNotFoundError):
            cwsandbox.get_runway("nonexistent-runway-xyz")


# ---------------------------------------------------------------------------
# Cross-reference consistency
# ---------------------------------------------------------------------------


class TestCrossReference:
    def test_tower_runways_match_list_runways(self, all_towers: list[Tower]) -> None:
        """For each tower, list_runways(tower_id=...) returns matching names."""
        for tower in all_towers[:3]:
            if not tower.runway_names:
                continue
            tower_runways = cwsandbox.list_runways(tower_id=tower.tower_id)
            tower_runway_names = {r.runway_name for r in tower_runways}
            for rn in tower.runway_names:
                assert rn in tower_runway_names, (
                    f"Tower {tower.tower_id} advertises runway {rn!r} "
                    f"but list_runways(tower_id=...) returned {tower_runway_names}"
                )
            # Also verify all returned runways belong to this tower
            for r in tower_runways:
                assert r.tower_id == tower.tower_id

    def test_runway_tower_ids_exist(
        self, all_towers: list[Tower], all_runways: list[Runway]
    ) -> None:
        """Every runway's tower_id should appear in list_towers."""
        tower_ids = {t.tower_id for t in all_towers}
        for runway in all_runways[:10]:
            assert runway.tower_id in tower_ids, (
                f"Runway {runway.runway_name} references tower "
                f"{runway.tower_id} which is not in list_towers"
            )
