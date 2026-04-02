# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._resources module."""

import pytest

from cwsandbox._resources import normalize_resources
from cwsandbox._types import ResourceOptions


class TestNormalizeResources:
    """Tests for normalize_resources."""

    def test_none_returns_none(self) -> None:
        assert normalize_resources(None) is None

    def test_resource_options_passthrough(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        result = normalize_resources(opts)
        assert result is not None
        assert result.requests == {"cpu": "1", "memory": "256Mi"}
        assert result.limits == {"cpu": "8", "memory": "2Gi"}

    def test_nested_dict(self) -> None:
        d = {
            "requests": {"cpu": "1", "memory": "256Mi"},
            "limits": {"cpu": "8", "memory": "2Gi"},
        }
        result = normalize_resources(d)
        assert result is not None
        assert result.requests == {"cpu": "1", "memory": "256Mi"}
        assert result.limits == {"cpu": "8", "memory": "2Gi"}
        assert result.gpu is None

    def test_nested_dict_with_gpu(self) -> None:
        d = {
            "requests": {"cpu": "1"},
            "limits": {"cpu": "4"},
            "gpu": {"count": 1, "type": "A100"},
        }
        result = normalize_resources(d)
        assert result is not None
        assert result.gpu == {"count": 1, "type": "A100"}

    def test_flat_dict_guaranteed_qos(self) -> None:
        d = {"cpu": "8", "memory": "2Gi"}
        result = normalize_resources(d)
        assert result is not None
        assert result.requests == {"cpu": "8", "memory": "2Gi"}
        assert result.limits == {"cpu": "8", "memory": "2Gi"}
        assert result.gpu is None

    def test_flat_dict_with_gpu_string_raises(self) -> None:
        d = {"cpu": "8", "gpu": "1"}
        with pytest.raises(ValueError, match="gpu count must be a positive int"):
            normalize_resources(d)

    def test_flat_dict_with_gpu_int(self) -> None:
        d = {"gpu": 1}
        result = normalize_resources(d)
        assert result is not None
        assert result.gpu == {"count": 1}
        assert result.requests is None
        assert result.limits is None

    def test_flat_dict_with_gpu_dict(self) -> None:
        d = {"gpu": {"count": 1, "type": "A100"}}
        result = normalize_resources(d)
        assert result is not None
        assert result.gpu == {"count": 1, "type": "A100"}

    def test_limits_only_copies_to_requests(self) -> None:
        opts = ResourceOptions(limits={"cpu": "4", "memory": "1Gi"})
        result = normalize_resources(opts)
        assert result is not None
        assert result.requests == {"cpu": "4", "memory": "1Gi"}
        assert result.limits == {"cpu": "4", "memory": "1Gi"}

    def test_requests_only_copies_to_limits(self) -> None:
        opts = ResourceOptions(requests={"cpu": "2", "memory": "512Mi"})
        result = normalize_resources(opts)
        assert result is not None
        assert result.requests == {"cpu": "2", "memory": "512Mi"}
        assert result.limits == {"cpu": "2", "memory": "512Mi"}

    def test_both_provided_burstable(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        result = normalize_resources(opts)
        assert result is not None
        assert result.requests == {"cpu": "1", "memory": "256Mi"}
        assert result.limits == {"cpu": "8", "memory": "2Gi"}

    def test_validation_failure_raises(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "16"},
            limits={"cpu": "8"},
        )
        with pytest.raises(ValueError, match="cpu.*exceeds limit"):
            normalize_resources(opts)

    def test_unknown_dict_format_raises(self) -> None:
        with pytest.raises(TypeError, match="unrecognized resources dict keys"):
            normalize_resources({"foo": "bar"})

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="resources must be"):
            normalize_resources("not a dict")  # type: ignore[arg-type]

    def test_empty_dicts_normalized_to_none(self) -> None:
        opts = ResourceOptions(requests={}, limits={}, gpu={})
        result = normalize_resources(opts)
        # Empty dicts normalize to None in __post_init__, then both sides
        # are None so no filling happens.
        assert result is not None
        assert result.requests is None
        assert result.limits is None
        assert result.gpu is None

    def test_gpu_preserved_through_fill(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "1"},
            gpu={"count": 2, "type": "H100"},
        )
        result = normalize_resources(opts)
        assert result is not None
        assert result.gpu == {"count": 2, "type": "H100"}
        assert result.limits == {"cpu": "1"}

    def test_nested_dict_unknown_keys_raises(self) -> None:
        d = {
            "requests": {"cpu": "1"},
            "limts": {"cpu": "4"},  # typo
        }
        with pytest.raises(ValueError, match="unrecognized resource keys"):
            normalize_resources(d)

    def test_nested_dict_extra_key_raises(self) -> None:
        d = {
            "requests": {"cpu": "1"},
            "limits": {"cpu": "4"},
            "extra": "stuff",
        }
        with pytest.raises(ValueError, match="unrecognized resource keys"):
            normalize_resources(d)

    def test_gpu_dict_unknown_keys_raises(self) -> None:
        d = {"gpu": {"count": 1, "typo_key": "A100"}}
        with pytest.raises(ValueError, match="unrecognized GPU keys"):
            normalize_resources(d)

    def test_flat_dict_no_shared_aliasing(self) -> None:
        d = {"cpu": "8", "memory": "2Gi"}
        result = normalize_resources(d)
        assert result is not None
        assert result.requests is not result.limits
        assert result.requests == result.limits

    def test_invalid_quantity_propagates_error(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "not-a-number"},
            limits={"cpu": "8"},
        )
        with pytest.raises(ValueError, match="unrecognized quantity"):
            normalize_resources(opts)

    def test_nested_dict_gpu_unknown_keys_raises(self) -> None:
        d = {
            "requests": {"cpu": "1"},
            "limits": {"cpu": "4"},
            "gpu": {"count": 1, "typo": "A100"},
        }
        with pytest.raises(ValueError, match="unrecognized GPU keys"):
            normalize_resources(d)

    def test_resource_options_gpu_unknown_keys_raises(self) -> None:
        opts = ResourceOptions(gpu={"count": 1, "typo": "A100"})
        with pytest.raises(ValueError, match="unrecognized GPU keys"):
            normalize_resources(opts)

    def test_flat_dict_unknown_extra_key_raises(self) -> None:
        d = {"cpu": "8", "memroy": "2Gi"}  # typo
        with pytest.raises(ValueError, match="unrecognized resource keys"):
            normalize_resources(d)
