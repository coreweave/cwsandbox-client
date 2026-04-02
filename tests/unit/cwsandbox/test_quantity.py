# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._quantity module."""

from decimal import Decimal

import pytest

from cwsandbox._quantity import parse_quantity, validate_resource_quantities
from cwsandbox._types import ResourceOptions


class TestParseQuantity:
    """Tests for parse_quantity."""

    def test_plain_integer(self) -> None:
        assert parse_quantity("100") == Decimal("100")

    def test_plain_decimal(self) -> None:
        assert parse_quantity("0.5") == Decimal("0.5")

    def test_millicores(self) -> None:
        assert parse_quantity("100m") == Decimal("0.1")

    def test_millicores_500(self) -> None:
        assert parse_quantity("500m") == Decimal("0.5")

    def test_millicores_1000(self) -> None:
        assert parse_quantity("1000m") == Decimal("1")

    def test_binary_ki(self) -> None:
        assert parse_quantity("1Ki") == Decimal("1024")

    def test_binary_mi(self) -> None:
        assert parse_quantity("256Mi") == Decimal(256 * 1024**2)

    def test_binary_gi(self) -> None:
        assert parse_quantity("1Gi") == Decimal(1024**3)

    def test_binary_ti(self) -> None:
        assert parse_quantity("1Ti") == Decimal(1024**4)

    def test_decimal_k(self) -> None:
        assert parse_quantity("1k") == Decimal("1000")

    def test_decimal_m_upper(self) -> None:
        assert parse_quantity("1M") == Decimal("1000000")

    def test_decimal_g(self) -> None:
        assert parse_quantity("1G") == Decimal("1000000000")

    def test_whitespace_stripped(self) -> None:
        assert parse_quantity("  500m  ") == Decimal("0.5")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty quantity"):
            parse_quantity("")

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized quantity"):
            parse_quantity("abc")

    def test_binary_pi(self) -> None:
        assert parse_quantity("1Pi") == Decimal(1024**5)

    def test_binary_ei(self) -> None:
        assert parse_quantity("1Ei") == Decimal(1024**6)

    def test_decimal_t(self) -> None:
        assert parse_quantity("1T") == Decimal("1000000000000")

    def test_decimal_p(self) -> None:
        assert parse_quantity("1P") == Decimal("1000000000000000")

    def test_decimal_e(self) -> None:
        assert parse_quantity("1E") == Decimal("1000000000000000000")

    def test_zero(self) -> None:
        assert parse_quantity("0") == Decimal("0")

    def test_exponent_notation(self) -> None:
        assert parse_quantity("1e3") == Decimal("1000")

    def test_malformed_binary_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized quantity"):
            parse_quantity("1.2.3Gi")

    def test_malformed_decimal_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized quantity"):
            parse_quantity("abcM")

    def test_bare_exponent_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized quantity"):
            parse_quantity("1e")

    def test_bare_millicore_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized quantity"):
            parse_quantity("m")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty quantity"):
            parse_quantity("   ")


class TestValidateResourceQuantities:
    """Tests for validate_resource_quantities."""

    def test_requests_within_limits_passes(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        validate_resource_quantities(opts)

    def test_requests_equal_limits_passes(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "4", "memory": "1Gi"},
            limits={"cpu": "4", "memory": "1Gi"},
        )
        validate_resource_quantities(opts)

    def test_cpu_request_exceeds_limit_raises(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "16"},
            limits={"cpu": "8"},
        )
        with pytest.raises(ValueError, match="cpu.*exceeds limit"):
            validate_resource_quantities(opts)

    def test_memory_request_exceeds_limit_raises(self) -> None:
        opts = ResourceOptions(
            requests={"memory": "4Gi"},
            limits={"memory": "2Gi"},
        )
        with pytest.raises(ValueError, match="memory.*exceeds limit"):
            validate_resource_quantities(opts)

    def test_partial_keys_only_validates_shared(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "1"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        validate_resource_quantities(opts)

    def test_no_requests_skips_validation(self) -> None:
        opts = ResourceOptions(limits={"cpu": "8", "memory": "2Gi"})
        validate_resource_quantities(opts)

    def test_no_limits_skips_validation(self) -> None:
        opts = ResourceOptions(requests={"cpu": "1", "memory": "256Mi"})
        validate_resource_quantities(opts)

    def test_both_none_skips_validation(self) -> None:
        opts = ResourceOptions()
        validate_resource_quantities(opts)

    def test_non_string_request_value_raises(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": 4},  # type: ignore[dict-item]
            limits={"cpu": "8"},
        )
        with pytest.raises(TypeError, match="resource value for cpu request must be a string"):
            validate_resource_quantities(opts)

    def test_non_string_limit_value_raises(self) -> None:
        opts = ResourceOptions(
            requests={"cpu": "1"},
            limits={"cpu": 8},  # type: ignore[dict-item]
        )
        with pytest.raises(TypeError, match="resource value for cpu limit must be a string"):
            validate_resource_quantities(opts)
