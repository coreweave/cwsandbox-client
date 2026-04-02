# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cwsandbox._types import ResourceOptions

_BINARY_SUFFIXES: dict[str, int] = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
}
_DECIMAL_SUFFIXES: dict[str, int] = {
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "E": 1000**6,
}


def parse_quantity(s: str) -> Decimal:
    """Parse a Kubernetes quantity string into a Decimal.

    Supports plain numbers, millicores (``m`` suffix), binary suffixes
    (``Ki``, ``Mi``, ``Gi``, ``Ti``, ``Pi``, ``Ei``), and decimal suffixes
    (``k``, ``M``, ``G``, ``T``, ``P``, ``E``).
    """
    s = s.strip()
    if not s:
        raise ValueError("empty quantity string")

    if s.startswith("-"):
        raise ValueError(f"negative quantities are not allowed: {s!r}")

    try:
        # Millicores: e.g. "500m"
        if s.endswith("m"):
            num_part = s[:-1]
            if "e" in num_part or "E" in num_part:
                raise ValueError(f"scientific notation with suffix is not allowed: {s!r}")
            return Decimal(num_part) / 1000

        # Binary suffixes: two-char check first
        if len(s) >= 3 and s[-2:] in _BINARY_SUFFIXES:
            num_part = s[:-2]
            if "e" in num_part or "E" in num_part:
                raise ValueError(f"scientific notation with suffix is not allowed: {s!r}")
            return Decimal(num_part) * _BINARY_SUFFIXES[s[-2:]]

        # Decimal suffixes: single-char
        if s[-1] in _DECIMAL_SUFFIXES:
            num_part = s[:-1]
            if "e" in num_part or "E" in num_part:
                raise ValueError(f"scientific notation with suffix is not allowed: {s!r}")
            return Decimal(num_part) * _DECIMAL_SUFFIXES[s[-1]]

        # Plain number (scientific notation OK without suffix)
        return Decimal(s)
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"unrecognized quantity format: {s!r}") from None


def validate_resource_quantities(options: ResourceOptions) -> None:
    """Validate that resource requests do not exceed limits.

    Compares CPU and memory quantities when both requests and limits
    contain the same key. Raises ``ValueError`` if any request exceeds
    its corresponding limit.
    """
    if options.requests is None or options.limits is None:
        return

    for key in ("cpu", "memory"):
        if key in options.requests and key in options.limits:
            req_raw = options.requests[key]
            lim_raw = options.limits[key]
            if not isinstance(req_raw, str):
                raise TypeError(
                    f"resource value for {key} request must be a string, "
                    f"got {type(req_raw).__name__}"
                )
            if not isinstance(lim_raw, str):
                raise TypeError(
                    f"resource value for {key} limit must be a string, got {type(lim_raw).__name__}"
                )
            req = parse_quantity(req_raw)
            lim = parse_quantity(lim_raw)
            if req > lim:
                raise ValueError(
                    f"resource request for {key} ({options.requests[key]}) "
                    f"exceeds limit ({options.limits[key]})"
                )
