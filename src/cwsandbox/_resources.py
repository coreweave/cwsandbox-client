# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

from typing import Any

from cwsandbox._quantity import validate_resource_quantities
from cwsandbox._types import ResourceOptions

# "gpu" alone does not signal nested form; it must appear with "requests" or "limits".
_NESTED_REQUIRED = {"requests", "limits"}
_NESTED_KEYS = {"requests", "limits", "gpu"}
_FLAT_KEYS = {"cpu", "memory", "gpu"}


def normalize_resources(
    resources: ResourceOptions | dict[str, Any] | None,
) -> ResourceOptions | None:
    """Normalize a resources value into a validated ``ResourceOptions``.

    Accepts ``None``, an existing ``ResourceOptions``, or a dict.  Dict
    inputs may use the new nested form (``requests``/``limits``/``gpu``
    keys) or the legacy flat form (``cpu``/``memory``/``gpu`` at the top
    level).  Flat dicts are treated as Guaranteed QoS: the values become
    both requests and limits.
    """
    if resources is None:
        return None

    if isinstance(resources, ResourceOptions):
        result = resources
    elif isinstance(resources, dict):
        result = _from_dict(resources)
    else:
        raise TypeError(
            f"resources must be ResourceOptions, dict, or None, got {type(resources).__name__}"
        )

    if result.gpu is not None and isinstance(result.gpu, dict):
        _validate_gpu_keys(result.gpu)
    result = _fill_missing_side(result)
    validate_resource_quantities(result)
    return result


def _from_dict(d: dict[str, Any]) -> ResourceOptions:
    """Convert a dict to ResourceOptions, handling nested and flat forms."""
    keys = set(d.keys())

    # Nested form: has "requests" or "limits" at top level (optionally "gpu")
    if keys & _NESTED_REQUIRED:
        unknown = keys - _NESTED_KEYS
        if unknown:
            raise ValueError(f"unrecognized resource keys: {unknown}")
        _INNER_KEYS = {"cpu", "memory"}
        for side in ("requests", "limits"):
            inner = d.get(side)
            if inner is not None and isinstance(inner, dict):
                bad = set(inner.keys()) - _INNER_KEYS
                if bad:
                    raise ValueError(f"unrecognized keys in {side}: {bad}")
        return ResourceOptions(**d)

    # Flat form (backward compat): has "cpu", "memory", or "gpu" at top level
    if keys & _FLAT_KEYS:
        unknown = keys - _FLAT_KEYS
        if unknown:
            raise ValueError(f"unrecognized resource keys: {unknown}")
        gpu_raw = d.get("gpu")
        gpu_dict = _normalize_gpu(gpu_raw) if gpu_raw is not None else None

        cpu_mem = {k: v for k, v in d.items() if k in ("cpu", "memory")}
        cpu_mem_or_none = cpu_mem if cpu_mem else None

        return ResourceOptions(
            requests=cpu_mem_or_none,
            limits=dict(cpu_mem) if cpu_mem_or_none is not None else None,
            gpu=gpu_dict,
        )

    raise TypeError(f"unrecognized resources dict keys: {keys}")


_GPU_KEYS = {"count", "type", "memory_gb"}


def _validate_gpu_keys(gpu: dict[str, Any]) -> None:
    """Raise ``ValueError`` if *gpu* contains unrecognized keys or invalid values."""
    unknown = set(gpu.keys()) - _GPU_KEYS
    if unknown:
        raise ValueError(f"unrecognized GPU keys: {unknown}")
    if "count" in gpu:
        if not isinstance(gpu["count"], int) or isinstance(gpu["count"], bool):
            raise ValueError(
                f"gpu 'count' must be a positive int, got {type(gpu['count']).__name__}"
            )
        if gpu["count"] < 1:
            raise ValueError(f"gpu 'count' must be a positive int, got {gpu['count']}")
    if "type" in gpu:
        if not isinstance(gpu["type"], str):
            raise ValueError(f"gpu 'type' must be a str, got {type(gpu['type']).__name__}")
    if "memory_gb" in gpu:
        if not isinstance(gpu["memory_gb"], int) or isinstance(gpu["memory_gb"], bool):
            raise ValueError(
                f"gpu 'memory_gb' must be a positive int, got {type(gpu['memory_gb']).__name__}"
            )
        if gpu["memory_gb"] < 1:
            raise ValueError(f"gpu 'memory_gb' must be a positive int, got {gpu['memory_gb']}")


def _normalize_gpu(value: Any) -> dict[str, Any]:
    """Normalize a GPU value to a dict with at least a ``count`` key."""
    if isinstance(value, dict):
        _validate_gpu_keys(value)
        return value
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"gpu count must be a positive int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"gpu count must be a positive int, got {value}")
    return {"count": value}


def _fill_missing_side(opts: ResourceOptions) -> ResourceOptions:
    """Copy requests to limits (or vice versa) when only one side is set.

    This produces Guaranteed QoS when the user only specifies one side.
    When both are provided, the values are left as-is (Burstable QoS).
    """
    if opts.requests is not None and opts.limits is None:
        return ResourceOptions(requests=opts.requests, limits=dict(opts.requests), gpu=opts.gpu)
    if opts.limits is not None and opts.requests is None:
        return ResourceOptions(requests=dict(opts.limits), limits=opts.limits, gpu=opts.gpu)
    return opts
