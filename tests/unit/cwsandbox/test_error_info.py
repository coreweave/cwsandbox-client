# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._error_info module."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import grpc
from google.protobuf import any_pb2
from google.protobuf.duration_pb2 import Duration
from google.rpc import error_details_pb2, status_pb2

from cwsandbox._error_info import ParsedError, parse_error_info


def _pack_status(
    *,
    reason: str | None = None,
    metadata: dict[str, str] | None = None,
    retry_seconds: int | None = None,
    retry_nanos: int | None = None,
) -> bytes:
    """Build a serialized google.rpc.Status with the given details."""
    status = status_pb2.Status(code=2, message="test")
    if reason is not None:
        info = error_details_pb2.ErrorInfo(
            reason=reason,
            domain="cwsandbox.com",
            metadata=metadata or {},
        )
        packed = any_pb2.Any()
        packed.Pack(info)
        status.details.append(packed)
    if retry_seconds is not None or retry_nanos is not None:
        retry = error_details_pb2.RetryInfo(
            retry_delay=Duration(seconds=retry_seconds or 0, nanos=retry_nanos or 0),
        )
        packed = any_pb2.Any()
        packed.Pack(retry)
        status.details.append(packed)
    return status.SerializeToString()


def _make_error(
    *,
    reason: str | None = None,
    metadata: dict[str, str] | None = None,
    retry_seconds: int | None = None,
    retry_nanos: int | None = None,
    trailing: object | None | type = ...,  # type: ignore[assignment]
    raise_on_trailing: Exception | None = None,
    no_trailing_method: bool = False,
) -> grpc.RpcError:
    """Build a mock grpc.RpcError with shaped trailing metadata."""
    err = MagicMock()
    if no_trailing_method:
        del err.trailing_metadata
        return err

    if raise_on_trailing is not None:
        err.trailing_metadata.side_effect = raise_on_trailing
        return err

    if trailing is ...:
        status_bytes = _pack_status(
            reason=reason,
            metadata=metadata,
            retry_seconds=retry_seconds,
            retry_nanos=retry_nanos,
        )
        err.trailing_metadata.return_value = [
            ("some-other-key", "irrelevant"),
            ("grpc-status-details-bin", status_bytes),
        ]
    else:
        err.trailing_metadata.return_value = trailing
    return err


class TestParsedErrorDefaults:
    def test_dataclass_defaults(self) -> None:
        parsed = ParsedError()
        assert parsed.reason is None
        assert parsed.metadata == {}
        assert parsed.retry_delay is None


class TestParseErrorInfo:
    def test_well_formed_error_info(self) -> None:
        err = _make_error(reason="CWSANDBOX_FILE_NOT_FOUND")
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason == "CWSANDBOX_FILE_NOT_FOUND"
        assert parsed.metadata == {}
        assert parsed.retry_delay is None

    def test_error_info_with_metadata(self) -> None:
        err = _make_error(
            reason="CWSANDBOX_FILE_NOT_FOUND",
            metadata={"filepath": "/data/missing.txt"},
        )
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.metadata["filepath"] == "/data/missing.txt"

    def test_metadata_is_dict_not_none_when_present(self) -> None:
        err = _make_error(reason="CWSANDBOX_FILE_NOT_FOUND", metadata={})
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.metadata == {}

    def test_retry_info_present(self) -> None:
        err = _make_error(reason="CWSANDBOX_BACKEND_UNAVAILABLE", retry_seconds=5)
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay == timedelta(seconds=5)

    def test_retry_info_with_nanos(self) -> None:
        err = _make_error(reason="CWSANDBOX_BACKEND_UNAVAILABLE", retry_nanos=500_000_000)
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay == timedelta(microseconds=500_000)

    def test_retry_info_without_error_info(self) -> None:
        err = _make_error(retry_seconds=2)
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason is None
        assert parsed.retry_delay == timedelta(seconds=2)

    def test_no_trailing_metadata_method(self) -> None:
        err = _make_error(no_trailing_method=True)
        assert parse_error_info(err) is None

    def test_trailing_metadata_empty(self) -> None:
        err = _make_error(trailing=[])
        assert parse_error_info(err) is None

    def test_trailing_metadata_missing_status_details(self) -> None:
        err = _make_error(trailing=[("x-other", b"bytes")])
        assert parse_error_info(err) is None

    def test_malformed_status_bytes_returns_none(self) -> None:
        err = _make_error(trailing=[("grpc-status-details-bin", b"not a valid protobuf")])
        assert parse_error_info(err) is None

    def test_trailing_metadata_raises(self) -> None:
        err = _make_error(raise_on_trailing=RuntimeError("no metadata available"))
        assert parse_error_info(err) is None

    def test_trailing_metadata_non_iterable(self) -> None:
        err = _make_error(trailing=42)
        assert parse_error_info(err) is None

    def test_status_without_any_details_returns_none(self) -> None:
        # Well-formed Status, but no ErrorInfo and no RetryInfo -> None
        err = _make_error(trailing=[("grpc-status-details-bin", _pack_status())])
        assert parse_error_info(err) is None

    def test_tuple_entries_work(self) -> None:
        status_bytes = _pack_status(reason="CWSANDBOX_SANDBOX_NOT_FOUND")
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason == "CWSANDBOX_SANDBOX_NOT_FOUND"

    def test_case_insensitive_key_match(self) -> None:
        status_bytes = _pack_status(reason="CWSANDBOX_SANDBOX_NOT_FOUND")
        err = _make_error(trailing=[("GRPC-Status-Details-Bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason == "CWSANDBOX_SANDBOX_NOT_FOUND"

    def test_metadata_entry_with_key_value_attributes(self) -> None:
        """Metadata entries exposing .key/.value (grpc.aio.Metadata pattern)."""
        status_bytes = _pack_status(reason="CWSANDBOX_COMMAND_TIMEOUT")
        entry = MagicMock()
        entry.key = "grpc-status-details-bin"
        entry.value = status_bytes
        err = _make_error(trailing=[entry])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason == "CWSANDBOX_COMMAND_TIMEOUT"

    def test_domain_populated_from_error_info(self) -> None:
        err = _make_error(reason="CWSANDBOX_FILE_NOT_FOUND")
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.domain == "cwsandbox.com"

    def test_retry_info_oversized_duration_returns_none(self) -> None:
        """A hostile server emitting an unrepresentable RetryInfo must not raise."""
        err = _make_error(retry_seconds=10**18)  # far beyond timedelta's range
        parsed = parse_error_info(err)
        # Either no retry_delay produced, or parser returned None entirely.
        if parsed is not None:
            assert parsed.retry_delay is None

    def test_malformed_first_entry_then_valid_second(self) -> None:
        """Duplicate metadata keys: malformed first, valid second still parses."""
        status_bytes = _pack_status(reason="CWSANDBOX_SANDBOX_NOT_FOUND")
        err = _make_error(
            trailing=[
                ("grpc-status-details-bin", b"not a valid protobuf"),
                ("grpc-status-details-bin", status_bytes),
            ]
        )
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason == "CWSANDBOX_SANDBOX_NOT_FOUND"


def _pack_error_info_detail(
    *,
    reason: str = "",
    domain: str = "cwsandbox.com",
    metadata: dict[str, str] | None = None,
) -> any_pb2.Any:
    info = error_details_pb2.ErrorInfo(
        reason=reason,
        domain=domain,
        metadata=metadata or {},
    )
    packed = any_pb2.Any()
    packed.Pack(info)
    return packed


def _pack_retry_info_detail(duration: Duration | None) -> any_pb2.Any:
    retry = error_details_pb2.RetryInfo()
    if duration is not None:
        retry.retry_delay.CopyFrom(duration)
    packed = any_pb2.Any()
    packed.Pack(retry)
    return packed


def _build_status_bytes(*details: any_pb2.Any) -> bytes:
    status = status_pb2.Status(code=2, message="test")
    status.details.extend(details)
    return status.SerializeToString()


class TestMultipleErrorInfoHandling:
    """M1: an empty ErrorInfo in the same Status must not block a later valid one."""

    def test_empty_reason_then_valid_reason_wins(self) -> None:
        status_bytes = _build_status_bytes(
            _pack_error_info_detail(reason=""),
            _pack_error_info_detail(reason="CWSANDBOX_FILE_NOT_FOUND"),
        )
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.reason == "CWSANDBOX_FILE_NOT_FOUND"

    def test_only_empty_reason_returns_none(self) -> None:
        status_bytes = _build_status_bytes(_pack_error_info_detail(reason=""))
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        assert parse_error_info(err) is None


class TestEmptyRetryInfoHandling:
    """M2: Default-constructed RetryInfo must parse to retry_delay=None."""

    def test_empty_retry_info_returns_none_retry_delay(self) -> None:
        status_bytes = _build_status_bytes(
            _pack_error_info_detail(reason="CWSANDBOX_BACKEND_UNAVAILABLE"),
            _pack_retry_info_detail(duration=None),
        )
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay is None

    def test_empty_retry_info_then_valid_retry_info_wins(self) -> None:
        status_bytes = _build_status_bytes(
            _pack_retry_info_detail(duration=None),
            _pack_retry_info_detail(duration=Duration(seconds=7)),
        )
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay == timedelta(seconds=7)

    def test_explicit_zero_duration_preserved(self) -> None:
        """RetryInfo(retry_delay=Duration()) is explicit zero, not absent."""
        status_bytes = _build_status_bytes(
            _pack_error_info_detail(reason="CWSANDBOX_BACKEND_UNAVAILABLE"),
            _pack_retry_info_detail(duration=Duration()),
        )
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay == timedelta(0)


class TestRetryDelayDurationConversion:
    """L1: Duration conversion delegates to protobuf's ToTimedelta().

    The parser trusts protobuf's native Duration -> timedelta conversion
    rather than applying its own validation rules. The backend is expected
    to emit well-formed Durations; when it doesn't, ToTimedelta()'s
    normalization is acceptable (we don't try to re-validate).
    """

    def test_valid_negative_duration_parses(self) -> None:
        status_bytes = _build_status_bytes(
            _pack_error_info_detail(reason="CWSANDBOX_BACKEND_UNAVAILABLE"),
            _pack_retry_info_detail(duration=Duration(seconds=-3, nanos=-500_000_000)),
        )
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay == timedelta(seconds=-3, microseconds=-500_000)

    def test_oversized_duration_out_of_range_returns_none(self) -> None:
        """Durations far outside timedelta's representable range fall back to None."""
        status_bytes = _build_status_bytes(
            _pack_error_info_detail(reason="CWSANDBOX_BACKEND_UNAVAILABLE"),
            _pack_retry_info_detail(duration=Duration(seconds=10**18)),
        )
        err = _make_error(trailing=[("grpc-status-details-bin", status_bytes)])
        parsed = parse_error_info(err)
        assert parsed is not None
        assert parsed.retry_delay is None


class TestTrailingMetadataIterationErrors:
    """L2: pathological trailing-metadata containers must not raise."""

    def test_iter_raises_returns_none(self) -> None:
        class RaisingIterable:
            def __iter__(self) -> RaisingIterable:
                raise RuntimeError("metadata iteration exploded")

        err = MagicMock()
        err.trailing_metadata.return_value = RaisingIterable()
        assert parse_error_info(err) is None

    def test_entry_with_raising_key_descriptor_returns_none(self) -> None:
        class RaisingEntry:
            @property
            def key(self) -> str:
                raise RuntimeError("key getter exploded")

            @property
            def value(self) -> bytes:
                return b""

        err = MagicMock()
        err.trailing_metadata.return_value = [RaisingEntry()]
        assert parse_error_info(err) is None
