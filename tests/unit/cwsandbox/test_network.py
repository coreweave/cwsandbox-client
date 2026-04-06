# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._network module."""

from unittest.mock import MagicMock, patch

import grpc
import pytest

from cwsandbox._network import create_channel, parse_grpc_target, translate_grpc_error
from cwsandbox.exceptions import (
    CWSandboxAuthenticationError,
    CWSandboxError,
    DiscoveryError,
    SandboxError,
)


class TestParseGrpcTarget:
    """Tests for parse_grpc_target function."""

    def test_https_with_explicit_port(self) -> None:
        """Test parsing HTTPS URL with explicit port."""
        target, is_secure = parse_grpc_target("https://atc.example.com:8443")

        assert target == "atc.example.com:8443"
        assert is_secure is True

    def test_https_default_port(self) -> None:
        """Test parsing HTTPS URL with default port 443."""
        target, is_secure = parse_grpc_target("https://atc.example.com")

        assert target == "atc.example.com:443"
        assert is_secure is True

    def test_http_with_explicit_port(self) -> None:
        """Test parsing HTTP URL with explicit port."""
        target, is_secure = parse_grpc_target("http://localhost:50051")

        assert target == "localhost:50051"
        assert is_secure is False

    def test_http_default_port(self) -> None:
        """Test parsing HTTP URL with default port 80."""
        target, is_secure = parse_grpc_target("http://localhost")

        assert target == "localhost:80"
        assert is_secure is False

    def test_https_with_trailing_slash(self) -> None:
        """Test parsing URL with trailing slash (allowed)."""
        target, is_secure = parse_grpc_target("https://atc.example.com/")

        assert target == "atc.example.com:443"
        assert is_secure is True

    def test_rejects_url_with_path(self) -> None:
        """Test that URLs with paths are rejected."""
        with pytest.raises(ValueError, match="gRPC does not support URL paths"):
            parse_grpc_target("https://atc.example.com/api/v1")

    def test_rejects_url_with_nested_path(self) -> None:
        """Test that URLs with nested paths are rejected."""
        with pytest.raises(ValueError, match="gRPC does not support URL paths"):
            parse_grpc_target("https://atc.example.com/foo/bar/baz")

    def test_rejects_invalid_scheme(self) -> None:
        """Test that non-HTTP schemes are rejected."""
        with pytest.raises(ValueError, match="URL must use http or https scheme"):
            parse_grpc_target("grpc://atc.example.com")

    def test_rejects_ftp_scheme(self) -> None:
        """Test that FTP scheme is rejected."""
        with pytest.raises(ValueError, match="URL must use http or https scheme"):
            parse_grpc_target("ftp://files.example.com")

    def test_rejects_missing_hostname(self) -> None:
        """Test that URLs without hostname are rejected."""
        with pytest.raises(ValueError, match="URL must have a hostname"):
            parse_grpc_target("https://")

    def test_production_url(self) -> None:
        """Test parsing the production CWSandbox URL."""
        target, is_secure = parse_grpc_target("https://api.cwsandbox.com")

        assert target == "api.cwsandbox.com:443"
        assert is_secure is True


class TestCreateChannel:
    """Tests for create_channel function."""

    @patch("cwsandbox._network.grpc.aio.secure_channel")
    @patch("cwsandbox._network.grpc.ssl_channel_credentials")
    def test_secure_channel_with_tls(
        self,
        mock_ssl_creds: MagicMock,
        mock_secure_channel: MagicMock,
    ) -> None:
        """Test creating a secure channel with TLS."""
        mock_creds = MagicMock()
        mock_ssl_creds.return_value = mock_creds
        mock_channel = MagicMock()
        mock_secure_channel.return_value = mock_channel

        result = create_channel("atc.example.com:443", is_secure=True)

        mock_ssl_creds.assert_called_once()
        mock_secure_channel.assert_called_once_with(
            "atc.example.com:443",
            mock_creds,
        )
        assert result is mock_channel

    @patch("cwsandbox._network.grpc.aio.insecure_channel")
    def test_insecure_channel(self, mock_insecure_channel: MagicMock) -> None:
        """Test creating an insecure channel."""
        mock_channel = MagicMock()
        mock_insecure_channel.return_value = mock_channel

        result = create_channel("localhost:50051", is_secure=False)

        mock_insecure_channel.assert_called_once_with("localhost:50051")
        assert result is mock_channel


def _make_rpc_error(code: grpc.StatusCode, details: str) -> grpc.RpcError:
    err = MagicMock()
    err.code.return_value = code
    err.details.return_value = details
    return err


class TestTranslateGrpcError:
    """Tests for translate_grpc_error shared helper."""

    def test_unauthenticated(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.UNAUTHENTICATED, "bad token")
        result = translate_grpc_error(err)
        assert isinstance(result, CWSandboxAuthenticationError)
        assert "bad token" in str(result)

    def test_permission_denied(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.PERMISSION_DENIED, "forbidden")
        result = translate_grpc_error(err)
        assert isinstance(result, CWSandboxAuthenticationError)
        assert "forbidden" in str(result)

    def test_deadline_exceeded(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "slow")
        result = translate_grpc_error(err, operation="List towers")
        assert isinstance(result, CWSandboxError)
        assert "timed out" in str(result)
        assert "slow" in str(result)

    def test_unavailable(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.UNAVAILABLE, "server down")
        result = translate_grpc_error(err)
        assert isinstance(result, CWSandboxError)
        assert "unavailable" in str(result).lower()

    def test_other_code(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.INTERNAL, "oops")
        result = translate_grpc_error(err, operation="Get tower")
        assert isinstance(result, CWSandboxError)
        assert "oops" in str(result)

    def test_returns_not_raises(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.INTERNAL, "oops")
        result = translate_grpc_error(err)
        assert isinstance(result, Exception)  # returned, not raised

    # -- fallback_cls parameter tests --

    def test_fallback_cls_sandbox_error_for_unavailable(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.UNAVAILABLE, "server down")
        result = translate_grpc_error(err, fallback_cls=SandboxError)
        assert isinstance(result, SandboxError)
        assert "unavailable" in str(result).lower()

    def test_fallback_cls_sandbox_error_for_deadline(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "slow")
        result = translate_grpc_error(err, fallback_cls=SandboxError)
        assert isinstance(result, SandboxError)
        assert "timed out" in str(result)

    def test_fallback_cls_sandbox_error_for_other(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.INTERNAL, "oops")
        result = translate_grpc_error(err, fallback_cls=SandboxError)
        assert isinstance(result, SandboxError)

    def test_fallback_cls_discovery_error_for_unavailable(self) -> None:
        err = _make_rpc_error(grpc.StatusCode.UNAVAILABLE, "server down")
        result = translate_grpc_error(err, fallback_cls=DiscoveryError)
        assert isinstance(result, DiscoveryError)

    def test_fallback_cls_does_not_affect_auth_errors(self) -> None:
        for code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
            err = _make_rpc_error(code, "denied")
            result = translate_grpc_error(err, fallback_cls=SandboxError)
            assert isinstance(result, CWSandboxAuthenticationError)
            assert not isinstance(result, SandboxError)
