"""Unit tests for aviato._network module."""

from unittest.mock import MagicMock, patch

import pytest

from aviato._network import create_channel, parse_grpc_target


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
        """Test parsing the production Aviato URL."""
        target, is_secure = parse_grpc_target("https://atc.cwaviato.com")

        assert target == "atc.cwaviato.com:443"
        assert is_secure is True


class TestCreateChannel:
    """Tests for create_channel function."""

    @patch("aviato._network.grpc.aio.secure_channel")
    @patch("aviato._network.grpc.ssl_channel_credentials")
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
            interceptors=None,
        )
        assert result is mock_channel

    @patch("aviato._network.grpc.aio.insecure_channel")
    def test_insecure_channel(self, mock_insecure_channel: MagicMock) -> None:
        """Test creating an insecure channel."""
        mock_channel = MagicMock()
        mock_insecure_channel.return_value = mock_channel

        result = create_channel("localhost:50051", is_secure=False)

        mock_insecure_channel.assert_called_once_with(
            "localhost:50051",
            interceptors=None,
        )
        assert result is mock_channel

    @patch("aviato._network.grpc.aio.secure_channel")
    @patch("aviato._network.grpc.ssl_channel_credentials")
    def test_secure_channel_with_interceptors(
        self,
        mock_ssl_creds: MagicMock,
        mock_secure_channel: MagicMock,
    ) -> None:
        """Test creating a secure channel with interceptors."""
        mock_creds = MagicMock()
        mock_ssl_creds.return_value = mock_creds
        interceptor1 = MagicMock()
        interceptor2 = MagicMock()

        create_channel(
            "atc.example.com:443",
            is_secure=True,
            interceptors=[interceptor1, interceptor2],
        )

        mock_secure_channel.assert_called_once_with(
            "atc.example.com:443",
            mock_creds,
            interceptors=[interceptor1, interceptor2],
        )

    @patch("aviato._network.grpc.aio.insecure_channel")
    def test_insecure_channel_with_interceptors(
        self,
        mock_insecure_channel: MagicMock,
    ) -> None:
        """Test creating an insecure channel with interceptors."""
        interceptor = MagicMock()

        create_channel(
            "localhost:50051",
            is_secure=False,
            interceptors=[interceptor],
        )

        mock_insecure_channel.assert_called_once_with(
            "localhost:50051",
            interceptors=[interceptor],
        )

    @patch("aviato._network.grpc.aio.insecure_channel")
    def test_empty_interceptors_list(
        self,
        mock_insecure_channel: MagicMock,
    ) -> None:
        """Test that empty interceptors list passes None."""
        create_channel("localhost:50051", is_secure=False, interceptors=[])

        mock_insecure_channel.assert_called_once_with(
            "localhost:50051",
            interceptors=None,
        )
