"""Unit tests for aviato._interceptor module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aviato._interceptor import AuthHeaderInterceptor, create_auth_interceptors


class TestAuthHeaderInterceptor:
    """Tests for AuthHeaderInterceptor class."""

    def test_stores_headers(self) -> None:
        """Test interceptor stores headers dict."""
        headers = {"Authorization": "Bearer token"}
        interceptor = AuthHeaderInterceptor(headers)
        assert interceptor._headers == headers

    def test_stores_empty_headers(self) -> None:
        """Test interceptor handles empty headers dict."""
        interceptor = AuthHeaderInterceptor({})
        assert interceptor._headers == {}

    @pytest.mark.asyncio
    async def test_on_start_adds_single_header(self) -> None:
        """Test on_start adds a single header to request context."""
        headers = {"Authorization": "Bearer token"}
        interceptor = AuthHeaderInterceptor(headers)

        # Create mock request context
        mock_headers: dict[str, str] = {}
        mock_ctx = MagicMock()
        mock_ctx.request_headers.return_value = mock_headers

        await interceptor.on_start(mock_ctx)

        assert mock_headers == {"Authorization": "Bearer token"}

    @pytest.mark.asyncio
    async def test_on_start_adds_multiple_headers(self) -> None:
        """Test on_start adds multiple headers to request context."""
        headers = {
            "x-api-key": "wandb-key",
            "x-entity-id": "my-entity",
            "x-project-name": "my-project",
        }
        interceptor = AuthHeaderInterceptor(headers)

        mock_headers: dict[str, str] = {}
        mock_ctx = MagicMock()
        mock_ctx.request_headers.return_value = mock_headers

        await interceptor.on_start(mock_ctx)

        assert mock_headers == headers

    @pytest.mark.asyncio
    async def test_on_start_with_empty_headers(self) -> None:
        """Test on_start handles empty headers gracefully."""
        interceptor = AuthHeaderInterceptor({})

        mock_headers: dict[str, str] = {}
        mock_ctx = MagicMock()
        mock_ctx.request_headers.return_value = mock_headers

        await interceptor.on_start(mock_ctx)

        assert mock_headers == {}

    @pytest.mark.asyncio
    async def test_on_start_overwrites_existing_headers(self) -> None:
        """Test on_start overwrites existing headers with same key."""
        headers = {"Authorization": "Bearer new-token"}
        interceptor = AuthHeaderInterceptor(headers)

        mock_headers: dict[str, str] = {"Authorization": "Bearer old-token"}
        mock_ctx = MagicMock()
        mock_ctx.request_headers.return_value = mock_headers

        await interceptor.on_start(mock_ctx)

        assert mock_headers["Authorization"] == "Bearer new-token"

    @pytest.mark.asyncio
    async def test_on_end_is_noop(self) -> None:
        """Test on_end does nothing (no-op)."""
        interceptor = AuthHeaderInterceptor({})
        mock_ctx = MagicMock()

        # Should not raise
        await interceptor.on_end(None, mock_ctx)


class TestCreateAuthInterceptors:
    """Tests for create_auth_interceptors factory function."""

    def test_creates_interceptor_with_aviato_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creates interceptor with Aviato API key headers."""
        monkeypatch.setenv("AVIATO_API_KEY", "test-api-key")

        interceptors = create_auth_interceptors()

        assert len(interceptors) == 1
        assert isinstance(interceptors[0], AuthHeaderInterceptor)
        assert interceptors[0]._headers == {"Authorization": "Bearer test-api-key"}

    def test_creates_interceptor_with_wandb_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creates interceptor with W&B headers."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")
        monkeypatch.setenv("WANDB_PROJECT_NAME", "my-project")

        interceptors = create_auth_interceptors()

        assert len(interceptors) == 1
        assert isinstance(interceptors[0], AuthHeaderInterceptor)
        assert interceptors[0]._headers == {
            "x-api-key": "wandb-key",
            "x-entity-id": "my-entity",
            "x-project-name": "my-project",
        }

    def test_creates_interceptor_with_empty_headers_when_no_auth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test creates interceptor with empty headers when no credentials."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            interceptors = create_auth_interceptors()

        assert len(interceptors) == 1
        assert isinstance(interceptors[0], AuthHeaderInterceptor)
        assert interceptors[0]._headers == {}

    def test_returns_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns a list (for use with connectrpc interceptors param)."""
        monkeypatch.setenv("AVIATO_API_KEY", "test-key")

        result = create_auth_interceptors()

        assert isinstance(result, list)


class TestMetadataInterceptorProtocol:
    """Tests verifying AuthHeaderInterceptor implements MetadataInterceptor protocol."""

    def test_has_on_start_method(self) -> None:
        """Test interceptor has on_start method."""
        interceptor = AuthHeaderInterceptor({})
        assert hasattr(interceptor, "on_start")
        assert callable(interceptor.on_start)

    def test_has_on_end_method(self) -> None:
        """Test interceptor has on_end method."""
        interceptor = AuthHeaderInterceptor({})
        assert hasattr(interceptor, "on_end")
        assert callable(interceptor.on_end)

    def test_is_runtime_checkable_as_metadata_interceptor(self) -> None:
        """Test interceptor passes runtime check for MetadataInterceptor protocol."""
        from connectrpc.interceptor import MetadataInterceptor

        interceptor = AuthHeaderInterceptor({})
        # MetadataInterceptor is a Protocol with @runtime_checkable
        assert isinstance(interceptor, MetadataInterceptor)
