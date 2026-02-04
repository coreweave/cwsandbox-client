"""Unit tests for aviato._auth module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from aviato._auth import (
    AuthHeaders,
    AuthInterceptor,
    WandbAuthError,
    _read_api_key_from_netrc,
    _try_aviato_auth,
    _try_wandb_auth,
    create_auth_interceptors,
    resolve_auth,
)
from aviato._defaults import DEFAULT_PROJECT_NAME, WANDB_NETRC_HOST


class TestAuthHeaders:
    """Tests for AuthHeaders dataclass."""

    def test_auth_headers_truthy_when_headers_present(self) -> None:
        """Test AuthHeaders is truthy when headers dict is non-empty."""
        auth = AuthHeaders(headers={"Authorization": "Bearer token"}, strategy="aviato")
        assert bool(auth) is True

    def test_auth_headers_falsy_when_empty(self) -> None:
        """Test AuthHeaders is falsy when headers dict is empty."""
        auth = AuthHeaders(headers={}, strategy="none")
        assert bool(auth) is False

    def test_auth_headers_is_frozen(self) -> None:
        """Test AuthHeaders is immutable."""
        auth = AuthHeaders(headers={}, strategy="none")
        with pytest.raises(AttributeError):
            auth.strategy = "aviato"  # type: ignore[misc]


class TestResolveAuth:
    """Tests for resolve_auth function."""

    def test_aviato_auth_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AVIATO_API_KEY takes priority over W&B credentials."""
        monkeypatch.setenv("AVIATO_API_KEY", "aviato-key")
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")

        auth = resolve_auth()

        assert auth.strategy == "aviato"
        assert auth.headers == {"Authorization": "Bearer aviato-key"}

    def test_wandb_auth_when_no_aviato(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test W&B auth is used when AVIATO_API_KEY is not set."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")
        monkeypatch.delenv("WANDB_PROJECT_NAME", raising=False)

        auth = resolve_auth()

        assert auth.strategy == "wandb"
        assert auth.headers == {
            "x-api-key": "wandb-key",
            "x-entity-id": "my-entity",
            "x-project-name": DEFAULT_PROJECT_NAME,
        }

    def test_wandb_auth_with_project_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test W&B auth includes custom project name when set."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")
        monkeypatch.setenv("WANDB_PROJECT_NAME", "my-project")

        auth = resolve_auth()

        assert auth.strategy == "wandb"
        assert auth.headers["x-project-name"] == "my-project"

    def test_no_auth_when_no_credentials(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test empty headers when no credentials are found."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)

        # Also mock netrc to ensure no credentials from there
        with patch("aviato._auth.Path.home", return_value=tmp_path):
            auth = resolve_auth()

        assert auth.strategy == "none"
        assert auth.headers == {}

    def test_raises_when_wandb_api_key_but_no_entity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_auth raises WandbAuthError when API key exists but entity missing."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)

        with pytest.raises(WandbAuthError, match="WANDB_ENTITY_NAME is not set"):
            resolve_auth()


class TestTryAviatoAuth:
    """Tests for _try_aviato_auth function."""

    def test_returns_auth_headers_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns AuthHeaders when AVIATO_API_KEY is set."""
        monkeypatch.setenv("AVIATO_API_KEY", "test-key")

        result = _try_aviato_auth()

        assert result is not None
        assert result.strategy == "aviato"
        assert result.headers == {"Authorization": "Bearer test-key"}

    def test_returns_none_when_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns None when AVIATO_API_KEY is not set."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)

        result = _try_aviato_auth()

        assert result is None


class TestTryWandbAuth:
    """Tests for _try_wandb_auth function."""

    def test_returns_auth_headers_with_all_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns AuthHeaders when all W&B env vars are set."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")
        monkeypatch.setenv("WANDB_PROJECT_NAME", "my-project")

        result = _try_wandb_auth()

        assert result is not None
        assert result.strategy == "wandb"
        assert result.headers == {
            "x-api-key": "wandb-key",
            "x-entity-id": "my-entity",
            "x-project-name": "my-project",
        }

    def test_raises_when_api_key_env_but_no_entity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test raises WandbAuthError when API key in env but entity is missing."""
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")

        with pytest.raises(WandbAuthError, match="WANDB_ENTITY_NAME is not set"):
            _try_wandb_auth()

    def test_raises_when_api_key_netrc_but_no_entity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test raises WandbAuthError when API key in netrc but entity is missing."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)

        # Create netrc with API key
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password netrc-key\n")

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            with pytest.raises(WandbAuthError, match="WANDB_ENTITY_NAME is not set"):
                _try_wandb_auth()

    def test_returns_none_when_no_api_key_and_no_entity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns None when neither API key nor entity is set."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)

        # Also mock netrc to ensure no credentials from there
        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is None

    def test_defaults_project_name_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test project name uses default when not set."""
        monkeypatch.delenv("WANDB_PROJECT_NAME", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")

        result = _try_wandb_auth()

        assert result is not None
        assert result.headers["x-project-name"] == DEFAULT_PROJECT_NAME

    def test_falls_back_to_netrc(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test falls back to netrc when WANDB_API_KEY is not set."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")

        # Create a mock netrc file
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password netrc-key\n")

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is not None
        assert result.headers["x-api-key"] == "netrc-key"

    def test_returns_none_when_no_api_key_anywhere(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns None when no API key in env or netrc."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_ENTITY_NAME", "my-entity")

        # No netrc file
        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is None


class TestReadApiKeyFromNetrc:
    """Tests for _read_api_key_from_netrc function."""

    def test_reads_password_from_netrc(self, tmp_path: Path) -> None:
        """Test successfully reads password from netrc file."""
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password my-api-key\n")

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result == "my-api-key"

    def test_returns_none_when_no_netrc_file(self, tmp_path: Path) -> None:
        """Test returns None when .netrc file doesn't exist."""
        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result is None

    def test_returns_none_when_no_wandb_entry(self, tmp_path: Path) -> None:
        """Test returns None when netrc has no api.wandb.ai entry."""
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text("machine other.host.com\n  login user\n  password key\n")

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result is None

    def test_handles_malformed_netrc(self, tmp_path: Path) -> None:
        """Test handles malformed netrc file gracefully."""
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text("this is not valid netrc format {{{}}")

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result is None


class TestAuthInterceptor:
    """Tests for AuthInterceptor class."""

    def test_lowercases_header_keys(self) -> None:
        """Test that header keys are normalized to lowercase."""
        headers = {"Authorization": "Bearer token", "X-Api-Key": "key"}
        interceptor = AuthInterceptor(headers)

        assert interceptor._metadata == (
            ("authorization", "Bearer token"),
            ("x-api-key", "key"),
        )

    def test_empty_headers(self) -> None:
        """Test interceptor works with empty headers."""
        interceptor = AuthInterceptor({})
        assert interceptor._metadata == ()

    def test_add_metadata_merges_with_existing(self) -> None:
        """Test _add_metadata merges auth metadata with existing metadata."""
        from unittest.mock import MagicMock

        headers = {"authorization": "Bearer token"}
        interceptor = AuthInterceptor(headers)

        mock_details = MagicMock()
        mock_details.metadata = (("x-existing", "value"),)
        mock_details._replace = MagicMock(return_value="new_details")

        result = interceptor._add_metadata(mock_details)

        mock_details._replace.assert_called_once_with(
            metadata=(("x-existing", "value"), ("authorization", "Bearer token"))
        )
        assert result == "new_details"

    def test_add_metadata_handles_none_existing(self) -> None:
        """Test _add_metadata handles None existing metadata."""
        from unittest.mock import MagicMock

        headers = {"authorization": "Bearer token"}
        interceptor = AuthInterceptor(headers)

        mock_details = MagicMock()
        mock_details.metadata = None
        mock_details._replace = MagicMock(return_value="new_details")

        result = interceptor._add_metadata(mock_details)

        mock_details._replace.assert_called_once_with(metadata=(("authorization", "Bearer token"),))
        assert result == "new_details"

    @pytest.mark.asyncio
    async def test_intercept_unary_unary_adds_metadata(self) -> None:
        """Test intercept_unary_unary adds auth metadata to calls."""
        from unittest.mock import AsyncMock, MagicMock

        headers = {"authorization": "Bearer test-token"}
        interceptor = AuthInterceptor(headers)

        mock_call_details = MagicMock()
        mock_call_details.metadata = None
        new_details = MagicMock()
        mock_call_details._replace = MagicMock(return_value=new_details)

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_continuation = AsyncMock(return_value=mock_response)

        result = await interceptor.intercept_unary_unary(
            mock_continuation, mock_call_details, mock_request
        )

        mock_continuation.assert_called_once_with(new_details, mock_request)
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_intercept_stream_stream_adds_metadata(self) -> None:
        """Test intercept_stream_stream adds auth metadata to calls."""
        from unittest.mock import MagicMock

        headers = {"authorization": "Bearer test-token"}
        interceptor = AuthInterceptor(headers)

        mock_call_details = MagicMock()
        mock_call_details.metadata = None
        new_details = MagicMock()
        mock_call_details._replace = MagicMock(return_value=new_details)

        mock_request_iterator = MagicMock()
        mock_stream_call = MagicMock()
        # StreamStreamCall is not awaitable - continuation returns it directly
        mock_continuation = MagicMock(return_value=mock_stream_call)

        result = await interceptor.intercept_stream_stream(
            mock_continuation, mock_call_details, mock_request_iterator
        )

        mock_continuation.assert_called_once_with(new_details, mock_request_iterator)
        assert result == mock_stream_call


class TestCreateAuthInterceptors:
    """Tests for create_auth_interceptors function."""

    def test_returns_list_with_single_interceptor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns a list with one interceptor."""
        monkeypatch.setenv("AVIATO_API_KEY", "test-key")

        result = create_auth_interceptors()

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], AuthInterceptor)

    def test_interceptor_has_resolved_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the interceptor contains resolved auth metadata with lowercase keys."""
        monkeypatch.setenv("AVIATO_API_KEY", "test-key")

        result = create_auth_interceptors()

        # Keys should be lowercased for gRPC
        assert result[0]._metadata == (("authorization", "Bearer test-key"),)

    def test_returns_interceptor_with_empty_metadata_when_no_auth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns interceptor with empty metadata when no credentials."""
        monkeypatch.delenv("AVIATO_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY_NAME", raising=False)

        with patch("aviato._auth.Path.home", return_value=tmp_path):
            result = create_auth_interceptors()

        assert len(result) == 1
        assert result[0]._metadata == ()
