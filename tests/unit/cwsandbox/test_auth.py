# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._auth module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cwsandbox._auth import (
    AuthHeaders,
    _read_api_key_from_netrc,
    _try_api_key_auth,
    _try_wandb_auth,
    register_auth_mode,
    resolve_auth,
    resolve_auth_metadata,
    unregister_auth_mode,
)
from cwsandbox._defaults import WANDB_NETRC_HOST


class TestAuthHeaders:
    """Tests for AuthHeaders dataclass."""

    def test_auth_headers_truthy_when_headers_present(self) -> None:
        """Test AuthHeaders is truthy when headers dict is non-empty."""
        auth = AuthHeaders(headers={"Authorization": "Bearer token"}, strategy="api_key")
        assert bool(auth) is True

    def test_auth_headers_falsy_when_empty(self) -> None:
        """Test AuthHeaders is falsy when headers dict is empty."""
        auth = AuthHeaders(headers={}, strategy="none")
        assert bool(auth) is False

    def test_auth_headers_is_frozen(self) -> None:
        """Test AuthHeaders is immutable."""
        auth = AuthHeaders(headers={}, strategy="none")
        with pytest.raises(AttributeError):
            auth.strategy = "api_key"  # type: ignore[misc]


class TestResolveAuth:
    """Tests for resolve_auth function."""

    def test_api_key_auth_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CWSANDBOX_API_KEY takes priority over W&B credentials."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")

        auth = resolve_auth()

        assert auth.strategy == "api_key"
        assert auth.headers == {"Authorization": "Bearer test-key"}

    def test_wandb_auth_when_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test W&B auth is used when CWSANDBOX_API_KEY is not set."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        auth = resolve_auth()

        assert auth.strategy == "wandb"
        assert auth.headers == {
            "x-api-key": "wandb-key",
            "x-entity-id": "my-entity",
        }

    def test_wandb_auth_with_project_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test W&B auth includes custom project name when set."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")
        monkeypatch.setenv("WANDB_PROJECT", "my-project")

        auth = resolve_auth()

        assert auth.strategy == "wandb"
        assert auth.headers["x-project-name"] == "my-project"

    def test_no_auth_when_no_credentials(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test empty headers when no credentials are found."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY", raising=False)

        # Also mock netrc to ensure no credentials from there
        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            auth = resolve_auth()

        assert auth.strategy == "none"
        assert auth.headers == {}

    def test_wandb_auth_succeeds_with_api_key_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_auth succeeds with only WANDB_API_KEY (entity and project optional)."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        auth = resolve_auth()

        assert auth.strategy == "wandb"
        assert auth.headers == {"x-api-key": "wandb-key"}


class TestRegisteredAuthModes:
    """Tests for registered auth modes in resolve_auth()."""

    @pytest.fixture
    def register_auth_mode_fixture(self):
        registered_names: list[str] = []

        def _register(name: str, try_auth) -> None:
            register_auth_mode(name, try_auth)
            registered_names.append(name)

        yield _register

        for name in reversed(registered_names):
            unregister_auth_mode(name)

    def test_registered_auth_mode_takes_priority_over_builtin_wandb_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test registered auth modes are checked before builtin W&B fallback."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")

        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(
                headers={"x-api-key": "mode-key", "x-project-name": "mode-project"},
                strategy="auth_mode",
            ),
        )

        auth = resolve_auth()

        assert auth.strategy == "auth_mode"
        assert auth.headers["x-api-key"] == "mode-key"
        assert auth.headers["x-project-name"] == "mode-project"

    def test_registered_auth_mode_does_not_override_api_key_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test explicit CoreWeave auth still wins over registered auth modes."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "cw-key")

        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "mode-key"}, strategy="auth_mode"),
        )

        auth = resolve_auth()

        assert auth.strategy == "api_key"
        assert auth.headers == {"Authorization": "Bearer cw-key"}

    def test_register_auth_mode_is_idempotent_by_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test registering the same auth mode name twice keeps the first mode."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "first-key"}, strategy="first"),
        )
        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "second-key"}, strategy="second"),
        )

        auth = resolve_auth()

        assert auth.strategy == "first"
        assert auth.headers == {"x-api-key": "first-key"}

    def test_unregister_auth_mode_removes_registered_auth_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
        tmp_path: Path,
    ) -> None:
        """Test unregister_auth_mode removes the auth mode from resolution."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "mode-key"}, strategy="auth_mode"),
        )
        unregister_auth_mode("auth-mode-test")

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            auth = resolve_auth()

        assert auth.strategy == "none"
        assert auth.headers == {}


class TestTryApiKeyAuth:
    """Tests for _try_api_key_auth function."""

    def test_returns_auth_headers_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns AuthHeaders when CWSANDBOX_API_KEY is set."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")

        result = _try_api_key_auth()

        assert result is not None
        assert result.strategy == "api_key"
        assert result.headers == {"Authorization": "Bearer test-key"}

    def test_returns_none_when_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns None when CWSANDBOX_API_KEY is not set."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        result = _try_api_key_auth()

        assert result is None


class TestTryWandbAuth:
    """Tests for _try_wandb_auth function."""

    def test_returns_auth_headers_with_all_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns AuthHeaders when all W&B env vars are set."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")
        monkeypatch.setenv("WANDB_PROJECT", "my-project")

        result = _try_wandb_auth()

        assert result is not None
        assert result.strategy == "wandb"
        assert result.headers == {
            "x-api-key": "wandb-key",
            "x-entity-id": "my-entity",
            "x-project-name": "my-project",
        }

    def test_returns_auth_with_api_key_only_no_entity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns AuthHeaders with only x-api-key when entity and project are not set."""
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")

        result = _try_wandb_auth()

        assert result is not None
        assert result.headers == {"x-api-key": "wandb-key"}

    def test_returns_auth_from_netrc_when_no_entity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns AuthHeaders from netrc with only x-api-key when entity/project not set."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password netrc-key\n")

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is not None
        assert result.headers == {"x-api-key": "netrc-key"}

    def test_returns_none_when_no_api_key_and_no_entity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns None when neither API key nor entity is set."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY", raising=False)

        # Also mock netrc to ensure no credentials from there
        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is None

    def test_omits_project_header_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test x-project-name header is omitted when WANDB_PROJECT not set."""
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")

        result = _try_wandb_auth()

        assert result is not None
        assert "x-project-name" not in result.headers

    def test_falls_back_to_netrc(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test falls back to netrc when WANDB_API_KEY is not set."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")

        # Create a mock netrc file
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password netrc-key\n")

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is not None
        assert result.headers["x-api-key"] == "netrc-key"

    def test_returns_none_when_no_api_key_anywhere(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns None when no API key in env or netrc."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")

        # No netrc file
        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _try_wandb_auth()

        assert result is None


class TestReadApiKeyFromNetrc:
    """Tests for _read_api_key_from_netrc function."""

    def test_reads_password_from_netrc(self, tmp_path: Path) -> None:
        """Test successfully reads password from netrc file."""
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(f"machine {WANDB_NETRC_HOST}\n  login user\n  password my-api-key\n")

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result == "my-api-key"

    def test_returns_none_when_no_netrc_file(self, tmp_path: Path) -> None:
        """Test returns None when .netrc file doesn't exist."""
        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result is None

    def test_returns_none_when_no_wandb_entry(self, tmp_path: Path) -> None:
        """Test returns None when netrc has no api.wandb.ai entry."""
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text("machine other.host.com\n  login user\n  password key\n")

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result is None

    def test_handles_malformed_netrc(self, tmp_path: Path) -> None:
        """Test handles malformed netrc file gracefully."""
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text("this is not valid netrc format {{{}}")

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = _read_api_key_from_netrc()

        assert result is None


class TestResolveAuthMetadata:
    """Tests for resolve_auth_metadata function."""

    def test_returns_lowercased_metadata_tuples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns metadata with lowercase keys as tuples."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")

        result = resolve_auth_metadata()

        assert result == (("authorization", "Bearer test-key"),)

    def test_returns_empty_tuple_when_no_auth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test returns empty tuple when no credentials found."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_ENTITY", raising=False)

        with patch("cwsandbox._auth.Path.home", return_value=tmp_path):
            result = resolve_auth_metadata()

        assert result == ()

    def test_wandb_metadata_has_correct_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test W&B metadata contains expected keys (no x-project-name when WANDB_PROJECT unset)."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
        monkeypatch.setenv("WANDB_ENTITY", "my-entity")
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        result = resolve_auth_metadata()

        keys = {k for k, v in result}
        assert keys == {"x-api-key", "x-entity-id"}
