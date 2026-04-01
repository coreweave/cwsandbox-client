# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._auth module."""

import pytest

from cwsandbox._auth import (
    AuthHeaders,
    _try_api_key_auth,
    register_auth_mode,
    resolve_auth,
    resolve_auth_metadata,
    unregister_auth_mode,
)


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

    @pytest.fixture
    def register_auth_mode_fixture(self):
        registered_names: list[str] = []

        def _register(name: str, try_auth) -> None:
            register_auth_mode(name, try_auth)
            registered_names.append(name)

        yield _register

        for name in reversed(registered_names):
            unregister_auth_mode(name)

    def test_api_key_auth_takes_priority_over_registered_auth_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test CWSANDBOX_API_KEY takes priority over registered auth modes."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")
        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "mode-key"}, strategy="auth_mode"),
        )

        auth = resolve_auth()

        assert auth.strategy == "api_key"
        assert auth.headers == {"Authorization": "Bearer test-key"}

    def test_registered_auth_mode_when_no_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test a registered auth mode is used when no API key is set."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(
                headers={"x-api-key": "mode-key", "x-project-name": "mode-project"},
                strategy="auth_mode",
            ),
        )

        auth = resolve_auth()

        assert auth.strategy == "auth_mode"
        assert auth.headers == {
            "x-api-key": "mode-key",
            "x-project-name": "mode-project",
        }

    def test_no_auth_when_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test empty headers when no credentials are found."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        auth = resolve_auth()

        assert auth.strategy == "none"
        assert auth.headers == {}


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

    def test_register_auth_mode_is_idempotent_by_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test registering the same auth mode name twice keeps the first mode."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

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
    ) -> None:
        """Test unregister_auth_mode removes the auth mode from resolution."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "mode-key"}, strategy="auth_mode"),
        )
        unregister_auth_mode("auth-mode-test")

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


class TestResolveAuthMetadata:
    """Tests for resolve_auth_metadata function."""

    @pytest.fixture
    def register_auth_mode_fixture(self):
        registered_names: list[str] = []

        def _register(name: str, try_auth) -> None:
            register_auth_mode(name, try_auth)
            registered_names.append(name)

        yield _register

        for name in reversed(registered_names):
            unregister_auth_mode(name)

    def test_returns_lowercased_metadata_tuples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns metadata with lowercase keys as tuples."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")

        result = resolve_auth_metadata()

        assert result == (("authorization", "Bearer test-key"),)

    def test_returns_registered_auth_mode_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        register_auth_mode_fixture,
    ) -> None:
        """Test registered auth mode metadata is lowercased for gRPC transport."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        register_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"X-Api-Key": "mode-key"}, strategy="auth_mode"),
        )

        result = resolve_auth_metadata()

        assert result == (("x-api-key", "mode-key"),)

    def test_returns_empty_tuple_when_no_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns empty tuple when no credentials found."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        result = resolve_auth_metadata()

        assert result == ()
