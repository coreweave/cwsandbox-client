# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._auth module."""

import pytest

from cwsandbox._auth import (
    AuthHeaders,
    _reset_auth_mode_for_testing,
    resolve_auth,
    resolve_auth_metadata,
    set_auth_mode,
)
from cwsandbox.exceptions import CWSandboxAuthenticationError


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
    def set_auth_mode_fixture(self):
        def _set(name: str, get_auth) -> None:
            set_auth_mode(name, get_auth)

        yield _set

        _reset_auth_mode_for_testing()

    def test_builtin_api_key_auth_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the builtin auth mode uses CWSANDBOX_API_KEY when present."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")

        auth = resolve_auth()

        assert auth.strategy == "api_key"
        assert auth.headers == {"Authorization": "Bearer test-key"}

    def test_registered_auth_mode_overrides_built_in_api_key_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        set_auth_mode_fixture,
    ) -> None:
        """Test a registered auth mode overrides the built-in API-key auth."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")
        set_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "mode-key"}, strategy="auth_mode"),
        )

        auth = resolve_auth()

        assert auth.strategy == "auth_mode"
        assert auth.headers == {"x-api-key": "mode-key"}

    def test_registered_auth_mode_when_no_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        set_auth_mode_fixture,
    ) -> None:
        """Test a registered auth mode is used when no API key is set."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        set_auth_mode_fixture(
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
    """Tests for the single registered auth override in resolve_auth()."""

    @pytest.fixture
    def set_auth_mode_fixture(self):
        def _set(name: str, get_auth) -> None:
            set_auth_mode(name, get_auth)

        yield _set

        _reset_auth_mode_for_testing()

    def test_set_auth_mode_replaces_existing_auth_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        set_auth_mode_fixture,
    ) -> None:
        """Test the most recently set auth mode becomes active."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        set_auth_mode_fixture(
            "first-auth-mode",
            lambda: AuthHeaders(headers={"x-api-key": "first-key"}, strategy="first"),
        )
        set_auth_mode_fixture(
            "second-auth-mode",
            lambda: AuthHeaders(headers={"x-api-key": "second-key"}, strategy="second"),
        )

        auth = resolve_auth()

        assert auth.strategy == "second"
        assert auth.headers == {"x-api-key": "second-key"}

    def test_reset_auth_mode_for_testing_restores_builtin_auth_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        set_auth_mode_fixture,
    ) -> None:
        """Test resetting the active auth mode restores builtin auth resolution."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")
        set_auth_mode_fixture(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"x-api-key": "mode-key"}, strategy="auth_mode"),
        )
        _reset_auth_mode_for_testing()

        auth = resolve_auth()

        assert auth.strategy == "api_key"
        assert auth.headers == {"Authorization": "Bearer test-key"}

    def test_reset_auth_mode_for_testing_restores_default_when_already_builtin(self) -> None:
        """Test resetting while already on the builtin mode keeps default auth behavior."""
        _reset_auth_mode_for_testing()
        assert resolve_auth().strategy == "none"

    def test_registered_auth_mode_callback_errors_propagate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        set_auth_mode_fixture,
    ) -> None:
        """Test the active auth mode cannot silently fall back to no auth."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        def _raise_missing_auth() -> AuthHeaders:
            raise CWSandboxAuthenticationError("auth-mode-test missing credentials")

        set_auth_mode_fixture("auth-mode-test", _raise_missing_auth)

        with pytest.raises(
            CWSandboxAuthenticationError,
            match="auth-mode-test missing credentials",
        ):
            resolve_auth()

    def test_registered_auth_mode_returning_none_raises_auth_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        set_auth_mode_fixture,
    ) -> None:
        """Test a misbehaving auth mode cannot return None silently."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        set_auth_mode_fixture("auth-mode-test", lambda: None)  # type: ignore[arg-type]

        with pytest.raises(
            CWSandboxAuthenticationError,
            match="Configured auth mode auth-mode-test returned no credentials",
        ):
            resolve_auth()


class TestResolveAuthMetadata:
    """Tests for resolve_auth_metadata function."""

    def test_returns_lowercased_metadata_tuples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns metadata with lowercase keys as tuples."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")

        result = resolve_auth_metadata()

        assert result == (("authorization", "Bearer test-key"),)

    def test_returns_registered_auth_mode_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test registered auth mode metadata is lowercased for gRPC transport."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
        set_auth_mode(
            "auth-mode-test",
            lambda: AuthHeaders(headers={"X-Api-Key": "mode-key"}, strategy="auth_mode"),
        )

        try:
            result = resolve_auth_metadata()
        finally:
            _reset_auth_mode_for_testing()

        assert result == (("x-api-key", "mode-key"),)

    def test_returns_empty_tuple_when_no_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns empty tuple when no credentials found."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        result = resolve_auth_metadata()

        assert result == ()
