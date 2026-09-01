# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._auth module."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cwsandbox._auth import (
    AuthHeaders,
    AuthStrategy,
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

    def test_default_does_not_attempt_wandb_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Omitted strategy keeps the existing CoreWeave-only behavior."""
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        with patch("cwsandbox._auth._resolve_wandb_auth") as resolve_wandb:
            auth = resolve_auth()

        assert auth.strategy == "none"
        assert auth.headers == {}
        resolve_wandb.assert_not_called()

    def test_explicit_coreweave_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CWSANDBOX_API_KEY", "test-key")

        auth = resolve_auth(AuthStrategy.COREWEAVE_API_KEY)

        assert auth.headers == {"Authorization": "Bearer test-key"}

    def test_explicit_coreweave_strategy_requires_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)

        with pytest.raises(CWSandboxAuthenticationError, match="CWSANDBOX_API_KEY"):
            resolve_auth(AuthStrategy.COREWEAVE_API_KEY)

    def test_explicit_wandb_strategy(self) -> None:
        expected = AuthHeaders(
            headers={"x-wandb-api-key": "wandb-key"},
            strategy="wandb_api_key",
        )

        with patch("cwsandbox._auth._resolve_wandb_auth", return_value=expected) as resolver:
            auth = resolve_auth(AuthStrategy.WANDB)

        assert auth is expected
        resolver.assert_called_once_with()

    def test_custom_provider_receives_base_url(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.base_url: str | None = None

            def resolve_auth(self, *, base_url: str) -> AuthHeaders:
                self.base_url = base_url
                return AuthHeaders(headers={"X-Test-Auth": "value"}, strategy="test")

        provider = Provider()

        auth = resolve_auth(provider, base_url="https://sandbox.example.test")

        assert auth.headers == {"X-Test-Auth": "value"}
        assert provider.base_url == "https://sandbox.example.test"

    def test_string_strategy_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="AuthStrategy"):
            resolve_auth("wandb")  # type: ignore[arg-type]

    def test_wandb_strategy_resolves_temp_netrc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """W&B's resolver supplies a host-scoped API key from .netrc."""
        pytest.importorskip("wandb")
        from wandb.sdk.lib import wbauth

        fake_api_key = "a" * 40
        netrc_path = tmp_path / "netrc"
        netrc_path.write_text(
            f"machine api.wandb.ai login user password {fake_api_key}\n",
            encoding="utf-8",
        )
        netrc_path.chmod(0o600)
        monkeypatch.setenv("NETRC", str(netrc_path))
        wbauth.unauthenticate_session(update_settings=False)

        try:
            auth = resolve_auth(AuthStrategy.WANDB)
        finally:
            wbauth.unauthenticate_session(update_settings=False)

        assert auth.headers["x-wandb-api-key"] == fake_api_key
        assert "x-wandb-sdk-version" in auth.headers
        assert auth.strategy == "wandb_api_key"

    def test_wandb_strategy_resolves_environment_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The existing WANDB_API_KEY flow remains supported."""
        pytest.importorskip("wandb")
        from wandb.sdk.lib import wbauth

        fake_api_key = "b" * 40
        monkeypatch.setenv("WANDB_API_KEY", fake_api_key)
        wbauth.unauthenticate_session(update_settings=False)

        try:
            auth = resolve_auth(AuthStrategy.WANDB)
        finally:
            wbauth.unauthenticate_session(update_settings=False)

        assert auth.headers["x-wandb-api-key"] == fake_api_key
        assert auth.strategy == "wandb_api_key"

    @pytest.mark.parametrize(
        ("base_url", "expected_host"),
        (
            ("https://api.wandb.ai", None),
            ("https://api.qa.wandb.ai", None),
            ("https://API.WANDB.AI", None),
            ("https://api.wandb.ai:443", None),
            ("https://acme.wandb.io", "acme.wandb.io"),
            ("https://user:password@acme.wandb.io", "acme.wandb.io"),
            ("https://acme.wandb.io:443", "acme.wandb.io"),
            ("https://acme.wandb.io:8443", "acme.wandb.io:8443"),
        ),
    )
    def test_wandb_strategy_forwards_dedicated_host(
        self,
        monkeypatch: pytest.MonkeyPatch,
        base_url: str,
        expected_host: str | None,
    ) -> None:
        """W&B's existing host classification drives dedicated routing metadata."""
        pytest.importorskip("wandb")
        from wandb.sdk import wandb_setup
        from wandb.sdk.lib import wbauth

        fake_api_key = "c" * 40
        monkeypatch.setenv("WANDB_API_KEY", fake_api_key)
        settings = SimpleNamespace(
            base_url=base_url,
            app_url=None,
            entity=None,
            project=None,
        )
        wbauth.unauthenticate_session(update_settings=False)

        try:
            with patch.object(
                wandb_setup,
                "singleton",
                return_value=SimpleNamespace(settings=settings),
            ):
                auth = resolve_auth(AuthStrategy.WANDB)
        finally:
            wbauth.unauthenticate_session(update_settings=False)

        if expected_host is None:
            assert "x-wandb-host" not in auth.headers
        else:
            assert auth.headers["x-wandb-host"] == expected_host

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
