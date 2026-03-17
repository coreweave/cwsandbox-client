# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox tower create-join-token CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cwsandbox.cli import cli

# Shared mock API response
_TOKEN_RESPONSE = {
    "tower_id": "my-tower",
    "tower_group_id": "default",
    "organization_id": "org-123",
    "token_id": "tok-abc",
    "expires_at": "2026-03-18T00:00:00Z",
    "token": "secret-join-token-value",
}


def _invoke(args: list[str], *, input: str | None = None) -> object:
    """Run the CLI with the given args."""
    runner = CliRunner()
    return runner.invoke(cli, ["tower", "create-join-token"] + args, input=input)


class TestCreateJoinTokenHelp:
    """Basic CLI registration tests."""

    def test_help_exits_zero(self) -> None:
        result = _invoke(["--help"])
        assert result.exit_code == 0
        assert "Create a join token" in result.output


class TestCreateJoinTokenValidation:
    """Input validation tests."""

    def test_missing_api_key(self) -> None:
        result = _invoke(["--tower-id=my-tower"])
        assert result.exit_code == 1
        assert "--api-key or CWSANDBOX_API_KEY" in result.output

    def test_missing_tower_id(self) -> None:
        result = _invoke(["--api-key=test-key"])
        assert result.exit_code == 1
        assert "--tower-id is required" in result.output

    def test_missing_tower_id_empty_json(self) -> None:
        result = _invoke(["--api-key=test-key", "--json={}"])
        assert result.exit_code == 1
        assert "--tower-id is required" in result.output

    def test_invalid_json(self) -> None:
        result = _invoke(["--api-key=test-key", "--json=not-json"])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CWSANDBOX_API_KEY env var is accepted."""
        monkeypatch.setenv("CWSANDBOX_API_KEY", "env-key")
        with (
            patch(
                "cwsandbox.cli.tower_create_join_token._create_token",
                return_value=_TOKEN_RESPONSE,
            ),
            patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret"),
        ):
            result = _invoke(["--tower-id=my-tower"])
        assert result.exit_code == 0


class TestCreateJoinTokenGenerateOnly:
    """Tests for --generate-only mode."""

    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_generate_only_prints_token_info(self, mock_create: MagicMock) -> None:
        result = _invoke(["--tower-id=my-tower", "--api-key=test-key", "--generate-only"])
        assert result.exit_code == 0
        assert "Tower ID:       my-tower" in result.output
        assert "Token:          secret-join-token-value" in result.output
        assert "Token ID:       tok-abc" in result.output

    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    def test_generate_only_does_not_store(
        self, mock_store: MagicMock, mock_create: MagicMock
    ) -> None:
        result = _invoke(["--tower-id=my-tower", "--api-key=test-key", "--generate-only"])
        assert result.exit_code == 0
        mock_store.assert_not_called()


class TestCreateJoinTokenNormalFlow:
    """Tests for the normal (non-generate-only) flow."""

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_creates_token_and_stores_secret(
        self, mock_create: MagicMock, mock_store: MagicMock
    ) -> None:
        result = _invoke(["--tower-id=my-tower", "--api-key=test-key"])
        assert result.exit_code == 0
        mock_create.assert_called_once()
        mock_store.assert_called_once_with(
            token="secret-join-token-value",
            kubeconfig=None,
            namespace="sandbox-system",
            secret_name="sandbox-tower-join-token",
            secret_key="token",
        )

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_custom_k8s_options(self, mock_create: MagicMock, mock_store: MagicMock) -> None:
        result = _invoke(
            [
                "--tower-id=my-tower",
                "--api-key=test-key",
                "--namespace=custom-ns",
                "--secret-name=my-secret",
                "--secret-key=my-key",
                "--kubeconfig=/tmp/kubeconfig",
            ]
        )
        assert result.exit_code == 0
        mock_store.assert_called_once_with(
            token="secret-join-token-value",
            kubeconfig="/tmp/kubeconfig",
            namespace="custom-ns",
            secret_name="my-secret",
            secret_key="my-key",
        )

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_request_body_includes_optional_fields(
        self, mock_create: MagicMock, mock_store: MagicMock
    ) -> None:
        result = _invoke(
            [
                "--tower-id=my-tower",
                "--api-key=test-key",
                "--tower-group-id=gpu-group",
                "--ttl=3600",
                "--description=test token",
            ]
        )
        assert result.exit_code == 0
        body = mock_create.call_args[0][2]
        assert body == {
            "tower_id": "my-tower",
            "tower_group_id": "gpu-group",
            "ttl_seconds": 3600,
            "description": "test token",
        }

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token")
    def test_missing_token_in_response(self, mock_create: MagicMock, mock_store: MagicMock) -> None:
        mock_create.return_value = {"tower_id": "my-tower"}
        result = _invoke(["--tower-id=my-tower", "--api-key=test-key"])
        assert result.exit_code == 1
        assert "missing 'token' field" in result.output
        mock_store.assert_not_called()


class TestJsonPayload:
    """Tests for --json payload parsing and flag precedence."""

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_json_provides_tower_id(self, mock_create: MagicMock, mock_store: MagicMock) -> None:
        result = _invoke(
            [
                "--api-key=test-key",
                '--json={"tower_id":"json-tower","ttl_seconds":7200}',
            ]
        )
        assert result.exit_code == 0
        body = mock_create.call_args[0][2]
        assert body["tower_id"] == "json-tower"
        assert body["ttl_seconds"] == 7200

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_flags_override_json(self, mock_create: MagicMock, mock_store: MagicMock) -> None:
        result = _invoke(
            [
                "--tower-id=flag-tower",
                "--api-key=test-key",
                '--json={"tower_id":"json-tower","ttl_seconds":7200}',
            ]
        )
        assert result.exit_code == 0
        body = mock_create.call_args[0][2]
        assert body["tower_id"] == "flag-tower"

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_json_stdin(self, mock_create: MagicMock, mock_store: MagicMock) -> None:
        result = _invoke(
            ["--api-key=test-key", "--json=-"],
            input='{"tower_id":"stdin-tower"}',
        )
        assert result.exit_code == 0
        body = mock_create.call_args[0][2]
        assert body["tower_id"] == "stdin-tower"

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_json_ttl_zero_not_dropped(self, mock_create: MagicMock, mock_store: MagicMock) -> None:
        """ttl_seconds=0 in JSON should not be silently dropped."""
        result = _invoke(
            [
                "--api-key=test-key",
                '--json={"tower_id":"my-tower","ttl_seconds":0}',
            ]
        )
        assert result.exit_code == 0
        body = mock_create.call_args[0][2]
        assert body["ttl_seconds"] == 0

    @patch("cwsandbox.cli.tower_create_join_token._store_token_in_secret")
    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_json_empty_string_description_not_dropped(
        self, mock_create: MagicMock, mock_store: MagicMock
    ) -> None:
        """Empty string description from JSON should be passed through."""
        result = _invoke(
            [
                "--api-key=test-key",
                '--json={"tower_id":"my-tower","description":""}',
            ]
        )
        assert result.exit_code == 0
        body = mock_create.call_args[0][2]
        assert body["description"] == ""


class TestCreateToken:
    """Tests for _create_token helper."""

    def test_http_status_error(self) -> None:
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_response
            )
            result = _invoke(["--tower-id=my-tower", "--api-key=test-key"])

        assert result.exit_code == 1
        assert "API error (403)" in result.output

    def test_request_error(self) -> None:
        import httpx

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = httpx.RequestError("connection refused")
            result = _invoke(["--tower-id=my-tower", "--api-key=test-key"])

        assert result.exit_code == 1
        assert "Failed to connect to ATC" in result.output


class TestStoreTokenInSecret:
    """Tests for _store_token_in_secret helper."""

    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_secret_creation_success(self, mock_create: MagicMock) -> None:
        from kubernetes.client import CoreV1Api

        with (
            patch.object(CoreV1Api, "create_namespaced_secret") as mock_create_secret,
            patch("kubernetes.config.load_kube_config"),
        ):
            result = _invoke(["--tower-id=my-tower", "--api-key=test-key"])

        assert result.exit_code == 0
        mock_create_secret.assert_called_once()

    @patch("cwsandbox.cli.tower_create_join_token._create_token", return_value=_TOKEN_RESPONSE)
    def test_secret_409_triggers_update(self, mock_create: MagicMock) -> None:
        from kubernetes.client import CoreV1Api
        from kubernetes.client.exceptions import ApiException

        api_exc = ApiException(status=409, reason="Conflict")

        with (
            patch.object(CoreV1Api, "create_namespaced_secret", side_effect=api_exc),
            patch.object(CoreV1Api, "replace_namespaced_secret") as mock_replace,
            patch("kubernetes.config.load_kube_config"),
        ):
            result = _invoke(["--tower-id=my-tower", "--api-key=test-key"])

        assert result.exit_code == 0
        mock_replace.assert_called_once()
