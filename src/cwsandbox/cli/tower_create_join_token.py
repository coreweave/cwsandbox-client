# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox tower create-join-token — create a tower join token and store it in K8s."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from cwsandbox._defaults import DEFAULT_BASE_URL

DEFAULT_SECRET_NAME = "sandbox-tower-join-token"
DEFAULT_SECRET_KEY = "token"
DEFAULT_NAMESPACE = "sandbox-system"
DEFAULT_ATC_SERVER = DEFAULT_BASE_URL


@click.command("create-join-token")
@click.option(
    "--tower-id",
    default=None,
    help="Tower ID to assign (required unless provided via --json).",
)
@click.option(
    "--tower-group-id",
    default=None,
    help='Tower group for scheduling affinity (default: "default").',
)
@click.option(
    "--ttl",
    "ttl_seconds",
    type=int,
    default=None,
    help="Token TTL in seconds (default: 86400 = 24 hours).",
)
@click.option("--description", default=None, help="Human-readable description for the token.")
@click.option(
    "--atc-server",
    envvar="CWSANDBOX_BASE_URL",
    default=DEFAULT_ATC_SERVER,
    show_envvar=True,
    help="ATC server address.",
)
@click.option(
    "--api-key",
    envvar="CWSANDBOX_API_KEY",
    default=None,
    show_envvar=True,
    help="API key for ATC authentication.",
)
@click.option(
    "--kubeconfig",
    envvar="KUBECONFIG",
    default=None,
    show_envvar=True,
    help="Path to kubeconfig file.",
)
@click.option(
    "--namespace",
    default=DEFAULT_NAMESPACE,
    help="Kubernetes namespace for the secret.",
)
@click.option(
    "--secret-name",
    default=DEFAULT_SECRET_NAME,
    help="Name of the Kubernetes secret.",
)
@click.option(
    "--secret-key",
    default=DEFAULT_SECRET_KEY,
    help="Key within the Kubernetes secret.",
)
@click.option(
    "--json",
    "json_payload",
    default=None,
    help="JSON payload matching the API request format (use '-' to read from stdin).",
)
@click.option(
    "--generate-only",
    is_flag=True,
    default=False,
    help="Only generate the token and print it; do not store in Kubernetes.",
)
def create_join_token(
    tower_id: str | None,
    tower_group_id: str | None,
    ttl_seconds: int | None,
    description: str | None,
    atc_server: str,
    api_key: str | None,
    kubeconfig: str | None,
    namespace: str,
    secret_name: str,
    secret_key: str,
    json_payload: str | None,
    generate_only: bool,
) -> None:
    """Create a join token for a tower and store it in a Kubernetes secret.

    Creates a join token by calling the ATC API, then stores the token
    in a Kubernetes secret on the target cluster. The tower's Helm chart
    reads this secret during the join process.

    Examples:

        cwsandbox tower create-join-token --tower-id=my-tower

        cwsandbox tower create-join-token --tower-id=my-tower --namespace=sandbox-system

        cwsandbox tower create-join-token --json='{"tower_id":"my-tower","ttl_seconds":3600}'

        echo '{"tower_id":"my-tower"}' | cwsandbox tower create-join-token --json=-

        cwsandbox tower create-join-token --tower-id=my-tower --generate-only
    """
    if not api_key:
        raise click.ClickException("--api-key or CWSANDBOX_API_KEY is required")

    # If --json is provided, parse and use as base values (flags override)
    if json_payload is not None:
        tower_id, tower_group_id, ttl_seconds, description = _apply_json_payload(
            json_payload, tower_id, tower_group_id, ttl_seconds, description
        )

    if not tower_id:
        raise click.ClickException("--tower-id is required (via flag or JSON payload)")

    # Build the API request body
    body: dict[str, Any] = {"tower_id": tower_id}
    if tower_group_id is not None:
        body["tower_group_id"] = tower_group_id
    if ttl_seconds is not None:
        body["ttl_seconds"] = ttl_seconds
    if description is not None:
        body["description"] = description

    token_resp = _create_token(atc_server, api_key, body)

    if generate_only:
        click.echo(f"Tower ID:       {token_resp.get('tower_id', '')}")
        click.echo(f"Tower Group:    {token_resp.get('tower_group_id', '')}")
        click.echo(f"Organization:   {token_resp.get('organization_id', '')}")
        click.echo(f"Token ID:       {token_resp.get('token_id', '')}")
        click.echo(f"Expires:        {token_resp.get('expires_at', '')}")
        click.echo(f"Token:          {token_resp.get('token', '')}")
        return

    click.echo(
        f"Created join token for tower {tower_id!r} "
        f"(token_id: {token_resp.get('token_id', '')}, "
        f"expires: {token_resp.get('expires_at', '')})",
        err=True,
    )

    token = token_resp.get("token")
    if not token:
        raise click.ClickException("ATC returned invalid response: missing 'token' field")

    _store_token_in_secret(
        token=token,
        kubeconfig=kubeconfig,
        namespace=namespace,
        secret_name=secret_name,
        secret_key=secret_key,
    )

    click.echo(
        f"Stored join token in secret {namespace}/{secret_name} (key: {secret_key})",
        err=True,
    )


def _apply_json_payload(
    json_payload: str,
    tower_id: str | None,
    tower_group_id: str | None,
    ttl_seconds: int | None,
    description: str | None,
) -> tuple[str | None, str | None, int | None, str | None]:
    """Parse JSON payload and merge with flag values. Flags take precedence."""
    if json_payload == "-":
        data = sys.stdin.read()
    else:
        data = json_payload

    try:
        req = json.loads(data)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON: {e}") from None

    if tower_id is None and req.get("tower_id") is not None:
        tower_id = req["tower_id"]
    if tower_group_id is None and req.get("tower_group_id") is not None:
        tower_group_id = req["tower_group_id"]
    if ttl_seconds is None and req.get("ttl_seconds") is not None:
        ttl_seconds = req["ttl_seconds"]
    if description is None and req.get("description") is not None:
        description = req["description"]

    return tower_id, tower_group_id, ttl_seconds, description


def _create_token(atc_server: str, api_key: str, body: dict[str, Any]) -> dict[str, Any]:
    """Call the ATC API to create a tower join token."""
    try:
        import httpx
    except ModuleNotFoundError:
        raise click.ClickException(
            "httpx is required for this command. Install it with: pip install httpx"
        ) from None

    url = f"{atc_server.rstrip('/')}/v1beta1/towers/tokens"

    try:
        resp = httpx.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"API error ({e.response.status_code}): {e.response.text}"
        ) from None
    except httpx.RequestError as e:
        raise click.ClickException(f"Failed to connect to ATC: {e}") from None

    return resp.json()  # type: ignore[no-any-return]


def _store_token_in_secret(
    *,
    token: str,
    kubeconfig: str | None,
    namespace: str,
    secret_name: str,
    secret_key: str,
) -> None:
    """Store the join token in a Kubernetes secret."""
    try:
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config
        from kubernetes.client.exceptions import ApiException
    except ModuleNotFoundError:
        raise click.ClickException(
            "kubernetes client is required for this command. "
            "Install it with: pip install kubernetes"
        ) from None

    # Load kubeconfig
    try:
        if kubeconfig:
            k8s_config.load_kube_config(config_file=kubeconfig)
        else:
            k8s_config.load_kube_config()
    except k8s_config.ConfigException:
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            raise click.ClickException(
                "Could not load Kubernetes configuration. "
                "Provide --kubeconfig or run inside a cluster."
            ) from None

    v1 = k8s_client.CoreV1Api()

    secret = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(
            name=secret_name,
            namespace=namespace,
            labels={
                "app.kubernetes.io/name": "sandbox-tower",
                "app.kubernetes.io/component": "join-token",
                "app.kubernetes.io/managed-by": "box",
            },
        ),
        type="Opaque",
        string_data={secret_key: token},
    )

    try:
        v1.create_namespaced_secret(namespace=namespace, body=secret)
    except ApiException as e:
        if e.status == 409:
            # Already exists — update it
            v1.replace_namespaced_secret(name=secret_name, namespace=namespace, body=secret)
            click.echo(f"Updated existing secret {namespace}/{secret_name}", err=True)
        else:
            raise click.ClickException(f"Failed to create Kubernetes secret: {e.reason}") from None
