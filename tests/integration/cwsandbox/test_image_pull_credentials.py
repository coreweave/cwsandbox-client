# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for image pull credentials on template creates.

These tests need a backend that accepts ``image_pull_credentials`` on
templates and a private-registry fixture the caller can pull. They skip
when that fixture is not configured, when the API key cannot create
templates, or when the template service is not reachable on the
configured URL.

Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY as for other integration
tests. See .env.example for the private-registry variables.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import grpc
import pytest

from cwsandbox import ImagePullCredentials, PlacementMode, Sandbox, SandboxDefaults
from cwsandbox._auth import resolve_auth_metadata
from cwsandbox._defaults import DEFAULT_ARGS, DEFAULT_BASE_URL, DEFAULT_COMMAND
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import create_channel, parse_grpc_target, translate_grpc_error
from cwsandbox._proto import sandbox_pb2, sandbox_template_pb2, sandbox_template_pb2_grpc
from cwsandbox.exceptions import CWSandboxAuthenticationError, CWSandboxError
from tests.integration.cwsandbox.conftest import _SESSION_TAG

_ENV_IMAGE = "CWSANDBOX_TEST_PRIVATE_REGISTRY_IMAGE"
_ENV_REGISTRY = "CWSANDBOX_TEST_PRIVATE_REGISTRY"
_ENV_STORE = "CWSANDBOX_TEST_PRIVATE_REGISTRY_STORE"
_ENV_SECRET = "CWSANDBOX_TEST_PRIVATE_REGISTRY_SECRET"
_ENV_FIELD = "CWSANDBOX_TEST_PRIVATE_REGISTRY_FIELD"
_ENV_TEMPLATE_URL = "CWSANDBOX_TEST_TEMPLATE_BASE_URL"

_TEMPLATE_LIFETIME_SECONDS = 180
_SKIP_CREATE_REASONS = ("CWSANDBOX_SECRET_STORE_NOT_PROVISIONED",)


@dataclass(frozen=True)
class _PrivateRegistry:
    image: str
    registry: str
    store: str
    secret: str
    field: str


@pytest.fixture
def private_registry() -> _PrivateRegistry:
    """Skip unless a private-registry pull fixture is fully configured."""
    image = os.environ.get(_ENV_IMAGE, "").strip()
    registry = os.environ.get(_ENV_REGISTRY, "").strip()
    store = os.environ.get(_ENV_STORE, "").strip()
    secret = os.environ.get(_ENV_SECRET, "").strip()
    missing = [
        name
        for name, value in (
            (_ENV_IMAGE, image),
            (_ENV_REGISTRY, registry),
            (_ENV_STORE, store),
            (_ENV_SECRET, secret),
        )
        if not value
    ]
    if missing:
        pytest.skip(
            "Private-registry template tests require " + ", ".join(missing) + " (see .env.example)"
        )
    return _PrivateRegistry(
        image=image,
        registry=registry,
        store=store,
        secret=secret,
        field=os.environ.get(_ENV_FIELD, "").strip(),
    )


def _credentials(registry: _PrivateRegistry) -> ImagePullCredentials:
    return ImagePullCredentials(
        registry=registry.registry,
        store=registry.store,
        name=registry.secret,
        field=registry.field,
    )


def _proto_credentials(registry: _PrivateRegistry) -> sandbox_pb2.ImagePullCredentials:
    return sandbox_pb2.ImagePullCredentials(
        registry=registry.registry,
        credentials=sandbox_pb2.SecretSource(
            store_name=registry.store,
            path=registry.secret,
            field=registry.field,
        ),
    )


def _keep_alive_container(
    image: str,
    *,
    credentials: sandbox_pb2.ImagePullCredentials | None = None,
) -> sandbox_pb2.PartialContainer:
    container = sandbox_pb2.PartialContainer(
        name="main",
        image=image,
        command=DEFAULT_COMMAND,
        args=list(DEFAULT_ARGS),
        resources=sandbox_pb2.Resources(cpu="500m", memory="256Mi"),
    )
    if credentials is not None:
        container.image_pull_credentials.CopyFrom(credentials)
    return container


def _template_spec(
    container: sandbox_pb2.PartialContainer,
) -> sandbox_pb2.PartialSandboxSpec:
    return sandbox_pb2.PartialSandboxSpec(
        containers=[container],
        max_lifetime_seconds=_TEMPLATE_LIFETIME_SECONDS,
        tags=[_SESSION_TAG, "integration-test"],
    )


def _template_base_url() -> str:
    return (
        os.environ.get(_ENV_TEMPLATE_URL)
        or os.environ.get("CWSANDBOX_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


async def _with_template_stub(operation: str, call: Any) -> Any:
    target, is_secure = parse_grpc_target(_template_base_url())
    channel = create_channel(target, is_secure)
    stub = sandbox_template_pb2_grpc.SandboxTemplateServiceStub(channel)
    try:
        try:
            return await call(stub)
        except grpc.RpcError as exc:
            raise translate_grpc_error(exc, operation=operation) from exc
    finally:
        await channel.close(grace=None)


def _create_template(spec: sandbox_pb2.PartialSandboxSpec) -> str:
    request = sandbox_template_pb2.CreateSandboxTemplateRequest(
        sandbox_template=sandbox_template_pb2.SandboxTemplate(
            display_name=f"sdk-e2e-ipc-{uuid.uuid4().hex[:8]}",
            spec=spec,
        ),
        request_id=str(uuid.uuid4()),
    )

    async def create(
        stub: sandbox_template_pb2_grpc.SandboxTemplateServiceStub,
    ) -> sandbox_template_pb2.SandboxTemplate:
        return await stub.CreateSandboxTemplate(
            request,
            timeout=30,
            metadata=resolve_auth_metadata(),
        )

    try:
        created = _LoopManager.get().run_sync(
            _with_template_stub("Create sandbox template", create)
        )
    except CWSandboxAuthenticationError as exc:
        pytest.skip(f"API key cannot create sandbox templates: {exc}")
    except CWSandboxError as exc:
        if exc.reason == "CWSANDBOX_NOT_IMPLEMENTED" or "unimplemented" in str(exc).lower():
            pytest.skip(
                "Template service is not available on "
                f"{_template_base_url()}; set {_ENV_TEMPLATE_URL} if it "
                f"listens on a different host. ({exc})"
            )
        raise
    return created.sandbox_template_id


def _delete_template(template_id: str) -> None:
    request = sandbox_template_pb2.DeleteSandboxTemplateRequest(
        sandbox_template_id=template_id,
    )

    async def delete(
        stub: sandbox_template_pb2_grpc.SandboxTemplateServiceStub,
    ) -> None:
        await stub.DeleteSandboxTemplate(
            request,
            timeout=30,
            metadata=resolve_auth_metadata(),
        )

    try:
        _LoopManager.get().run_sync(_with_template_stub("Delete sandbox template", delete))
    except CWSandboxError:
        pass


@contextmanager
def _created_template(spec: sandbox_pb2.PartialSandboxSpec) -> Iterator[str]:
    template_id = _create_template(spec)
    try:
        yield template_id
    finally:
        _delete_template(template_id)


def _from_template_kwargs(
    sandbox_defaults: SandboxDefaults,
    configured_runner_ids: tuple[str, ...] | None,
    **extra: object,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "defaults": sandbox_defaults,
        "max_lifetime_seconds": _TEMPLATE_LIFETIME_SECONDS,
        "tags": list(sandbox_defaults.tags or ()),
    }
    if configured_runner_ids is not None:
        kwargs["runner_ids"] = list(configured_runner_ids)
        kwargs["placement_mode"] = PlacementMode.CKS
    kwargs.update(extra)
    return kwargs


def _assert_private_image_runs(sandbox: Sandbox) -> None:
    sandbox.wait()
    assert sandbox.status == "running"
    result = sandbox.exec(["echo", "template-pull-ok"]).result()
    assert result.returncode == 0
    assert result.stdout.strip() == "template-pull-ok"


def test_run_from_template_uses_stored_image_pull_credentials(
    sandbox_defaults: SandboxDefaults,
    configured_runner_ids: tuple[str, ...] | None,
    private_registry: _PrivateRegistry,
) -> None:
    """Omit the kwarg so create-from-template keeps the template credential."""
    spec = _template_spec(
        _keep_alive_container(
            private_registry.image,
            credentials=_proto_credentials(private_registry),
        )
    )
    with _created_template(spec) as template_id:
        try:
            with Sandbox.run_from_template(
                template_id,
                **_from_template_kwargs(sandbox_defaults, configured_runner_ids),
            ) as sandbox:
                _assert_private_image_runs(sandbox)
        except CWSandboxError as exc:
            if exc.reason in _SKIP_CREATE_REASONS:
                pytest.skip(f"Private-registry secret is not usable: {exc}")
            raise


def test_run_from_template_sends_image_pull_credentials_overlay(
    sandbox_defaults: SandboxDefaults,
    configured_runner_ids: tuple[str, ...] | None,
    private_registry: _PrivateRegistry,
) -> None:
    """Replace the template container and send credentials on the overlay."""
    spec = _template_spec(_keep_alive_container(sandbox_defaults.container_image or "python:3.11"))
    with _created_template(spec) as template_id:
        try:
            with Sandbox.run_from_template(
                template_id,
                *DEFAULT_ARGS,
                command=DEFAULT_COMMAND,
                container_image=private_registry.image,
                image_pull_credentials=_credentials(private_registry),
                **_from_template_kwargs(sandbox_defaults, configured_runner_ids),
            ) as sandbox:
                _assert_private_image_runs(sandbox)
        except CWSandboxError as exc:
            if exc.reason in _SKIP_CREATE_REASONS:
                pytest.skip(f"Private-registry secret is not usable: {exc}")
            raise
