# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for Sandbox.run_from_file (CreateSandboxFromFile).

These tests require a running CWSandbox backend that admits Compose ingest
and more than one user container, and a runner that supports container
engine mode. Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY before running.
"""

from __future__ import annotations

from pathlib import Path

from cwsandbox import ResourceOptions, Sandbox, SandboxControl, SandboxDefaults
from cwsandbox._sandbox import SandboxStatus

_COMPOSE = Path(__file__).parent / "testdata" / "cache-api.docker-compose.yaml"
_RESOURCES = ResourceOptions(
    requests={"cpu": "500m", "memory": "256Mi"},
    limits={"cpu": "500m", "memory": "256Mi"},
)
_HTTP_GET = """\
set -euo pipefail
exec 3<>/dev/tcp/api/8080
printf 'GET /health HTTP/1.0\\r\\nHost: api\\r\\n\\r\\n' >&3
cat <&3
"""
_ENGINE_COMPOSE = b"""\
services:
  main:
    image: alpine:3.20
    command: ["sleep", "3600"]
  helper:
    image: alpine:3.20
    command: ["sh", "-c", "echo ready > /tmp/evidence && exec sleep 3600"]
    healthcheck:
      test: ["CMD", "test", "-f", "/tmp/evidence"]
      interval: 2s
      retries: 30
"""


def test_run_from_file_dependent_services(sandbox_defaults: SandboxDefaults) -> None:
    """Redis + Python API + Ubuntu primary: RUNNING means the health chain passed.

    Exec from main resolves service hostnames on loopback, pings Redis, and
    fetches /health on the api hostname (PONG from Redis).
    """
    defaults = sandbox_defaults.with_overrides(max_lifetime_seconds=600)
    with Sandbox.run_from_file(
        _COMPOSE,
        primary_service="main",
        default_resources=_RESOURCES,
        defaults=defaults,
    ) as sandbox:
        sandbox.wait()
        assert sandbox.status == SandboxStatus.RUNNING

        names = [row.name for row in sandbox.containers]
        assert names == ["cache", "api", "main"]
        by_name = {row.name: row for row in sandbox.containers}
        assert by_name["main"].primary is True
        assert by_name["cache"].primary is False
        assert by_name["api"].primary is False
        assert (by_name["api"].environment_variables or {})["REDIS_HOST"] == "cache"
        assert (by_name["main"].environment_variables or {})["API_HOST"] == "api"

        hosts_cache = sandbox.exec(["getent", "hosts", "cache"]).result()
        assert hosts_cache.returncode == 0, hosts_cache.stderr or hosts_cache.stdout
        assert "127.0.0.1" in hosts_cache.stdout

        hosts_api = sandbox.exec(["getent", "hosts", "api"]).result()
        assert hosts_api.returncode == 0, hosts_api.stderr or hosts_api.stdout
        assert "127.0.0.1" in hosts_api.stdout

        ping = sandbox.exec(["redis-cli", "ping"], container="cache").result()
        assert ping.returncode == 0, ping.stderr or ping.stdout
        assert ping.stdout.strip() == "PONG"

        health = sandbox.exec(["bash", "-c", _HTTP_GET]).result()
        assert health.returncode == 0, health.stderr or health.stdout
        assert "PONG" in health.stdout


def test_run_from_file_container_engine(sandbox_defaults: SandboxDefaults) -> None:
    """Engine mode: Compose runs in cws-engine and services stop independently.

    Stopping ``main`` leaves ``helper`` running and reachable through
    ``docker compose exec``.
    """
    defaults = sandbox_defaults.with_overrides(max_lifetime_seconds=600)
    with Sandbox.run_from_file(
        _ENGINE_COMPOSE,
        control=SandboxControl.CONTAINER_ENGINE,
        default_resources=_RESOURCES,
        defaults=defaults,
    ) as sandbox:
        sandbox.wait(timeout=540)
        assert sandbox.status == SandboxStatus.RUNNING
        assert sandbox.control == SandboxControl.CONTAINER_ENGINE
        assert [row.name for row in sandbox.containers] == ["cws-engine"]

        stop = sandbox.exec(["docker", "compose", "stop", "-t", "1", "main"]).result()
        assert stop.returncode == 0, stop.stderr or stop.stdout

        running = sandbox.exec(
            ["docker", "compose", "ps", "--services", "--status", "running"],
            container="cws-engine",
        ).result()
        assert running.returncode == 0, running.stderr or running.stdout
        assert running.stdout.split() == ["helper"]

        evidence = sandbox.exec(
            ["docker", "compose", "exec", "-T", "helper", "cat", "/tmp/evidence"]
        ).result()
        assert evidence.returncode == 0, evidence.stderr or evidence.stdout
        assert evidence.stdout.strip() == "ready"
