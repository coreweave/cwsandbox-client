# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for v1 multi-container create, targeting, and echo."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cwsandbox import (
    Container,
    ContainerStatus,
    FileSystemSnapshotOptions,
    ResourceOptions,
    Sandbox,
    SandboxDefaults,
    ScratchVolumeOptions,
    Secret,
    VolumeMount,
)
from cwsandbox._proto import sandbox_pb2
from cwsandbox._sandbox import SandboxStatus, _Running, _SandboxView, _Terminal
from tests.unit.cwsandbox.test_sandbox import (
    MockStreamCall,
    create_mock_channel_and_stub,
)


def _cpu(cpu: str = "1") -> ResourceOptions:
    return ResourceOptions(requests={"cpu": cpu}, limits={"cpu": cpu})


def _run(*args: str, **kwargs: Any) -> tuple[Sandbox, MagicMock]:
    from tests.unit.cwsandbox.test_sandbox import TestSandboxRun

    sandbox, stub = TestSandboxRun._run_with_mock_stub(*args, **kwargs)
    sandbox._state = _Terminal(sandbox_id="matrix-id", status=SandboxStatus.COMPLETED)
    return sandbox, stub


def _running_sandbox() -> Sandbox:
    sandbox = Sandbox(command="sleep", args=["infinity"])
    sandbox._sandbox_id = "sb-mc"
    sandbox._state = _Running(sandbox_id="sb-mc")
    sandbox._channel = MagicMock()
    sandbox._stub = MagicMock()
    sandbox._auth_metadata = ()
    return sandbox


class TestVolumeMount:
    def test_valid(self) -> None:
        mount = VolumeMount(volume="workspace", mount_path="/data", read_only=True, sub_path="x")
        assert mount.volume == "workspace"
        assert mount.mount_path == "/data"
        assert mount.read_only is True
        assert mount.sub_path == "x"

    def test_empty_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="VolumeMount.volume"):
            VolumeMount(volume="", mount_path="/data")

    @pytest.mark.parametrize("path", ["", "relative", "/"])
    def test_invalid_mount_path_raises(self, path: str) -> None:
        with pytest.raises(ValueError, match="VolumeMount.mount_path"):
            VolumeMount(volume="workspace", mount_path=path)

    def test_empty_sub_path_normalized(self) -> None:
        assert VolumeMount(volume="workspace", mount_path="/data", sub_path="").sub_path is None


class TestContainerValidation:
    def test_empty_image_raises(self) -> None:
        with pytest.raises(ValueError, match="Container.image"):
            Container(image="")

    def test_empty_name_normalized(self) -> None:
        assert Container(image="python:3.11", name="").name is None

    def test_working_dir_must_be_absolute(self) -> None:
        with pytest.raises(ValueError, match="Container.working_dir"):
            Container(image="python:3.11", working_dir="app")

    def test_reserved_name_raises(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            Container(image="python:3.11", name="dns-egress")

    def test_reserved_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            Container(image="python:3.11", name="cw-object-store-agent-restore-1")

    def test_dns1123_rejects_uppercase_and_dots(self) -> None:
        with pytest.raises(ValueError, match="DNS-1123 label"):
            Container(image="python:3.11", name="Main")
        with pytest.raises(ValueError, match="DNS-1123 label"):
            Container(image="python:3.11", name="helper.one")

    def test_dict_coercion_and_volume_mounts(self) -> None:
        row = Container(
            image="python:3.11",
            volume_mounts=[{"volume": "workspace", "mount_path": "/workspace"}],
        )
        assert row.volume_mounts == (VolumeMount(volume="workspace", mount_path="/workspace"),)

    def test_string_args_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="Container.args"):
            Container(image="busybox", args="--save")  # type: ignore[arg-type]

    def test_bytes_args_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="Container.args"):
            Container(image="busybox", args=b"--save")  # type: ignore[arg-type]

    def test_non_string_args_entries_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="Container.args\\[1\\]"):
            Container(image="busybox", args=["--save", 1])  # type: ignore[list-item]

    def test_from_dict_string_args_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="Container.args"):
            SandboxDefaults.from_dict({"containers": [{"image": "busybox", "args": "--save"}]})

    def test_conflicting_secrets_on_one_container_raise(self) -> None:
        with pytest.raises(ValueError, match="Conflicting secrets for env_var 'TOKEN'"):
            Container(
                image="python:3.11",
                secrets=[
                    Secret(store="wandb", name="a", env_var="TOKEN"),
                    Secret(store="vault", name="b", env_var="TOKEN"),
                ],
            )

    def test_identical_secrets_on_one_container_dedupe(self) -> None:
        secret = Secret(store="wandb", name="HF_TOKEN")
        row = Container(image="python:3.11", secrets=[secret, secret])
        assert row.secrets == (secret,)

    def test_same_env_var_on_different_containers_is_allowed(self) -> None:
        secret = Secret(store="wandb", name="HF_TOKEN", env_var="TOKEN")
        sandbox = Sandbox(
            containers=[
                Container(
                    image="python:3.11",
                    name="main",
                    primary=True,
                    resources=_cpu(),
                    secrets=[secret],
                ),
                Container(
                    image="redis:7",
                    name="cache",
                    resources=_cpu(),
                    secrets=[secret],
                ),
            ]
        )
        rows = sandbox._start_kwargs["containers"]
        assert rows[0].secrets == (secret,)
        assert rows[1].secrets == (secret,)

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="containers cannot be empty"):
            Sandbox(containers=[])

    def test_n_gt_1_requires_exactly_one_primary(self) -> None:
        rows = [
            Container(image="python:3.11", name="main", resources=_cpu()),
            Container(image="redis:7", name="cache", resources=_cpu()),
        ]
        with pytest.raises(ValueError, match="exactly one primary=True"):
            Sandbox(containers=rows)

    def test_n_gt_1_requires_names(self) -> None:
        with pytest.raises(ValueError, match="every container must have a name"):
            Sandbox(
                containers=[
                    Container(image="python:3.11", primary=True, resources=_cpu()),
                    Container(image="redis:7", name="cache", resources=_cpu()),
                ]
            )

    def test_duplicate_names_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate container name"):
            Sandbox(
                containers=[
                    Container(image="python:3.11", name="main", primary=True, resources=_cpu()),
                    Container(image="redis:7", name="main", resources=_cpu()),
                ]
            )

    def test_n_gt_1_requires_resources(self) -> None:
        with pytest.raises(ValueError, match="requires resources"):
            Sandbox(
                containers=[
                    Container(image="python:3.11", name="main", primary=True, resources=_cpu()),
                    Container(image="redis:7", name="cache"),
                ]
            )

    def test_gpu_only_on_primary(self) -> None:
        with pytest.raises(ValueError, match="GPU is only allowed on the primary"):
            Sandbox(
                containers=[
                    Container(image="python:3.11", name="main", primary=True, resources=_cpu()),
                    Container(
                        image="redis:7",
                        name="cache",
                        resources=ResourceOptions(gpu={"count": 1}),
                    ),
                ]
            )

    def test_exclusive_with_single_container_kwargs(self) -> None:
        with pytest.raises(TypeError, match="mutually exclusive"):
            Sandbox(
                containers=[Container(image="python:3.11")],
                container_image="python:3.12",
            )

    def test_defaults_containers_conflict_with_container_image(self) -> None:
        defaults = SandboxDefaults(
            containers=[
                Container(image="python:3.11", name="main", primary=True, resources=_cpu()),
                Container(image="redis:7", name="cache", resources=_cpu()),
            ]
        )
        with pytest.raises(TypeError, match="mutually exclusive"):
            Sandbox(container_image="python:3.11", defaults=defaults)


class TestCreateRequest:
    def test_kwargs_path_names_main_and_omits_primary(self) -> None:
        _, stub = _run("sleep", "infinity")
        container = stub.CreateSandbox.call_args.args[0].sandbox.spec.containers[0]
        assert container.name == "main"
        assert not container.HasField("primary")
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        assert [fd.name for fd, _ in spec.ListFields() if fd.name == "primary_container"] == []

    def test_single_container_omits_name_and_primary(self) -> None:
        _, stub = _run(containers=[Container(image="python:3.11", working_dir="/app")])
        container = stub.CreateSandbox.call_args.args[0].sandbox.spec.containers[0]
        assert container.image == "python:3.11"
        assert container.name == ""
        assert container.working_dir == "/app"
        assert not container.HasField("primary")

    def test_multi_container_sets_primary_and_per_container_fields(self) -> None:
        _, stub = _run(
            containers=[
                Container(
                    image="python:3.11",
                    name="main",
                    primary=True,
                    resources=_cpu("2"),
                    volume_mounts=[VolumeMount(volume="workspace", mount_path="/workspace")],
                    environment_variables={"ROLE": "primary"},
                ),
                Container(
                    image="redis:7",
                    name="cache",
                    command="redis-server",
                    args=["--save", ""],
                    resources=_cpu("1"),
                ),
            ],
            volumes=[ScratchVolumeOptions(name="workspace")],
        )
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        assert [row.name for row in spec.containers] == ["main", "cache"]
        assert spec.containers[0].primary is True
        assert spec.containers[0].HasField("primary")
        assert not spec.containers[1].HasField("primary")
        assert spec.containers[0].resource_requirements.requests.cpu == "2"
        assert spec.containers[1].command == "redis-server"
        assert list(spec.containers[1].args) == ["--save", ""]
        assert spec.containers[0].environment_variables["ROLE"] == "primary"
        mounts = [(m.volume, m.mount_path) for m in spec.containers[0].volume_mounts]
        assert mounts == [("workspace", "/workspace")]
        assert list(spec.containers[1].volume_mounts) == []
        assert spec.volumes[0].name == "workspace"
        assert [fd.name for fd, _ in spec.ListFields() if fd.name == "primary_container"] == []

    def test_declare_only_volume_does_not_mount(self) -> None:
        _, stub = _run(volumes=[ScratchVolumeOptions(name="cache")])
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        assert spec.volumes[0].name == "cache"
        assert list(spec.containers[0].volume_mounts) == []

    def test_convenience_mount_attaches_to_primary(self) -> None:
        _, stub = _run(
            containers=[
                Container(image="python:3.11", name="main", primary=True, resources=_cpu()),
                Container(image="redis:7", name="cache", resources=_cpu()),
            ],
            volumes=[ScratchVolumeOptions(name="workspace", mount_path="/workspace")],
        )
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        primary_mounts = [(m.volume, m.mount_path) for m in spec.containers[0].volume_mounts]
        helper_mounts = list(spec.containers[1].volume_mounts)
        assert primary_mounts == [("workspace", "/workspace")]
        assert helper_mounts == []

    def test_kwargs_path_convenience_mount(self) -> None:
        _, stub = _run(volumes=[ScratchVolumeOptions(name="cache", mount_path="/cache")])
        mount = stub.CreateSandbox.call_args.args[0].sandbox.spec.containers[0].volume_mounts[0]
        assert mount.volume == "cache"
        assert mount.mount_path == "/cache"

    def test_fss_without_mount_path_declares_volume_only(self) -> None:
        _, stub = _run(file_system_snapshot=FileSystemSnapshotOptions(size="10Gi"))
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        assert spec.volumes[0].name == "workspace"
        assert spec.volumes[0].scratch.size == "10Gi"
        assert list(spec.containers[0].volume_mounts) == []

    def test_template_containers_replace_without_container_image(self) -> None:
        from tests.unit.cwsandbox.test_sandbox import TestSandboxRun

        sandbox, stub = TestSandboxRun._run_with_mock_stub(
            template_id="template-123",
            containers=[
                Container(image="python:3.11", name="main", primary=True, resources=_cpu()),
                Container(image="redis:7", name="cache", resources=_cpu()),
            ],
        )
        sandbox._state = _Terminal(sandbox_id="template-id", status=SandboxStatus.COMPLETED)
        request = stub.CreateSandboxFromTemplate.call_args.args[0]
        assert [row.name for row in request.overrides.containers] == ["main", "cache"]
        assert request.overrides.containers[0].primary is True
        assert request.overrides.containers[0].image == "python:3.11"


class TestContainerTarget:
    def test_exec_sets_container_when_passed(self) -> None:
        sandbox = _running_sandbox()
        exit_response = sandbox_pb2.ExecStreamResponse(exit=sandbox_pb2.ExecStreamExit(exit_code=0))
        mock_call = MockStreamCall(responses=[exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub",
                return_value=mock_stub,
            ),
        ):
            sandbox.exec(["echo", "hi"], container="cache").result()

        init = mock_call._writes[0].init
        assert init.container == "cache"

    def test_exec_omits_container_when_empty(self) -> None:
        sandbox = _running_sandbox()
        exit_response = sandbox_pb2.ExecStreamResponse(exit=sandbox_pb2.ExecStreamExit(exit_code=0))
        mock_call = MockStreamCall(responses=[exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub",
                return_value=mock_stub,
            ),
        ):
            sandbox.exec(["echo", "hi"]).result()

        init = mock_call._writes[0].init
        assert init.container == ""

    def test_read_file_sets_container(self) -> None:
        sandbox = _running_sandbox()
        mock_read_response = MagicMock()
        mock_read_response.content = b"ok"
        sandbox._stub.ReadFile = AsyncMock(return_value=mock_read_response)

        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_read_file_via_exec_streaming", new_callable=AsyncMock),
        ):
            data = sandbox.read_file("/tmp/x", container="cache").result()

        assert data == b"ok"
        request = sandbox._stub.ReadFile.await_args.args[0]
        assert request.container == "cache"

    def test_read_file_omits_container_when_unset(self) -> None:
        sandbox = _running_sandbox()
        mock_read_response = MagicMock()
        mock_read_response.content = b"ok"
        sandbox._stub.ReadFile = AsyncMock(return_value=mock_read_response)

        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_read_file_via_exec_streaming", new_callable=AsyncMock),
        ):
            sandbox.read_file("/tmp/x").result()

        request = sandbox._stub.ReadFile.await_args.args[0]
        assert request.container == ""

    @pytest.mark.asyncio
    async def test_stream_logs_sets_container(self) -> None:
        sandbox = _running_sandbox()
        entries = [sandbox_pb2.LogEntry(data=b"line\n", log_session_id="s1", next_log_offset=1)]

        class _Call:
            def __init__(self, items: list[Any]) -> None:
                self._items = items

            def __aiter__(self) -> _Call:
                return self

            async def __anext__(self) -> Any:
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.StreamLogs = MagicMock(return_value=_Call(list(entries)))
        output_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch.object(
                sandbox,
                "_get_or_create_streaming_channel",
                new_callable=AsyncMock,
                return_value=mock_channel,
            ),
            patch("cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub", return_value=mock_stub),
        ):
            await sandbox._stream_logs_async(output_queue, follow=False, container="cache")

        req = mock_stub.StreamLogs.call_args[0][0]
        assert req.container == "cache"


class TestStatusEcho:
    def test_infers_primary_and_maps_container_statuses(self) -> None:
        proto = sandbox_pb2.Sandbox(
            sandbox_id="sb-1",
            spec=sandbox_pb2.SandboxSpec(
                containers=[
                    sandbox_pb2.Container(name="main", image="python:3.11"),
                    sandbox_pb2.Container(name="cache", image="redis:7"),
                ]
            ),
            status=sandbox_pb2.SandboxStatus(
                state=sandbox_pb2.STATE_RUNNING,
                container_statuses=[
                    sandbox_pb2.ContainerStatus(name="main", state=sandbox_pb2.STATE_RUNNING),
                    sandbox_pb2.ContainerStatus(
                        name="cache",
                        state=sandbox_pb2.STATE_RUNNING,
                        restart_count=2,
                    ),
                ],
            ),
        )
        sandbox = Sandbox._from_sandbox_info(
            _SandboxView(proto),
            base_url="https://api.cwsandbox.com",
            timeout_seconds=30.0,
        )
        assert [row.name for row in sandbox.containers] == ["main", "cache"]
        assert sandbox.containers[0].primary is True
        assert sandbox.containers[1].primary is False
        assert sandbox.containers[0].image == "python:3.11"
        assert [row.name for row in sandbox.container_statuses] == ["main", "cache"]
        assert sandbox.container_statuses[1].restart_count == 2
        assert sandbox.container_statuses[1].state == SandboxStatus.RUNNING
        assert sandbox.container_statuses[0].exit_code is None
        assert sandbox.container_statuses[1].exit_code is None
        assert isinstance(sandbox.container_statuses[0], ContainerStatus)
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)

    def test_exit_code_only_for_terminal_container_status(self) -> None:
        proto = sandbox_pb2.Sandbox(
            sandbox_id="sb-1",
            spec=sandbox_pb2.SandboxSpec(
                containers=[sandbox_pb2.Container(name="main", image="python:3.11", primary=True)]
            ),
            status=sandbox_pb2.SandboxStatus(
                state=sandbox_pb2.STATE_RUNNING,
                container_statuses=[
                    sandbox_pb2.ContainerStatus(
                        name="main",
                        state=sandbox_pb2.STATE_COMPLETED,
                        exit_code=0,
                    ),
                    sandbox_pb2.ContainerStatus(
                        name="cache",
                        state=sandbox_pb2.STATE_FAILED,
                        exit_code=1,
                    ),
                    sandbox_pb2.ContainerStatus(
                        name="helper",
                        state=sandbox_pb2.STATE_RUNNING,
                        exit_code=0,
                    ),
                ],
            ),
        )
        sandbox = Sandbox._from_sandbox_info(
            _SandboxView(proto),
            base_url="https://api.cwsandbox.com",
            timeout_seconds=30.0,
        )
        by_name = {row.name: row for row in sandbox.container_statuses}
        assert by_name["main"].exit_code == 0
        assert by_name["cache"].exit_code == 1
        assert by_name["helper"].exit_code is None
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)

    def test_echo_keeps_platform_names_and_root_working_dir(self) -> None:
        proto = sandbox_pb2.Sandbox(
            sandbox_id="sb-1",
            spec=sandbox_pb2.SandboxSpec(
                containers=[
                    sandbox_pb2.Container(
                        name="dns-egress",
                        image="platform/dns-egress:1",
                        working_dir="/",
                    ),
                    sandbox_pb2.Container(name="Main", image="python:3.11"),
                ]
            ),
            status=sandbox_pb2.SandboxStatus(state=sandbox_pb2.STATE_RUNNING),
        )
        sandbox = Sandbox._from_sandbox_info(
            _SandboxView(proto),
            base_url="https://api.cwsandbox.com",
            timeout_seconds=30.0,
        )
        assert [row.name for row in sandbox.containers] == ["dns-egress", "Main"]
        assert sandbox.containers[0].working_dir == "/"
        assert sandbox.containers[0].primary is True
        assert sandbox.containers[1].primary is False
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)

    def test_preserves_explicit_primary_on_helper(self) -> None:
        proto = sandbox_pb2.Sandbox(
            sandbox_id="sb-1",
            spec=sandbox_pb2.SandboxSpec(
                containers=[
                    sandbox_pb2.Container(name="sidecars", image="busybox"),
                    sandbox_pb2.Container(name="main", image="python:3.11", primary=True),
                ]
            ),
            status=sandbox_pb2.SandboxStatus(state=sandbox_pb2.STATE_RUNNING),
        )
        sandbox = Sandbox._from_sandbox_info(
            _SandboxView(proto),
            base_url="https://api.cwsandbox.com",
            timeout_seconds=30.0,
        )
        assert sandbox.containers[0].primary is False
        assert sandbox.containers[1].primary is True
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)
