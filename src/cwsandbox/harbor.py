# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Harbor environment adapter for CW Sandbox.

Requires ``cwsandbox[harbor]`` to be installed. Importing this module without
the ``harbor`` extra raises :class:`ImportError`.

Usage::

    pip install "harbor>=0.6.6" "cwsandbox[harbor]"

    harbor run ... \\
      --environment-import-path cwsandbox.harbor:CWSandboxEnvironment
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from cwsandbox import Sandbox, Session

_logger = logging.getLogger(__name__)


# BaseEnvironment resolves to Any because harbor.* is treated as untyped
# (see [tool.mypy.overrides] for "harbor.*"). Subclassing Any is a strict-mode
# error, but here it's the only honest option: we genuinely don't have stubs
# for Harbor and don't want to pull harbor into the dev venv (it conflicts
# with cli/dev via click).
class CWSandboxEnvironment(BaseEnvironment):  # type: ignore[misc]
    """Harbor environment backend for CW Sandbox.

    Uses pre-built container images only — does not support building from
    Dockerfiles. Images must be specified via docker_image in task.toml or
    the docker_image constructor kwarg (--ek docker_image=...).

    A shared Session is lazily created on first start() and tracks all
    sandboxes for automatic cleanup when the process exits (via the SDK's
    signal handlers) or when individual trials call stop(). No external
    lifecycle management is needed — the Session is fully self-contained.
    """

    # Shared Session across all instances. Lazily created on first start().
    # Access is safe: registration happens on Harbor's asyncio loop,
    # deregistration happens on _LoopManager's background loop, both
    # protected by the GIL for dict operations.
    _shared_session: ClassVar[Session | None] = None
    _session_lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def _get_session_lock(cls) -> asyncio.Lock:
        """Get or create the asyncio lock for session initialization."""
        if cls._session_lock is None:
            cls._session_lock = asyncio.Lock()
        return cls._session_lock

    @classmethod
    async def _ensure_session(
        cls,
        base_url: str | None = None,
    ) -> Session:
        """Lazily create the shared Session on first use."""
        if cls._shared_session is not None:
            return cls._shared_session

        async with cls._get_session_lock():
            # Double-check after acquiring lock
            if cls._shared_session is not None:
                return cls._shared_session

            from cwsandbox import SandboxDefaults, Session

            defaults_kwargs: dict[str, Any] = {"runner_ids": ["prod-east-14-managed"]}
            if base_url:
                defaults_kwargs["base_url"] = base_url

            cls._shared_session = Session(
                defaults=SandboxDefaults(**defaults_kwargs),
                report_to=[],  # Disable wandb reporting
            )
            _logger.info("Created shared CW Sandbox session")
            return cls._shared_session

    @staticmethod
    def type() -> str:
        return "cwsandbox"

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    _SHUTDOWN_BUFFER_SECONDS = 600  # 10 minutes

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        docker_image: str | None = None,
        base_url: str | None = None,
        agent_timeout_sec: float = 600.0,
        verifier_timeout_sec: float = 600.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Resolve image before super().__init__() which calls _validate_definition()
        self._docker_image: str | None = docker_image or task_env_config.docker_image
        self._base_url = base_url
        self._max_lifetime_seconds = int(
            task_env_config.build_timeout_sec
            + agent_timeout_sec
            + verifier_timeout_sec
            + self._SHUTDOWN_BUFFER_SECONDS
        )
        self._sandbox: Sandbox | None = None

        # Import cwsandbox here (main thread) so signal handlers install
        # correctly. The SDK installs SIGINT/SIGTERM handlers on import, which
        # fails if first imported from a thread pool worker.
        from cwsandbox import (
            NetworkOptions,
            Sandbox,  # noqa: F401
            SandboxDefaults,
        )

        # `_Sandbox` and `_SandboxDefaults` are intentionally captured but
        # unused; the import above is what's load-bearing for signal-handler
        # install order. Held as Any since they're class objects from a lazy
        # import path.
        self._Sandbox: Any = Sandbox
        self._SandboxDefaults: Any = SandboxDefaults
        self._NetworkOptions: Any = NetworkOptions

        super().__init__(
            *args,
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

    def _validate_definition(self) -> None:
        if not self._docker_image:
            raise ValueError(
                "CW Sandbox requires a pre-built container image. "
                "Set docker_image in task.toml [environment] section, or pass "
                "--ek docker_image=<image> on the CLI."
            )

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError(
                "CW Sandbox does not support building images. "
                "Use a pre-built image via docker_image instead."
            )

        session = await self._ensure_session(base_url=self._base_url)
        resources = {
            "cpu": f"{self.task_env_config.cpus * 1000}m",
            "memory": f"{self.task_env_config.memory_mb}Mi",
        }
        sandbox = session.sandbox(
            command="sleep",
            args=["infinity"],
            container_image=self._docker_image,
            network=self._NetworkOptions(egress_mode="internet"),
            resources=resources,
        )
        self._sandbox = sandbox
        # Set per-sandbox lifetime before start() sends the request.
        # session.sandbox() doesn't expose max_lifetime_seconds, so we
        # set it directly on the unstarted sandbox object.
        sandbox._max_lifetime_seconds = self._max_lifetime_seconds
        # session.sandbox() returns an unstarted sandbox. Use the sync API
        # via asyncio.to_thread to avoid cross-loop issues — the SDK's
        # _LoopManager runs its own event loop, and awaiting its internal
        # async methods from Harbor's loop causes "Future attached to a
        # different loop" errors.
        await asyncio.to_thread(sandbox.start().result)
        await asyncio.to_thread(sandbox.wait, timeout=self.task_env_config.build_timeout_sec)

        # Create log directories
        await self.exec(f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _stop_sandbox(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.stop(missing_ok=True)

    async def stop(self, delete: bool) -> None:
        if self._sandbox is None:
            return

        try:
            await self._stop_sandbox()
        except Exception as e:
            self.logger.warning(f"Error stopping CW Sandbox: {e}")
        finally:
            self._sandbox = None

    def _ensure_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sandbox

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        sandbox = self._ensure_sandbox()

        # Wrap env vars into the command since CW Sandbox doesn't support
        # per-exec environment variables
        if env:
            exports = " ".join(f"{k}={_shell_quote(v)}" for k, v in env.items())
            command = f"export {exports} && {command}"

        # CW Sandbox's exec has no per-call user switch; commands run as the
        # container image's default user (root for the pre-built images used
        # here). Honor `user` by wrapping via `su -c` when a non-root user is
        # requested; skip the wrap for root/None to avoid shelling out through
        # su on images that may not have it fully configured.
        if user is not None and str(user) != "root" and user != 0:
            command = f"su -s /bin/bash {_shell_quote(str(user))} -c {_shell_quote(command)}"

        result = await sandbox.exec(
            ["bash", "-c", command],
            cwd=cwd,
            timeout_seconds=timeout_sec,
        )

        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        sandbox = self._ensure_sandbox()

        data = Path(source_path).read_bytes()
        await sandbox.write_file(target_path, data)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        sandbox = self._ensure_sandbox()

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        # Create a tar archive in memory
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for file_path in source_path.rglob("*"):
                arcname = str(file_path.relative_to(source_path))
                tar.add(str(file_path), arcname=arcname)
        tar_bytes = buf.getvalue()

        # Upload and extract
        xfer_path = "/tmp/harbor-upload.tar.gz"
        await sandbox.write_file(xfer_path, tar_bytes)
        await self.exec(
            f"mkdir -p {target_dir} && tar xzf {xfer_path} -C {target_dir} && rm -f {xfer_path}"
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        sandbox = self._ensure_sandbox()

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = await sandbox.read_file(source_path)
        target.write_bytes(data)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        sandbox = self._ensure_sandbox()

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        # Tar the remote directory and download
        xfer_path = "/tmp/harbor-download.tar.gz"
        result = await self.exec(f"tar czf {xfer_path} -C {source_dir} . 2>/dev/null; echo $?")

        # If tar failed (e.g., empty directory), just return
        if result.return_code != 0:
            self.logger.debug(
                f"tar of {source_dir} returned {result.return_code}, directory may be empty"
            )
            return

        tar_bytes = await sandbox.read_file(xfer_path)

        # Extract locally with safety checks
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            for member in tar.getmembers():
                # Reject path traversal
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    self.logger.warning(f"Skipping unsafe tar member: {member.name}")
                    continue
                tar.extract(member, path=target)

        # Clean up remote tar
        await self.exec(f"rm -f {xfer_path}")


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell use."""
    return "'" + s.replace("'", "'\\''") + "'"
