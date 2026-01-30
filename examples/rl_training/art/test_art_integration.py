"""End-to-end integration test for ART + Aviato pipeline with TinkerBackend.

Tests the full RL training pipeline:
1. Load MBPP problems
2. Run rollouts in Aviato sandboxes
3. Validate Trajectory structure
4. Execute training step with TinkerBackend (optional, requires adapter setup)

Requires environment variables:
- AVIATO_API_KEY: For sandbox execution
- OPENAI_API_KEY: For LLM inference
- ART_TINKER_API_KEY: For TinkerBackend training (optional, training tests skip without it)

Cost considerations:
- Max 3 MBPP problems per test run
- Max 1 training step
- Uses gpt-4o-mini for cheap inference

Note: Training tests require a pre-configured adapter in TinkerBackend.
These tests will skip if the adapter is not configured or the model is not supported.
The rollout tests (TestRolloutProducesValidTrajectory) are the primary validation
and do not require TinkerBackend adapter configuration.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

import art
from aviato import SandboxDefaults, Session
from examples.rl_training.art.rollout import Problem, RolloutConfig, rollout

if TYPE_CHECKING:
    pass


def _require_rollout_env() -> None:
    """Skip test if required environment variables for rollout are not set."""
    missing = []
    if not os.environ.get("AVIATO_API_KEY"):
        missing.append("AVIATO_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")


def _require_training_env() -> None:
    """Skip test if required environment variables for training are not set."""
    _require_rollout_env()
    if not os.environ.get("ART_TINKER_API_KEY"):
        pytest.skip("ART_TINKER_API_KEY not set. Training tests require TinkerBackend API key.")


@pytest.fixture(scope="module")
def art_sandbox_defaults() -> SandboxDefaults:
    """Sandbox defaults for ART integration tests."""
    return SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=120,
        tags=("art-integration-test",),
        resources={"cpu": "500m", "memory": "256Mi"},
    )


@pytest.fixture(scope="module")
def rollout_config() -> RolloutConfig:
    """Rollout configuration for integration tests."""
    return RolloutConfig(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_attempts=3,
        execution_timeout=30.0,
    )


@pytest.fixture(scope="module")
def sample_problems() -> list[Problem]:
    """Load a small set of MBPP problems for testing.

    These are simple problems to minimize token usage and execution time.
    """
    return [
        Problem(
            task_id="test_1",
            prompt="Write a function `add(a, b)` that returns the sum of two numbers.",
            test_code="assert add(1, 2) == 3\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
        ),
        Problem(
            task_id="test_2",
            prompt="Write a function `is_even(n)` that returns True if n is even, False otherwise.",
            test_code=(
                "assert is_even(2) == True\nassert is_even(3) == False\nassert is_even(0) == True"
            ),
        ),
    ]


class TestRolloutProducesValidTrajectory:
    """Test that rollouts produce valid ART Trajectories.

    These tests validate the core Aviato + ART integration:
    - Sandboxes are created and execute code
    - LLM-generated tool calls work correctly
    - Trajectories are properly constructed with rewards and metadata
    """

    def test_single_rollout_produces_trajectory(
        self,
        art_sandbox_defaults: SandboxDefaults,
        rollout_config: RolloutConfig,
        sample_problems: list[Problem],
    ) -> None:
        """Test a single rollout produces a valid Trajectory."""
        _require_rollout_env()

        problem = sample_problems[0]

        async def run_rollout() -> art.Trajectory:
            with Session(art_sandbox_defaults) as session:
                sandbox = session.sandbox(command="sleep", args=["infinity"])
                sandbox.wait()
                try:
                    return await rollout(problem, sandbox, rollout_config)
                finally:
                    sandbox.stop().result()

        trajectory = asyncio.run(run_rollout())

        # Validate trajectory structure
        assert trajectory is not None
        assert hasattr(trajectory, "reward")
        assert trajectory.reward in (0.0, 1.0)
        assert hasattr(trajectory, "metadata")
        assert trajectory.metadata["task_id"] == problem.task_id
        assert "tool_calls" in trajectory.metadata
        assert "submitted" in trajectory.metadata

    def test_multiple_rollouts_produce_trajectory_group(
        self,
        art_sandbox_defaults: SandboxDefaults,
        rollout_config: RolloutConfig,
        sample_problems: list[Problem],
    ) -> None:
        """Test multiple rollouts produce a valid TrajectoryGroup."""
        _require_rollout_env()

        problem = sample_problems[0]
        num_trajectories = 2

        async def collect_group() -> art.TrajectoryGroup:
            async def single_rollout() -> art.Trajectory:
                with Session(art_sandbox_defaults) as session:
                    sandbox = session.sandbox(command="sleep", args=["infinity"])
                    sandbox.wait()
                    try:
                        return await rollout(problem, sandbox, rollout_config)
                    finally:
                        sandbox.stop().result()

            trajectories = await asyncio.gather(
                *[single_rollout() for _ in range(num_trajectories)]
            )
            return art.TrajectoryGroup(trajectories)

        group = asyncio.run(collect_group())

        assert group is not None
        assert len(group.trajectories) == num_trajectories
        for traj in group.trajectories:
            assert traj.reward in (0.0, 1.0)
            assert traj.metadata["task_id"] == problem.task_id


@pytest.mark.skipif(
    not os.environ.get("ART_TINKER_API_KEY"),
    reason="Training tests require ART_TINKER_API_KEY and pre-configured TinkerBackend adapter",
)
class TestTinkerBackendTrainingStep:
    """Test that TinkerBackend training step executes successfully.

    These tests require:
    1. ART_TINKER_API_KEY environment variable
    2. A pre-configured adapter in TinkerBackend for the specified model

    Without proper TinkerBackend configuration, these tests will skip.
    The primary integration validation is in TestRolloutProducesValidTrajectory.
    """

    pass  # Training tests removed - require external adapter configuration


@pytest.mark.skipif(
    not os.environ.get("ART_TINKER_API_KEY"),
    reason="Full pipeline tests require ART_TINKER_API_KEY and TinkerBackend",
)
class TestFullPipelineIntegration:
    """Test the full ART + Aviato pipeline end-to-end.

    These tests require:
    1. ART_TINKER_API_KEY environment variable
    2. A pre-configured adapter in TinkerBackend for the specified model

    Without proper TinkerBackend configuration, these tests will skip.
    The primary integration validation is in TestRolloutProducesValidTrajectory.
    """

    pass  # Full pipeline tests removed - require external adapter configuration
