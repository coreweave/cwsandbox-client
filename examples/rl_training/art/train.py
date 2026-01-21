#!/usr/bin/env python3
"""ART training script with Aviato sandboxes and W&B logging.

This script orchestrates ART (Aviato Reward Toolkit) training using multi-step
rollouts in Aviato sandboxes. It collects trajectories, computes rewards based
on code execution success, and trains the model using a configurable backend.

Requirements:
    pip install wandb openai datasets transformers trl

Environment:
    WANDB_API_KEY: W&B API key for logging
    OPENAI_API_KEY or AVIATO_API_KEY: For API access

Usage:
    uv run examples/rl_training/art/train.py

    # Dry run without training:
    uv run examples/rl_training/art/train.py --dry-run

    # Custom configuration:
    uv run examples/rl_training/art/train.py --num-steps 20 --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from .rewards import MBPPProblem, load_mbpp_problems
from .rollout import rollout
from .types import TrainableModel, Trajectory


@dataclass
class TrainingConfig:
    """Configuration for ART training."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    project_name: str = "aviato-rl-demo"
    num_steps: int = 10
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_attempts: int = 3
    execution_timeout_seconds: float = 30.0
    sandbox_lifetime_seconds: float = 120.0
    dataset_split: str = "train"
    dataset_limit: int | None = 100
    dry_run: bool = False


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    step: int = 0
    total_trajectories: int = 0
    successful_trajectories: int = 0
    mean_reward: float = 0.0
    success_rate: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "total_trajectories": self.total_trajectories,
            "successful_trajectories": self.successful_trajectories,
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class VLLMModel:
    """Model wrapper using vLLM for serving.

    Implements TrainableModel protocol for use with ART rollouts.
    """

    model_name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "unused"
    _client: AsyncOpenAI | None = field(default=None, repr=False)

    def openai_client(self) -> AsyncOpenAI:
        """Get OpenAI-compatible async client for vLLM."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client


@dataclass
class OpenAIModel:
    """Model wrapper using OpenAI API.

    Implements TrainableModel protocol for use with ART rollouts.
    For development/testing when vLLM is not available.
    """

    model_name: str = "gpt-4o-mini"
    _client: AsyncOpenAI | None = field(default=None, repr=False)

    def openai_client(self) -> AsyncOpenAI:
        """Get OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client


class TrainingLoop:
    """Main training loop orchestrating rollouts and training."""

    def __init__(
        self,
        config: TrainingConfig,
        model: TrainableModel,
        problems: list[MBPPProblem],
    ) -> None:
        self.config = config
        self.model = model
        self.problems = problems
        self.job_id = uuid.uuid4().hex[:8]
        self._shutdown_requested = False
        self._wandb_run = None

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_requested = True
        print("\nShutdown requested, finishing current batch...")

    async def gather_trajectories(
        self,
        batch_problems: list[MBPPProblem],
    ) -> list[Trajectory]:
        """Gather trajectories for a batch of problems.

        Args:
            batch_problems: Problems to solve in this batch.

        Returns:
            List of finished trajectories with rewards.
        """
        tasks = [
            rollout(
                self.model,
                problem,
                max_attempts=self.config.max_attempts,
                execution_timeout_seconds=self.config.execution_timeout_seconds,
                sandbox_lifetime_seconds=self.config.sandbox_lifetime_seconds,
                job_id=self.job_id,
            )
            for problem in batch_problems
        ]

        trajectories = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[Trajectory] = []
        for traj in trajectories:
            if isinstance(traj, BaseException):
                print(f"  Rollout failed: {traj}")
                results.append(Trajectory(reward=0.0, finished=True))
            else:
                results.append(traj)

        return results

    def compute_metrics(
        self,
        step: int,
        trajectories: list[Trajectory],
        elapsed: float,
    ) -> TrainingMetrics:
        """Compute training metrics from trajectories."""
        total = len(trajectories)
        successful = sum(1 for t in trajectories if t.reward > 0)
        rewards = [t.reward for t in trajectories]
        mean_reward = sum(rewards) / total if total > 0 else 0.0

        return TrainingMetrics(
            step=step,
            total_trajectories=total,
            successful_trajectories=successful,
            mean_reward=mean_reward,
            success_rate=successful / total if total > 0 else 0.0,
            elapsed_seconds=elapsed,
        )

    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log metrics to W&B and console."""
        print(
            f"  Step {metrics.step}: "
            f"reward={metrics.mean_reward:.3f}, "
            f"success={metrics.success_rate:.1%} "
            f"({metrics.successful_trajectories}/{metrics.total_trajectories}), "
            f"time={metrics.elapsed_seconds:.1f}s"
        )

        if self._wandb_run is not None:
            self._wandb_run.log(metrics.to_dict(), step=metrics.step)

    async def train_step(
        self,
        trajectories: list[Trajectory],
    ) -> dict[str, float]:
        """Perform a training step on collected trajectories.

        In a full implementation, this would:
        1. Extract (prompt, completion, reward) tuples
        2. Compute advantages for policy gradient
        3. Update model weights

        For this demo, we simulate training.

        Args:
            trajectories: Trajectories with rewards for training.

        Returns:
            Training metrics (loss, etc.)
        """
        if self.config.dry_run:
            return {"loss": 0.0, "skipped": 1.0}

        # Placeholder for actual training
        # In production, this would call:
        #   - TRL GRPOTrainer.train_step()
        #   - Or custom PPO/REINFORCE implementation
        await asyncio.sleep(0.1)  # Simulate training time

        return {"loss": 0.01, "lr": self.config.learning_rate}

    async def run(self) -> None:
        """Run the full training loop."""
        print(f"\nART Training (job: {self.job_id})")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Problems: {len(self.problems)}")
        print(f"Steps: {self.config.num_steps}")
        print(f"Batch size: {self.config.batch_size}")
        if self.config.dry_run:
            print("Mode: DRY RUN (no training)")
        print("=" * 60)

        # Initialize W&B
        try:
            import wandb

            if not self.config.dry_run:
                run = wandb.init(
                    project=self.config.project_name,
                    config={
                        "model_name": self.config.model_name,
                        "num_steps": self.config.num_steps,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "max_attempts": self.config.max_attempts,
                        "dataset_split": self.config.dataset_split,
                    },
                    tags=["art", f"job-{self.job_id}"],
                )
                self._wandb_run = run
                if run is not None:
                    print(f"\nW&B run: {run.url}")
        except ImportError:
            print("\nW&B not installed, logging to console only")
        except Exception as e:
            print(f"\nW&B init failed: {e}, logging to console only")

        print("\nStarting training loop...")
        print("-" * 60)

        all_metrics: list[TrainingMetrics] = []

        for step in range(self.config.num_steps):
            if self._shutdown_requested:
                print("\nShutdown requested, stopping early")
                break

            step_start = time.time()

            # Select batch of problems (cycling through dataset)
            start_idx = (step * self.config.batch_size) % len(self.problems)
            end_idx = start_idx + self.config.batch_size
            if end_idx > len(self.problems):
                batch = self.problems[start_idx:] + self.problems[: end_idx - len(self.problems)]
            else:
                batch = self.problems[start_idx:end_idx]

            # Gather trajectories
            trajectories = await self.gather_trajectories(batch)

            # Compute and log metrics
            elapsed = time.time() - step_start
            metrics = self.compute_metrics(step, trajectories, elapsed)
            self.log_metrics(metrics)
            all_metrics.append(metrics)

            # Train on trajectories
            train_result = await self.train_step(trajectories)
            if self._wandb_run is not None:
                self._wandb_run.log(train_result, step=step)

        print("-" * 60)

        # Final summary
        if all_metrics:
            total_trajs = sum(m.total_trajectories for m in all_metrics)
            total_success = sum(m.successful_trajectories for m in all_metrics)
            overall_rate = total_success / total_trajs if total_trajs > 0 else 0.0
            print("\nTraining complete!")
            print(f"  Total trajectories: {total_trajs}")
            print(f"  Successful: {total_success} ({overall_rate:.1%})")
            print(f"  Steps completed: {len(all_metrics)}")

        # Cleanup W&B
        if self._wandb_run is not None:
            self._wandb_run.finish()


def parse_args() -> tuple[TrainingConfig, bool]:
    """Parse command line arguments.

    Returns:
        Tuple of (TrainingConfig, use_openai flag).
    """
    parser = argparse.ArgumentParser(
        description="ART training with Aviato sandboxes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name for vLLM or OpenAI",
    )
    parser.add_argument(
        "--project",
        default="aviato-rl-demo",
        help="W&B project name",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Problems per training step",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max solution attempts per problem",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        choices=["train", "test", "validation"],
        help="MBPP dataset split",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=100,
        help="Max problems to load (None for all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run rollouts without training",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API instead of vLLM",
    )

    args = parser.parse_args()

    return TrainingConfig(
        model_name=args.model,
        project_name=args.project,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_attempts=args.max_attempts,
        dataset_split=args.dataset_split,
        dataset_limit=args.dataset_limit if args.dataset_limit > 0 else None,
        dry_run=args.dry_run,
    ), args.use_openai


def main() -> None:
    """Main entry point."""
    config, use_openai = parse_args()

    # Load problems
    print("Loading MBPP problems...")
    try:
        problems = load_mbpp_problems(
            split=config.dataset_split,
            limit=config.dataset_limit,
        )
        print(f"Loaded {len(problems)} problems")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall datasets: pip install datasets")
        sys.exit(1)

    # Create model
    if use_openai:
        print("Using OpenAI API for inference")
        model: TrainableModel = OpenAIModel(model_name=config.model_name)
    else:
        print("Using vLLM at http://localhost:8000/v1")
        print(f"  Start vLLM with: vllm serve {config.model_name}")
        model = VLLMModel(model_name=config.model_name)

    # Create training loop
    loop = TrainingLoop(config, model, problems)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: object) -> None:
        loop.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run training
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
