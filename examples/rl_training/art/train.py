#!/usr/bin/env python3
"""ART training script with Aviato sandbox execution.

This script trains language models on MBPP coding problems using:
- Aviato sandboxes for code execution
- ART (Agent Reinforcement Trainer) for RL training
- LocalBackend (requires GPU) or TinkerBackend (no GPU) for training

Usage:
    uv run examples/rl_training/art/train.py --backend tinker --num-problems 10

Environment Variables:
    OPENAI_API_KEY: API key for inference
    ART_TINKER_API_KEY: API key for TinkerBackend (required if --backend=tinker)
    AVIATO_API_KEY: API key for Aviato sandboxes
    WANDB_API_KEY: (optional) API key for W&B logging
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import TYPE_CHECKING

from datasets import load_dataset

import art
from art.local import LocalBackend
from art.tinker import TinkerBackend
from aviato import SandboxDefaults, Session

from .rollout import Problem, RolloutConfig, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


DEFAULT_MODEL = "gpt-4o"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_NUM_PROBLEMS = 10
DEFAULT_NUM_STEPS = 5
DEFAULT_TRAJECTORIES_PER_PROBLEM = 2


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train on MBPP with ART and Aviato sandboxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  OPENAI_API_KEY        API key for inference
  ART_TINKER_API_KEY    API key for TinkerBackend (required if --backend=tinker)
  AVIATO_API_KEY        API key for Aviato sandboxes
  WANDB_API_KEY         (optional) API key for W&B logging
""",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "tinker"],
        default="local",
        help="Training backend: local (requires GPU) or tinker (no GPU)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for inference (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model for training (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=DEFAULT_NUM_PROBLEMS,
        help=f"Number of MBPP problems to use (default: {DEFAULT_NUM_PROBLEMS})",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of training steps (default: {DEFAULT_NUM_STEPS})",
    )
    parser.add_argument(
        "--trajectories-per-problem",
        type=int,
        default=DEFAULT_TRAJECTORIES_PER_PROBLEM,
        help=f"Trajectories per problem per step (default: {DEFAULT_TRAJECTORIES_PER_PROBLEM})",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Inference API base URL (default: OpenAI)",
    )
    parser.add_argument(
        "--project",
        default="aviato-mbpp",
        help="W&B project name (default: aviato-mbpp)",
    )
    parser.add_argument(
        "--run-name",
        default="train-001",
        help="Training run name (default: train-001)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for training (default: 1e-5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running training",
    )
    return parser.parse_args()


def load_mbpp_problems(num_problems: int) -> list[Problem]:
    """Load MBPP problems from HuggingFace datasets."""
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split=f"train[:{num_problems}]",
    )

    problems = []
    for row in ds:
        test_code = "\n".join(row["test_list"])
        test_imports = row.get("test_setup_code", "") or ""
        problems.append(
            Problem(
                task_id=str(row["task_id"]),
                prompt=row["text"],
                test_code=test_code,
                test_imports=test_imports,
            )
        )
    return problems


def create_backend(backend_type: str) -> LocalBackend | TinkerBackend:
    """Create the training backend."""
    if backend_type == "tinker":
        api_key = os.environ.get("ART_TINKER_API_KEY")
        if not api_key:
            print("Error: ART_TINKER_API_KEY environment variable required for tinker backend")
            sys.exit(1)
        return TinkerBackend(tinker_api_key=api_key)
    else:
        return LocalBackend()


async def collect_trajectory_group(
    problem: Problem,
    session: Session,
    config: RolloutConfig,
    num_trajectories: int,
) -> art.TrajectoryGroup:
    """Collect multiple trajectories for a single problem."""

    async def single_rollout() -> art.Trajectory:
        sandbox = session.sandbox(command="sleep", args=["infinity"])
        sandbox.wait()
        try:
            return await rollout(problem, sandbox, config)
        finally:
            sandbox.stop().result()

    trajectories = await asyncio.gather(
        *[single_rollout() for _ in range(num_trajectories)]
    )
    return art.TrajectoryGroup(trajectories)


async def generate_trajectory_groups(
    problems: list[Problem],
    session: Session,
    config: RolloutConfig,
    trajectories_per_problem: int,
) -> AsyncIterator[art.TrajectoryGroup]:
    """Generate trajectory groups for all problems."""
    for problem in problems:
        yield await collect_trajectory_group(
            problem, session, config, trajectories_per_problem
        )


async def train_step(
    model: art.TrainableModel,
    problems: list[Problem],
    session: Session,
    config: RolloutConfig,
    trajectories_per_problem: int,
    learning_rate: float,
    step: int,
) -> None:
    """Execute a single training step."""
    print(f"\n=== Step {step + 1} ===")
    print(f"Collecting trajectories for {len(problems)} problems...")

    try:
        groups = await art.gather_trajectory_groups(
            (
                collect_trajectory_group(problem, session, config, trajectories_per_problem)
                for problem in problems
            ),
            pbar_desc=f"step {step + 1}",
            max_exceptions=0,
        )

        if not groups:
            print("Warning: No trajectory groups collected")
            return

        total_trajectories = sum(len(g.trajectories) for g in groups)
        total_reward = sum(
            sum(t.reward for t in g.trajectories) for g in groups
        )
        avg_reward = total_reward / total_trajectories if total_trajectories > 0 else 0

        print(f"Collected {total_trajectories} trajectories, avg reward: {avg_reward:.2f}")

        print("Training...")
        await model.train(groups, config=art.TrainConfig(learning_rate=learning_rate))
        current_step = await model.get_step()
        print(f"Training complete: step={current_step}")

        await model.log(groups, split="train")
    finally:
        # Log aviato sandbox execution metrics (success/failure/error rates)
        # Use post-training step to align with ART's model.log() convention:
        # step N metrics = data used to train TO step N
        metric_step = await model.get_step()
        session.log_metrics(step=metric_step, reset=True)


async def main() -> int:
    """Main training entry point."""
    args = parse_args()

    print("ART Training with Aviato Sandboxes")
    print("=" * 40)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Base model: {args.base_model}")
    print(f"Problems: {args.num_problems}")
    print(f"Steps: {args.num_steps}")
    print(f"Trajectories per problem: {args.trajectories_per_problem}")
    print(f"Project: {args.project}")
    print(f"Run name: {args.run_name}")

    if args.dry_run:
        print("\n[Dry run] Validating setup...")

    print("\nLoading MBPP problems...")
    problems = load_mbpp_problems(args.num_problems)
    print(f"Loaded {len(problems)} problems")

    print(f"\nCreating {args.backend} backend...")
    backend = create_backend(args.backend)

    print("Creating trainable model...")
    model = art.TrainableModel(
        name=args.run_name,
        project=args.project,
        base_model=args.base_model,
        _internal_config=art.dev.InternalModelConfig(
            tinker_args=art.dev.TinkerArgs(renderer_name="qwen3"),
        ),
    )

    if args.dry_run:
        print("\n[Dry run] Setup validated successfully!")
        print("Note: Model registration and training require backend connectivity.")
        print("Supported base models depend on backend:")
        print("  - TinkerBackend: meta-llama/*, Qwen/Qwen3-*, deepseek-ai/DeepSeek-V3")
        print("  - LocalBackend: Any HuggingFace model (requires GPU)")
        await backend.close()
        return 0

    print("Registering model with backend...")
    await model.register(backend)

    config = RolloutConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    sandbox_defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("art-training", args.project, args.run_name),
    )

    print("\nStarting training...")
    try:
        with Session(sandbox_defaults) as session:
            start_step = await model.get_step()
            for step in range(start_step, args.num_steps):
                await train_step(
                    model=model,
                    problems=problems,
                    session=session,
                    config=config,
                    trajectories_per_problem=args.trajectories_per_problem,
                    learning_rate=args.learning_rate,
                    step=step,
                )
    finally:
        try:
            await backend.close()
        except RuntimeError:
            pass  # ART backend.close() has a dictionary iteration bug

    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
