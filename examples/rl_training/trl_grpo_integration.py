#!/usr/bin/env python3
"""TRL GRPOTrainer integration example with Aviato sandboxes.

This example demonstrates integrating Aviato sandboxes with TRL's GRPOTrainer
for reinforcement learning with code execution rewards.

Key patterns demonstrated:
- GRPOTrainer reward_funcs parameter
- Batch reward computation with parallel Aviato sandboxes
- Tagging with training step for tracking
- Sync API usage (TRL is sync)

Requirements:
    pip install trl transformers datasets torch

Usage:
    uv run examples/rl_training/trl_grpo_integration.py

Note: Requires GPU for training. Without GPU, the script will run but be slow.
"""

from __future__ import annotations

import re
import uuid

import aviato
from aviato import Sandbox, SandboxDefaults, SandboxTimeoutError

JOB_ID = uuid.uuid4().hex[:8]

EXECUTION_TIMEOUT_SECONDS = 10.0
SANDBOX_LIFETIME_SECONDS = 60.0


def extract_code_block(text: str) -> str:
    """Extract Python code from a completion.

    Handles both fenced code blocks and raw code.
    """
    # Try to extract from markdown code block
    match = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to extract from <code> tags
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return raw text, stripping common prefixes
    lines = text.strip().split("\n")
    code_lines = []
    for line in lines:
        # Skip lines that look like prompts or explanations
        if line.startswith("#") or line.startswith(">>>"):
            continue
        code_lines.append(line)
    return "\n".join(code_lines).strip()


def make_reward_function(training_step: int = 0):
    """Create a reward function with step-based tagging.

    Args:
        training_step: Current training step for sandbox tagging

    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """

    def code_execution_reward(completions: list[str], **kwargs) -> list[float]:
        """Compute rewards by executing code completions in sandboxes.

        Args:
            completions: List of model-generated completions
            **kwargs: Additional arguments from GRPOTrainer (prompts, etc.)

        Returns:
            List of rewards (1.0 for successful execution, 0.0 for failure)
        """
        codes = [extract_code_block(c) for c in completions]

        defaults = SandboxDefaults(
            container_image="python:3.11",
            max_lifetime_seconds=SANDBOX_LIFETIME_SECONDS,
            tags=(
                "rl-training",
                "trl-grpo",
                f"job-{JOB_ID}",
                f"step-{training_step}",
            ),
        )

        # Create sandboxes in parallel for the batch
        sandboxes = [Sandbox.run(defaults=defaults) for _ in codes]
        aviato.wait(sandboxes)

        # Execute code in parallel
        processes = []
        for sandbox, code in zip(sandboxes, codes, strict=True):
            if not code:
                processes.append(None)
                continue
            processes.append(
                sandbox.exec(
                    ["python", "-c", code],
                    timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
                )
            )

        # Collect rewards
        rewards = []
        for process in processes:
            if process is None:
                rewards.append(0.0)
                continue
            try:
                result = process.result()
                rewards.append(1.0 if result.returncode == 0 else 0.0)
            except SandboxTimeoutError:
                rewards.append(0.0)
            except Exception:
                rewards.append(0.0)

        # Cleanup all sandboxes
        aviato.result([sb.stop(missing_ok=True) for sb in sandboxes])

        return rewards

    return code_execution_reward


def cleanup_job_sandboxes() -> None:
    """Clean up any remaining sandboxes from this job."""
    try:
        sandboxes = Sandbox.list(tags=[f"job-{JOB_ID}"]).result()
        if sandboxes:
            print(f"\nCleaning up {len(sandboxes)} sandbox(es)...")
            for sb in sandboxes:
                sb.stop(missing_ok=True).result()
    except Exception as e:
        print(f"Cleanup error: {e}")


def create_toy_dataset():
    """Create a toy dataset of simple coding problems."""
    from datasets import Dataset

    problems = [
        {
            "prompt": (
                "Write a Python function that adds two numbers and prints "
                "the result. Call it with 2 and 3.\n\n```python\n"
            ),
            "expected_output": "5",
        },
        {
            "prompt": "Write Python code that prints 'Hello, World!'.\n\n```python\n",
            "expected_output": "Hello, World!",
        },
        {
            "prompt": (
                "Write Python code that prints the sum of numbers from 1 to 10."
                "\n\n```python\n"
            ),
            "expected_output": "55",
        },
        {
            "prompt": (
                "Write Python code that prints the length of the string 'aviato'."
                "\n\n```python\n"
            ),
            "expected_output": "6",
        },
        {
            "prompt": (
                "Write Python code that prints the maximum of [3, 1, 4, 1, 5, 9]."
                "\n\n```python\n"
            ),
            "expected_output": "9",
        },
    ]

    return Dataset.from_list(problems)


def main() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install trl transformers datasets torch")
        return

    print(f"TRL GRPO Integration Example (job: {JOB_ID})")
    print("=" * 60)

    # Use a small model for quick testing
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure you have enough memory and transformers installed.")
        return

    print("Creating toy dataset...")
    dataset = create_toy_dataset()
    print(f"Dataset size: {len(dataset)} problems")

    # Create reward function with initial step
    print("\nSetting up GRPOTrainer...")
    reward_fn = make_reward_function(training_step=0)

    # Configure GRPO for minimal training (proof of concept)
    config = GRPOConfig(
        output_dir="./grpo_output",
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=128,
        max_steps=1,  # Single step to prove integration
        logging_steps=1,
        report_to="none",  # Disable W&B/tensorboard for example
    )

    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_fn],
            args=config,
            train_dataset=dataset,
        )

        print("\nStarting training (1 step)...")
        print("-" * 60)
        trainer.train()
        print("-" * 60)
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining error: {e}")
        print("\nNote: Full training requires GPU. The integration pattern is still valid.")
        raise

    finally:
        cleanup_job_sandboxes()


if __name__ == "__main__":
    main()
