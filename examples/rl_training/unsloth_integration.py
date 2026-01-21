#!/usr/bin/env python3
"""Unsloth + TRL GRPOTrainer integration example with Aviato sandboxes.

This example demonstrates integrating Aviato sandboxes with Unsloth's
memory-efficient model loading and TRL's GRPOTrainer for reinforcement
learning with code execution rewards.

Unsloth provides 2-4x faster training with 60% less memory through optimized
CUDA kernels and 4-bit quantization. Combined with Aviato sandboxes for
code execution rewards, this enables efficient GRPO training on code generation.

Key patterns demonstrated:
- FastLanguageModel with 4-bit quantization
- Memory-efficient training with LoRA adapters
- GRPOTrainer integration with sandbox-based rewards
- Tagging with model name for tracking
- Parallel sandbox execution for batch rewards

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install trl transformers datasets torch

Usage:
    uv run examples/rl_training/unsloth_integration.py

Note: Requires GPU with CUDA support. Unsloth optimizations are CUDA-only.
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


def make_reward_function(model_name: str, training_step: int = 0):
    """Create a reward function with model and step-based tagging.

    Args:
        model_name: Model name for sandbox tagging
        training_step: Current training step for sandbox tagging

    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    # Sanitize model name for use in tags
    model_tag = model_name.replace("/", "-").replace(".", "-")

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
                "unsloth-grpo",
                f"job-{JOB_ID}",
                f"step-{training_step}",
                f"model-{model_tag}",
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
        from unsloth import FastLanguageModel
    except ImportError as e:
        print(f"Missing Unsloth: {e}")
        print("\nInstall Unsloth:")
        print(
            '  pip install "unsloth[colab-new] @ '
            'git+https://github.com/unslothai/unsloth.git"'
        )
        return

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"Missing TRL: {e}")
        print("\nInstall required packages:")
        print("  pip install trl transformers datasets torch")
        return

    print(f"Unsloth GRPO Integration Example (job: {JOB_ID})")
    print("=" * 60)

    # Use Unsloth's pre-quantized 4-bit model for memory efficiency
    model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    print(f"\nLoading model with Unsloth: {model_name}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,  # Auto-detect
        )

        # Apply LoRA for parameter-efficient fine-tuning
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nEnsure you have:")
        print("  1. A CUDA-capable GPU")
        print("  2. Unsloth installed correctly")
        print("  3. Sufficient GPU memory")
        return

    print("Creating toy dataset...")
    dataset = create_toy_dataset()
    print(f"Dataset size: {len(dataset)} problems")

    # Create reward function with model name for tagging
    print("\nSetting up GRPOTrainer with Unsloth model...")
    reward_fn = make_reward_function(model_name=model_name, training_step=0)

    # Configure GRPO for minimal training (proof of concept)
    config = GRPOConfig(
        output_dir="./unsloth_grpo_output",
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
        print("\nUnsloth optimizations applied:")
        print("  - 4-bit quantization for memory efficiency")
        print("  - LoRA adapters for parameter-efficient training")
        print("  - Gradient checkpointing for reduced memory")

    except Exception as e:
        print(f"\nTraining error: {e}")
        print("\nNote: This example requires a CUDA GPU with Unsloth support.")
        print("The integration pattern remains valid for GPU environments.")
        raise

    finally:
        cleanup_job_sandboxes()


if __name__ == "__main__":
    main()
