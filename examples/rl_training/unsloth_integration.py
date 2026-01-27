#!/usr/bin/env python3
"""Unsloth + TRL GRPOTrainer integration example with Aviato sandboxes.

This example demonstrates integrating Aviato sandboxes with Unsloth's
memory-efficient model loading and TRL's GRPOTrainer for reinforcement
learning with code execution rewards.

Unsloth provides 2-4x faster training with 60% less memory through optimized
CUDA kernels and 4-bit quantization. Combined with Aviato sandboxes for
code execution rewards, this enables efficient GRPO training on code generation.

Key patterns demonstrated:
- Session-based sandbox management for automatic cleanup
- Parallel sandbox creation and execution with session.sandbox().exec()
- FastLanguageModel with 4-bit quantization
- Memory-efficient training with LoRA adapters
- GRPOTrainer integration with sandbox-based rewards
- Tagging with model name for tracking

Requirements:
    uv pip install unsloth==2026.1.4 trl==0.27.1 transformers==5.0.0 datasets==4.5.0 torch==2.10.0

Usage:
    uv run examples/rl_training/unsloth_integration.py

Note: Requires GPU with CUDA support. Unsloth optimizations are CUDA-only.
"""

from __future__ import annotations

import uuid

from aviato import SandboxDefaults, Session

JOB_ID = uuid.uuid4().hex[:8]

EXECUTION_TIMEOUT_SECONDS = 10.0
SANDBOX_LIFETIME_SECONDS = 60.0


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-style <answer> tags.

    This follows the standard pattern used in GRPO training examples.
    The model is prompted to put its code inside <answer>...</answer> tags.
    """
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def make_reward_function(session: Session, model_name: str, training_step: int = 0):
    """Create a reward function with model and step-based tagging.

    Args:
        session: Session for sandbox management
        model_name: Model name for sandbox tagging
        training_step: Current training step for sandbox tagging

    Returns:
        A reward function compatible with TRL's GRPOTrainer
    """
    # Track cumulative stats across reward function calls
    call_count = [0]  # Use list to allow mutation in closure
    total_executions = [0]
    total_successes = [0]

    def code_execution_reward(completions: list[str], **kwargs) -> list[float]:
        """Compute rewards by executing code completions in sandboxes.

        Args:
            completions: List of model-generated completions
            **kwargs: Additional arguments from GRPOTrainer (prompts, etc.)

        Returns:
            List of rewards (1.0 for successful execution, 0.0 for failure)
        """
        call_count[0] += 1
        codes = [extract_xml_answer(c) for c in completions]

        # Track which indices have empty code (reward 0.0)
        code_indices = [(i, code) for i, code in enumerate(codes) if code]

        # Create sandboxes and execute non-empty code in parallel
        processes = [
            (i, session.sandbox().exec(
                ["python", "-c", code],
                timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
            ))
            for i, code in code_indices
        ]

        # Collect rewards, defaulting to 0.0
        rewards = [0.0] * len(codes)
        successes = 0
        exceptions = 0
        for i, process in processes:
            try:
                result = process.result()
                if result.returncode == 0:
                    rewards[i] = 1.0
                    successes += 1
            except Exception as e:
                exceptions += 1
                print(f"  [Aviato] WARNING: Sandbox exception for completion {i}: {type(e).__name__}: {e}")

        total_executions[0] += len(code_indices)
        total_successes[0] += successes

        # Show sandbox usage for this batch
        skipped = len(codes) - len(code_indices)
        skip_note = f", {skipped} skipped (no code)" if skipped else ""
        exception_note = f", {exceptions} exceptions" if exceptions else ""
        print(
            f"  [Aviato] Reward call {call_count[0]}: "
            f"{len(code_indices)} sandboxes, "
            f"{successes}/{len(code_indices)} passed{skip_note}{exception_note}"
        )

        # Session handles sandbox cleanup automatically
        return rewards

    return code_execution_reward


SYSTEM_PROMPT = """You solve coding problems by writing Python code.
Put your code inside <answer> tags like this: <answer>print("hello")</answer>
Only include the code, no explanations."""


def create_toy_dataset():
    """Create a toy dataset of simple coding problems."""
    from datasets import Dataset

    problems = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Write code that adds 2 and 3, then prints the result."},
            ],
            "expected_output": "5",
        },
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Write code that prints 'Hello, World!'."},
            ],
            "expected_output": "Hello, World!",
        },
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Write code that prints the sum of numbers from 1 to 10."},
            ],
            "expected_output": "55",
        },
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Write code that prints the length of the string 'aviato'."},
            ],
            "expected_output": "6",
        },
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Write code that prints the maximum of [3, 1, 4, 1, 5, 9]."},
            ],
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
        print("  uv pip install unsloth==2026.1.4")
        return

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"Missing TRL: {e}")
        print("\nInstall required packages:")
        print("  uv pip install trl==0.27.1 transformers==5.0.0 datasets==4.5.0 torch==2.10.0")
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

    # Sanitize model name for use in tags
    model_tag = model_name.replace("/", "-").replace(".", "-")

    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=SANDBOX_LIFETIME_SECONDS,
        tags=(
            "rl-training",
            "unsloth-grpo",
            f"job-{JOB_ID}",
            f"model-{model_tag}",
        ),
    )

    # Configure GRPO for minimal training (proof of concept)
    config = GRPOConfig(
        output_dir="./examples/output/rl_training/unsloth_grpo",
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=128,
        max_steps=1,  # Single step to prove integration
        logging_steps=1,
        report_to="none",  # Disable W&B/tensorboard for example
    )

    # Session manages sandbox lifecycle; cleanup happens automatically on exit
    with Session(defaults=defaults) as session:
        reward_fn = make_reward_function(session, model_name=model_name, training_step=0)

        print("\nSetting up GRPOTrainer with Unsloth model...")
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


if __name__ == "__main__":
    main()
