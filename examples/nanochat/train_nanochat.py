#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""End-to-end LLM training pipeline using nanochat inside a CWSandbox.

Replicates karpathy/nanochat's runcpu.sh: tokenizer training, base model
pretraining, evaluation, and supervised fine-tuning — all running on CPU
inside a single sandbox.

Demonstrates:
- Long-running sequential pipeline with per-step exec() streaming
- Shell-wrapped commands with venv activation across exec() calls
- Per-step error handling with clear step boundaries
- Configurable iteration count for quick smoke tests
"""

from __future__ import annotations

import argparse
import sys
import time

from cwsandbox import Sandbox, SandboxDefaults

REPO_URL = "https://github.com/karpathy/nanochat.git"
WORKSPACE = "/workspace/nanochat"
VENV_ACTIVATE = f"source {WORKSPACE}/.venv/bin/activate"

# SFT dataset URL (from runcpu.sh)
SFT_DATASET_URL = (
    "https://huggingface.co/datasets/karpathy/llm_hero_data/resolve/main"
    "/identity_conversations.jsonl"
)
SFT_DATASET_PATH = "/root/.cache/nanochat/identity_conversations.jsonl"


def run_step(
    sandbox: Sandbox,
    name: str,
    command: str,
    timeout_seconds: float,
) -> None:
    """Execute a pipeline step with streaming output and error handling."""
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print(f"{'='*60}")

    start = time.monotonic()
    process = sandbox.exec(
        ["bash", "-c", command],
        timeout_seconds=timeout_seconds,
    )

    for line in process.stdout:
        print(line, end="")

    result = process.result()
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        if result.stderr:
            print(f"stderr:\n{result.stderr}")
        sys.exit(1)

    print(f"\nCompleted: {name} ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nanochat training pipeline in a CWSandbox.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5000,
        help="Number of base training iterations (default: 5000)",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip supervised fine-tuning steps",
    )
    args = parser.parse_args()

    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=14400.0,  # 4 hours
        tags=("example", "nanochat"),
        resources={"cpu": "2", "memory": "8Gi"},
        environment_variables={
            "PYTHONUNBUFFERED": "1",
            "NANOCHAT_BASE_DIR": "/root/.cache/nanochat",
        },
    )

    with Sandbox.run("sleep", "infinity", defaults=defaults) as sandbox:
        print(f"Sandbox {sandbox.sandbox_id} is running.")

        # 0. Setup: install uv, clone repo, create venv
        run_step(
            sandbox,
            "Setup environment",
            (
                "apt-get update -qq && apt-get install -y -qq curl git > /dev/null"
                " && curl -LsSf https://astral.sh/uv/install.sh | sh"
                " && export PATH=$HOME/.local/bin:$PATH"
                f" && git clone {REPO_URL} {WORKSPACE}"
                f" && cd {WORKSPACE}"
                ' && printf \'\\n[tool.setuptools.packages.find]\\ninclude = ["nanochat*"]\\n\''
                " >> pyproject.toml"
                " && uv venv"
                " && uv pip install '.[cpu]'"
            ),
            timeout_seconds=600,
        )

        # 1. Download dataset
        run_step(
            sandbox,
            "Download dataset",
            f"cd {WORKSPACE} && {VENV_ACTIVATE} && python -m nanochat.dataset -n 8",
            timeout_seconds=600,
        )

        # 2. Train tokenizer
        run_step(
            sandbox,
            "Train tokenizer",
            (
                f"cd {WORKSPACE} && {VENV_ACTIVATE}"
                " && python -m scripts.tok_train --max-chars=2000000000"
            ),
            timeout_seconds=600,
        )

        # 3. Evaluate tokenizer
        run_step(
            sandbox,
            "Evaluate tokenizer",
            f"cd {WORKSPACE} && {VENV_ACTIVATE} && python -m scripts.tok_eval",
            timeout_seconds=300,
        )

        # 4. Base model training
        run_step(
            sandbox,
            f"Base model training ({args.num_iterations} iterations)",
            (
                f"cd {WORKSPACE} && {VENV_ACTIVATE}"
                " && python -m scripts.base_train"
                " --depth=6 --head-dim=64 --window-pattern=L"
                " --max-seq-len=512 --device-batch-size=4"
                " --total-batch-size=2048 --eval-every=100"
                " --eval-tokens=8192 --core-metric-every=-1"
                " --sample-every=100"
                f" --num-iterations={args.num_iterations}"
                " --run=dummy"
            ),
            timeout_seconds=7200,
        )

        # 5. Base model evaluation
        run_step(
            sandbox,
            "Base model evaluation",
            (
                f"cd {WORKSPACE} && {VENV_ACTIVATE}"
                " && python -m scripts.base_eval"
                " --device-batch-size=1 --split-tokens=4096"
                " --max-per-task=16"
            ),
            timeout_seconds=600,
        )

        if not args.skip_sft:
            # 6. Download SFT dataset
            run_step(
                sandbox,
                "Download SFT dataset",
                f"curl -L -o {SFT_DATASET_PATH} {SFT_DATASET_URL}",
                timeout_seconds=120,
            )

            # 7. SFT training
            run_step(
                sandbox,
                "Supervised fine-tuning",
                (
                    f"cd {WORKSPACE} && {VENV_ACTIVATE}"
                    " && python -m scripts.chat_sft"
                    " --max-seq-len=512 --device-batch-size=32"
                    " --total-batch-size=16384 --eval-every=200"
                    " --eval-tokens=524288 --num-iterations=1500"
                    " --run=dummy"
                ),
                timeout_seconds=3600,
            )

        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
