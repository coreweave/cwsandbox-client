# ART Integration with Aviato Sandboxes

Multi-step RL training using [ART (Agent Reinforcement Trainer)](https://github.com/OpenPipe/ART) from OpenPipe with Aviato sandboxes for code execution.

## Overview

This example demonstrates training language models on MBPP (Mostly Basic Python Problems) using:

- **Aviato sandboxes** for secure code execution
- **ART framework** for multi-step agent training
- **LocalBackend** (requires GPU) or **TinkerBackend** (no GPU) for training

## Prerequisites

### Environment Variables

```bash
# Aviato authentication
export AVIATO_API_KEY="your-aviato-key"

# OpenAI-compatible inference
export OPENAI_API_KEY="your-openai-key"

# For TinkerBackend (no local GPU required)
export ART_TINKER_API_KEY="your-tinker-key"

# Optional: W&B logging
export WANDB_API_KEY="your-wandb-key"
```

## Installation

```bash
# Install aviato from source first
uv pip install -e .

# Install ART dependencies (includes TinkerBackend support)
uv pip install -r examples/rl_training/art/requirements.txt
```

For LocalBackend (requires GPU), also install:

```bash
uv pip install "openpipe-art[backend]==0.5.7"
```

## Usage

### TinkerBackend (No GPU)

```bash
uv run python examples/rl_training/art/train.py \
    --backend tinker \
    --num-problems 10 \
    --num-steps 5
```

### LocalBackend (GPU Required)

```bash
uv run python examples/rl_training/art/train.py \
    --backend local \
    --num-problems 10 \
    --num-steps 5
```

### CLI Options

```
--backend {local,tinker}   Training backend (default: local)
--model TEXT               Model for inference (default: gpt-5.1-codex-mini)
--base-model TEXT          Base model for training (default: Qwen/Qwen3-8B)
--num-problems INT         Number of MBPP problems (default: 10)
--num-steps INT            Training steps (default: 5)
--trajectories-per-problem Number of trajectories per problem (default: 2)
--base-url TEXT            Inference API base URL
--project TEXT             W&B project name (default: aviato-mbpp)
--run-name TEXT            Training run name (default: train-001)
--learning-rate FLOAT      Learning rate (default: 1e-5)
--dry-run                  Validate setup without training
```

## Architecture

1. **Trajectory Collection**: For each problem, the agent uses tools (execute_code, submit_solution) in Aviato sandboxes to explore and solve the problem
2. **Reward Computation**: Binary reward (1.0 if tests pass, 0.0 otherwise)
3. **Training Step**: Collected trajectories are grouped and used to train the model via ART backend

## W&B Metrics

When `WANDB_API_KEY` is set, aviato automatically logs sandbox execution metrics (success rate, error rate, exec counts) to your wandb run. See the [W&B Metrics Integration](../../../docs/guides/rl-training.md#wb-metrics-integration) section of the RL Training Guide for details.

## Further Reading

- [ART Documentation](https://github.com/OpenPipe/ART)
- [RL Training Guide](../../../docs/guides/rl-training.md)
- [MBPP Dataset](https://huggingface.co/datasets/google-research-datasets/mbpp)
