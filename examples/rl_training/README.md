# RL Training Examples

Examples demonstrating Aviato sandboxes for reinforcement learning training with code execution rewards.

**Full documentation**: [RL Training Guide](../../docs/guides/rl-training.md)

## Quick Start

```bash
# Set credentials
export AVIATO_API_KEY="your-api-key"

# Install aviato from source (from repo root)
uv pip install -e .

# Run standalone reward function (no GPU needed)
uv run examples/rl_training/reward_function.py

# Run TRL GRPO integration (GPU recommended)
uv pip install trl==0.27.1 transformers==5.0.0 datasets==4.5.0 torch==2.10.0
uv run examples/rl_training/trl_grpo_integration.py

# Run Unsloth integration (CUDA required)
uv pip install unsloth==2026.1.4
uv run examples/rl_training/unsloth_integration.py
```

## Examples

| Script | Description | GPU |
|--------|-------------|-----|
| `reward_function.py` | Standalone reward function with toy completions | No |
| `trl_grpo_integration.py` | TRL GRPOTrainer with sandbox rewards | Recommended |
| `unsloth_integration.py` | Unsloth + GRPO with 4-bit quantization | Required |
| `art/` | Multi-step rollouts on MBPP benchmark | Recommended |
