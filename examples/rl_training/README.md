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
```

## Examples

| Script | Description | GPU |
|--------|-------------|-----|
| `reward_function.py` | Standalone reward function with toy completions | No |
| `trl_grpo_integration.py` | TRL GRPOTrainer with sandbox rewards | Recommended |
| `art/` | ART training with Aviato sandboxes | See below |

### ART Training

The `art/` directory contains ART (Agent Reinforcement Trainer) integration:

```bash
# Install ART dependencies
uv pip install -r examples/rl_training/art/requirements.txt

# Run training with TinkerBackend (no GPU required)
uv run python examples/rl_training/art/train.py --backend tinker

# Run training with LocalBackend (requires GPU)
uv run python examples/rl_training/art/train.py --backend local
```

See [`art/README.md`](art/README.md) for detailed options and configuration.
