# ART Integration with Aviato Sandboxes

Multi-step RL training using [ART (Agent Reinforcement Trainer)](https://github.com/OpenPipe/ART) from OpenPipe with Aviato sandboxes for code execution.

## Status

This example integrates the real ART package for multi-step agent training on MBPP (Mostly Basic Python Problems). Implementation in progress.

## Prerequisites

### Environment Variables

Set these in your `.env` file (values available in repo):

```bash
# Aviato authentication
AVIATO_API_KEY="your-aviato-key"

# ART Tinker backend (no local GPU required)
ART_TINKER_API_KEY="your-tinker-key"

# OpenAI-compatible inference
OPENAI_API_KEY="your-openai-key"

# Optional: W&B logging
WANDB_API_KEY="your-wandb-key"
```

## Installation

```bash
uv pip install -r examples/rl_training/art/requirements.txt
```

## Further Reading

- [ART Documentation](https://github.com/OpenPipe/ART)
- [RL Training Guide](../../../docs/guides/rl-training.md)
- [MBPP Dataset](https://huggingface.co/datasets/google-research-datasets/mbpp)
