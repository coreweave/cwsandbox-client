# ART: Aviato Reward Toolkit

Multi-step RL training with code execution rewards using Aviato sandboxes.

## Overview

This example demonstrates a complete reinforcement learning training loop using Aviato sandboxes for code execution rewards. The system:

- Loads problems from the MBPP (Mostly Basic Python Problems) benchmark
- Generates solutions using an LLM (vLLM or OpenAI)
- Executes solutions in isolated Aviato sandboxes
- Computes binary rewards based on test case results
- Supports multi-step rollouts with error feedback between attempts

The multi-step approach allows models to iteratively refine solutions based on execution feedback, improving success rates compared to single-shot generation.

## Prerequisites

### Hardware

| Mode | Requirements |
|------|-------------|
| Dry run | CPU only |
| OpenAI backend | CPU only (inference via API) |
| vLLM backend | GPU required (A100/H100 recommended) |
| Full training | GPU required (A100/H100 recommended) |

### Environment Variables

```bash
# Required: Aviato authentication (one of these)
export AVIATO_API_KEY="your-aviato-key"
# OR use W&B-based auth:
export WANDB_API_KEY="your-wandb-key"
export WANDB_ENTITY_NAME="your-entity"

# Required: W&B for logging
export WANDB_API_KEY="your-wandb-key"

# Optional: OpenAI API (when using --use-openai)
export OPENAI_API_KEY="your-openai-key"
```

## Installation

Install dependencies using the provided requirements file:

```bash
pip install -r examples/rl_training/art/requirements.txt
```

Or install individually:

```bash
pip install aviato wandb openai datasets transformers
```

For vLLM serving (optional):

```bash
pip install vllm
```

## Usage

### Basic Run

```bash
# Run with vLLM backend (requires vLLM server running)
uv run examples/rl_training/art/train.py

# Run with OpenAI backend
uv run examples/rl_training/art/train.py --use-openai

# Dry run (rollouts only, no training)
uv run examples/rl_training/art/train.py --dry-run --use-openai
```

### Starting vLLM Server

When using the vLLM backend, start the server first:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
```

The training script connects to `http://localhost:8000/v1` by default.

### Custom Configuration

```bash
uv run examples/rl_training/art/train.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --num-steps 20 \
    --batch-size 8 \
    --max-attempts 5 \
    --learning-rate 1e-5 \
    --project "my-rl-project"
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Model name for vLLM or OpenAI |
| `--project` | `aviato-rl-demo` | W&B project name |
| `--num-steps` | `10` | Number of training steps |
| `--batch-size` | `4` | Problems per training step |
| `--learning-rate` | `2e-5` | Learning rate |
| `--max-attempts` | `3` | Max solution attempts per problem |
| `--dataset-split` | `train` | MBPP split: train, test, validation |
| `--dataset-limit` | `100` | Max problems to load (0 for all) |
| `--dry-run` | `false` | Run rollouts without training |
| `--use-openai` | `false` | Use OpenAI API instead of vLLM |

## Expected Output

```
Loading MBPP problems...
Loaded 100 problems
Using vLLM at http://localhost:8000/v1
  Start vLLM with: vllm serve Qwen/Qwen2.5-3B-Instruct

ART Training (job: a1b2c3d4)
============================================================
Model: Qwen/Qwen2.5-3B-Instruct
Problems: 100
Steps: 10
Batch size: 4
============================================================

W&B run: https://wandb.ai/your-entity/aviato-rl-demo/runs/abc123

Starting training loop...
------------------------------------------------------------
  Step 0: reward=0.500, success=50.0% (2/4), time=45.2s
  Step 1: reward=0.750, success=75.0% (3/4), time=38.1s
  Step 2: reward=0.500, success=50.0% (2/4), time=42.3s
  ...
------------------------------------------------------------

Training complete!
  Total trajectories: 40
  Successful: 22 (55.0%)
  Steps completed: 10
```

## Architecture

```
art/
├── __init__.py      # Package exports
├── train.py         # Training loop and CLI
├── rollout.py       # Multi-step sandbox execution
├── rewards.py       # MBPP loading and reward calculation
├── types.py         # Trajectory and TrainableModel protocols
├── requirements.txt # Dependencies
└── README.md        # This file
```

### Components

**TrainingLoop** (`train.py`): Orchestrates the training process:
- Loads MBPP problems from HuggingFace
- Batches problems for parallel rollouts
- Collects trajectories with rewards
- Logs metrics to W&B
- Manages graceful shutdown on SIGINT/SIGTERM

**rollout()** (`rollout.py`): Executes a single problem with multi-step refinement:
1. Creates a fresh Aviato sandbox
2. Prompts model for solution
3. Executes code and checks test results
4. On failure, provides error feedback for next attempt
5. Returns trajectory with messages and final reward

**MBPPProblem** (`rewards.py`): Dataclass representing a coding problem:
- `task_id`: Unique identifier
- `prompt`: Problem description
- `test_cases`: Assertion statements to verify solutions

**Trajectory** (`types.py`): Captures the full interaction history:
- `messages_and_choices`: All prompts and model responses
- `reward`: Final reward (1.0 for success, 0.0 for failure)
- Enables credit assignment for policy gradient training

**TrainableModel** (`types.py`): Protocol for model backends:
- `VLLMModel`: Connects to local vLLM server
- `OpenAIModel`: Uses OpenAI API directly
- Any model providing an OpenAI-compatible async client

### Data Flow

```
MBPP Dataset
     │
     ▼
┌─────────────────┐
│  TrainingLoop   │
│  (batch problems)│
└────────┬────────┘
         │ parallel
         ▼
┌─────────────────┐     ┌─────────────────┐
│    rollout()    │────▶│  Aviato Sandbox │
│ (multi-step)    │◀────│  (code exec)    │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Trajectory    │
│ (messages+reward)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  W&B Logging    │
│  + Training     │
└─────────────────┘
```

### Sandbox Lifecycle

Each rollout creates a fresh sandbox to ensure isolation:

1. **Create**: `Sandbox.run()` with `python:3.11` image
2. **Wait**: Block until sandbox reaches RUNNING state
3. **Execute**: Write test script and run with timeout
4. **Feedback**: On failure, capture errors for model
5. **Cleanup**: `sandbox.stop(missing_ok=True)` in finally block

Sandboxes are tagged with job ID and problem ID for tracking:

```python
tags=(
    "art-rollout",
    f"job-{job_id}",
    f"problem-{problem.task_id}",
)
```

## Troubleshooting

**vLLM connection refused**: Ensure the vLLM server is running and accessible at `http://localhost:8000`.

**W&B authentication failed**: Set `WANDB_API_KEY` or run `wandb login`.

**Sandbox creation timeout**: Check Aviato API credentials and network connectivity.

**Low success rate**: Try increasing `--max-attempts` or using a larger model.

**Memory issues with vLLM**: Use a smaller model or enable tensor parallelism.

## Further Reading

- [RL Training Guide](../../../docs/guides/rl-training.md): Comprehensive documentation
- [MBPP Dataset](https://huggingface.co/datasets/google-research-datasets/mbpp): Benchmark details
- [vLLM Documentation](https://docs.vllm.ai/): Server setup and optimization
