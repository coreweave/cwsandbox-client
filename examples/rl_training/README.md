# RL Training Examples

Examples demonstrating Aviato sandboxes for reinforcement learning training with code execution rewards.

For comprehensive documentation, see the [RL Training Guide](../../docs/guides/rl-training.md).

## Overview

These examples show how to integrate Aviato sandboxes into RL training loops for computing code execution rewards. The core pattern is executing model-generated code in isolated sandboxes and returning binary rewards based on execution success.

Key patterns demonstrated:
- Fresh sandbox per execution for isolation
- Parallel sandbox creation for batch evaluation
- Tagging for job tracking and cleanup
- Timeout handling with zero reward fallback
- Integration with TRL GRPOTrainer

## Prerequisites

### Credentials

Set your Aviato API key:

```bash
export AVIATO_API_KEY="your-api-key"
```

### Dependencies

All examples require the base `aviato` package:

```bash
pip install aviato
```

Additional dependencies vary by example:

| Example | Dependencies |
|---------|-------------|
| `reward_function.py` | None (aviato only) |
| `trl_grpo_integration.py` | `pip install trl transformers datasets torch` |
| `unsloth_integration.py` | `pip install trl transformers datasets torch` and Unsloth (see below) |

For Unsloth:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Hardware

| Example | GPU Required |
|---------|-------------|
| `reward_function.py` | No |
| `trl_grpo_integration.py` | Yes (for reasonable performance) |
| `unsloth_integration.py` | Yes (CUDA required) |

## Examples

### reward_function.py

Standalone reward function demonstrating the simplest integration pattern.

```bash
uv run examples/rl_training/reward_function.py
```

**What it does**: Executes a set of toy code completions (arithmetic, string operations, syntax errors, runtime errors) and computes binary rewards based on execution success.

**Expected output**:

```
RL Training Reward Function Example (job: abc12345)
============================================================

Evaluating 5 completions...


Results:
------------------------------------------------------------
  Problem 0 (arithmetic): reward=1.0 [PASS] OK
  Problem 1 (string-ops): reward=1.0 [PASS] OK
  Problem 2 (syntax-error): reward=0.0 [FAIL] OK
  Problem 3 (runtime-error): reward=0.0 [FAIL] OK
  Problem 4 (list-comp): reward=1.0 [PASS] OK
------------------------------------------------------------
Total reward: 3.0/5
Pass rate: 3/5 (60%)

Cleaning up 0 sandbox(es)...
```

### trl_grpo_integration.py

Integration with TRL's GRPOTrainer for reinforcement learning training.

```bash
uv run examples/rl_training/trl_grpo_integration.py
```

**What it does**: Sets up a GRPOTrainer with a sandbox-based reward function and runs a single training step on a toy dataset of coding problems.

**Expected output**:

```
TRL GRPO Integration Example (job: abc12345)
============================================================

Loading model: Qwen/Qwen2.5-0.5B-Instruct
Creating toy dataset...
Dataset size: 5 problems

Setting up GRPOTrainer...

Starting training (1 step)...
------------------------------------------------------------
[Training logs from TRL]
------------------------------------------------------------

Training completed successfully!
```

### unsloth_integration.py

Memory-efficient training combining Unsloth's optimizations with sandbox-based rewards.

```bash
uv run examples/rl_training/unsloth_integration.py
```

**What it does**: Uses Unsloth's FastLanguageModel with 4-bit quantization and LoRA adapters for memory-efficient GRPO training. The reward function uses parallel sandbox execution.

**Expected output**:

```
Unsloth GRPO Integration Example (job: abc12345)
============================================================

Loading model with Unsloth: unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
Creating toy dataset...
Dataset size: 5 problems

Setting up GRPOTrainer with Unsloth model...

Starting training (1 step)...
------------------------------------------------------------
[Training logs from TRL]
------------------------------------------------------------

Training completed successfully!

Unsloth optimizations applied:
  - 4-bit quantization for memory efficiency
  - LoRA adapters for parameter-efficient training
  - Gradient checkpointing for reduced memory
```

## Key Concepts

### Reward Function Pattern

All examples follow the same core pattern:

```python
def reward_fn(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for code in completions:
        with Sandbox.run(defaults=defaults) as sandbox:
            result = sandbox.exec(
                ["python", "-c", code],
                timeout_seconds=30.0,
            ).result()
            rewards.append(1.0 if result.returncode == 0 else 0.0)
    return rewards
```

### Parallel Execution

For better throughput, create sandboxes in parallel:

```python
sandboxes = [Sandbox.run(defaults=defaults) for _ in codes]
aviato.wait(sandboxes)

processes = [sb.exec(["python", "-c", code]) for sb, code in zip(sandboxes, codes)]
rewards = [1.0 if p.result().returncode == 0 else 0.0 for p in processes]

aviato.result([sb.stop() for sb in sandboxes])
```

### Tagging for Tracking

Tag sandboxes with job metadata for debugging and cleanup:

```python
defaults = SandboxDefaults(
    tags=(
        "rl-training",
        f"job-{job_id}",
        f"step-{training_step}",
    ),
)
```

## Troubleshooting

**Missing GPU**: The TRL and Unsloth examples require a GPU for reasonable performance. Without one, training will be slow but the integration pattern remains valid.

**Unsloth installation fails**: Unsloth requires CUDA. Ensure you have a compatible GPU and CUDA installed.

**Sandbox timeouts**: If sandboxes timeout frequently, increase `EXECUTION_TIMEOUT_SECONDS` in the example scripts.

**Cleanup failures**: Sandboxes are tagged with job IDs. If cleanup fails, use `Sandbox.list(tags=[f"job-{job_id}"])` to find and stop orphaned sandboxes manually.

## Further Reading

- [RL Training Guide](../../docs/guides/rl-training.md) - Comprehensive documentation
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) - TRL documentation
- [Unsloth](https://github.com/unslothai/unsloth) - Unsloth documentation
