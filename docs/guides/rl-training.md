# RL Training Guide

This guide covers using Aviato sandboxes for reinforcement learning training workflows, specifically for computing code execution rewards.

## Why Sandboxes for Code Execution Rewards

Training LLMs to generate code requires executing model outputs to verify correctness. Running untrusted code from a model presents security and reproducibility challenges: the code might perform file system operations, make network requests, or produce non-deterministic results. Sandboxes solve these problems by providing isolated, ephemeral environments where code runs without affecting the host system. Each execution starts from a clean state, ensuring consistent reward computation across training runs.

Aviato sandboxes are designed for this use case. They start quickly, handle concurrent executions, and integrate with Python training code through a sync/async hybrid API. The tagging and listing APIs enable cleanup and monitoring, which matters when running thousands of executions during a training job.

## Core Pattern

The fundamental pattern is creating sandboxes within your training loop to evaluate model-generated code:

```python
import aviato
from aviato import Sandbox, SandboxDefaults

def compute_reward(generated_code: str, test_cases: list[dict]) -> float:
    with Sandbox.run() as sandbox:
        # Write the generated code to the sandbox
        sandbox.write_file("/tmp/solution.py", generated_code.encode()).result()

        # Run test cases
        passed = 0
        for test in test_cases:
            result = sandbox.exec(
                ["python", "/tmp/solution.py"],
                timeout_seconds=10.0,
            ).result()

            if result.returncode == 0 and test["expected"] in result.stdout:
                passed += 1

        return passed / len(test_cases)
```

This pattern creates a fresh sandbox for each reward computation. The context manager ensures cleanup happens even if an exception occurs during execution.

## Tagging for Job Metadata

Tags enable filtering and discovery of sandboxes created by your training jobs. Include metadata that helps identify sandboxes when debugging or cleaning up:

```python
from aviato import SandboxDefaults

def make_defaults(
    wandb_run_id: str,
    training_step: int,
    model_name: str,
) -> SandboxDefaults:
    return SandboxDefaults(
        container_image="python:3.11",
        tags=(
            f"run:{wandb_run_id}",
            f"step:{training_step}",
            f"model:{model_name}",
            "rl-training",
        ),
    )
```

Useful metadata to include in tags:

| Tag Pattern | Purpose |
|-------------|---------|
| `run:{id}` | W&B run ID or job identifier for filtering by training run |
| `step:{n}` | Training step number for debugging specific iterations |
| `model:{name}` | Model name or checkpoint for multi-model experiments |
| `env:{name}` | Environment (dev, staging, prod) for resource management |

Tags propagate to the backend, enabling queries like "find all sandboxes from run abc123" without maintaining local state.

## Cleanup Patterns

Training jobs can create thousands of sandboxes. Proper cleanup prevents resource accumulation and simplifies debugging.

### Session-Based Cleanup

Sessions provide automatic cleanup of all sandboxes when the session closes:

```python
import aviato
from aviato import SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("rl-training", f"run:{run_id}"),
)

with aviato.Session(defaults=defaults) as session:
    for step in range(num_steps):
        # Sandboxes created through session are tracked
        sandbox = session.sandbox()
        reward = evaluate_in_sandbox(sandbox, model_output)
        sandbox.stop().result()

# All remaining sandboxes cleaned up when session exits
```

### Cleanup on Job Failure

When training fails, sandboxes from the failed job may remain running. Query by tags to find and clean them:

```python
from aviato import Sandbox

def cleanup_run(run_id: str) -> int:
    orphans = Sandbox.list(tags=[f"run:{run_id}"]).result()
    for sandbox in orphans:
        sandbox.stop(missing_ok=True).result()
    return len(orphans)
```

Call this function in your job's finally block or error handler to ensure cleanup happens regardless of how the job terminates.

### Cleanup Old Sandboxes

For scheduled cleanup of sandboxes older than a threshold:

```python
from datetime import datetime, timedelta, timezone
from aviato import Sandbox

def cleanup_old_sandboxes(max_age_hours: int = 24) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    sandboxes = Sandbox.list(tags=["rl-training"]).result()

    cleaned = 0
    for sandbox in sandboxes:
        if sandbox.started_at and sandbox.started_at < cutoff:
            sandbox.stop(missing_ok=True).result()
            cleaned += 1

    return cleaned
```

## Parallelism for Concurrent Rollouts

RL training benefits from evaluating multiple rollouts concurrently. Aviato supports several parallelism patterns.

### Parallel Sandbox Creation

Create sandboxes in parallel to reduce startup latency:

```python
import aviato
from aviato import Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("rl-training",),
)

def parallel_evaluate(code_samples: list[str]) -> list[float]:
    # Start all sandboxes concurrently
    sandboxes = [Sandbox.run(defaults=defaults) for _ in code_samples]

    # Wait for all to be running
    aviato.wait(sandboxes)

    # Execute in parallel
    processes = [
        sb.exec(["python", "-c", code])
        for sb, code in zip(sandboxes, code_samples)
    ]

    # Collect results
    rewards = []
    for process in processes:
        result = process.result()
        rewards.append(1.0 if result.returncode == 0 else 0.0)

    # Cleanup
    aviato.result([sb.stop() for sb in sandboxes])

    return rewards
```

### Session with Worker Pool

For sustained parallel evaluation, maintain a pool of sandboxes:

```python
import aviato
from aviato import SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("rl-training", "worker-pool"),
)

with aviato.Session(defaults=defaults) as session:
    # Create worker pool
    pool_size = 8
    workers = [session.sandbox() for _ in range(pool_size)]

    # Distribute work across workers
    for batch in batches:
        processes = [
            worker.exec(["python", "-c", code])
            for worker, code in zip(workers, batch)
        ]
        results = [p.result() for p in processes]
        # Process results...
```

### Waiting for First N Results

When you need results from a subset of parallel executions:

```python
import aviato

processes = [sb.exec(["python", "evaluate.py"]) for sb in sandboxes]

# Wait for first 5 to complete
done, pending = aviato.wait(processes, num_returns=5)

# Use completed results immediately
for process in done:
    result = process.result()
    # Process result...

# Cancel or wait for remaining
for process in pending:
    process.cancel()
```

## TRL GRPOTrainer Integration

TRL's GRPOTrainer accepts a reward function that receives model completions and returns scores. Wrap Aviato execution in this function:

```python
from aviato import Sandbox, SandboxDefaults
from trl import GRPOConfig, GRPOTrainer

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("trl-grpo", f"run:{wandb_run_id}"),
)

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        code = extract_code(completion)
        with Sandbox.run(defaults=defaults) as sandbox:
            result = sandbox.exec(
                ["python", "-c", code],
                timeout_seconds=30.0,
            ).result()
            rewards.append(1.0 if result.returncode == 0 else 0.0)
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=GRPOConfig(
        output_dir="./grpo_output",
        num_generations=4,
    ),
    train_dataset=dataset,
)

trainer.train()
```

For better throughput, parallelize the reward computation:

```python
import aviato
from aviato import Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("trl-grpo", f"run:{wandb_run_id}"),
)

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    codes = [extract_code(c) for c in completions]

    # Create sandboxes in parallel
    sandboxes = [Sandbox.run(defaults=defaults) for _ in codes]
    aviato.wait(sandboxes)

    # Execute in parallel
    processes = [
        sb.exec(["python", "-c", code], timeout_seconds=30.0)
        for sb, code in zip(sandboxes, codes)
    ]

    # Collect results
    rewards = [
        1.0 if p.result().returncode == 0 else 0.0
        for p in processes
    ]

    # Cleanup
    aviato.result([sb.stop() for sb in sandboxes])

    return rewards
```

## Unsloth + GRPO Integration

Unsloth provides optimized model training. The reward function pattern is similar:

```python
from unsloth import FastLanguageModel
import aviato
from aviato import Sandbox, SandboxDefaults

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B",
    max_seq_length=2048,
    load_in_4bit=True,
)

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("unsloth-grpo", f"run:{wandb_run_id}"),
)

def code_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []

    # Parallel execution for batch
    sandboxes = [Sandbox.run(defaults=defaults) for _ in completions]
    aviato.wait(sandboxes)

    processes = []
    for sandbox, completion in zip(sandboxes, completions):
        code = extract_code_block(completion)
        processes.append(
            sandbox.exec(["python", "-c", code], timeout_seconds=30.0)
        )

    for process in processes:
        try:
            result = process.result()
            rewards.append(1.0 if result.returncode == 0 else 0.0)
        except Exception:
            rewards.append(0.0)

    aviato.result([sb.stop() for sb in sandboxes])

    return rewards
```

When using Unsloth's GRPO implementation, pass the reward function to the trainer:

```python
from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[code_reward],
    args=GRPOConfig(
        output_dir="./unsloth_grpo",
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=512,
    ),
    train_dataset=dataset,
)

trainer.train()
```

## Error Handling in Reward Functions

Reward functions must not raise exceptions, as this interrupts training. Handle errors gracefully:

```python
from aviato import Sandbox, SandboxDefaults, SandboxTimeoutError

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("rl-training",),
)

def robust_reward(completion: str) -> float:
    code = extract_code(completion)
    if not code:
        return 0.0

    try:
        with Sandbox.run(defaults=defaults) as sandbox:
            result = sandbox.exec(
                ["python", "-c", code],
                timeout_seconds=30.0,
            ).result()
            return 1.0 if result.returncode == 0 else 0.0
    except SandboxTimeoutError:
        return 0.0
    except Exception:
        return 0.0
```

## Monitoring and Debugging

### Counting Active Sandboxes

Monitor sandbox usage during training:

```python
from aviato import Sandbox, SandboxStatus

def count_active_sandboxes(run_id: str) -> dict:
    sandboxes = Sandbox.list(
        tags=[f"run:{run_id}"],
        status=[SandboxStatus.RUNNING, SandboxStatus.PENDING],
    ).result()

    return {
        "running": sum(1 for s in sandboxes if s.status == SandboxStatus.RUNNING),
        "pending": sum(1 for s in sandboxes if s.status == SandboxStatus.PENDING),
        "total": len(sandboxes),
    }
```

### Logging Execution Details

Capture execution details for debugging reward computation:

```python
import logging

logger = logging.getLogger(__name__)

def logged_reward(completion: str, step: int) -> float:
    code = extract_code(completion)

    with Sandbox.run(defaults=defaults) as sandbox:
        result = sandbox.exec(
            ["python", "-c", code],
            timeout_seconds=30.0,
        ).result()

        logger.debug(
            "Reward computation",
            extra={
                "step": step,
                "sandbox_id": sandbox.sandbox_id,
                "returncode": result.returncode,
                "stdout_len": len(result.stdout),
                "stderr_len": len(result.stderr),
            },
        )

        return 1.0 if result.returncode == 0 else 0.0
```
