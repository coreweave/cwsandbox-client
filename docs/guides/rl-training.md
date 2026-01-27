# RL Training Guide

This guide covers using Aviato sandboxes for reinforcement learning training workflows, focusing on agent training with tool execution.

## Contents

- [Why Sandboxes for Agent Tool Execution](#why-sandboxes-for-agent-tool-execution)
- [Prerequisites](#prerequisites)
- [Core Pattern](#core-pattern)
- [Tagging for Job Metadata](#tagging-for-job-metadata)
- [Try it: reward_function.py](#try-it-reward_functionpy)
- [TRL GRPOTrainer Integration](#trl-grpotrainer-integration)
- [Try it: trl_grpo_integration.py](#try-it-trl_grpo_integrationpy)
- [Error Handling in Agent Episodes](#error-handling-in-agent-episodes)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Multi-step Rollouts with ART](#multi-step-rollouts-with-art)

## Why Sandboxes for Agent Tool Execution

Training code agents with reinforcement learning requires executing tool calls (bash commands, file operations) in isolated environments. Running untrusted commands from a model is risky: the code might modify the host filesystem, make network requests, or produce non-deterministic results. Sandboxes solve this by providing isolated, ephemeral environments where tool calls execute without affecting the host or other rollouts.

In an agent training loop, the model generates actions (tool calls), the sandbox executes them, and observations flow back to the model. The sandbox persists across multiple tool calls within an episode, maintaining state as the agent works through a task. Reward is computed based on the final sandbox state (e.g., tests passing) or trajectory quality.

Aviato sandboxes start quickly and handle concurrent executions. The tagging and listing APIs help with cleanup and monitoring when you're running thousands of episodes.

## Prerequisites

Set your API key:

```bash
export AVIATO_API_KEY="your-api-key"
```

Install aviato from source (from repo root):

```bash
uv pip install -e .
```

## Core Pattern

The basic setup: an agent loop runs on your training infrastructure, and tool calls execute in a sandbox.

```python
import aviato
from aviato import Sandbox

def run_agent_episode(model, task: dict, sandbox: Sandbox) -> tuple[list, float]:
    """Run one agent episode, returning trajectory and reward."""
    messages = [{"role": "user", "content": task["prompt"]}]

    for step in range(task.get("max_steps", 10)):
        # Model generates next action
        response = model.generate(messages)
        messages.append({"role": "assistant", "content": response})

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            break

        # Execute tool calls in sandbox
        for tool in tool_calls:
            if tool.name == "bash":
                result = sandbox.exec(
                    ["bash", "-c", tool.command],
                    timeout_seconds=30.0,
                ).result()
                observation = f"exit={result.returncode}\n{result.stdout}{result.stderr}"
            elif tool.name == "read_file":
                content = sandbox.read_file(tool.path).result()
                observation = content.decode()
            elif tool.name == "write_file":
                sandbox.write_file(tool.path, tool.content.encode()).result()
                observation = "File written successfully"

            messages.append({"role": "tool", "name": tool.name, "content": observation})

    # Compute reward from final sandbox state
    test_result = sandbox.exec(task["test_command"]).result()
    reward = 1.0 if test_result.returncode == 0 else 0.0

    return messages, reward
```

The sandbox persists across tool calls within an episode, so file changes accumulate as the agent works.

### Training step with parallel episodes

Process a batch of tasks with one sandbox per episode:

```python
def training_step(model, batch: list[dict], session) -> list[float]:
    """Run agent episodes for a batch of tasks."""

    # Create sandboxes in parallel
    sandboxes = [session.sandbox() for _ in batch]

    trajectories = []
    rewards = []

    for task, sandbox in zip(batch, sandboxes):
        trajectory, reward = run_agent_episode(model, task, sandbox)
        trajectories.append(trajectory)
        rewards.append(reward)
        sandbox.stop()  # Non-blocking cleanup

    # trajectories and rewards go to policy update
    return rewards
```

## Tagging for Job Metadata

Tags let you filter and find sandboxes created by your training jobs. Include metadata that helps identify sandboxes when debugging or cleaning up:

```python
import os
from aviato import SandboxDefaults

def make_defaults(model_name: str) -> SandboxDefaults:
    return SandboxDefaults(
        container_image="python:3.11",
        tags=(
            f"wandb-run:{os.environ.get('WANDB_RUN_ID', 'local')}",
            f"slurm-job:{os.environ.get('SLURM_JOB_ID', 'interactive')}",
            f"model:{model_name}",
            "rl-training",
        ),
    )
```

Useful metadata to include in tags:

| Tag Pattern | Purpose |
|-------------|---------|
| `wandb-run:{id}` | W&B run ID (from `WANDB_RUN_ID` env var) for filtering by training run |
| `slurm-job:{id}` | Slurm job ID (from `SLURM_JOB_ID` env var) for cluster job tracking |
| `model:{name}` | Model name or checkpoint for multi-model experiments |
| `env:{name}` | Environment (dev, staging, prod) for resource management |

Sandbox tags become Kubernetes pod labels, which the CoreWeave observability platform uses for filtering and dashboards.

## Try it: reward_function.py

The simplest integration: compute code execution rewards with parallel sandbox execution.

**What it does:**
- Executes a set of toy code completions (arithmetic, string operations, syntax errors, runtime errors)
- Creates one sandbox per completion for isolation
- Computes binary rewards: 1.0 for successful execution, 0.0 for failure
- Shows progress as results arrive (faster executions complete first)

**How it uses Aviato:**

The example uses `aviato.wait()` to process results as they complete:

```python
# Create sandboxes and execute all completions in parallel
processes = [
    session.sandbox().exec(
        ["python", "-c", code],
        timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
    )
    for code in completions
]

# Collect results as they complete
while pending:
    [process], pending = aviato.wait(pending, num_returns=1)
    result = process.result()
    reward = 1.0 if result.returncode == 0 else 0.0
```

**Run it:**

```bash
uv run examples/rl_training/reward_function.py
```

No additional dependencies required. No GPU needed.

**Expected output:**

Results arrive as executions complete, so faster problems finish first:

```
RL Training Reward Function Example (job: 4768f471)
============================================================

Evaluating 5 completions...

Progress (results arrive as executions complete):
------------------------------------------------------------
  [1/5] Problem 0 (slow-sum): PASS
  [2/5] Problem 1 (string-ops): PASS
  [3/5] Problem 2 (delayed-error): FAIL
  [4/5] Problem 3 (syntax-error): FAIL
  [5/5] Problem 4 (slow-list): PASS
------------------------------------------------------------

Final summary (original order):
------------------------------------------------------------
  Problem 0 (slow-sum): reward=1.0 [PASS] OK
  Problem 1 (string-ops): reward=1.0 [PASS] OK
  Problem 2 (delayed-error): reward=0.0 [FAIL] OK
  Problem 3 (syntax-error): reward=0.0 [FAIL] OK
  Problem 4 (slow-list): reward=1.0 [PASS] OK
------------------------------------------------------------
Total reward: 3.0/5
Pass rate: 3/5 (60%)
```

## TRL GRPOTrainer Integration

TRL uses a reward function interface where completions map directly to rewards. The agent generates a completion, and the reward function executes it in a sandbox.

The standard pattern uses `<answer>` XML tags for code extraction (matching the format used in GRPO math examples with `\boxed{}`):

```python
import aviato
from aviato import SandboxDefaults

session = aviato.Session(defaults=SandboxDefaults(
    container_image="python:3.11",
    tags=("trl-grpo",),
))

def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    codes = [extract_xml_answer(c) for c in completions]
    code_indices = [(i, code) for i, code in enumerate(codes) if code]

    processes = [
        (i, session.sandbox().exec(
            ["python", "-c", code],
            timeout_seconds=30.0,
        ))
        for i, code in code_indices
    ]

    rewards = [0.0] * len(codes)
    for i, process in processes:
        try:
            rewards[i] = 1.0 if process.result().returncode == 0 else 0.0
        except Exception:
            pass

    return rewards
```

This pattern works for training models to generate correct code in a single turn.

## Try it: trl_grpo_integration.py

Aviato sandboxes with TRL's GRPOTrainer for code execution rewards.

**What it does:**
- Loads a small model (`Qwen/Qwen2.5-0.5B-Instruct`)
- Creates a toy dataset of simple coding problems
- Trains the model using GRPO with sandbox-based reward computation
- Runs 10 training steps to demonstrate the integration

**How it uses Aviato:**

The reward function extracts code from `<answer>` tags (the standard GRPO pattern), creates sandboxes in parallel through a Session, executes each completion, and returns binary rewards:

```python
def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-style <answer> tags."""
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def code_execution_reward(completions: list[str], **kwargs) -> list[float]:
    codes = [extract_xml_answer(c) for c in completions]

    # Create sandboxes and execute non-empty code in parallel
    processes = [
        (i, session.sandbox().exec(
            ["python", "-c", code],
            timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
        ))
        for i, code in enumerate(codes) if code
    ]

    # Collect rewards, defaulting to 0.0
    rewards = [0.0] * len(codes)
    for i, process in processes:
        try:
            result = process.result()
            rewards[i] = 1.0 if result.returncode == 0 else 0.0
        except Exception:
            rewards[i] = 0.0

    return rewards
```

The prompts use a system message instructing the model to format code with `<answer>` tags:

```python
SYSTEM_PROMPT = """You solve coding problems by writing Python code.
Put your code inside <answer> tags like this: <answer>print("hello")</answer>
Only include the code, no explanations."""
```

The Session tracks sandboxes and cleans them up when it closes.

**Run it:**

```bash
uv pip install trl==0.27.1 transformers==5.0.0 datasets==4.5.0 torch==2.10.0
uv run examples/rl_training/trl_grpo_integration.py
```

GPU is recommended for reasonable performance. Without one, training works but is slow.

**Expected output:**

```
TRL GRPO Integration Example (job: def67890)
============================================================

Loading model: Qwen/Qwen2.5-0.5B-Instruct
Creating toy dataset...
Dataset size: 5 problems

Setting up GRPOTrainer...

Starting training (10 steps)...
------------------------------------------------------------
  [Aviato] Reward call 1: 2 sandboxes, 0/2 passed
  [Aviato] Reward call 2: 1 sandboxes, 0/1 passed, 1 skipped (no code)
  [Aviato] Reward call 3: 0 sandboxes, 0/0 passed, 2 skipped (no code)
  [Aviato] Reward call 4: 2 sandboxes, 0/2 passed
  ...
  [Aviato] Reward call 8: 2 sandboxes, 1/2 passed
  [Aviato] Reward call 9: 2 sandboxes, 1/2 passed
  [Aviato] Reward call 10: 1 sandboxes, 0/1 passed, 1 skipped (no code)
[training logs]
------------------------------------------------------------

Training completed successfully!
```

**Understanding the output:**

The number of sandboxes varies per step because we only create sandboxes when `extract_code_block()` finds extractable Python code in the model's completion. When the model generates text without recognizable code (no markdown fences like ` ```python `, no `<code>` tags), that completion is skipped and receives a reward of 0.0.

- `2 sandboxes, 0/2 passed` - Model generated 2 code blocks, both failed execution
- `1 sandboxes, 0/1 passed, 1 skipped (no code)` - Model generated 1 code block (failed) and 1 text-only completion
- `0 sandboxes, 0/0 passed, 2 skipped (no code)` - Model generated no extractable code in either completion

Expected with a small, untrained model. As training progresses, you should see fewer skipped completions and more passes.

## Error Handling in Agent Episodes

Sandbox operations can fail (timeouts, missing files, sandbox termination). Return observations that help the agent understand what went wrong:

```python
from aviato import SandboxTimeoutError, SandboxFileError

def execute_tool(sandbox, tool) -> str:
    """Execute a tool call, returning an observation string."""
    try:
        if tool.name == "bash":
            result = sandbox.exec(
                ["bash", "-c", tool.command],
                timeout_seconds=30.0,
            ).result()
            return f"exit={result.returncode}\n{result.stdout}{result.stderr}"
        elif tool.name == "read_file":
            content = sandbox.read_file(tool.path).result()
            return content.decode()
        elif tool.name == "write_file":
            sandbox.write_file(tool.path, tool.content.encode()).result()
            return "File written successfully"
    except SandboxTimeoutError:
        return "Error: command timed out after 30 seconds"
    except SandboxFileError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
```

For reward computation, catch exceptions and return a fallback reward instead of propagating to the training loop.

## Monitoring and Debugging

### Counting Active Sandboxes

Monitor sandbox usage during training:

```python
from aviato import Sandbox, SandboxStatus

def count_active_sandboxes(run_id: str) -> dict:
    sandboxes = Sandbox.list(
        tags=[f"wandb-run:{run_id}"],
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

# Assumes `session` is created at module level

def logged_reward(completion: str, step: int) -> float:
    code = extract_code(completion)

    sandbox = session.sandbox()
    sandbox.wait()
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

    sandbox.stop()
    return 1.0 if result.returncode == 0 else 0.0
```

## Multi-step Rollouts with ART

The TRL example above uses sandboxes for **single-shot execution**: one sandbox per completion, execute once, return a reward. This works for training models to generate correct code in one attempt.

**Stateful multi-step rollouts** are different: the agent takes multiple actions within a single sandbox, and the sandbox maintains state between actions. The agent can write a file, run it, see the error, edit the file, and try again - all within the same sandbox. Ephemeral execution environments can't do this.

The `examples/rl_training/art/` directory demonstrates this pattern on the MBPP benchmark. When a solution fails, the agent receives error feedback and can modify its approach while the sandbox preserves all prior file changes and environment state.

### Overview

[ART (Agent Reinforcement Trainer)](https://github.com/OpenPipe/ART) is an open-source RL framework by OpenPipe for training multi-step agents using GRPO. This example uses Aviato sandboxes with ART:

- Loads problems from the MBPP benchmark
- Generates solutions using an LLM (vLLM or OpenAI)
- Executes solutions in stateful Aviato sandboxes that persist across attempts
- Computes binary rewards based on test case results
- Supports multi-step rollouts where the agent iterates on failures

### Prerequisites

| Mode | Requirements |
|------|-------------|
| Dry run | CPU only |
| OpenAI backend | CPU only (inference via API) |
| vLLM backend | GPU required (A100/H100 recommended) |

Environment variables:

```bash
export AVIATO_API_KEY="your-aviato-key"
export WANDB_API_KEY="your-wandb-key"
export OPENAI_API_KEY="your-openai-key"  # if using --use-openai
```

### Installation

```bash
uv pip install -r examples/rl_training/art/requirements.txt
```

Or install individually:

```bash
uv pip install wandb==0.24.0 openai==2.15.0 datasets==4.5.0 transformers==5.0.0
```

### Running the Example

```bash
# Dry run with OpenAI (no training, just rollouts)
uv run examples/rl_training/art/train.py --use-openai --dry-run

# Full training with vLLM (start vLLM server first)
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
uv run examples/rl_training/art/train.py
```

Expected output:

```
Loading MBPP problems...
Loaded 100 problems

ART Training (job: a1b2c3d4)
============================================================
Model: gpt-4o-mini
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

### Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Model name for vLLM or OpenAI |
| `--project` | `aviato-rl-demo` | W&B project name |
| `--num-steps` | `10` | Number of training steps |
| `--batch-size` | `4` | Problems per training step |
| `--max-attempts` | `3` | Max solution attempts per problem |
| `--dry-run` | `false` | Run rollouts without training |
| `--use-openai` | `false` | Use OpenAI API instead of vLLM |

### Architecture

```
art/
├── train.py         # Training loop and CLI
├── rollout.py       # Multi-step sandbox execution
├── rewards.py       # MBPP loading and reward calculation
└── types.py         # Trajectory and TrainableModel protocols
```

Each rollout creates a fresh sandbox with `Sandbox.run()`. It writes the solution and test script, runs with a timeout, and captures errors on failure for the next attempt. Cleanup happens via `sandbox.stop()` in a finally block.

Sandboxes are tagged with job ID and problem ID for tracking:

```python
tags=(
    "art-rollout",
    f"job-{job_id}",
    f"problem-{problem.task_id}",
)
```

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
