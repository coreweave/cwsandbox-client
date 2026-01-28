# RL Training Guide

!!! warning "Work in Progress"
    This guide is under active development. APIs and examples may change.

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
- [W&B Metrics Integration](#wb-metrics-integration)
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

## W&B Metrics Integration

When using W&B (Weights & Biases) for training, aviato Sessions automatically log sandbox usage metrics to your active wandb run. You can see how your training uses sandboxes without writing extra instrumentation code.

### Auto-detection

If `WANDB_API_KEY` is set and a wandb run is active (`wandb.run` exists), metrics logging is enabled automatically:

```python
import wandb
from aviato import Session, SandboxDefaults

wandb.init(project="my-rl-training")

# Metrics logging enabled automatically
with Session(defaults) as session:
    for step in range(num_steps):
        sandbox = session.sandbox()
        # Exec results are automatically tracked - no manual calls needed
        result = sandbox.exec(["python", "-c", code]).result()
        # Log metrics at this training step for correlation
        session.log_metrics(step=step)
# Final metrics logged on session close
```

### Explicit Control

Control metrics reporting with the `report_to` parameter:

```python
# Explicit opt-in (reports even without active wandb run)
session = Session(defaults, report_to=["wandb"])

# Disable reporting (even if wandb run exists)
session = Session(defaults, report_to=[])

# Auto-detect (default behavior)
session = Session(defaults, report_to=None)
```

### Metrics

Execution metrics are tracked automatically when `exec()` completes:

| Metric | Description |
|--------|-------------|
| `aviato/sandboxes_created` | Total sandboxes created via session |
| `aviato/executions` | Total exec() calls |
| `aviato/exec_successes` | Successful executions (returncode=0) |
| `aviato/exec_failures` | Failed executions (returncode!=0) |
| `aviato/exec_errors` | Errors (timeouts, transport failures) |
| `aviato/success_rate` | Fraction of exec() with returncode=0 |
| `aviato/error_rate` | Fraction of exec() that errored |
| `aviato/avg_execs_per_sandbox` | Average exec() calls per sandbox |
| `aviato/min_execs_per_sandbox` | Minimum exec() calls in any sandbox |
| `aviato/max_execs_per_sandbox` | Maximum exec() calls in any sandbox |

Tracking is automatic: just call `exec()` on any sandbox associated with a session. Call `session.log_metrics(step=N)` to log at specific training steps:

```python
def training_step(session, model, batch, step: int) -> list[float]:
    rewards = []
    for task in batch:
        sandbox = session.sandbox()
        # Metrics tracked automatically on exec() completion
        result = sandbox.exec(["python", "-c", task["code"]]).result()
        reward = 1.0 if result.returncode == 0 else 0.0
        rewards.append(reward)
        sandbox.stop()

    # Log metrics at this training step for correlation
    session.log_metrics(step=step)
    return rewards
```

You can also access per-sandbox statistics via the `execution_stats` property:

```python
sandbox = session.sandbox()
result = sandbox.exec(["echo", "hello"]).result()
print(sandbox.execution_stats)  # {"total": 1, "successes": 1, "failures": 0, "errors": 0}
```

### Per-Sandbox Exec Metrics

When using W&B integration with Sessions, the following per-sandbox metrics are automatically tracked:

| Metric | Description |
|--------|-------------|
| `aviato/avg_execs_per_sandbox` | Average exec() calls per sandbox (useful for "tool calls per rollout") |
| `aviato/min_execs_per_sandbox` | Minimum exec() calls in any sandbox |
| `aviato/max_execs_per_sandbox` | Maximum exec() calls in any sandbox |

These metrics help understand agent behavior during RL training:

- **High avg_execs_per_sandbox** may indicate verbose agents that make many tool calls per episode
- **Large variance (max-min)** may indicate inconsistent rollout behavior across episodes
- **Trends over training steps** show how agent behavior evolves as the policy improves

Example dashboard usage:

- Plot `avg_execs_per_sandbox` vs training step to see tool usage trends over training
- Alert if `max_execs_per_sandbox` exceeds a threshold (runaway agent making excessive tool calls)
- Compare min/max spread to detect episodes where agents get stuck in loops vs complete quickly

By default, `log_metrics()` resets the counters after logging. Set `reset=False` to keep accumulating:

```python
session.log_metrics(step=step, reset=False)  # Keep accumulating
```

Metrics are also logged automatically when the session closes, so you get final summary metrics even without explicit logging.

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

**Stateful multi-step rollouts** are different: the agent takes multiple actions within a single sandbox, and the sandbox maintains state between actions. The agent can write a file, run it, see the error, edit the file, and try again - all within the same sandbox.

The `examples/rl_training/art/` directory demonstrates this pattern on the MBPP benchmark. When a solution fails, the agent receives error feedback and can iterate on its approach.

### Overview

[ART (Agent Reinforcement Trainer)](https://github.com/OpenPipe/ART) is an open-source RL framework by OpenPipe for training multi-step agents using GRPO. This example integrates Aviato sandboxes with ART:

- Uses the `art` package (`openpipe-art`) for trajectory collection and training
- Supports two backends: `LocalBackend` (requires GPU) or `TinkerBackend` (no GPU)
- Executes code via tool calling in Aviato sandboxes
- Computes binary rewards based on MBPP test case results

### Training Approach: GRPO with Distillation

This example uses **distillation with reinforcement learning**: a stronger model generates demonstrations, and a smaller model learns to replicate the successful ones.

**Two models are involved:**

1. **Inference model** (`--model`, default: `gpt-5.1-codex-mini`): Generates trajectories during rollouts. This model makes tool calls, sees sandbox results, iterates on errors, and submits solutions. It does not get trained.

2. **Base model** (`--base-model`, default: `Qwen/Qwen3-8B`): The model being trained. It receives the trajectories generated by the inference model and learns from them via GRPO (Group Relative Policy Optimization).

**How it works:**

1. The inference model generates multiple trajectories per problem, each with tool calls executed in Aviato sandboxes
2. Each trajectory receives a binary reward: 1.0 if tests pass, 0.0 otherwise
3. Trajectories for the same problem form a group - GRPO compares trajectories within each group
4. The base model (Qwen3-8B) is trained to prefer higher-reward trajectories over lower-reward ones

After training, you deploy Qwen3-8B with the same tool definitions. It will have learned to make similar tool calls by imitating the successful trajectories from the inference model.

### Prerequisites

| Mode | Requirements |
|------|-------------|
| Dry run | CPU only |
| TinkerBackend | CPU only (training via API) |
| LocalBackend | GPU required |

Environment variables:

```bash
export AVIATO_API_KEY="your-aviato-key"
export OPENAI_API_KEY="your-openai-key"
export ART_TINKER_API_KEY="your-tinker-key"  # required for --backend=tinker
export WANDB_API_KEY="your-wandb-key"        # optional, for logging
```

### Installation

```bash
uv pip install -r examples/rl_training/art/requirements.txt
```

This installs:

```text
openpipe-art==0.5.7       # ART framework
openai==2.15.0            # LLM inference
datasets==4.5.0           # MBPP loading
wandb==0.24.0             # Optional logging
```

For LocalBackend with GPU support, also install:

```bash
uv pip install "openpipe-art[backend]==0.5.7"
```

### Running the Example

```bash
# Dry run - validate setup without training
uv run examples/rl_training/art/train.py --dry-run

# Train with TinkerBackend (no GPU required)
uv run examples/rl_training/art/train.py --backend tinker --num-problems 10

# Train with LocalBackend (requires GPU)
uv run examples/rl_training/art/train.py --backend local --num-problems 10
```

Expected output:

```
ART Training with Aviato Sandboxes
========================================
Backend: tinker
Model: gpt-5.1-codex-mini
Base model: Qwen/Qwen3-8B-Instruct
Problems: 10
Steps: 5
Trajectories per problem: 2
Project: aviato-mbpp
Run name: train-001

Loading MBPP problems...
Loaded 10 problems

Creating tinker backend...
Creating trainable model...
Registering model with backend...

Starting training...

=== Step 1 ===
Collecting trajectories for 10 problems...
step 1: 100%|██████████| 10/10 [00:45<00:00]
Collected 20 trajectories, avg reward: 0.35
Training...
Training complete: step=1, metrics={'loss': 0.42}
...
```

### Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `local` | Training backend: `local` (GPU) or `tinker` (no GPU) |
| `--model` | `gpt-5.1-codex-mini` | Model for inference |
| `--base-model` | `Qwen/Qwen3-8B-Instruct` | Base model for training |
| `--num-problems` | `10` | Number of MBPP problems |
| `--num-steps` | `5` | Training steps |
| `--trajectories-per-problem` | `2` | Trajectories collected per problem per step |
| `--base-url` | `None` | OpenAI-compatible API base URL |
| `--project` | `aviato-mbpp` | W&B project name |
| `--run-name` | `train-001` | Training run name |
| `--learning-rate` | `1e-5` | Learning rate |
| `--dry-run` | `false` | Validate setup without training |

### Architecture

```
art/
├── train.py         # Training loop, CLI, and ART backend setup
├── rollout.py       # Multi-step sandbox execution, builds Trajectory
├── tools.py         # Tool schemas for execute_code and submit_solution
└── __init__.py
```

**Key ART imports:**

```python
import art
from art.local import LocalBackend
from art.tinker import TinkerBackend

# Create trainable model
model = art.TrainableModel(
    name="train-001",
    project="aviato-mbpp",
    base_model="Qwen/Qwen3-8B-Instruct",
)

# Collect trajectories
groups = await art.gather_trajectory_groups(
    (collect_trajectories(problem) for problem in problems),
    pbar_desc="collecting",
)

# Train
result = await backend.train(model, groups, learning_rate=1e-5)
```

**Rollout returns `art.Trajectory`:**

```python
from openai.types.chat import ChatCompletionToolParam

trajectory = art.Trajectory(
    messages_and_choices=messages_and_choices,  # Conversation history
    tools=ROLLOUT_TOOLS,                        # Tool definitions
    reward=1.0 if passed else 0.0,              # Binary reward
    metadata={"task_id": problem.task_id},
)
return trajectory.finish()
```

**Tool-calling pattern:**

The rollout uses OpenAI-compatible tool calling with two tools:
- `execute_code`: Test code in sandbox, returns stdout/stderr
- `submit_solution`: Final submission, runs all test cases

```python
ROLLOUT_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in an isolated sandbox...",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    # submit_solution tool...
]
```

Sandboxes are tagged for tracking:

```python
sandbox_defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("art-training", args.project, args.run_name),
)
```

### Data Flow

The training pipeline has several components:

1. **Local machine**: The training script (`train.py`) runs on your machine or training server. It loads problems from the MBPP dataset and orchestrates the training loop.

2. **OpenAI API (inference)**: During trajectory collection, the rollout code calls the OpenAI API (or compatible endpoint) to generate model responses. The model receives tool definitions and returns tool calls that the rollout executes.

3. **Aviato sandboxes (code execution)**: Each rollout uses a single Aviato sandbox that persists across all tool calls. When the model calls `execute_code` or `submit_solution`, the code runs in that sandbox. This means file changes and state accumulate as the agent iterates - it can write a file, run it, see an error, and fix it. The sandbox provides isolation so untrusted model-generated code cannot affect the host. Results (stdout, stderr, exit code) flow back to the rollout.

4. **Trajectory collection**: The rollout accumulates the conversation history (messages and tool results) along with the final reward into an `art.Trajectory` object. Multiple trajectories for the same problem form an `art.TrajectoryGroup`.

5. **Training backend**: The collected trajectory groups are sent to the training backend. With `LocalBackend`, training happens on your local GPU. With `TinkerBackend`, trajectories are uploaded to Thinking Machines's Tinker service which handles training remotely - no local GPU required.
