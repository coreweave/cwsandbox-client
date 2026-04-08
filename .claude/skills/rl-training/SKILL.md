---
name: rl-training
description: "Use when working with RL training workflows using CoreWeave Sandbox. Covers agent tool execution in sandboxes, parallel episode processing, GRPOTrainer integration with TRL, reward function patterns, W&B metrics integration, tagging for job metadata, and monitoring training runs. Relevant for requests involving reinforcement learning, GRPO, multi-step rollouts, agent training, or tool-calling models."
disable-model-invocation: false
---

# RL Training with CoreWeave Sandbox

Use sandboxes for reinforcement learning training environments where models execute tool calls in isolated environments.

## Why Sandboxes for RL

Training code agents with RL requires executing tool calls (bash, file ops) in isolated environments. Sandboxes give you:
- Isolated, ephemeral environments for untrusted model-generated code
- State persistence across tool calls within an episode
- File changes and installed packages carry over between steps
- Tagging and listing APIs for cleanup and monitoring thousands of sandboxes

## Core Pattern

```python
import cwsandbox
from cwsandbox import Sandbox

def run_agent_episode(model, task: dict, sandbox: Sandbox) -> tuple[list, float]:
    messages = [{"role": "user", "content": task["prompt"]}]

    for step in range(task.get("max_steps", 10)):
        response = model.generate(messages)
        messages.append({"role": "assistant", "content": response})

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            break

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

    test_result = sandbox.exec(task["test_command"]).result()
    reward = 1.0 if test_result.returncode == 0 else 0.0

    return messages, reward
```

## Parallel Episode Processing

```python
def training_step(model, batch: list[dict], session) -> list[float]:
    sandboxes = [session.sandbox() for _ in batch]
    refs = [sb.start() for sb in sandboxes]
    [r.result() for r in refs]  # Wait for all backends to accept

    trajectories = []
    rewards = []

    for task, sandbox in zip(batch, sandboxes):
        trajectory, reward = run_agent_episode(model, task, sandbox)
        trajectories.append(trajectory)
        rewards.append(reward)
        sandbox.stop()

    return rewards
```

## Tagging for Job Metadata

```python
import os
from cwsandbox import SandboxDefaults

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

| Tag pattern | Purpose |
|-------------|---------|
| `wandb-run:{id}` | W&B run ID for filtering by training run |
| `slurm-job:{id}` | Slurm job ID for cluster job tracking |
| `model:{name}` | Model name or checkpoint |
| `env:{name}` | Environment (dev, staging, prod) |

## TRL GRPOTrainer Integration

Standard GRPO pattern with `<answer>` XML tags:

```python
import cwsandbox
from cwsandbox import SandboxDefaults

session = cwsandbox.Session(defaults=SandboxDefaults(
    container_image="python:3.11",
    tags=("trl-grpo",),
))

def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def reward_fn(completions, **kwargs) -> list[float]:
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    codes = [extract_xml_answer(t) for t in texts]
    code_indices = [(i, code) for i, code in enumerate(codes) if code]

    processes = [
        (i, session.sandbox().exec(["python", "-c", code], timeout_seconds=30.0))
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

## Error Handling

```python
from cwsandbox import SandboxTimeoutError, SandboxFileError

def execute_tool(sandbox, tool) -> str:
    try:
        if tool.name == "bash":
            result = sandbox.exec(["bash", "-c", tool.command], timeout_seconds=30.0).result()
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

## W&B Metrics Integration

Auto-enabled when `WANDB_API_KEY` is set and wandb run is active:

```python
import wandb
from cwsandbox import Session, SandboxDefaults

wandb.init(project="my-rl-training")

with Session(defaults) as session:
    for step in range(num_steps):
        sandbox = session.sandbox()
        result = sandbox.exec(["python", "-c", code]).result()
        session.log_metrics(step=step)
```

| Metric | Description |
|--------|-------------|
| `cwsandbox/sandboxes_created` | Total sandboxes created |
| `cwsandbox/executions` | Total exec() calls |
| `cwsandbox/exec_completed_ok` | Successful executions (returncode=0) |
| `cwsandbox/exec_completed_nonzero` | Completed with returncode!=0 |
| `cwsandbox/exec_failures` | Failed executions (timeouts, transport) |
| `cwsandbox/avg_execs_per_sandbox` | Average exec() calls per sandbox |

## Single-Step vs Multi-Step

| Pattern | Use Case |
|---------|----------|
| **Single-shot** | One sandbox per completion, execute once, return reward |
| **Multi-step** | Agent takes multiple actions in same sandbox, state persists |

Multi-step enables: write file → run → see error → edit → try again

## Monitoring Active Sandboxes

```python
from cwsandbox import Sandbox, SandboxStatus

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

## Examples

See `examples/rl_training/`:
- `reward_function.py` — Binary code execution rewards with parallel sandboxes
- `trl_grpo_integration.py` — TRL GRPOTrainer integration
- `art/` — Multi-step rollouts with ART framework

## References

- [CoreWeave Sandbox RL Training Docs](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/rl-training)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl)
- [ART (Agent Reinforcement Trainer)](https://github.com/OpenPipe/ART)
