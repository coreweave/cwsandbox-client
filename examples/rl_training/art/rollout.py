"""ART rollout function for multi-step code execution with Aviato sandboxes.

This module provides the core rollout function that integrates Aviato sandboxes
with ART's trajectory system. Each rollout creates a fresh sandbox, allows
multiple solution attempts with error feedback, and returns a trajectory
capturing the full interaction history.

Key patterns:
- Fresh sandbox per rollout for isolation
- Multi-step attempts (configurable, default 3)
- Error feedback between attempts enables iterative refinement
- Proper cleanup on success, failure, or timeout
"""

from __future__ import annotations

import uuid

from aviato import Sandbox, SandboxDefaults, SandboxTimeoutError

from .rewards import MBPPProblem, extract_code, passes_tests
from .types import TrainableModel, Trajectory

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30.0
DEFAULT_SANDBOX_LIFETIME_SECONDS = 120.0


async def rollout(
    model: TrainableModel,
    problem: MBPPProblem,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    sandbox_lifetime_seconds: float = DEFAULT_SANDBOX_LIFETIME_SECONDS,
    job_id: str | None = None,
) -> Trajectory:
    """Execute a multi-step rollout for solving an MBPP problem.

    Creates a fresh sandbox, then iteratively:
    1. Prompts the model for a solution
    2. Executes the code in the sandbox
    3. If tests pass, returns reward 1.0
    4. If tests fail, feeds error back to model for next attempt

    The sandbox is stateful: each attempt can see files from previous attempts,
    enabling the model to build on or debug previous work.

    Args:
        model: TrainableModel providing an OpenAI-compatible async client.
        problem: MBPP problem with prompt and test cases.
        max_attempts: Maximum solution attempts (default 3).
        execution_timeout_seconds: Timeout for each code execution.
        sandbox_lifetime_seconds: Maximum sandbox lifetime.
        job_id: Optional job ID for tagging. Generated if not provided.

    Returns:
        Trajectory with messages, choices, and final reward (1.0 if solved, 0.0 otherwise).
    """
    if job_id is None:
        job_id = uuid.uuid4().hex[:8]

    traj = Trajectory(
        messages_and_choices=[],
        reward=0.0,
    )

    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=sandbox_lifetime_seconds,
        tags=(
            "art-rollout",
            f"job-{job_id}",
            f"problem-{problem.task_id}",
        ),
    )

    sandbox: Sandbox | None = None
    try:
        sandbox = Sandbox.run(defaults=defaults)
        await sandbox.wait()

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a Python programming assistant. "
                    "Write clean, correct Python code to solve the given problem. "
                    "Return only the code in a ```python code block."
                ),
            },
            {
                "role": "user",
                "content": _build_problem_prompt(problem),
            },
        ]

        client = model.openai_client()

        for attempt in range(max_attempts):
            choice = await client.chat.completions.create(
                model="gpt-4",
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=1024,
            )

            assistant_message = choice.choices[0].message
            traj.messages_and_choices.append(choice)

            code = extract_code(assistant_message.content or "")
            if not code:
                error_msg = {
                    "role": "user",
                    "content": (
                        "No code found in your response. "
                        "Please provide Python code in a ```python code block."
                    ),
                }
                assistant_content = assistant_message.content or ""
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append(error_msg)
                traj.messages_and_choices.append(error_msg)
                continue

            script_path = f"/tmp/attempt_{attempt}.py"
            test_script = _build_test_script(code, problem.test_cases)

            await sandbox.write_file(script_path, test_script.encode())

            try:
                result = await sandbox.exec(
                    ["python", script_path],
                    timeout_seconds=execution_timeout_seconds,
                )
            except SandboxTimeoutError:
                timeout_msg = (
                    f"Execution timed out after {execution_timeout_seconds}s. "
                    "Please optimize your solution."
                )
                error_msg = {"role": "user", "content": timeout_msg}
                assistant_content = assistant_message.content or ""
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append(error_msg)
                traj.messages_and_choices.append(error_msg)
                continue

            if result.returncode == 0 and passes_tests(
                result.stdout, result.stderr, problem.test_cases
            ):
                traj.reward = 1.0
                break

            error_output = result.stderr or result.stdout or "Unknown error"
            fail_msg = (
                f"Tests failed. Error output:\n```\n{error_output}\n```\n"
                "Please fix your solution."
            )
            error_msg = {"role": "user", "content": fail_msg}
            assistant_content = assistant_message.content or ""
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(error_msg)
            traj.messages_and_choices.append(error_msg)

    except Exception:
        traj.reward = 0.0
    finally:
        if sandbox is not None:
            try:
                await sandbox.stop(missing_ok=True)
            except Exception:
                pass

    return traj.finish()


def _build_problem_prompt(problem: MBPPProblem) -> str:
    """Build the initial problem prompt for the model.

    Args:
        problem: MBPP problem to format.

    Returns:
        Formatted prompt string.
    """
    test_examples = "\n".join(problem.test_cases[:2])
    return f"""{problem.prompt}

Example test cases:
```python
{test_examples}
```

Write a Python function that passes these test cases."""


def _build_test_script(code: str, test_cases: tuple[str, ...] | list[str]) -> str:
    """Build a test script combining code with test assertions.

    Args:
        code: The solution code.
        test_cases: Test assertions to run.

    Returns:
        Complete Python script with code and tests.
    """
    tests = "\n".join(test_cases)
    return f"""{code}

# Test cases
{tests}
print("All tests passed!")
"""
