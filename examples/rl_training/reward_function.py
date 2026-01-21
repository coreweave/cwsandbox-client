#!/usr/bin/env python3
"""Standalone reward function example for RL training.

This example demonstrates the simplest integration pattern: using Aviato sandboxes
to compute code execution rewards for reinforcement learning training loops.

The reward function executes model-generated code completions against test cases
in isolated sandboxes, returning binary rewards (1.0 for pass, 0.0 for fail).

Key patterns demonstrated:
- Fresh sandbox per execution for isolation
- Tagging with job ID and problem index for tracking
- Timeout handling (zero reward on timeout)
- Cleanup of all sandboxes on exit

Usage:
    uv run examples/rl_training/reward_function.py
"""

from __future__ import annotations

import uuid

from aviato import Sandbox, SandboxDefaults, SandboxTimeoutError

JOB_ID = uuid.uuid4().hex[:8]

EXECUTION_TIMEOUT_SECONDS = 10.0
SANDBOX_LIFETIME_SECONDS = 60.0


def code_execution_reward(
    completions: list[str],
    test_cases: list[str],
) -> list[float]:
    """Compute rewards by executing code completions against test cases.

    Each completion is executed in a fresh sandbox. A reward of 1.0 is given
    if execution succeeds (returncode 0), otherwise 0.0.

    Args:
        completions: List of code strings to execute
        test_cases: List of test case identifiers (used for tagging)

    Returns:
        List of rewards (1.0 for success, 0.0 for failure/timeout)
    """
    rewards = []

    for i, (code, test_id) in enumerate(zip(completions, test_cases, strict=True)):
        defaults = SandboxDefaults(
            container_image="python:3.11",
            max_lifetime_seconds=SANDBOX_LIFETIME_SECONDS,
            tags=(
                "rl-training",
                f"job-{JOB_ID}",
                f"problem-{i}",
                f"test-{test_id}",
            ),
        )

        reward = 0.0
        sandbox = None
        try:
            sandbox = Sandbox.run(defaults=defaults)
            sandbox.wait()
            result = sandbox.exec(
                ["python", "-c", code],
                timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
            ).result()
            reward = 1.0 if result.returncode == 0 else 0.0
        except SandboxTimeoutError:
            reward = 0.0
        except Exception:
            reward = 0.0
        finally:
            if sandbox is not None:
                sandbox.stop(missing_ok=True).result()

        rewards.append(reward)

    return rewards


def cleanup_job_sandboxes() -> None:
    """Clean up any remaining sandboxes from this job."""
    try:
        sandboxes = Sandbox.list(tags=[f"job-{JOB_ID}"]).result()
        if sandboxes:
            print(f"\nCleaning up {len(sandboxes)} sandbox(es)...")
            for sb in sandboxes:
                sb.stop(missing_ok=True).result()
    except Exception as e:
        print(f"Cleanup error: {e}")


def main() -> None:
    print(f"RL Training Reward Function Example (job: {JOB_ID})")
    print("=" * 60)

    # Toy problems: (code completion, test case ID, expected result)
    problems = [
        # Problem 0: Simple arithmetic (should pass)
        (
            "print(2 + 2)",
            "arithmetic",
            "pass",
        ),
        # Problem 1: String manipulation (should pass)
        (
            "print('hello'.upper())",
            "string-ops",
            "pass",
        ),
        # Problem 2: Syntax error (should fail)
        (
            "print('missing paren'",
            "syntax-error",
            "fail",
        ),
        # Problem 3: Runtime error (should fail)
        (
            "x = 1 / 0",
            "runtime-error",
            "fail",
        ),
        # Problem 4: List comprehension (should pass)
        (
            "print([x * 2 for x in range(5)])",
            "list-comp",
            "pass",
        ),
    ]

    completions = [p[0] for p in problems]
    test_cases = [p[1] for p in problems]
    expected = [p[2] for p in problems]

    print(f"\nEvaluating {len(completions)} completions...\n")

    try:
        rewards = code_execution_reward(completions, test_cases)

        print("\nResults:")
        print("-" * 60)
        for i, (test_id, reward, exp) in enumerate(
            zip(test_cases, rewards, expected, strict=True)
        ):
            status = "PASS" if reward == 1.0 else "FAIL"
            match = "OK" if (reward == 1.0) == (exp == "pass") else "UNEXPECTED"
            print(f"  Problem {i} ({test_id}): reward={reward:.1f} [{status}] {match}")

        print("-" * 60)
        print(f"Total reward: {sum(rewards):.1f}/{len(rewards)}")

        passed = sum(1 for r in rewards if r == 1.0)
        print(f"Pass rate: {passed}/{len(rewards)} ({100 * passed / len(rewards):.0f}%)")
    finally:
        cleanup_job_sandboxes()


if __name__ == "__main__":
    main()
