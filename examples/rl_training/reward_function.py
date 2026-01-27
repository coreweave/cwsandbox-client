#!/usr/bin/env python3
"""Standalone reward function example for RL training.

This example demonstrates the simplest integration pattern: using Aviato sandboxes
to compute code execution rewards for reinforcement learning training loops.

The reward function executes model-generated code completions against test cases
in isolated sandboxes, returning binary rewards (1.0 for pass, 0.0 for fail).

Key patterns demonstrated:
- Session-based sandbox management for automatic cleanup
- Parallel sandbox creation and execution
- Progress tracking with aviato.wait() as results complete
- Tagging with job ID and problem index for tracking
- Timeout handling (zero reward on timeout)

Usage:
    uv run examples/rl_training/reward_function.py
"""

from __future__ import annotations

import uuid

import aviato
from aviato import SandboxDefaults, Session

JOB_ID = uuid.uuid4().hex[:8]

EXECUTION_TIMEOUT_SECONDS = 10.0
SANDBOX_LIFETIME_SECONDS = 300.0


def code_execution_reward(
    session: Session,
    completions: list[str],
    test_cases: list[str],
) -> list[tuple[int, float]]:
    """Compute rewards by executing code completions against test cases.

    Each completion is executed in a fresh sandbox. A reward of 1.0 is given
    if execution succeeds (returncode 0), otherwise 0.0.

    Args:
        session: Session for sandbox management
        completions: List of code strings to execute
        test_cases: List of test case identifiers (used for tagging)

    Returns:
        List of (index, reward) tuples in completion order
    """
    # Create sandboxes and execute all completions in parallel
    processes = [
        session.sandbox().exec(
            ["python", "-c", code],
            timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
        )
        for code in completions
    ]

    # Map processes to their indices for tracking completion order
    process_to_idx = {id(p): i for i, p in enumerate(processes)}

    # Collect results as they complete, showing progress
    results: list[tuple[int, float]] = []
    pending = list(processes)
    total = len(processes)

    while pending:
        [process], pending = aviato.wait(pending, num_returns=1)
        idx = process_to_idx[id(process)]
        test_id = test_cases[idx]

        try:
            result = process.result()
            reward = 1.0 if result.returncode == 0 else 0.0
        except Exception:
            reward = 0.0

        results.append((idx, reward))
        status = "PASS" if reward == 1.0 else "FAIL"
        print(f"  [{len(results)}/{total}] Problem {idx} ({test_id}): {status}")

    # Session handles sandbox cleanup automatically
    return results


def main() -> None:
    print(f"RL Training Reward Function Example (job: {JOB_ID})")
    print("=" * 60)

    # Toy problems: (code completion, test case ID, expected result)
    # Some include sleeps to demonstrate out-of-order completion
    problems = [
        # Problem 0: Slow computation (should pass, 2s delay)
        (
            "import time; time.sleep(2); print(sum(range(100)))",
            "slow-sum",
            "pass",
        ),
        # Problem 1: Fast string op (should pass, instant)
        (
            "print('hello'.upper())",
            "string-ops",
            "pass",
        ),
        # Problem 2: Medium delay with error (should fail, 1s delay)
        (
            "import time; time.sleep(1); x = 1 / 0",
            "delayed-error",
            "fail",
        ),
        # Problem 3: Syntax error (should fail, instant)
        (
            "print('missing paren'",
            "syntax-error",
            "fail",
        ),
        # Problem 4: Slowest computation (should pass, 3s delay)
        (
            "import time; time.sleep(3); print([x * 2 for x in range(5)])",
            "slow-list",
            "pass",
        ),
    ]

    completions = [p[0] for p in problems]
    test_cases = [p[1] for p in problems]
    expected = [p[2] for p in problems]

    print(f"\nEvaluating {len(completions)} completions...\n")

    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=SANDBOX_LIFETIME_SECONDS,
        tags=(
            "rl-training",
            f"job-{JOB_ID}",
        ),
    )

    with Session(defaults=defaults) as session:
        print("Progress (results arrive as executions complete):")
        print("-" * 60)
        results = code_execution_reward(session, completions, test_cases)
        print("-" * 60)

        # Sort by original index for final summary
        results.sort(key=lambda x: x[0])
        rewards = [r for _, r in results]

        print("\nFinal summary (original order):")
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


if __name__ == "__main__":
    main()
