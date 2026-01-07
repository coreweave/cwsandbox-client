#!/usr/bin/env python3
"""Example: Parallel batch processing with progress tracking.

This example shows how to:
- Create multiple sandboxes in parallel
- Submit long-running commands concurrently
- Use aviato.wait() to process results as they complete
- Track progress through a large batch job

Usage:
    uv run examples/parallel_batch_job.py
"""

import aviato
from aviato import SandboxDefaults, Session


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "batch-job"),
    )

    # Simulate tasks with varying durations (in seconds)
    task_durations = [1, 3, 1, 4, 2, 3, 1, 2]
    total_tasks = len(task_durations)

    print(f"Starting batch job with {total_tasks} tasks")
    print(f"Task durations: {task_durations}\n")

    with Session(defaults) as session:
        # Create all sandboxes in parallel (returns immediately)
        print("Creating sandboxes...")
        sandboxes = [session.sandbox() for _ in range(total_tasks)]
        print(f"Created {len(sandboxes)} sandboxes\n")

        # Submit all commands in parallel (returns Process immediately)
        print("Submitting tasks...")
        processes = []
        for i, (sb, duration) in enumerate(zip(sandboxes, task_durations, strict=True)):
            # Simulate work that takes varying time
            cmd = f"sleep {duration} && echo 'Task {i} done ({duration}s)'"
            process = sb.exec(["sh", "-c", cmd])
            processes.append(process)
        print(f"Submitted {len(processes)} tasks\n")

        # Process results as they complete using aviato.wait()
        print("Waiting for results (processing in batches of 2)...")
        print("-" * 50)

        completed = 0
        pending = processes

        while pending:
            # Wait for next 2 tasks to complete (or fewer if less remaining)
            batch_size = min(2, len(pending))
            done, pending = aviato.wait(pending, num_returns=batch_size)

            # Process completed tasks
            for process in done:
                result = process.result()
                completed += 1
                print(f"[{completed}/{total_tasks}] {result.stdout.strip()}")

        print("-" * 50)
        print(f"\nBatch job complete: {completed}/{total_tasks} tasks succeeded")

    # Session exit automatically stops all sandboxes
    print("Session closed - all sandboxes cleaned up")


if __name__ == "__main__":
    main()
