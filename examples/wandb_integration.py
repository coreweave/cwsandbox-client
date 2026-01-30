"""W&B (Weights & Biases) metrics integration example.

This example demonstrates:
- Auto-detection of active wandb runs
- Explicit opt-in and opt-out for metrics reporting
- Automatic tracking of exec() success/failure/error (no manual calls needed)
- Using log_metrics() to log at specific training steps

Metrics logged to wandb:
- aviato/sandboxes_created: Total sandboxes created via session
- aviato/executions: Total exec() calls
- aviato/exec_completed_ok: Completed executions (returncode=0)
- aviato/exec_completed_nonzero: Completed executions (returncode!=0)
- aviato/exec_failures: Failed executions (timeouts, transport failures)
- aviato/exec_completion_rate: Fraction of exec() that completed with returncode=0
- aviato/exec_failure_rate: Fraction of exec() that failed to complete

Prerequisites:
- Set WANDB_API_KEY environment variable
- Run wandb.init() before creating a Session to enable auto-detection

Run with wandb:
    export WANDB_API_KEY="your-api-key"
    uv run examples/wandb_integration.py

Run without wandb (metrics logged to console instead):
    uv run examples/wandb_integration.py --no-wandb
"""

import argparse
import sys

from aviato import SandboxDefaults, Session


def main() -> None:
    parser = argparse.ArgumentParser(description="W&B metrics integration example")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Run without wandb (prints metrics to console)",
    )
    args = parser.parse_args()

    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "wandb-integration"),
    )

    if args.no_wandb:
        run_without_wandb(defaults)
    else:
        run_with_wandb(defaults)


def run_with_wandb(defaults: SandboxDefaults) -> None:
    """Run with wandb metrics logging."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: uv pip install wandb")
        print("Or use --no-wandb to run without wandb")
        sys.exit(1)

    print("W&B Metrics Integration Example")
    print("=" * 50)
    print()

    wandb.init(
        project="aviato-examples",
        name="wandb-integration-demo",
        tags=["aviato", "example"],
    )
    print(f"Initialized wandb run: {wandb.run.name}")
    print()

    with Session(defaults) as session:
        simulate_training_loop(session, num_steps=3, problems_per_step=3)

    wandb.finish()
    print()
    print("Wandb run finished. Check your wandb dashboard for metrics.")


def run_without_wandb(defaults: SandboxDefaults) -> None:
    """Run without wandb, printing metrics to console."""
    print("W&B Metrics Integration Example (no wandb)")
    print("=" * 50)
    print()
    print("Running with report_to=['wandb'] to demonstrate metrics collection.")
    print("Metrics will be collected but not sent to wandb (no active run).")
    print()

    with Session(defaults, report_to=["wandb"]) as session:
        simulate_training_loop(session, num_steps=3, problems_per_step=3)

        print()
        print("Final metrics (would be logged on session close):")
        metrics = session._reporter.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


def simulate_training_loop(session: Session, num_steps: int, problems_per_step: int) -> None:
    """Simulate a training loop that creates sandboxes and executes code.

    Exec results are automatically tracked - no manual calls needed.
    Only call log_metrics(step=N) to correlate with training steps.
    """
    print(f"Simulating {num_steps} training steps with {problems_per_step} problems each")
    print("-" * 50)
    print()

    code_samples = [
        ("add", "print(1 + 2)"),
        ("syntax_error", "print(1 +"),
        ("multiply", "print(3 * 4)"),
        ("divide_by_zero", "print(1 / 0)"),
        ("string_ops", "print('hello'.upper())"),
        ("list_comp", "print([x**2 for x in range(5)])"),
        ("import_error", "import nonexistent_module"),
        ("dict_ops", "d = {'a': 1}; print(d['a'])"),
        ("recursion", "def f(): return f(); f()"),
    ]

    for step in range(num_steps):
        print(f"Step {step + 1}:")

        successes = 0
        failures = 0

        for i in range(problems_per_step):
            idx = (step * problems_per_step + i) % len(code_samples)
            name, code = code_samples[idx]

            with session.sandbox() as sandbox:
                # Exec results are automatically tracked via sandbox completion callback
                result = sandbox.exec(
                    ["python", "-c", code],
                    timeout_seconds=5.0,
                ).result()

                if result.returncode == 0:
                    successes += 1
                    status = "PASS"
                else:
                    failures += 1
                    status = "FAIL"

                print(f"  [{status}] {name}")

        # Log metrics at this training step for correlation
        session.log_metrics(step=step)
        print(f"  -> Logged metrics at step {step}")
        print()


if __name__ == "__main__":
    main()
