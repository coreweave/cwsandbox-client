"""Weights & Biases metrics integration for sandbox usage tracking.

Demonstrates:
- Automatic W&B metrics reporting during sandbox operations
- Explicit opt-in/out of W&B reporting via report_to parameter
- Manual metrics logging at each training step
- Lazy detection: wandb.init() can happen after Session creation

Requirements:
    uv pip install wandb
    export WANDB_API_KEY="your-api-key"
"""

import wandb
from aviato import SandboxDefaults, Session


def main() -> None:
    """Demonstrate W&B integration patterns."""
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "wandb-integration"),
    )

    # Pattern 1: Auto-detection (default)
    # If wandb is installed, WANDB_API_KEY is set, and there's an active run,
    # metrics are automatically reported to W&B.
    print("Pattern 1: Auto-detection")
    wandb.init(project="aviato-examples", name="pattern-1-auto-detection")
    with Session(defaults) as session:
        sb = session.sandbox(command="sleep", args=["infinity"])
        sb.wait()

        result = sb.exec(["echo", "hello"]).result()
        print(f"  Output: {result.stdout.strip()}")

        result = sb.exec(["python", "-c", "print(1+1)"]).result()
        print(f"  Calculation: {result.stdout.strip()}")

        print(f"  Exec stats: {sb.exec_stats}")
    wandb.finish()

    # Pattern 2: Explicit opt-in
    # Force W&B reporting even without auto-detection
    print("\nPattern 2: Explicit W&B opt-in")
    wandb.init(project="aviato-examples", name="pattern-2-explicit-opt-in")
    with Session(defaults, report_to=["wandb"]) as session:
        sb = session.sandbox(command="sleep", args=["infinity"])
        sb.wait()

        for step in range(3):
            result = sb.exec(["echo", f"step {step}"]).result()
            print(f"  Step {step}: {result.stdout.strip()}")

            logged = session.log_metrics(step=step)
            print(f"  Metrics logged: {logged}")
    wandb.finish()

    # Pattern 3: Explicit opt-out
    # Disable W&B reporting even if auto-detection would enable it
    print("\nPattern 3: Explicit opt-out (empty report_to)")
    wandb.init(project="aviato-examples", name="pattern-3-explicit-opt-out")
    with Session(defaults, report_to=[]) as session:
        sb = session.sandbox(command="sleep", args=["infinity"])
        sb.wait()

        result = sb.exec(["echo", "no metrics reported"]).result()
        print(f"  Output: {result.stdout.strip()}")

        # Metrics are collected but NOT logged (report_to=[])
        logged = session.log_metrics(step=0)
        print(f"  Metrics logged: {logged}")
    wandb.finish()

    # Pattern 4: Lazy detection with training loop wrapper
    # Session is created BEFORE wandb.init() - detection happens dynamically
    print("\nPattern 4: Lazy detection (wandb.init inside training loop)")
    with Session(defaults) as session:
        train(session)

    print("\nDone!")


def train(session: Session) -> None:
    """Example training loop that initializes wandb after session creation.

    This pattern is common when:
    - Your training framework initializes wandb
    - You wrap the training call with Session for sandbox support
    - wandb.init() happens inside the training loop, not before
    """
    # Initialize wandb here - Session detects it lazily
    wandb.init(project="aviato-examples", name="pattern-4-lazy-detection")

    sb = session.sandbox(command="sleep", args=["infinity"])
    sb.wait()

    for step in range(3):
        result = sb.exec(["echo", f"training step {step}"]).result()
        print(f"  Step {step}: {result.stdout.strip()}")

        # Session detects wandb.run dynamically - works even though
        # Session was created before wandb.init()
        logged = session.log_metrics(step=step)
        print(f"  Metrics logged: {logged}")

    wandb.finish()


if __name__ == "__main__":
    main()
