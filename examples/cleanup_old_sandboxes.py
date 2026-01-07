#!/usr/bin/env python3
"""Example: Clean up sandboxes older than a threshold.

This pattern is useful for automated cleanup jobs that run periodically
to remove sandboxes that have been running too long.

Since Sandbox.list() returns actual Sandbox instances with started_at
populated, we can filter by age and call stop() directly.

Usage:
    # Dry run - just list what would be stopped
    uv run examples/cleanup_old_sandboxes.py --dry-run

    # Actually stop old sandboxes
    uv run examples/cleanup_old_sandboxes.py --max-age-hours 2
"""

import argparse
from datetime import UTC, datetime, timedelta

from aviato import Sandbox, SandboxError


def cleanup_old_sandboxes(
    max_age: timedelta,
    tags: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Stop sandboxes older than max_age.

    Args:
        max_age: Maximum age for sandboxes
        tags: Optional tag filter
        dry_run: If True, only report what would be stopped

    Returns:
        Tuple of (stopped_count, failed_count)
    """
    # Get all running sandboxes (optionally filtered by tags)
    # list() returns OperationRef; use .get() to block for results
    sandboxes = Sandbox.list(status="running", tags=tags).get()

    # Filter to old sandboxes (client-side)
    cutoff = datetime.now(UTC) - max_age
    old_sandboxes = [
        sb for sb in sandboxes if sb.started_at and sb.started_at.replace(tzinfo=UTC) < cutoff
    ]

    if not old_sandboxes:
        print("No sandboxes older than the threshold.")
        return 0, 0

    print(f"Found {len(old_sandboxes)} sandbox(es) older than {max_age}:\n")

    stopped = 0
    failed = 0

    for sb in old_sandboxes:
        age = datetime.now(UTC) - sb.started_at.replace(tzinfo=UTC)
        if dry_run:
            print(f"  [DRY RUN] Would stop: {sb.sandbox_id} (age: {age})")
            stopped += 1
        else:
            try:
                # sb is a Sandbox instance, so we can call stop() directly
                sb.stop().get()
                print(f"  Stopped: {sb.sandbox_id} (age: {age})")
                stopped += 1
            except SandboxError as e:
                print(f"  Failed: {sb.sandbox_id} - {e}")
                failed += 1

    return stopped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up old sandboxes")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=1.0,
        help="Maximum age in hours (default: 1)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Filter by tag (can be repeated)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be stopped",
    )
    args = parser.parse_args()

    max_age = timedelta(hours=args.max_age_hours)

    print(f"Cleaning up sandboxes older than {max_age}")
    if args.tags:
        print(f"Filtering by tags: {args.tags}")
    if args.dry_run:
        print("DRY RUN MODE - no sandboxes will be stopped\n")

    stopped, failed = cleanup_old_sandboxes(
        max_age=max_age,
        tags=args.tags,
        dry_run=args.dry_run,
    )

    print(f"\nSummary: {stopped} stopped, {failed} failed")


if __name__ == "__main__":
    main()
