"""Output formatters for CLI commands."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aviato import Sandbox


def format_sandbox_table(sandboxes: list[Sandbox], verbose: bool = False) -> str:
    """Format sandboxes as a human-readable table.

    Args:
        sandboxes: List of Sandbox objects to format.
        verbose: If True, show additional columns.

    Returns:
        Formatted table string.
    """
    if not sandboxes:
        return "No sandboxes found."

    # Define columns
    headers = ["ID", "STATUS", "AGE"]
    if verbose:
        headers.append("IMAGE")

    # Build rows
    rows: list[list[str]] = []
    for sb in sandboxes:
        sandbox_id = sb.sandbox_id or "-"
        status = str(sb.status) if sb.status else "-"
        age = _format_age(sb.started_at) if sb.started_at else "-"
        row: list[str] = [sandbox_id, status, age]
        if verbose:
            image = getattr(sb, "container_image", None) or "-"
            row.append(image)
        rows.append(row)

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format output
    lines: list[str] = []

    # Header
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)

    # Rows
    for row in rows:
        row_line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)

    return "\n".join(lines)


def format_sandbox_json(sandboxes: list[Sandbox]) -> str:
    """Format sandboxes as JSON.

    Args:
        sandboxes: List of Sandbox objects to format.

    Returns:
        JSON string.
    """
    data = [
        {
            "id": sb.sandbox_id,
            "status": sb.status,
            "started_at": sb.started_at.isoformat() if sb.started_at else None,
        }
        for sb in sandboxes
    ]
    return json.dumps(data, indent=2)


def format_sandbox_quiet(sandboxes: list[Sandbox]) -> str:
    """Format sandboxes as a list of IDs only.

    Args:
        sandboxes: List of Sandbox objects to format.

    Returns:
        Newline-separated sandbox IDs.
    """
    return "\n".join(sb.sandbox_id for sb in sandboxes if sb.sandbox_id)


def is_tty() -> bool:
    """Check if stdout is a TTY."""
    return sys.stdout.isatty()


def _format_age(started_at: datetime) -> str:
    """Format a datetime as a human-readable age string.

    Args:
        started_at: The start time.

    Returns:
        Human-readable age (e.g., "2h", "5m", "3d").
    """
    now = datetime.now(UTC)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)

    delta = now - started_at
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        return "0s"

    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes}m"
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        return f"{hours}h"
    else:
        days = total_seconds // 86400
        return f"{days}d"
