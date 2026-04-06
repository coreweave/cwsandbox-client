# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Discover available infrastructure before running sandboxes.

Demonstrates:
- Listing runways and their networking capabilities
- Filtering runways by egress mode
- Listing towers with live resource availability
- Filtering towers by minimum capacity
- Using format_cpu() and format_bytes() for readable output
"""

import cwsandbox
from cwsandbox import Runway, Tower, format_bytes, format_cpu


def main() -> None:
    # --- Available Runways ---
    print("--- Available Runways ---")
    runways: list[Runway] = cwsandbox.list_runways()

    if not runways:
        print("  No runways available.")
    for rw in runways:
        ingress = ", ".join(m.name for m in rw.ingress_modes) or "none"
        egress = ", ".join(m.name for m in rw.egress_modes) or "none"
        print(f"  {rw.runway_name}  tower={rw.tower_id}")
        print(f"    ingress: {ingress}")
        print(f"    egress:  {egress}")

    # --- Runways With Internet Egress ---
    print("\n--- Runways With Internet Egress ---")
    internet_runways = cwsandbox.list_runways(egress_mode="internet")

    if not internet_runways:
        print("  No runways with internet egress.")
    for rw in internet_runways:
        ingress = ", ".join(m.name for m in rw.ingress_modes) or "none"
        print(f"  {rw.runway_name}  tower={rw.tower_id}  ingress: {ingress}")

    # --- Towers With Capacity ---
    print("\n--- Towers With At Least 2 CPU, 4 GiB Available ---")
    towers: list[Tower] = cwsandbox.list_towers(
        include_resources=True,
        min_available_cpu_millicores=2000,
        min_available_memory_bytes=4 * 1024**3,
    )

    if not towers:
        print("  No towers match the capacity requirements.")
    for t in towers:
        print(f"  {t.tower_id}  healthy={t.healthy}")
        print(f"    max: {format_cpu(t.max_cpu_millicores)}, {format_bytes(t.max_memory_bytes)}")
        if t.resources:
            avail_cpu = format_cpu(t.resources.available_cpu_millicores)
            avail_mem = format_bytes(t.resources.available_memory_bytes)
            print(f"    available: {avail_cpu}, {avail_mem}")
            print(f"    running sandboxes: {t.resources.running_sandboxes}")
        print(f"    runways: {', '.join(t.runway_names)}")

    # --- Use With Sandbox.run ---
    print("\n--- Use With Sandbox.run ---")
    if runways:
        name = runways[0].runway_name
        print("  # Use a discovered runway with Sandbox.run:")
        print(f'  # sandbox = Sandbox.run(runway_ids=["{name}"])')
    else:
        print("  # No runways discovered - check credentials and connectivity.")


if __name__ == "__main__":
    main()
