# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Discover available infrastructure before running sandboxes.

Demonstrates:
- Listing runners and their service visibilities / capacity
- Filtering runners by minimum available CPU and memory
- Filtering by supported service visibility (public / private / custom)
- Using format_cpu() and format_bytes() for readable output
- How discovery maps to Sandbox.run placement kwargs
"""

import cwsandbox
from cwsandbox import Runner, format_bytes, format_cpu


def main() -> None:
    # --- All Runners ---
    print("--- Available Runners ---")
    runners: list[Runner] = cwsandbox.list_runners()

    if not runners:
        print("  No runners available.")
    for r in runners:
        vis = ", ".join(r.supported_service_visibilities) or "none"
        print(f"  {r.runner_id}  healthy={r.healthy}  org={r.organization_id}")
        print(f"    max: {format_cpu(r.max_cpu_millicores)}, {format_bytes(r.max_memory_bytes)}")
        print(f"    visibilities: {vis}")
        print(f"    architectures: {', '.join(r.supported_architectures) or 'none'}")

    # --- Runners That Support Public Services ---
    print("\n--- Runners Supporting Public Visibility ---")
    public_runners = cwsandbox.list_runners(service_visibility="public")

    if not public_runners:
        print("  No runners advertise public service visibility.")
    for r in public_runners:
        vis = ", ".join(r.supported_service_visibilities) or "none"
        print(f"  {r.runner_id}  visibilities: {vis}")

    # --- Runners With Capacity ---
    print("\n--- Runners With At Least 2 CPU, 4 GiB Available ---")
    capacious: list[Runner] = cwsandbox.list_runners(
        include_resources=True,
        min_available_cpu_millicores=2000,
        min_available_memory_bytes=4 * 1024**3,
    )

    if not capacious:
        print("  No runners match the capacity requirements.")
    for r in capacious:
        print(f"  {r.runner_id}  healthy={r.healthy}")
        print(f"    max: {format_cpu(r.max_cpu_millicores)}, {format_bytes(r.max_memory_bytes)}")
        if r.resources:
            avail_cpu = format_cpu(r.resources.available_cpu_millicores)
            avail_mem = format_bytes(r.resources.available_memory_bytes)
            print(f"    available: {avail_cpu}, {avail_mem}")
            print(f"    running sandboxes: {r.resources.running_sandboxes}")

    # --- Use With Sandbox.run ---
    print("\n--- Use With Sandbox.run ---")
    if runners:
        # Shared (serverless pool) runners cannot be pinned in CKS mode.
        pin_candidates = [r for r in runners if not r.is_shared]
        runner = pin_candidates[0] if pin_candidates else None
        print("  # Serverless (default): place by capabilities, no runner pin")
        print("  # from cwsandbox import PlacementMode, Service, ServiceVisibility")
        print("  # sandbox = Sandbox.run(")
        print("  #     services=[Service(port=8080, visibility=ServiceVisibility.PUBLIC)],")
        print("  # )")
        print()
        if runner is not None:
            print("  # CKS: pin to a non-shared discovered runner")
            print("  # sandbox = Sandbox.run(")
            print("  #     placement_mode=PlacementMode.CKS,")
            print(f'  #     runner_ids=["{runner.runner_id}"],')
            print("  # )")
            print()
            print("  # Optional: spill to serverless once if CKS is at capacity")
            print("  # from cwsandbox import PlacementSpillover")
            print("  # sandbox = Sandbox.run(")
            print("  #     placement_mode=PlacementMode.CKS,")
            print(f'  #     runner_ids=["{runner.runner_id}"],')
            print("  #     placement_spillover=PlacementSpillover.CKS_THEN_SERVERLESS,")
            print("  # )")
        else:
            print("  # No non-shared runners available to pin for a CKS example.")
    else:
        print("  # No runners discovered - check credentials and connectivity.")


if __name__ == "__main__":
    main()
