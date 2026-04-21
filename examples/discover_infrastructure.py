# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Discover available infrastructure before running sandboxes.

Demonstrates:
- Listing profiles and their networking capabilities
- Filtering profiles by egress mode
- Listing runners with live resource availability
- Filtering runners by minimum capacity
- Using format_cpu() and format_bytes() for readable output
"""

import cwsandbox
from cwsandbox import Profile, Runner, format_bytes, format_cpu


def main() -> None:
    # --- Available Profiles ---
    print("--- Available Profiles ---")
    profiles: list[Profile] = cwsandbox.list_profiles()

    if not profiles:
        print("  No profiles available.")
    for p in profiles:
        exposure = ", ".join(m.name for m in p.service_exposure_modes) or "none"
        egress = ", ".join(m.name for m in p.egress_modes) or "none"
        print(f"  {p.profile_name}  runner={p.runner_id}")
        print(f"    service exposure: {exposure}")
        print(f"    egress:  {egress}")

    # --- Profiles With Internet Egress ---
    print("\n--- Profiles With Internet Egress ---")
    internet_profiles = cwsandbox.list_profiles(egress_mode="internet")

    if not internet_profiles:
        print("  No profiles with internet egress.")
    for p in internet_profiles:
        exposure = ", ".join(m.name for m in p.service_exposure_modes) or "none"
        print(f"  {p.profile_name}  runner={p.runner_id}  service exposure: {exposure}")

    # --- Runners With Capacity ---
    print("\n--- Runners With At Least 2 CPU, 4 GiB Available ---")
    runners: list[Runner] = cwsandbox.list_runners(
        include_resources=True,
        min_available_cpu_millicores=2000,
        min_available_memory_bytes=4 * 1024**3,
    )

    if not runners:
        print("  No runners match the capacity requirements.")
    for r in runners:
        print(f"  {r.runner_id}  healthy={r.healthy}")
        print(f"    max: {format_cpu(r.max_cpu_millicores)}, {format_bytes(r.max_memory_bytes)}")
        if r.resources:
            avail_cpu = format_cpu(r.resources.available_cpu_millicores)
            avail_mem = format_bytes(r.resources.available_memory_bytes)
            print(f"    available: {avail_cpu}, {avail_mem}")
            print(f"    running sandboxes: {r.resources.running_sandboxes}")
        print(f"    profiles: {', '.join(r.profile_names)}")

    # --- Use With Sandbox.run ---
    print("\n--- Use With Sandbox.run ---")
    if profiles:
        name = profiles[0].profile_name
        print("  # Pass a discovered profile name to Sandbox.run:")
        print(f'  # sandbox = Sandbox.run(profile_names=["{name}"])')
    else:
        print("  # No profiles discovered - check credentials and connectivity.")


if __name__ == "__main__":
    main()
