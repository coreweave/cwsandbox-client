# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Create sandboxes from an organization template.

Demonstrates:
- Sandbox.run_from_template() starting a sandbox from a template with no
  overrides at all
- Replace-on-presence overrides: container_image replaces the whole container
- Tags supplied via defaults are an override too, replacing the template's
  tags rather than merging with them

Requires CWSANDBOX_TEMPLATE_ID or a template id as the first argument.
Templates are created by an org admin (for example with
``cwic sandbox template create``); this example only consumes one.
"""

import os
import sys

from cwsandbox import Sandbox, SandboxDefaults


def main() -> None:
    template_id = os.environ.get("CWSANDBOX_TEMPLATE_ID", "").strip() or (
        sys.argv[1].strip() if len(sys.argv) > 1 else ""
    )
    if not template_id:
        raise SystemExit("Set CWSANDBOX_TEMPLATE_ID or pass a template id as the first argument.")

    # --- Start from the template with no overrides ---
    # Nothing but the template id: container, resources, services, network,
    # and tags all come from the template. No exec here; the template's image
    # is whatever the org admin put in it, so this block assumes nothing
    # about the binaries inside.
    print("=== Start from the template ===")
    with Sandbox.run_from_template(template_id) as sb:
        print(f"Sandbox ID: {sb.sandbox_id}")
        sb.wait()
        print(f"Status: {sb.status}")

    # --- Override the container (replace-on-presence) ---
    # Overrides replace whole fields, they are not merged: passing
    # container_image replaces the entire template container (command, args,
    # env, files, resources). Any container-field override -- command, args,
    # environment_variables, resources, and friends -- therefore requires
    # container_image; the API rejects a sparse patch. Tags are an override
    # too: this sandbox carries the tags below instead of the template's own.
    defaults = SandboxDefaults(tags=("example", "run-from-template"))
    print("=== Replace the template container ===")
    with Sandbox.run_from_template(
        template_id,
        "infinity",
        command="sleep",
        container_image="python:3.11",
        defaults=defaults,
    ) as sb:
        print(f"Sandbox ID: {sb.sandbox_id}")
        result = sb.exec(["python", "-c", "print('overridden container')"]).result()
        print(f"Output: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
