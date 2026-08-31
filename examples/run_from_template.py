# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Create sandboxes from an organization template.

Demonstrates:
- Sandbox.run_from_template() inheriting the template spec unchanged
- Replace-on-presence overrides: container_image replaces the whole container
- Tag merging so template sandboxes stay discoverable via list()

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

    # Default container_image/command/args are ignored in template mode; only
    # tags are merged onto the created sandbox.
    defaults = SandboxDefaults(tags=("example", "run-from-template"))

    # --- Inherit the template as-is ---
    print("=== Inherit the template spec ===")
    with Sandbox.run_from_template(template_id, defaults=defaults) as sb:
        print(f"Sandbox ID: {sb.sandbox_id}")
        result = sb.exec(["sh", "-c", "echo hello from the template"]).result()
        print(f"Output: {result.stdout.strip()}")

    # --- Override the container (replace-on-presence) ---
    # Overrides replace whole fields, they are not merged: passing
    # container_image replaces the entire template container (command, args,
    # env, files, resources). Any container-field override -- command, args,
    # environment_variables, resources, and friends -- therefore requires
    # container_image; the API rejects a sparse patch.
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
