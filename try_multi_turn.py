"""Try multi-turn persistent interpreter via stdin relay."""

import json

from cwsandbox import Sandbox, SandboxDefaults

RELAY_SCRIPT = r'''
import sys, json, io, traceback, ast

SENTINEL = "\x04CELL_DONE\x04"
cell_globals = {"__builtins__": __builtins__}

for raw_line in sys.stdin:
    msg = json.loads(raw_line)
    code = msg["code"]

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = out_buf = io.StringIO()
    sys.stderr = err_buf = io.StringIO()
    try:
        tree = ast.parse(code)
        # If last statement is an expression, print its value (like REPL)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Interactive(body=[tree.body.pop()])
            if tree.body:
                exec(compile(tree, "<cell>", "exec"), cell_globals)
            exec(compile(last, "<cell>", "single"), cell_globals)
        else:
            exec(compile(tree, "<cell>", "exec"), cell_globals)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    result = json.dumps({
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
    })
    print(result, flush=True)
    print(SENTINEL, flush=True)
'''

SENTINEL = "\x04CELL_DONE\x04"


def execute_cell(process, code: str) -> dict:
    """Send a cell and collect output until sentinel."""
    process.stdin.writeline(json.dumps({"code": code})).result()
    buf = ""
    for chunk in process.stdout:
        buf += chunk
        if SENTINEL in buf:
            break
    # Everything before the sentinel is the JSON result
    result_text = buf.split(SENTINEL)[0].strip()
    return json.loads(result_text)


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "multi-turn"),
        max_lifetime_seconds=120.0,
    )

    with Sandbox.run(defaults=defaults) as sb:
        print(f"Sandbox: {sb.sandbox_id}")
        proc = sb.exec(["python3", "-u", "-c", RELAY_SCRIPT], stdin=True)

        # Cell 1: define state
        print("--- Cell 1: define x ---")
        r = execute_cell(proc, "x = 42\nprint(f'x = {x}')")
        print(f"  stdout: {r['stdout']!r}")
        print(f"  stderr: {r['stderr']!r}")

        # Cell 2: state persists
        print("--- Cell 2: mutate x ---")
        r = execute_cell(proc, "x += 1\nprint(f'x = {x}')")
        print(f"  stdout: {r['stdout']!r}")

        # Cell 3: expression result (REPL-style)
        print("--- Cell 3: expression ---")
        r = execute_cell(proc, "x * 2")
        print(f"  stdout: {r['stdout']!r}")

        # Cell 4: error doesn't kill session
        print("--- Cell 4: error ---")
        r = execute_cell(proc, "1 / 0")
        print(f"  stderr: {r['stderr']!r}")

        # Cell 5: state survives error
        print("--- Cell 5: state after error ---")
        r = execute_cell(proc, "print(f'x is still {x}')")
        print(f"  stdout: {r['stdout']!r}")

        # Cell 6: imports persist
        print("--- Cell 6: import + use ---")
        r = execute_cell(proc, "import math\nprint(math.pi)")
        print(f"  stdout: {r['stdout']!r}")

        print("--- Cell 7: imported module persists ---")
        r = execute_cell(proc, "print(math.e)")
        print(f"  stdout: {r['stdout']!r}")

        proc.stdin.close().result()
        print("Done!")


if __name__ == "__main__":
    main()
