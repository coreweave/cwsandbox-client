<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: aviato-client
-->

# Interactive Shells and TTY

Use `Sandbox.shell()` for interactive TTY sessions. It returns a `TerminalSession` with raw byte streaming and no output buffering — purpose-built for interactive use. For batch command execution, use `exec()` instead.

## Shell session (recommended)

```python
session = sandbox.shell(
    ["/bin/bash"],
    width=80,
    height=24,
)
```

`shell()` always allocates a TTY and enables stdin. The returned `TerminalSession` streams raw bytes (no UTF-8 decode/encode round-trip), making it safe for terminal escape sequences and binary output.

## Send commands and read output

```python
session.stdin.writeline("echo hello").result()
session.stdin.writeline("ls -la /tmp").result()
session.stdin.writeline("exit").result()

# Iterate raw byte output
for chunk in session.output:
    sys.stdout.buffer.write(chunk)

# Get exit code
exit_code = session.wait(timeout=5.0)
```

`TerminalSession` does not buffer output — there is no `stdout`/`stderr` on the result. Consume `session.output` as it arrives or the data is lost.

## Terminal resize

Send resize messages when the terminal dimensions change.

```python
session.resize(120, 40)
```

`resize()` is fire-and-forget on `TerminalSession`.

## Exiting

End a shell session by closing stdin or sending an exit command:

```python
session.stdin.writeline("exit").result()  # Ask the remote shell to exit
# or
session.stdin.close().result()            # Close the stdin stream
```

The session completes when the remote process exits. Call `session.wait()` or `session.result()` to block for the exit code.

## CLI

The `aviato shell` command wraps `Sandbox.shell()` for terminal use:

```bash
aviato shell <sandbox-id>                          # Default: /bin/bash
aviato shell <sandbox-id> --cmd /bin/zsh           # Custom shell
aviato shell <sandbox-id> --cmd "python main.py"   # Run command with PTY
```

The CLI handles all terminal management — the SDK stays portable with no terminal dependencies.

The CLI runs in raw mode, so **Ctrl+C is forwarded to the remote process** instead of exiting locally. To exit, type `exit` and press Enter, or press **Ctrl+D** (sends EOF).

**Note**: `aviato shell` is Unix-only. It exits with an error on Windows.

## When to use what

| Use case | Approach |
|----------|----------|
| One-off command from terminal | `aviato exec <id> echo hello` |
| Interactive shell from terminal | `aviato shell <id>` |
| Run a script, capture output (SDK) | `exec(["python", "script.py"])` |
| Interactive shell session (SDK) | `shell(["/bin/bash"])` |
| Send input to a command (SDK) | `exec(["cat"], stdin=True)` |
| Terminal application (vim, htop) | `shell(["vim"])` + CLI raw mode |

## See also

- [Command Execution](execution.md) — running commands with `exec()`
- [Sandbox Logging](logging.md) — streaming container logs
