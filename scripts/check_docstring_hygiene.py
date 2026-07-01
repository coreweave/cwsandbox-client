# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Docstring hygiene guard for the cwsandbox public API surface.

This package's public docstrings are the source for an auto-generated API
reference. The documentation build rejects certain typographic characters
(em dashes, en dashes, smart quotes, ellipsis), which otherwise pass straight
through into the rendered output. This guard flags those characters in
docstrings under ``src/`` so they are caught before merge.

Both the leading docstring of every module/class/function and PEP 257 attribute
docstrings (a string literal following an assignment in a module or class body)
are inspected, since the docs generator extracts all of them. Comments and
ordinary string literals are left alone, and the generated ``_proto`` stubs are
skipped (mirroring ruff's extend-exclude), since their content regenerates from
upstream proto comments.

Run via ``mise run lint:docstrings`` or ``python scripts/check_docstring_hygiene.py``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED_CHARS: dict[str, str] = {
    "—": "em dash (U+2014)",
    "–": "en dash (U+2013)",
    "“": "left double quotation mark (U+201C)",
    "”": "right double quotation mark (U+201D)",
    "‘": "left single quotation mark (U+2018)",
    "’": "right single quotation mark (U+2019)",
    "…": "horizontal ellipsis (U+2026)",
}

_REMEDIATION = "use a hyphen, colon, or straight quote"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (str(REPO_ROOT / "src"),)
EXCLUDED_PARTS = ("_proto",)

_DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)

_ATTRIBUTE_OWNERS = (ast.Module, ast.ClassDef)
_ASSIGNMENTS = (ast.Assign, ast.AnnAssign)


class Violation:
    """One banned character found inside one docstring."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message

    def render(self) -> str:
        try:
            shown: Path | str = self.path.relative_to(REPO_ROOT)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}: {self.message}"


def _string_expr(node: ast.stmt) -> ast.Constant | None:
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return node.value
    return None


def _iter_docstrings(tree: ast.Module) -> list[ast.Constant]:
    """Collect every docstring node the API-reference generator extracts.

    That is the leading docstring of each module/class/function, plus attribute
    docstrings: a bare string statement immediately following an assignment in a
    module or class body.
    """
    docstrings: list[ast.Constant] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body:
            continue
        if isinstance(node, _DOCSTRING_OWNERS):
            leading = _string_expr(body[0])
            if leading is not None:
                docstrings.append(leading)
        if isinstance(node, _ATTRIBUTE_OWNERS):
            for previous, statement in zip(body, body[1:], strict=False):
                if isinstance(previous, _ASSIGNMENTS):
                    attribute_doc = _string_expr(statement)
                    if attribute_doc is not None:
                        docstrings.append(attribute_doc)
    return docstrings


def _scan_docstring(path: Path, doc: ast.Constant) -> list[Violation]:
    """Scan the decoded docstring text.

    Scanning the decoded value rather than the raw source span means a banned
    character written as an escape sequence is caught too, since the decoded
    form is what ships to the rendered API reference. The line number is the
    literal's start line plus the newlines preceding the match, which is exact
    for the triple-quoted docstrings this repo uses.
    """
    violations: list[Violation] = []
    value = doc.value
    for char, name in BANNED_CHARS.items():
        index = value.find(char)
        if index != -1:
            line_no = doc.lineno + value.count("\n", 0, index)
            violations.append(
                Violation(path, line_no, f"banned character {name} in docstring; {_REMEDIATION}")
            )
    return violations


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 0, f"failed to parse: {exc.msg}")]

    violations: list[Violation] = []
    for doc in _iter_docstrings(tree):
        violations.extend(_scan_docstring(path, doc))
    return violations


def iter_python_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = Path(root)
        candidates = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for candidate in candidates:
            if candidate.suffix != ".py":
                continue
            if any(part in EXCLUDED_PARTS for part in candidate.parts):
                continue
            files.append(candidate)
    return files


def main(argv: list[str]) -> int:
    roots = argv or list(DEFAULT_ROOTS)

    files = iter_python_files(roots)
    if not files:
        print(f"docstring hygiene: no Python files found under {roots}", file=sys.stderr)
        return 1

    all_violations: list[Violation] = []
    for path in files:
        all_violations.extend(check_file(path))

    if not all_violations:
        print("docstring hygiene: OK")
        return 0

    for violation in all_violations:
        print(violation.render())

    count = len(all_violations)
    noun = "violation" if count == 1 else "violations"
    print(f"\ndocstring hygiene: {count} {noun} found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
