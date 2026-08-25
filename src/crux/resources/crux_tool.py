#!/usr/bin/env python3
"""crux — file tools for an agent that only has a shell.

mini-SWE-agent talks to the environment through bash and nothing else, so the
model reads files with `sed -n`, searches with `grep`, and edits with `sed -i`.
That works until it doesn't: sed edits fail by silently matching the wrong
line, ad-hoc reads come back without line numbers to refer to, and an
unbounded `cat` of a large file evicts the context the model needed.

The mature terminal agents all solve this the same way, with a small set of
tools whose contracts are precise. This is that set, delivered as a shell
command because a shell is the only channel available. The specifics follow
the designs in opencode and pi:

* reads are line-numbered and paginated, so the model can refer to a location
  and ask for more of it;
* output is bounded by lines *and* bytes, whichever binds first, and never cut
  mid-line, with the elision reported so the model knows what it has not seen;
* edits require their anchor text to be unique in the file, which turns "I
  matched the wrong line" from a silent corruption into an error message.

Pure standard library, single file, so it runs in whatever the task image
happens to provide.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Two independent limits, whichever binds first (pi's design). Lines alone let
# one enormous line through; bytes alone truncate a file of short lines far too
# early.
MAX_LINES = 2000
MAX_BYTES = 50 * 1024
MAX_LINE_CHARS = 2000
GREP_MAX_LINE_CHARS = 500
GREP_MAX_MATCHES = 200

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", ".cache",
}


def _fail(message):
    print(f"crux: {message}", file=sys.stderr)
    sys.exit(1)


def _truncate_line(text, limit=MAX_LINE_CHARS):
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [{len(text) - limit} more chars on this line]"


def cmd_read(args):
    """Print a file with line numbers, paginated and bounded."""
    if os.path.isdir(args.path):
        entries = sorted(os.listdir(args.path))
        for name in entries[:MAX_LINES]:
            full = os.path.join(args.path, name)
            print(f"{name}/" if os.path.isdir(full) else name)
        if len(entries) > MAX_LINES:
            print(f"… [{len(entries) - MAX_LINES} more entries]")
        return
    if not os.path.exists(args.path):
        _fail(f"{args.path}: no such file")

    try:
        with open(args.path, errors="replace") as fh:
            lines = fh.read().split("\n")
    except (IsADirectoryError, PermissionError) as exc:
        _fail(f"{args.path}: {exc}")

    # Trailing newline produces a final empty element that is not a line.
    if lines and lines[-1] == "":
        lines.pop()

    total = len(lines)
    start = max(args.offset - 1, 0)
    limit = args.limit or MAX_LINES
    window = lines[start : start + limit]

    emitted = 0
    used_bytes = 0
    for i, line in enumerate(window, start=start + 1):
        rendered = f"{i}: {_truncate_line(line)}"
        size = len(rendered.encode("utf-8", "replace")) + 1
        if used_bytes + size > MAX_BYTES:
            print(
                f"… [stopped at line {i - 1}: {MAX_BYTES // 1024}KB limit reached. "
                f"Continue with --offset {i}]"
            )
            break
        print(rendered)
        used_bytes += size
        emitted += 1

    shown_to = start + emitted
    if shown_to < total:
        print(
            f"… [{total - shown_to} more lines of {total}. "
            f"Continue with --offset {shown_to + 1}]"
        )


def _walk(root, include):
    pattern = None
    if include:
        # Translate a shell glob to a regex once, rather than fnmatch per file.
        pattern = re.compile(
            "^"
            + include.replace(".", r"\.")
            .replace("**", "\0")
            .replace("*", "[^/]*")
            .replace("\0", ".*")
            .replace("?", ".")
            + "$"
        )
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if pattern and not (pattern.match(name) or pattern.match(full)):
                continue
            yield full


def cmd_grep(args):
    """Search file contents, reporting path:line: match."""
    try:
        regex = re.compile(args.pattern)
    except re.error as exc:
        _fail(f"bad pattern: {exc}")

    matches = 0
    files_with_matches = 0
    for path in _walk(args.path, args.include):
        try:
            with open(path, errors="replace") as fh:
                hit_here = False
                for lineno, line in enumerate(fh, start=1):
                    if not regex.search(line):
                        continue
                    if not hit_here:
                        files_with_matches += 1
                        hit_here = True
                    print(
                        f"{path}:{lineno}: "
                        f"{_truncate_line(line.rstrip(), GREP_MAX_LINE_CHARS)}"
                    )
                    matches += 1
                    if matches >= GREP_MAX_MATCHES:
                        print(
                            f"… [stopped at {GREP_MAX_MATCHES} matches. "
                            f"Narrow the pattern or use --include]"
                        )
                        return
        except (OSError, UnicodeDecodeError):
            continue
    if not matches:
        print(f"no matches for {args.pattern!r} under {args.path}")
    else:
        print(f"— {matches} matches in {files_with_matches} files")


def cmd_files(args):
    """List file paths matching a glob."""
    found = sorted(_walk(args.path, args.include))
    for path in found[:MAX_LINES]:
        print(path)
    if not found:
        print(f"no files matching {args.include!r} under {args.path}")
    elif len(found) > MAX_LINES:
        print(f"… [{len(found) - MAX_LINES} more]")


def cmd_edit(args):
    """Apply exact-text replacements read as JSON on stdin.

    Every anchor must occur exactly once. Uniqueness is the whole point: an
    anchor that matches twice is precisely the case where sed would edit the
    wrong one and say nothing, and nothing is written unless every edit in the
    batch resolves.
    """
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        _fail(f"stdin is not valid JSON: {exc}")

    edits = payload if isinstance(payload, list) else payload.get("edits")
    if not edits:
        _fail('expected {"edits": [{"old": ..., "new": ...}, ...]}')

    if not os.path.exists(args.path):
        _fail(f"{args.path}: no such file")
    with open(args.path, errors="replace") as fh:
        content = fh.read()

    updated = content
    for index, edit in enumerate(edits, start=1):
        old, new = edit.get("old"), edit.get("new")
        if old is None or new is None:
            _fail(f"edit {index}: needs both 'old' and 'new'")
        count = updated.count(old)
        if count == 0:
            _fail(
                f"edit {index}: anchor not found. Nothing was written.\n"
                f"--- looked for ---\n{old[:400]}"
            )
        if count > 1:
            _fail(
                f"edit {index}: anchor occurs {count} times and must be unique. "
                f"Nothing was written. Extend it with surrounding lines until "
                f"it matches once.\n--- looked for ---\n{old[:400]}"
            )
        updated = updated.replace(old, new, 1)

    with open(args.path, "w") as fh:
        fh.write(updated)
    print(f"{args.path}: applied {len(edits)} edit(s)")


def cmd_write(args):
    """Write stdin to a file, creating parent directories."""
    content = sys.stdin.read()
    parent = os.path.dirname(args.path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.path, "w") as fh:
        fh.write(content)
    print(f"{args.path}: wrote {len(content.splitlines())} lines")


# ---- todo ------------------------------------------------------------------
#
# Ported from the plan/todo tools in Claude Code and Codex, and aimed at the
# failure this project measured: across 70 tasks the agent declared completion
# 28 times and was right 3 times. It was not lying — it had done most of the
# work and lost track of the rest, and scoring is per-task all-or-nothing, so
# the requirement it forgot cost the whole task. A list it writes down and has
# to check off turns "I think I'm done" into something checkable.

TODO_PATH = os.environ.get("CRUX_TODO_PATH", "/tmp/.crux-todo.json")


def _load_todo():
    try:
        with open(TODO_PATH) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []


def _save_todo(items):
    with open(TODO_PATH, "w") as fh:
        json.dump(items, fh)


def _print_todo(items):
    if not items:
        print("todo list is empty")
        return
    for i, item in enumerate(items, start=1):
        mark = "x" if item["done"] else " "
        print(f"  {i}. [{mark}] {item['text']}")
        if item.get("verify") and not item["done"]:
            print(f"        check: {item['verify']}")
    remaining = sum(1 for i in items if not i["done"])
    if remaining:
        print(f"— {remaining} of {len(items)} still open")
    else:
        print(f"— all {len(items)} done")


def cmd_todo(args):
    items = _load_todo()

    if args.action == "add":
        if not args.text:
            _fail("todo add needs at least one item")
        for text in args.text:
            items.append({"text": text, "done": False, "verify": args.verify})
        _save_todo(items)
    elif args.action == "done":
        if not args.text:
            _fail("todo done needs an item number")
        for raw in args.text:
            try:
                index = int(raw) - 1
            except ValueError:
                _fail(f"not an item number: {raw!r}")
            if not 0 <= index < len(items):
                _fail(f"no item {raw} (list has {len(items)})")
            item = items[index]
            verify = item.get("verify")
            if verify:
                # Re-run the check now rather than trusting that it passed when
                # the item was written. Later edits break earlier requirements
                # constantly, and this is the whole point of binding a command
                # to the item: closing it is an observation, not a claim.
                result = subprocess.run(
                    verify, shell=True, capture_output=True, text=True
                )
                if result.returncode != 0:
                    output = (result.stdout + result.stderr).strip()[:600]
                    _fail(
                        f"item {raw} not closed: its check still fails "
                        f"(exit {result.returncode})\n"
                        f"--- {verify} ---\n{output}"
                    )
            item["done"] = True
        _save_todo(items)
    elif args.action == "verify":
        # Re-check everything at once. Edits made for one requirement routinely
        # break another, and the model has no other way to notice before it
        # submits.
        failed = []
        for i, item in enumerate(items, start=1):
            check = item.get("verify")
            if not check:
                continue
            result = subprocess.run(check, shell=True, capture_output=True, text=True)
            status = "pass" if result.returncode == 0 else "FAIL"
            print(f"  {i}. [{status}] {item['text']}")
            if result.returncode != 0:
                failed.append((i, check, (result.stdout + result.stderr).strip()[:300]))
        if failed:
            print(f"\n{len(failed)} check(s) failing:")
            for i, check, output in failed:
                print(f"  item {i}: {check}\n    {output}")
            sys.exit(2)
        print("all checks pass")
        return
    elif args.action == "clear":
        items = []
        _save_todo(items)
    elif args.action != "list":
        _fail(f"unknown todo action {args.action!r}")

    _print_todo(items)

    # Exit non-zero when anything is still open, so the model cannot read a
    # `todo list` as confirmation that it is finished.
    if args.action == "list" and any(not i["done"] for i in items):
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(prog="crux", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("read", help="print a file with line numbers")
    p.add_argument("path")
    p.add_argument("--offset", type=int, default=1, help="first line (1-indexed)")
    p.add_argument("--limit", type=int, default=0, help="max lines")
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("grep", help="search file contents")
    p.add_argument("pattern")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--include", help="glob filter, e.g. '*.py'")
    p.set_defaults(func=cmd_grep)

    p = sub.add_parser("files", help="list files matching a glob")
    p.add_argument("include", help="glob, e.g. '*.py'")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(func=cmd_files)

    p = sub.add_parser("edit", help="exact-text replacement (JSON on stdin)")
    p.add_argument("path")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("write", help="write stdin to a file")
    p.add_argument("path")
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("todo", help="track the task's requirements")
    p.add_argument("action", choices=["add", "done", "list", "clear", "verify"])
    p.add_argument("text", nargs="*", help="items to add, or numbers to close")
    p.add_argument(
        "--verify",
        help=(
            "shell command that proves this item holds. Re-run when the item "
            "is closed, and by `todo verify`; a non-zero exit refuses the close."
        ),
    )
    p.set_defaults(func=cmd_todo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
