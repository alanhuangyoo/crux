#!/usr/bin/env python3
"""apply_patch — structured file edits from a stdin patch.

Implements the patch envelope Codex uses. The point of the format over a plain
unified diff is that it carries no line numbers: hunks are located by their
context, so a patch stays valid when the model's idea of the file is a few
lines out of date. That is the common failure mode when an agent edits a file
it read several commands ago.

Pure standard library, single file, so it runs in whatever the task image
happens to provide.

Usage:  apply_patch < patch.txt   |   apply_patch -   |   apply_patch 'PATCH'
"""

import os
import sys

BEGIN = "*** Begin Patch"
END = "*** End Patch"
ADD = "*** Add File: "
DELETE = "*** Delete File: "
UPDATE = "*** Update File: "
MOVE = "*** Move to: "
EOF_MARKER = "*** End of File"


class PatchError(Exception):
    pass


def _split_hunk(lines):
    """Split hunk body lines into (before, after) blocks."""
    before, after = [], []
    for line in lines:
        if not line:
            # A completely empty line means an unchanged empty line: the
            # leading marker space is routinely lost in transit, so treating
            # it as malformed would reject otherwise-good patches.
            before.append("")
            after.append("")
            continue
        marker, text = line[0], line[1:]
        if marker == " ":
            before.append(text)
            after.append(text)
        elif marker == "-":
            before.append(text)
        elif marker == "+":
            after.append(text)
        else:
            raise PatchError(f"bad hunk line: {line!r}")
    return before, after


def _find(haystack, needle, start):
    """Locate `needle` in `haystack` at or after `start`.

    Tried exact first, then with trailing whitespace ignored, then with all
    indentation stripped. Models reproduce content faithfully but reflow
    whitespace constantly; refusing those patches would fail edits that are
    unambiguously correct about which lines they mean.
    """
    if not needle:
        return start
    n = len(needle)
    for norm in (
        lambda s: s,
        lambda s: s.rstrip(),
        lambda s: s.strip(),
    ):
        target = [norm(x) for x in needle]
        for i in range(start, len(haystack) - n + 1):
            if [norm(x) for x in haystack[i : i + n]] == target:
                return i
    return -1


def _apply_hunks(original, hunks, path):
    lines = list(original)
    cursor = 0
    for header, body in hunks:
        if header:
            # An @@ header names the enclosing function or class. It is a
            # disambiguator for context that repeats in the file, so search
            # for it first and anchor the hunk after it.
            idx = _find(lines, [header], cursor)
            if idx != -1:
                cursor = idx + 1
        before, after = _split_hunk(body)
        at = _find(lines, before, cursor)
        if at == -1:
            at = _find(lines, before, 0)
        if at == -1:
            preview = "\n".join(before[:5])
            raise PatchError(
                f"{path}: could not locate context:\n{preview}"
            )
        lines[at : at + len(before)] = after
        cursor = at + len(after)
    return lines


def parse_and_apply(patch_text, root="."):
    lines = patch_text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines or not lines[0].startswith(BEGIN):
        raise PatchError(f"patch must start with {BEGIN!r}")

    i = 1
    changed = []
    while i < len(lines):
        line = lines[i]
        if line.startswith(END):
            break
        if not line.strip():
            i += 1
            continue

        if line.startswith(ADD):
            path = line[len(ADD) :].strip()
            i += 1
            body = []
            while i < len(lines) and not lines[i].startswith("*** "):
                body.append(lines[i][1:] if lines[i].startswith("+") else lines[i])
                i += 1
            full = os.path.join(root, path)
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w") as fh:
                fh.write("\n".join(body) + ("\n" if body else ""))
            changed.append(f"added {path}")

        elif line.startswith(DELETE):
            path = line[len(DELETE) :].strip()
            full = os.path.join(root, path)
            if not os.path.exists(full):
                raise PatchError(f"{path}: no such file")
            os.remove(full)
            changed.append(f"deleted {path}")
            i += 1

        elif line.startswith(UPDATE):
            path = line[len(UPDATE) :].strip()
            i += 1
            dest = path
            if i < len(lines) and lines[i].startswith(MOVE):
                dest = lines[i][len(MOVE) :].strip()
                i += 1

            full = os.path.join(root, path)
            if not os.path.exists(full):
                raise PatchError(f"{path}: no such file")
            with open(full) as fh:
                original = fh.read().split("\n")

            hunks = []
            while i < len(lines) and not lines[i].startswith("*** "):
                if lines[i].startswith("@@"):
                    hunks.append((lines[i][2:].strip(), []))
                    i += 1
                    continue
                if not hunks:
                    hunks.append(("", []))
                hunks[-1][1].append(lines[i])
                i += 1
            if i < len(lines) and lines[i].startswith(EOF_MARKER):
                i += 1

            result = _apply_hunks(original, hunks, path)
            out = os.path.join(root, dest)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as fh:
                fh.write("\n".join(result))
            if dest != path:
                os.remove(full)
                changed.append(f"updated {path} -> {dest}")
            else:
                changed.append(f"updated {path}")
        else:
            raise PatchError(f"unexpected line: {line!r}")

    if not changed:
        raise PatchError("patch contained no file operations")
    return changed


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        patch_text = sys.argv[1]
    else:
        patch_text = sys.stdin.read()
    try:
        for entry in parse_and_apply(patch_text):
            print(entry)
    except PatchError as exc:
        # Non-zero exit and a specific message: the agent's next observation is
        # this output, and "could not locate context" plus the lines it looked
        # for is actionable in a way that "patch failed" is not.
        print(f"apply_patch: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
