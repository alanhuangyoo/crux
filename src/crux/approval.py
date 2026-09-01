"""Decide which commands may run on the user's own machine without asking.

`crux solve` runs with the invoking user's full authority -- no container, no
sandbox. That is deliberate: the point is to change real files in a real repo.
It also means a wrong command is not a failed trial, it is a lost afternoon.

Two ideas taken from the reference agents:

pi's permission gate matches a few dangerous patterns and, crucially, **blocks
by default when there is no UI to ask** -- a non-interactive run must not
silently escalate because nobody was there to say no.

codex frames it as levels rather than a boolean (`AskForApproval`: Never,
OnRequest, UnlessTrusted) and separately constrains *writes* to the project
directory, so "is this command dangerous" and "is this path mine to touch" stay
different questions. Both are worth having; a `git push` is not dangerous by
pattern and is still not something to do unasked.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Approval(str, Enum):
    """How much to ask. Mirrors codex's AskForApproval."""

    NEVER = "never"          # ask for nothing; for sandboxes and benchmarks
    DANGEROUS = "dangerous"  # ask before destructive or outward-facing commands
    ALWAYS = "always"        # ask before every command


class Verdict(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    BLOCK = "block"


@dataclass
class Decision:
    verdict: Verdict
    reason: str = ""


# Destroys data. The check is on the whole command line because these appear
# inside pipelines and `sh -c` strings where argument parsing does not reach.
_DESTRUCTIVE = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR]|--recursive)"),
     "recursive delete"),
    (re.compile(r"\b(mkfs|fdisk|parted)\b"), "disk formatting"),
    (re.compile(r"\bdd\b[^|;]*\bof=/dev/"), "raw write to a device"),
    (re.compile(r">\s*/dev/[sh]d[a-z]"), "raw write to a device"),
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)"), "discards uncommitted work"),
    (re.compile(r"\bgit\s+push\b[^|;]*(--force|-f)\b"), "force push"),
    (re.compile(r"\btruncate\b|\bshred\b"), "destroys file contents"),
]

# Reaches outside this machine, or changes it beyond the project. Not
# "dangerous" by pattern, but not something to do unasked either.
_OUTWARD = [
    (re.compile(r"\bsudo\b|\bsu\s"), "runs as another user"),
    (re.compile(r"\bgit\s+push\b"), "publishes to a remote"),
    (re.compile(r"\b(apt-get|apt|yum|dnf|brew|pacman)\s+(install|remove|purge)\b"),
     "changes system packages"),
    (re.compile(r"\bnpm\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b"),
     "publishes a package"),
    (re.compile(r"\b(chmod|chown)\b[^|;]*\b777\b"), "world-writable permissions"),
    (re.compile(r"\bcurl\b[^|;]*\|\s*(ba)?sh\b|\bwget\b[^|;]*\|\s*(ba)?sh\b"),
     "pipes a download into a shell"),
]


def _writes_outside(command: str, project_root: Path) -> str | None:
    """Whether an obvious write target escapes the project directory.

    Deliberately shallow: it reads redirect targets and the arguments of a few
    well-known writers. A shell can hide a path in a variable and this will not
    see it, which is why it is one signal among several rather than a boundary
    anything should be trusted to.
    """
    root = project_root.resolve()
    candidates: list[str] = []

    for m in re.finditer(r">>?\s*([^\s;|&]+)", command):
        candidates.append(m.group(1))

    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    writers = {"rm", "mv", "cp", "touch", "mkdir", "tee", "install", "truncate"}
    for i, p in enumerate(parts):
        if Path(p).name in writers:
            candidates += [a for a in parts[i + 1:] if not a.startswith("-")]

    for c in candidates:
        c = c.strip("\"'")
        if not c or c.startswith(("$", "(")):
            continue
        p = Path(c).expanduser()
        if not p.is_absolute():
            p = root / p
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if root not in resolved.parents and resolved != root:
            return str(resolved)
    return None


def classify(command: str, project_root: Path, level: Approval) -> Decision:
    """What to do with one command, before it runs."""
    if level is Approval.NEVER:
        return Decision(Verdict.ALLOW)
    if level is Approval.ALWAYS:
        return Decision(Verdict.ASK, "every command is confirmed at this level")

    for pattern, why in _DESTRUCTIVE:
        if pattern.search(command):
            return Decision(Verdict.ASK, why)
    for pattern, why in _OUTWARD:
        if pattern.search(command):
            return Decision(Verdict.ASK, why)

    outside = _writes_outside(command, project_root)
    if outside:
        return Decision(Verdict.ASK, f"writes outside the project: {outside}")

    return Decision(Verdict.ALLOW)


def decide(command: str, project_root: Path, level: Approval, interactive: bool) -> Decision:
    """classify(), with pi's rule that no UI means no.

    A run with nobody watching must not take a destructive action because there
    was no way to ask. Blocking is recoverable -- the user reruns with a
    different level, or approves it themselves. Proceeding is not.
    """
    d = classify(command, project_root, level)
    if d.verdict is Verdict.ASK and not interactive:
        return Decision(Verdict.BLOCK, f"{d.reason} (nothing to confirm with; use --approval never to allow)")
    return d
