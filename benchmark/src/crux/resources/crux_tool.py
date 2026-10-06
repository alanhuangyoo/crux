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
import pathlib
import os
import re
import shutil
import subprocess
import sys

# A check is run when it is bound, and a bound check is meant to be quick;
# anything slower than this is a build, and is left for `todo done` to run.
BIND_PROBE_TIMEOUT_SEC = 20

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
# Modelled on the plan/todo tools in Claude Code and Codex. Scoring is per-task
# all-or-nothing, so a requirement the agent loses track of costs the whole
# task; a list it writes down and has to check off turns "I think I'm done" into
# something checkable.

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


def _report_green_at_bind(verify, number):
    """Run a check the moment it is bound, and say if it passes already.

    A check written before the work is red when it is written and green when the
    work is done, and that transition is what makes it worth anything. A check
    bound after the last edit only describes what was built, and a description
    cannot fail.

    This refuses nothing. It states the one fact the agent cannot otherwise see,
    that this check has never been observed to fail, and leaves the judgement
    with the agent.
    """
    if not verify:
        return
    try:
        result = subprocess.run(verify, shell=True, capture_output=True, text=True,
                                timeout=BIND_PROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        print(f"item {number}: its check did not finish in "
              f"{BIND_PROBE_TIMEOUT_SEC}s, so it was not run at bind time")
        return
    except OSError as exc:
        print(f"item {number}: could not run its check ({exc})")
        return
    if result.returncode != 0:
        print(f"item {number}: its check fails now (exit {result.returncode}), "
              "which is what a check bound before the work should do")
        return
    print(
        f"item {number}: its check already passes.\n"
        "  A check that has never failed has not shown it can tell right from\n"
        "  wrong -- it may be testing what you did rather than what was asked.\n"
        "  Worth one look: does it use the task's own wording and data, or\n"
        "  yours? Would it still pass against a deliberately broken version?"
    )


# Paths a check names, which are the files it might actually be testing.
_CHECK_PATHS = re.compile(r"(?<![\w-])((?:\.{0,2}/)?[\w.-]+(?:/[\w.-]+)*\.[A-Za-z0-9_]{1,8})")

# Never touch these, whatever a check names.
_NEVER_BREAK = re.compile(r"^/(etc|usr|bin|sbin|lib|proc|sys|dev|boot|var/lib)/|(^|/)\.git(/|$)")


def _breakable_targets(command, cwd):
    """Files a check names that exist, are ordinary, and are safe to disturb."""
    out = []
    for m in _CHECK_PATHS.finditer(command or ""):
        raw = m.group(1)
        path = raw if os.path.isabs(raw) else os.path.join(cwd, raw)
        path = os.path.normpath(path)
        if _NEVER_BREAK.search(path):
            continue
        try:
            if not os.path.isfile(path) or os.path.islink(path):
                continue
            if os.path.getsize(path) > 4_000_000:
                continue
        except OSError:
            continue
        out.append(path)
    return list(dict.fromkeys(out))


def _falsifies(command, cwd, target):
    """Whether the check notices `target` being wrong.

    A check that still passes against a deliberately corrupted deliverable is not
    testing the deliverable, and deciding that needs no judgement about whether
    the work is right.

    The file is restored from a byte-for-byte copy in a finally block, and the
    copy is made before anything is written. A failed restore is reported loudly,
    because a silently corrupted deliverable is far worse than an unverified
    check.
    """
    backup = target + ".crux-falsify-backup"
    try:
        shutil.copy2(target, backup)
    except OSError as exc:
        return None, f"could not back up {target} ({exc}); not touching it"
    try:
        with open(target, "wb") as fh:
            fh.write(b"")           # emptied, not deleted: a missing file and a
                                    # wrong one fail differently
        try:
            r = subprocess.run(command, shell=True, capture_output=True, text=True,
                               timeout=BIND_PROBE_TIMEOUT_SEC, cwd=cwd)
            noticed = r.returncode != 0
        except subprocess.TimeoutExpired:
            return None, f"the check did not finish in {BIND_PROBE_TIMEOUT_SEC}s"
        except OSError as exc:
            return None, f"could not run the check ({exc})"
        return noticed, ""
    finally:
        try:
            shutil.copy2(backup, target)
            os.unlink(backup)
        except OSError as exc:
            print(f"!! could not restore {target} from {backup}: {exc}", file=sys.stderr)
            print(f"!! restore it by hand before doing anything else", file=sys.stderr)


def cmd_falsify(args):
    """Check that each bound check can fail, by breaking what it tests.

    Empty the file a check names, run the check, put the file back. A check that
    still passes was not testing that file.

    Read-only in effect: every file is restored from a copy taken first, and a
    failure to restore is reported loudly rather than swallowed.
    """
    items = _load_todo()
    if not items:
        print("no checklist to falsify")
        return
    cwd = os.getcwd()
    vacuous = tested = 0
    for i, item in enumerate(items, start=1):
        check = item.get("verify")
        if not check:
            print(f"  {i}. {item['text'][:60]}\n      no bound check")
            continue
        targets = _breakable_targets(check, cwd)
        if not targets:
            print(f"  {i}. {item['text'][:60]}\n"
                  f"      names no file that exists here, so nothing to break.\n"
                  f"      That is worth knowing: {check[:70]}")
            continue
        target = targets[0]
        noticed, err = _falsifies(check, cwd, target)
        tested += 1
        rel = os.path.relpath(target, cwd)
        if err:
            print(f"  {i}. {item['text'][:60]}\n      inconclusive: {err}")
        elif noticed:
            print(f"  {i}. {item['text'][:60]}\n"
                  f"      ok -- fails when {rel} is emptied")
        else:
            vacuous += 1
            print(f"  {i}. {item['text'][:60]}\n"
                  f"      VACUOUS -- still passes with {rel} emptied.\n"
                  f"      {check[:70]}")
    print()
    if vacuous:
        print(f"{vacuous} of {tested} checks pass whatever the file says.")
        print("They are not evidence. Replace them with something that reads the")
        print("deliverable and compares it against what the task asked for.")
    elif tested:
        print(f"all {tested} checks noticed a broken deliverable")


def cmd_todo(args):
    items = _load_todo()

    if args.action == "add":
        if not args.text:
            _fail("todo add needs at least one item")
        for text in args.text:
            items.append({"text": text, "done": False, "verify": args.verify})
        _save_todo(items)
        _report_green_at_bind(args.verify, len(items))
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


SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# Where a repository states how its own tests are run. Ordered by how
# specific each source is, because the first hit wins and a CI workflow says
# more than a Makefile target called "test".
_TEST_SOURCES = (
    (".github/workflows", "GitHub Actions"),
    ("tox.ini", "tox.ini"),
    ("noxfile.py", "noxfile.py"),
    ("Makefile", "Makefile"),
    ("setup.cfg", "setup.cfg"),
    ("pyproject.toml", "pyproject.toml"),
    ("CONTRIBUTING.rst", "CONTRIBUTING"),
    ("CONTRIBUTING.md", "CONTRIBUTING"),
    ("tests/README.rst", "tests/README"),
    ("docs/internals/contributing/writing-code/unit-tests.txt", "django docs"),
)

# A test command, matched only where a command can stand: the start of a line,
# or after a shell connective. Matching it anywhere finds `pytest>=8.0` in a
# dependency list and `[tool.pytest.ini_options]` in a section header, which is
# how the first version reported crux's own pyproject.toml as three ways to run
# the tests.
_RUNNER = (
    r"(?:python\S*\s+-m\s+pytest|pytest|"
    r"\.?/?(?:tests/)?runtests\.py|python\S*\s+\S*runtests\.py|"
    r"tox|nox|python\S*\s+setup\.py\s+test|"
    r"go\s+test|cargo\s+test|npm\s+(?:run\s+)?test)"
)
# What may precede it and still leave it in command position.
# A YAML step is `- run: pytest -q`: a bullet AND a key, not one or the other.
# Allowing only one was why nothing matched a GitHub Actions workflow.
_LEAD = (r"(?:[-*$>]\s*)?(?:\d+\.\s*)?"
         r"(?:(?:-\s*)?(?:run|script|commands?|cmd|entrypoint)\s*[:=]\s*)?")
_CD = r"(?:cd\s+\S+\s*&&\s*)?"
_INVOCATION = re.compile(
    r"^[ \t]*" + _LEAD + _CD + _RUNNER + r"(?:[ \t][^\n]{0,180})?$",
    re.M,
)
# Lines that mention a runner but are declaring rather than invoking.
_NOT_A_COMMAND = re.compile(
    r"^\s*[\[#]"                      # section header or comment
    r"|[\w.-]+\s*(?:==|>=|<=|~=)"     # a pinned dependency
    r"|^\s*\w[\w.-]*\s*=\s*[\[\{\"']"  # key = [ ... ] or key = "..."
)


def _repo_root(start):
    """Walk up to the checkout root.

    Run from `/testbed/tests`, the first version looked only there, found
    nothing, and said so -- on a repository whose CI config was one directory
    up. Where the agent happens to be standing is not where the project
    describes itself.
    """
    cur = os.path.abspath(start)
    for _ in range(12):
        for marker in (".git", "tox.ini", "setup.py", "pyproject.toml", ".github"):
            if os.path.exists(os.path.join(cur, marker)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start)


# CI templating that has not been expanded. `tox ${{ matrix.toxenv }}` is not a
# command anyone can run, and printing it as one is worse than printing nothing.
_UNEXPANDED = re.compile(r"\$\{\{|\$\(\(|\{posargs\}|<[A-Z_]+>|\{\{")

# Which ecosystem a runner belongs to, so a python repo is not told `npm test`
# because its tox.ini happens to have a javascript env.
_ECOSYSTEM = (
    (re.compile(r"\b(npm|yarn|pnpm|jest|vitest)\b"), "js"),
    (re.compile(r"\b(go\s+test)\b"), "go"),
    (re.compile(r"\b(cargo\s+test)\b"), "rust"),
    (re.compile(r"(pytest|runtests\.py|\btox\b|\bnox\b|setup\.py\s+test)"), "py"),
)
_ECOSYSTEM_MARKERS = (
    ("py", ("setup.py", "pyproject.toml", "setup.cfg", "tox.ini")),
    ("js", ("package.json",)),
    ("go", ("go.mod",)),
    ("rust", ("Cargo.toml",)),
)


def _ecosystems(root):
    """Which ecosystems the repository actually is."""
    out = set()
    for name, markers in _ECOSYSTEM_MARKERS:
        if any(os.path.exists(os.path.join(root, m)) for m in markers):
            out.add(name)
    return out or {"py", "js", "go", "rust"}


def _ecosystem_of(line):
    for rx, name in _ECOSYSTEM:
        if rx.search(line):
            return name
    return None


def _scan_for_invocations(root):
    """Every test invocation the repository states about itself."""
    found = []
    for rel, label in _TEST_SOURCES:
        path = os.path.join(root, rel)
        files = []
        if os.path.isdir(path):
            for name in sorted(os.listdir(path))[:20]:
                if name.endswith((".yml", ".yaml")):
                    files.append(os.path.join(path, name))
        elif os.path.isfile(path):
            files = [path]
        for f in files:
            try:
                text = open(f, errors="replace").read(200_000)
            except OSError:
                continue
            for m in _INVOCATION.finditer(text):
                raw = m.group(0)
                if _NOT_A_COMMAND.search(raw):
                    continue
                line = raw.strip()
                line = re.sub(r"^[-*$>]\s*|^\d+\.\s*", "", line)
                line = re.sub(r"^(run|script|commands?|cmd|entrypoint)\s*[:=]\s*", "", line)
                line = line.strip("'\"` ")
                if not (4 < len(line) < 200):
                    continue
                if _UNEXPANDED.search(line):
                    continue
                found.append((line, label, os.path.relpath(f, root)))
    return found


def cmd_tests(args):
    """Print how this repository runs its own tests.

    Running the right tests with the wrong invocation is a common way to pass
    locally and fail the grader: a different settings module or parallelism
    policy makes the same tests not the same tests. Django's grader, for
    example, runs

        ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 <modules>

    Whether the work is right is a judgement; how this repository runs its tests
    is a fact written down in the repository, so it is worth a command.
    """
    root = _repo_root(args.path or ".")
    mine = _ecosystems(root)
    found = [f for f in _scan_for_invocations(root)
             if (_ecosystem_of(f[0]) or "py") in mine]
    if not found:
        print(f"no test invocation stated under {root}")
        print("Looked in: " + ", ".join(rel for rel, _ in _TEST_SOURCES))
        return
    seen = {}
    for line, label, rel in found:
        seen.setdefault(line, (label, rel))
    print(f"How {os.path.basename(root)} says it runs its tests ({root}):\n")
    for line, (label, rel) in list(seen.items())[:12]:
        print(f"  {line}")
        print(f"      {label} -- {rel}")
    print("\nRun the suite the way the repository does, not the way that is quickest.")
    print("A pass under a different settings module or a different isolation")
    print("policy is not the pass the grader will see.")
    print("Nothing here is authoritative about flags the repository leaves to you")
    print("-- a settings module, a parallelism setting. Those you still have to")
    print("choose, and choosing the default is a choice.")


def cmd_submit(args):
    """Print the submit sentinel, but only if every bound check passes.

    Finishing is the judgement this agent gets wrong most often: it submits
    work that does not pass, having done most of it and lost track of the rest.
    A checklist it ticks off itself does not help, because ticking is the same
    judgement. This makes the last step mechanical — the sentinel is emitted by
    a command that has just re-run the checks, not by the model deciding it is
    done.
    """
    items = _load_todo()
    open_items = [i for i in items if not i["done"]]
    if open_items:
        print("not submitting: %d item(s) still open" % len(open_items), file=sys.stderr)
        for i, item in enumerate(items, start=1):
            if not item["done"]:
                print(f"  {i}. {item['text']}", file=sys.stderr)
        sys.exit(1)

    failures = []
    for i, item in enumerate(items, start=1):
        check = item.get("verify")
        if not check:
            continue
        result = subprocess.run(check, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            failures.append((i, item["text"], check,
                             (result.stdout + result.stderr).strip()[:300]))

    if failures:
        print(
            "not submitting: %d check(s) that passed earlier now fail"
            % len(failures),
            file=sys.stderr,
        )
        for i, text, check, output in failures:
            print(f"  {i}. {text}\n     {check}\n     {output}", file=sys.stderr)
        sys.exit(1)

    if not items:
        print(
            "not submitting: the checklist is empty. List the task's "
            "requirements first — submitting without having enumerated them is "
            "how most of these tasks are lost.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Off by default. Most failed runs are voluntary stops with much of the budget
    # unspent, made on a green light the agent wrote itself, so the gate asks for
    # more coverage only when the run can afford it.
    if os.environ.get("CRUX_SUBMIT_GATE", "0") not in ("1", "true", "yes"):
        print(f"all {len(items)} item(s) verified")
        print(SUBMIT_SENTINEL)
        return

    # Kept beside the checklist rather than in it, so the list's shape -- which
    # the prompt, the tests and `todo list` all depend on -- does not change to
    # carry one integer.
    marker = pathlib.Path(TODO_PATH + ".gate")
    try:
        seen_at = int(marker.read_text(encoding="utf-8"))
        gate_fired = True
    except (OSError, ValueError):
        seen_at, gate_fired = len(items), False

    if args.confirm and gate_fired and len(items) <= seen_at:
        # `--confirm` alone does not close the gate: the agent can enumerate the
        # requirements correctly and still bind none of them. Closing it takes a new
        # bound check.
        print(
            f"not submitting: still {len(items)} check(s), the same as when the "
            "last submit asked for more.\n"
            "\n--confirm records that you looked again; it does not record what "
            "you found. Bind the requirement you checked:\n"
            "\n    crux todo add \"<requirement>\" --verify \"<command>\"\n"
            "\nthen `crux submit --confirm`.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.confirm:
        # Every check passing is where the agent tends to stop early. A gate that
        # listed generic categories (empty input, exit codes, tolerances) was dismissed
        # as generically as it asked, so this one asks about the task's own words and
        # is closed by binding a check rather than by reading a list.
        if not gate_fired:
            try:
                marker.write_text(str(len(items)), encoding="utf-8")
            except OSError:
                pass  # a read-only /tmp is not a reason to block submitting

        if gate_fired and len(items) > seen_at:
            print(
                f"all {len(items)} check(s) pass, {len(items) - seen_at} added "
                "since the last submit. Run `crux submit --confirm` to finish."
            )
            sys.exit(1)

        print(f"all {len(items)} check(s) pass.")
        print(
            "\nThat says the checks you bound hold. It does not say they cover "
            "what is graded: measured here, graders ran about six times as many "
            "checks as this agent bound, and every failure was inside that gap.\n"
            "\nRe-read the task statement now and list every separate thing it "
            "asks for -- each sentence that says the work must do, produce, "
            "accept, reject or preserve something is its own requirement. Then, "
            "for each one with no check above, bind it:\n"
            "\n    crux todo add \"<the requirement, in the task's words>\" "
            "--verify \"<command that exits non-zero if it does not hold>\"\n"
            "\nBind at least one. If every requirement really is covered, the "
            "check to add is the one that runs the whole deliverable end to end "
            "the way the task describes running it, from a clean state.\n"
            "\nThen `crux submit` again."
        )
        sys.exit(1)

    print(f"all {len(items)} item(s) verified")
    print(SUBMIT_SENTINEL)


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

    p = sub.add_parser(
        "submit", help="finish the task, if every check still passes"
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="finish, after the first call has asked what is not covered",
    )
    p.set_defaults(func=cmd_submit)

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

    p = sub.add_parser(
        "falsify",
        help="check that each bound check can fail, by breaking what it tests",
    )
    p.set_defaults(func=cmd_falsify)

    p = sub.add_parser("tests", help="how this repository runs its own tests")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(func=cmd_tests)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
