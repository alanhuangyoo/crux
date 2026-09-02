"""The in-container toolkit.

These run where we cannot watch them, and a bug surfaces as the model behaving
strangely several turns later. The contracts worth pinning are the ones that
make the tools safer than the shell commands they replace: uniqueness on edit,
and bounded output on read and grep.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).parent.parent / "src" / "crux" / "resources" / "crux_tool.py"


def run(args, cwd, stdin=""):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


# ---- read ------------------------------------------------------------------

def test_read_numbers_lines(tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbeta\ngamma\n")
    r = run(["read", "a.py"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "1: alpha\n2: beta\n3: gamma\n"


def test_read_paginates_and_says_how_to_continue(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 101)))
    r = run(["read", "big.txt", "--limit", "10"], tmp_path)
    assert "1: line1" in r.stdout and "10: line10" in r.stdout
    assert "11: line11" not in r.stdout
    # The model has to know both that it is missing content and how to get it.
    assert "90 more lines" in r.stdout
    assert "--offset 11" in r.stdout


def test_read_offset(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\nd\n")
    r = run(["read", "f.txt", "--offset", "3"], tmp_path)
    assert r.stdout == "3: c\n4: d\n"


def test_read_truncates_a_pathological_line(tmp_path):
    (tmp_path / "min.js").write_text("x" * 9000 + "\n")
    r = run(["read", "min.js"], tmp_path)
    assert len(r.stdout) < 4000
    assert "more chars on this line" in r.stdout


def test_read_directory_marks_subdirectories(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "file.py").write_text("x")
    r = run(["read", "."], tmp_path)
    assert "pkg/" in r.stdout and "file.py" in r.stdout


def test_read_missing_file_fails(tmp_path):
    r = run(["read", "nope.txt"], tmp_path)
    assert r.returncode == 1 and "no such file" in r.stderr


# ---- grep ------------------------------------------------------------------

def test_grep_reports_path_line_and_text(tmp_path):
    (tmp_path / "a.py").write_text("import os\ndef handler():\n    return None\n")
    r = run(["grep", "def handler"], tmp_path)
    assert "./a.py:2: def handler():" in r.stdout


def test_grep_include_filter(tmp_path):
    (tmp_path / "a.py").write_text("target\n")
    (tmp_path / "b.txt").write_text("target\n")
    r = run(["grep", "target", ".", "--include", "*.py"], tmp_path)
    assert "a.py" in r.stdout and "b.txt" not in r.stdout


def test_grep_skips_noise_directories(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("target\n")
    (tmp_path / "src.js").write_text("target\n")
    r = run(["grep", "target"], tmp_path)
    assert "src.js" in r.stdout and "node_modules" not in r.stdout


def test_grep_no_match_says_so(tmp_path):
    (tmp_path / "a.py").write_text("nothing here\n")
    r = run(["grep", "absent"], tmp_path)
    assert r.returncode == 0 and "no matches" in r.stdout


def test_grep_bad_pattern_fails_clearly(tmp_path):
    r = run(["grep", "[unclosed"], tmp_path)
    assert r.returncode == 1 and "bad pattern" in r.stderr


# ---- edit ------------------------------------------------------------------

def test_edit_replaces_exact_text(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    r = run(
        ["edit", "app.py"],
        tmp_path,
        stdin=json.dumps({"edits": [{"old": "return 1", "new": "return 42"}]}),
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "app.py").read_text() == "def f():\n    return 42\n"


def test_edit_refuses_an_ambiguous_anchor(tmp_path):
    """The case where sed edits the wrong line and says nothing."""
    (tmp_path / "app.py").write_text("x = 1\ny = 2\nx = 1\n")
    original = (tmp_path / "app.py").read_text()
    r = run(
        ["edit", "app.py"],
        tmp_path,
        stdin=json.dumps({"edits": [{"old": "x = 1", "new": "x = 9"}]}),
    )
    assert r.returncode == 1
    assert "occurs 2 times" in r.stderr and "must be unique" in r.stderr
    assert (tmp_path / "app.py").read_text() == original


def test_edit_batch_is_all_or_nothing(tmp_path):
    """A half-applied batch leaves a file no one can reason about."""
    (tmp_path / "app.py").write_text("a = 1\nb = 2\n")
    original = (tmp_path / "app.py").read_text()
    r = run(
        ["edit", "app.py"],
        tmp_path,
        stdin=json.dumps(
            {"edits": [{"old": "a = 1", "new": "a = 9"}, {"old": "zzz", "new": "!"}]}
        ),
    )
    assert r.returncode == 1
    assert "Nothing was written" in r.stderr
    assert (tmp_path / "app.py").read_text() == original


def test_edit_applies_several_disjoint_edits(tmp_path):
    (tmp_path / "app.py").write_text("a = 1\nb = 2\nc = 3\n")
    r = run(
        ["edit", "app.py"],
        tmp_path,
        stdin=json.dumps(
            {"edits": [{"old": "a = 1", "new": "a = 9"}, {"old": "c = 3", "new": "c = 7"}]}
        ),
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "app.py").read_text() == "a = 9\nb = 2\nc = 7\n"


def test_edit_missing_anchor_shows_what_it_looked_for(tmp_path):
    (tmp_path / "app.py").write_text("a = 1\n")
    r = run(
        ["edit", "app.py"],
        tmp_path,
        stdin=json.dumps({"edits": [{"old": "nonexistent", "new": "x"}]}),
    )
    assert r.returncode == 1
    # Without the anchor echoed back the model cannot tell what went wrong.
    assert "nonexistent" in r.stderr


def test_edit_rejects_malformed_stdin(tmp_path):
    (tmp_path / "app.py").write_text("a\n")
    r = run(["edit", "app.py"], tmp_path, stdin="not json")
    assert r.returncode == 1 and "not valid JSON" in r.stderr


# ---- write / files ---------------------------------------------------------

def test_write_creates_parent_directories(tmp_path):
    r = run(["write", "a/b/c.txt"], tmp_path, stdin="hello\nworld\n")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "a/b/c.txt").read_text() == "hello\nworld\n"


def test_files_lists_matching_paths(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x")
    (tmp_path / "readme.md").write_text("x")
    r = run(["files", "*.py"], tmp_path)
    assert "mod.py" in r.stdout and "readme.md" not in r.stdout


@pytest.mark.parametrize("command", ["read", "grep", "files", "edit", "write"])
def test_every_subcommand_has_help(command):
    r = subprocess.run(
        [sys.executable, str(TOOL), command, "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0 and r.stdout


# ---- todo ------------------------------------------------------------------

def todo(args, tmp_path, stdin=""):
    """Run a todo subcommand against a list scoped to this test."""
    import os

    env = dict(os.environ, CRUX_TODO_PATH=str(tmp_path / "todo.json"))
    return subprocess.run(
        [sys.executable, str(TOOL), "todo", *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )


def test_todo_add_and_list(tmp_path):
    todo(["add", "block XSS", "preserve clean HTML"], tmp_path)
    r = todo(["list"], tmp_path)
    assert "1. [ ] block XSS" in r.stdout
    assert "2. [ ] preserve clean HTML" in r.stdout
    assert "2 of 2 still open" in r.stdout


def test_todo_list_exits_nonzero_while_work_remains(tmp_path):
    """So `crux todo list` cannot be read as confirmation of being finished."""
    todo(["add", "one thing"], tmp_path)
    assert todo(["list"], tmp_path).returncode == 2

    todo(["done", "1"], tmp_path)
    r = todo(["list"], tmp_path)
    assert r.returncode == 0
    assert "all 1 done" in r.stdout


def test_todo_done_marks_the_item(tmp_path):
    todo(["add", "a", "b"], tmp_path)
    todo(["done", "2"], tmp_path)
    out = todo(["list"], tmp_path).stdout
    assert "1. [ ] a" in out and "2. [x] b" in out


def test_todo_done_accepts_several_numbers(tmp_path):
    todo(["add", "a", "b", "c"], tmp_path)
    todo(["done", "1", "3"], tmp_path)
    out = todo(["list"], tmp_path).stdout
    assert "1. [x] a" in out and "2. [ ] b" in out and "3. [x] c" in out


def test_todo_rejects_an_out_of_range_number(tmp_path):
    todo(["add", "only one"], tmp_path)
    r = todo(["done", "5"], tmp_path)
    assert r.returncode == 1 and "no item 5" in r.stderr


def test_todo_rejects_a_non_number(tmp_path):
    todo(["add", "x"], tmp_path)
    r = todo(["done", "first"], tmp_path)
    assert r.returncode == 1 and "not an item number" in r.stderr


def test_todo_persists_across_invocations(tmp_path):
    todo(["add", "survives"], tmp_path)
    assert "survives" in todo(["list"], tmp_path).stdout


def test_todo_clear(tmp_path):
    todo(["add", "a", "b"], tmp_path)
    todo(["clear"], tmp_path)
    r = todo(["list"], tmp_path)
    assert "empty" in r.stdout and r.returncode == 0


def test_empty_todo_list_is_not_an_error(tmp_path):
    r = todo(["list"], tmp_path)
    assert r.returncode == 0 and "empty" in r.stdout


# ---- todo with bound checks -------------------------------------------------

def test_closing_an_item_reruns_its_check(tmp_path):
    """Closing is an observation, not a claim."""
    todo(["add", "file exists", "--verify", "test -f target.txt"], tmp_path)
    r = todo(["done", "1"], tmp_path)
    assert r.returncode == 1
    assert "its check still fails" in r.stderr
    assert "[ ] file exists" in todo(["list"], tmp_path).stdout

    (tmp_path / "target.txt").write_text("x")
    assert todo(["done", "1"], tmp_path).returncode == 0
    assert "[x] file exists" in todo(["list"], tmp_path).stdout


def test_a_failing_check_shows_its_output(tmp_path):
    """"It failed" is not actionable; the command's own output is."""
    todo(["add", "grep works", "--verify", "grep NOPE missing.txt"], tmp_path)
    r = todo(["done", "1"], tmp_path)
    assert "grep NOPE missing.txt" in r.stderr


def test_items_without_a_check_still_close(tmp_path):
    """Not every requirement has a one-line command behind it."""
    todo(["add", "reviewed the spec"], tmp_path)
    assert todo(["done", "1"], tmp_path).returncode == 0


def test_verify_reruns_every_check(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    todo(["add", "b exists", "--verify", "test -f b.txt"], tmp_path)
    r = todo(["verify"], tmp_path)
    assert r.returncode == 2
    assert "[pass] a exists" in r.stdout
    assert "[FAIL] b exists" in r.stdout


def test_verify_catches_a_regression_from_a_later_edit(tmp_path):
    """The case this exists for: fixing one requirement breaks another."""
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    assert todo(["done", "1"], tmp_path).returncode == 0

    (tmp_path / "a.txt").unlink()
    r = todo(["verify"], tmp_path)
    assert r.returncode == 2 and "[FAIL] a exists" in r.stdout


def test_verify_passes_when_everything_holds(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    r = todo(["verify"], tmp_path)
    assert r.returncode == 0 and "all checks pass" in r.stdout


def test_list_shows_the_pending_check(tmp_path):
    todo(["add", "tests pass", "--verify", "pytest -q"], tmp_path)
    assert "check: pytest -q" in todo(["list"], tmp_path).stdout


# ---- submit gate ------------------------------------------------------------

def crux(args, tmp_path, stdin=""):
    import os

    env = dict(os.environ, CRUX_TODO_PATH=str(tmp_path / "todo.json"))
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        input=stdin, capture_output=True, text=True, cwd=tmp_path, env=env,
    )


def test_submit_refuses_while_items_are_open(tmp_path):
    todo(["add", "still to do"], tmp_path)
    r = crux(["submit"], tmp_path)
    assert r.returncode == 1
    assert "1 item(s) still open" in r.stderr
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in r.stdout


def test_submit_rechecks_closed_items(tmp_path):
    """The regression case: an item passed earlier, a later edit broke it."""
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    todo(["done", "1"], tmp_path)
    (tmp_path / "a.txt").unlink()

    r = crux(["submit"], tmp_path)
    assert r.returncode == 1
    assert "passed earlier now fail" in r.stderr
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in r.stdout


def test_submit_emits_the_sentinel_when_everything_holds(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    todo(["done", "1"], tmp_path)
    r = crux(["submit", "--confirm"], tmp_path)
    assert r.returncode == 0
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in r.stdout


def test_submit_refuses_an_empty_checklist(tmp_path):
    """Submitting without having enumerated anything proves nothing."""
    r = crux(["submit"], tmp_path)
    assert r.returncode == 1 and "checklist is empty" in r.stderr


def test_submit_is_closed_by_a_check_not_by_reading_a_list(tmp_path, monkeypatch):
    """The gate has to cost a bound check, because that is the measured gap.

    First version listed generic categories -- empty input, exit codes,
    tolerances -- and let the agent through if none applied. The trajectory
    from that run says what it bought, in the model's own words: "The nudge to
    add more checks lists generic categories ... none of which this task
    actually specifies". It then ran --confirm with one check bound, and failed.
    The dismissal was correct on its own terms, which is why the fix is not a
    sterner list.
    """
    monkeypatch.setenv("CRUX_SUBMIT_GATE", "1")
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    todo(["done", "1"], tmp_path)

    first = crux(["submit"], tmp_path)
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in first.stdout
    assert "crux todo add" in first.stdout, "it must say how to close the gate"
    assert "end to end" in first.stdout, "and what to bind when all else is covered"

    # The flag on its own is what the first version let through.
    bypass = crux(["submit", "--confirm"], tmp_path)
    assert bypass.returncode == 1
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in bypass.stdout
    assert "does not record what you found" in bypass.stderr

    # Binding one is what closes it.
    (tmp_path / "b.txt").write_text("y")
    todo(["add", "b exists", "--verify", "test -f b.txt"], tmp_path)
    todo(["done", "2"], tmp_path)

    nudge = crux(["submit"], tmp_path)
    assert "--confirm" in nudge.stdout
    assert "1 added" in nudge.stdout

    done = crux(["submit", "--confirm"], tmp_path)
    assert done.returncode == 0
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in done.stdout


def test_confirm_is_a_second_look_not_a_bypass(tmp_path):
    # The extra phase must not become a way around a check that regressed.
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    todo(["done", "1"], tmp_path)
    (tmp_path / "a.txt").unlink()

    r = crux(["submit", "--confirm"], tmp_path)
    assert r.returncode != 0
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in r.stdout


def test_the_gate_is_off_unless_asked_for(tmp_path):
    """An unverified change must not sit in the path of the next measurement.

    The gate rests partly on a reading the finished runs disproved -- crux was
    said to stop earlier than the arm that solved the same task, and on the
    full 89 it stops later on nine of the ten it loses. Leaving it on would put
    two changes in one experiment.
    """
    (tmp_path / "a.txt").write_text("x")
    todo(["add", "a exists", "--verify", "test -f a.txt"], tmp_path)
    todo(["done", "1"], tmp_path)

    out = crux(["submit"], tmp_path)
    assert out.returncode == 0
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in out.stdout
