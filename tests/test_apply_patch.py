"""apply_patch runs inside the task container, where a bug is nearly invisible.

A failed edit surfaces to the agent as a confusing error several turns later,
and to us as a lost task with no obvious cause. So the format is pinned here.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "src" / "crux" / "resources" / "apply_patch.py"


def run(patch: str, cwd: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "-"],
        input=patch,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_add_file(tmp_path):
    r = run(
        "*** Begin Patch\n*** Add File: hello.txt\n+Hello\n+world\n*** End Patch\n",
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "hello.txt").read_text() == "Hello\nworld\n"


def test_add_file_creates_parent_dirs(tmp_path):
    r = run(
        "*** Begin Patch\n*** Add File: a/b/c.txt\n+x\n*** End Patch\n", tmp_path
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "a/b/c.txt").read_text() == "x\n"


def test_update_file(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    pass\n\ndef g():\n    pass\n")
    r = run(
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@ def f():\n"
        "-    pass\n"
        "+    return 123\n"
        "*** End Patch\n",
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "app.py").read_text()
    assert "return 123" in text
    # The @@ header must disambiguate: g's identical body stays untouched.
    assert text.count("pass") == 1


def test_delete_file(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    r = run("*** Begin Patch\n*** Delete File: old.txt\n*** End Patch\n", tmp_path)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "old.txt").exists()


def test_move_file(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    r = run(
        "*** Begin Patch\n"
        "*** Update File: a.py\n"
        "*** Move to: b.py\n"
        "@@\n"
        "-x = 1\n"
        "+x = 2\n"
        "*** End Patch\n",
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "a.py").exists()
    assert (tmp_path / "b.py").read_text() == "x = 2\n"


def test_tolerates_reflowed_indentation(tmp_path):
    """The model's copy of a line often differs from the file only in spacing."""
    (tmp_path / "f.py").write_text("class A:\n        val = 1\n")
    r = run(
        "*** Begin Patch\n"
        "*** Update File: f.py\n"
        "@@ class A:\n"
        "-    val = 1\n"
        "+    val = 2\n"
        "*** End Patch\n",
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert "val = 2" in (tmp_path / "f.py").read_text()


def test_missing_context_fails_loudly(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n")
    r = run(
        "*** Begin Patch\n"
        "*** Update File: f.py\n"
        "@@\n"
        "-nonexistent line\n"
        "+replacement\n"
        "*** End Patch\n",
        tmp_path,
    )
    assert r.returncode == 1
    # The message must name what it looked for, or the agent cannot recover.
    assert "could not locate context" in r.stderr
    assert "nonexistent line" in r.stderr
    assert (tmp_path / "f.py").read_text() == "a = 1\n"


def test_multiple_operations_in_one_patch(tmp_path):
    (tmp_path / "keep.py").write_text("v = 1\n")
    (tmp_path / "gone.txt").write_text("x")
    r = run(
        "*** Begin Patch\n"
        "*** Add File: new.txt\n"
        "+fresh\n"
        "*** Update File: keep.py\n"
        "@@\n"
        "-v = 1\n"
        "+v = 99\n"
        "*** Delete File: gone.txt\n"
        "*** End Patch\n",
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "new.txt").read_text() == "fresh\n"
    assert "v = 99" in (tmp_path / "keep.py").read_text()
    assert not (tmp_path / "gone.txt").exists()


def test_rejects_patch_without_envelope(tmp_path):
    r = run("*** Update File: x.py\n@@\n-a\n+b\n", tmp_path)
    assert r.returncode == 1
    assert "Begin Patch" in r.stderr
