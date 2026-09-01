"""The local environment is what lets the CLI use the better agent.

`crux solve` ran on mini-swe-agent while every benchmark result came from the
Terminus base, so the CLI shipped the scaffold the measurements had rejected.
Terminus needs an environment, and every harbor environment is a container.
These pin the seven members it actually touches.
"""

import asyncio
from pathlib import Path

import pytest

from crux.local_env import LocalEnvironment


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_only_the_surface_terminus_uses():
    # Subclassing BaseEnvironment would mean supplying a TrialPaths, an
    # EnvironmentConfig and a trial id that do not exist outside a benchmark.
    # Terminus reads seven members; this asserts all seven are present so a
    # harbor upgrade that reaches for an eighth fails here and not mid-task.
    e = LocalEnvironment()
    for name in ("exec", "is_dir", "upload_file", "download_file",
                 "default_user", "session_id", "trial_paths"):
        assert hasattr(e, name), name
    assert e.trial_paths.agent_dir.exists()


def test_exec_reports_exit_code_and_output(tmp_path):
    e = LocalEnvironment(cwd=tmp_path)
    r = run(e.exec("echo out; echo err >&2; exit 7"))
    assert r.return_code == 7
    assert "out" in r.stdout
    assert "err" in r.stderr


def test_exec_runs_in_the_given_directory(tmp_path):
    # The CLI's whole point is acting on the user's repo, so cwd is not
    # cosmetic.
    (tmp_path / "marker.txt").write_text("x")
    e = LocalEnvironment(cwd=tmp_path)
    r = run(e.exec("ls"))
    assert "marker.txt" in r.stdout


def test_timeout_returns_124_rather_than_raising():
    # Terminus checks return_code; an exception here would abort the turn
    # instead of letting the agent see that its command ran long.
    e = LocalEnvironment()
    r = run(e.exec("sleep 5", timeout_sec=0.5))
    assert r.return_code == 124
    assert "timed out" in r.stderr


def test_user_is_accepted_and_ignored():
    # Honouring it would need sudo. A CLI that silently escalates is worse
    # than one that cannot, so the parameter exists for interface
    # compatibility and does nothing.
    e = LocalEnvironment()
    r = run(e.exec("echo ok", user="root"))
    assert r.return_code == 0
    assert e.default_user is None


def test_upload_and_download_are_copies(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("payload")
    e = LocalEnvironment(cwd=tmp_path)
    run(e.upload_file(src, str(tmp_path / "sub" / "b.txt")))
    assert (tmp_path / "sub" / "b.txt").read_text() == "payload"
    run(e.download_file(str(tmp_path / "sub" / "b.txt"), tmp_path / "c.txt"))
    assert (tmp_path / "c.txt").read_text() == "payload"


def test_start_requires_tmux(monkeypatch):
    # Terminus drives a live tmux session; without it the failure should name
    # the cause up front rather than surfacing as a broken first turn.
    monkeypatch.setattr("crux.local_env.shutil.which", lambda _: None)
    e = LocalEnvironment()
    with pytest.raises(RuntimeError, match="tmux"):
        run(e.start())
