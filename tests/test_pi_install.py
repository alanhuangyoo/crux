"""27% of pi's benchmark was decided by a download.

Upstream installs the agent from scratch in every container -- curl the nvm
installer, `nvm install 22`, `npm install -g` the agent -- under `set -euo
pipefail`. Any one of those failing raises NonZeroAgentExitCodeError before the
agent runs, and the trial scores zero.

Measured on Terminal-Bench 2.1: 24 of pi's 89 trials died there, every one in
`curl ... nvm/install.sh`, none with a single tool call recorded. pi's headline
53.9% was not a number about pi.
"""

import asyncio
from pathlib import Path

import pytest


class _Exec:
    def __init__(self, stdout="", stderr=None, return_code=0):
        self.stdout, self.stderr, self.return_code = stdout, stderr, return_code


class FakeEnv:
    def __init__(self, *, unpack_ok=True):
        self.unpack_ok = unpack_ok
        self.uploaded = []
        self.commands = []

    async def upload_file(self, source_path=None, target_path=None, **kw):
        self.uploaded.append((str(source_path), target_path))

    async def exec(self, command, *a, **k):
        self.commands.append(command)
        if self.unpack_ok:
            return _Exec("0.85.1\n")
        return _Exec("tar: unexpected end of file\n", return_code=2)


def _agent(tmp_path, bundle: Path | None, monkeypatch):
    import crux.pi_agent as pi_agent

    monkeypatch.setattr(pi_agent, "_PI_BUNDLE", str(bundle) if bundle else "/nope.tar.gz")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    calls = []

    async def fake_super_install(env):
        calls.append("network")

    monkeypatch.setattr(
        pi_agent.Pi, "install", lambda self, environment: fake_super_install(environment)
    )
    return agent, calls


def test_a_present_bundle_replaces_the_network_install(tmp_path, monkeypatch):
    bundle = tmp_path / "pi-nvm.tar.gz"
    bundle.write_bytes(b"x")
    agent, calls = _agent(tmp_path, bundle, monkeypatch)
    env = FakeEnv()
    asyncio.run(agent.install(env))

    assert calls == []                                  # no curl, no npm
    assert env.uploaded and env.uploaded[0][0] == str(bundle)
    joined = " ".join(env.commands)
    assert "tar xzf" in joined and "pi --version" in joined


def test_a_missing_bundle_still_installs(tmp_path, monkeypatch, caplog):
    """A missing file should cost a slower setup, not the run."""
    agent, calls = _agent(tmp_path, None, monkeypatch)
    env = FakeEnv()
    with caplog.at_level("WARNING"):
        asyncio.run(agent.install(env))
    assert calls == ["network"]
    assert env.uploaded == []
    assert "27%" in caplog.text


def test_a_broken_bundle_says_so_before_falling_back(tmp_path, monkeypatch, caplog):
    """A broken bundle must be distinguishable from a missing one."""
    bundle = tmp_path / "pi-nvm.tar.gz"
    bundle.write_bytes(b"not a tarball")
    agent, calls = _agent(tmp_path, bundle, monkeypatch)
    env = FakeEnv(unpack_ok=False)
    with caplog.at_level("WARNING"):
        asyncio.run(agent.install(env))
    assert calls == ["network"]
    assert "unexpected end of file" in caplog.text


def test_the_bundle_path_is_overridable_by_env():
    """So a run on another box does not need the code changed."""
    import importlib
    import os

    os.environ["CRUX_PI_BUNDLE"] = "/tmp/elsewhere.tar.gz"
    try:
        import crux.pi_agent as pi_agent

        importlib.reload(pi_agent)
        assert pi_agent._PI_BUNDLE == "/tmp/elsewhere.tar.gz"
    finally:
        del os.environ["CRUX_PI_BUNDLE"]
        importlib.reload(pi_agent)
