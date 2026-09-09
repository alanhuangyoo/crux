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
    # The unpack must run as the agent user, not root: on an image whose agent
    # is not root, `environment.exec` puts the bundle in root's home and the
    # agent never sees it.
    agent.exec_as_agent = lambda environment, command=None, **kw: environment.exec(command)

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
    # `"$b/pi" --version`, not `pi --version`: the check must not depend on
    # nvm putting pi on PATH, which is what failed the install on images where
    # the bundle had in fact unpacked cleanly.
    assert "tar xzf" in joined and '"$b/node" "$b/pi" --version' in joined


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


def test_the_bundle_is_selectable_per_arm(tmp_path, monkeypatch):
    """A control arm and a treatment arm must differ only in the bundle.

    Reading the path from the environment would make the two arms differ in how
    they were launched as well as in what they run, which is the confound this
    project spends most of its time avoiding.
    """
    import crux.pi_agent as pi_agent

    stock = tmp_path / "stock.tar.gz"
    patched = tmp_path / "patched.tar.gz"
    for f in (stock, patched):
        f.write_bytes(b"x")

    for chosen in (stock, patched):
        agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
        agent.exec_as_agent = lambda environment, command=None, **kw: environment.exec(command)
        agent._bundle_path = str(chosen)
        env = FakeEnv()
        asyncio.run(agent.install(env))
        assert env.uploaded[0][0] == str(chosen)


def test_the_unpack_runs_as_the_agent_not_as_root(tmp_path, monkeypatch):
    """`environment.exec` is root; the agent may not be.

    The same bundle installed cleanly on Terminal-Bench and failed on every
    SWE-Atlas image with `pi: command not found` -- the tar had gone into
    root's home while harbor starts the agent with `. ~/.nvm/nvm.sh` as a
    different user. Upstream's own install uses exec_as_agent; matching it is
    the fix.
    """
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)
    seen = []

    async def as_agent(environment, command=None, **kw):
        seen.append(command)
        return _Exec("0.85.1\nCRUX_PI_BIN=/home/agent/.nvm/versions/node/v22.23.2/bin\n")

    agent.exec_as_agent = as_agent

    env = FakeEnv()
    asyncio.run(agent.install(env))
    # The unpack goes through the agent user...
    assert seen and "tar xzf" in seen[0]
    # ...and root is used for one thing only: the symlink onto the default
    # PATH, which needs /usr/local/bin and nothing else.
    assert all("tar xzf" not in c for c in env.commands)
    assert any("ln -sf" in c for c in env.commands)


def test_a_successful_unpack_also_puts_pi_on_the_default_path(tmp_path):
    """nvm resolution is one assumption too many.

    harbor starts the agent with `. ~/.nvm/nvm.sh; ... pi ...`. On the
    SWE-Atlas images that line found no `pi` even after the bundle unpacked and
    reported 0.85.1 -- a different user, a different $HOME, or a shell where
    nvm's default alias does not resolve. A symlink on the default PATH is the
    one thing every one of those agrees on.
    """
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)

    async def as_agent(environment, command=None, **kw):
        return _Exec("0.85.1\nCRUX_PI_BIN=/home/agent/.nvm/versions/node/v22.23.2/bin\n")

    agent.exec_as_agent = as_agent
    env = FakeEnv()
    asyncio.run(agent.install(env))
    joined = " ".join(env.commands)
    assert "/usr/local/bin" in joined and "ln -sf" in joined
    # The absolute path the agent reported, not `$HOME` re-expanded as root --
    # which is the same mistake one layer down.
    assert "/home/agent/.nvm/versions/node/v22.23.2/bin" in joined
    assert "$HOME" not in joined


def test_a_failed_unpack_does_not_try_to_link(tmp_path, caplog):
    """Nothing to link, and the fallback needs the path left alone."""
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)

    async def as_agent(environment, command=None, **kw):
        return _Exec("tar: broken\n", return_code=2)

    agent.exec_as_agent = as_agent
    calls = []

    async def fake_install(environment):
        calls.append("network")

    agent.__class__.__mro__[1].install  # keep the base reachable
    import crux.pi_agent as m

    orig = m.Pi.install
    m.Pi.install = lambda self, environment: fake_install(environment)
    try:
        env = FakeEnv()
        with caplog.at_level("WARNING"):
            asyncio.run(agent.install(env))
    finally:
        m.Pi.install = orig
    assert "ln -sf" not in " ".join(env.commands)
    assert calls == ["network"]


def test_a_bundle_that_reports_no_bin_dir_says_so(tmp_path, caplog):
    """Silence here would leave `pi` unreachable with nothing to read."""
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)

    async def as_agent(environment, command=None, **kw):
        return _Exec("0.85.1\n")          # no CRUX_PI_BIN line

    agent.exec_as_agent = as_agent
    env = FakeEnv()
    with caplog.at_level("WARNING"):
        asyncio.run(agent.install(env))
    assert "bin dir" in caplog.text
    assert all("ln -sf" not in c for c in env.commands)


def test_the_unpack_does_not_require_tar(tmp_path):
    """Nine of forty SWE-Atlas trials failed on exit 127: no `tar`.

    The network fallback could not run either, because those images have no
    curl. Python's tarfile module needs neither a package manager nor the
    network, and an image holding a Python repository has Python.
    """
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)
    seen = []

    async def as_agent(environment, command=None, **kw):
        seen.append(command)
        return _Exec("0.85.1\nCRUX_PI_BIN=/root/.nvm/versions/node/v22/bin\n")

    agent.exec_as_agent = as_agent
    asyncio.run(agent.install(FakeEnv()))
    cmd = seen[0]
    assert "command -v tar" in cmd            # preferred when present
    assert "python3 -m tarfile -e" in cmd     # and a way through when not
    assert "exit 127" in cmd                  # says so when neither exists


def test_the_install_check_does_not_depend_on_nvms_path(tmp_path):
    """The check that proved the install had worked was what failed it.

    Sourcing nvm.sh and calling `pi` needs nvm to select a version, which it
    does not do on every image. The bundle unpacked cleanly and then died on
    `pi: command not found` under `set -eu`, so the install reported a broken
    bundle and fell back to a network path those images cannot run either.
    """
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)
    seen = []

    async def as_agent(environment, command=None, **kw):
        seen.append(command)
        return _Exec("0.85.1\nCRUX_PI_BIN=/root/.nvm/versions/node/v22/bin\n")

    agent.exec_as_agent = as_agent
    asyncio.run(agent.install(FakeEnv()))
    cmd = seen[0]
    assert 'ls -d "$HOME"/.nvm/versions/node/*/bin' in cmd   # resolved by glob
    assert '"$b/node" "$b/pi" --version' in cmd              # run by absolute path
    assert "nvm.sh" not in cmd                               # nvm never consulted


def test_an_alpine_image_gets_the_musl_node(tmp_path):
    """The bundled node is glibc-linked; three of eleven SWE-Atlas images are Alpine.

    musl reports the mismatch as `env: can't execute 'node': No such file or
    directory`, which reads like a missing binary rather than an incompatible
    one. pi's package is pure JavaScript, so one package and two node binaries
    cover both.
    """
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)
    seen = []

    async def as_agent(environment, command=None, **kw):
        seen.append(command)
        return _Exec("0.85.1\nCRUX_PI_LIBC=musl\nCRUX_PI_BIN=/root/.nvm/versions/node/v22/bin\n")

    agent.exec_as_agent = as_agent
    asyncio.run(agent.install(FakeEnv()))
    cmd = seen[0]
    assert "/lib/ld-musl-x86_64.so.1" in cmd      # detected, not guessed
    assert 'cp "$HOME/.nvm/crux-musl/node"' in cmd
    assert "CRUX_PI_LIBC" in cmd                  # and recorded either way


def test_a_glibc_image_keeps_the_bundled_node(tmp_path):
    import crux.pi_agent as pi_agent

    bundle = tmp_path / "b.tar.gz"
    bundle.write_bytes(b"x")
    agent = pi_agent.CruxPiAgent.__new__(pi_agent.CruxPiAgent)
    agent._bundle_path = str(bundle)
    seen = []

    async def as_agent(environment, command=None, **kw):
        seen.append(command)
        return _Exec("0.85.1\nCRUX_PI_LIBC=glibc\nCRUX_PI_BIN=/root/.nvm/versions/node/v22/bin\n")

    agent.exec_as_agent = as_agent
    asyncio.run(agent.install(FakeEnv()))
    # The swap is conditional in the shell, so the glibc path is the else.
    assert "CRUX_PI_LIBC=glibc" in seen[0]
