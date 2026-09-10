"""tmux 3.1c has no `-e`, and one variable cost 38 trials.

`confirm_gate` sets CRUX_SUBMIT_GATE inside the task container. Upstream
delivers extra_env as `tmux new-session -e KEY=value`, which arrived in tmux
3.2; the two qemu images ship Debian 11's 3.1c. Across the corpus 434 trials
ran with confirm_gate and exactly the 38 qemu ones died in setup -- and the
error they died with, `Failed to start tmux session. Error: None`, is the same
string every other tmux failure produces, because harbor merges stderr into
stdout and then prints stderr.

So the mechanism looked ineffective on two tasks it had silently deleted.
"""

import asyncio
import shlex

import pytest

from crux.terminus_agent import CruxTerminusAgent


class _Exec:
    def __init__(self, stdout="", stderr=None, return_code=0):
        self.stdout, self.stderr, self.return_code = stdout, stderr, return_code


class FakeEnv:
    """A container whose tmux may or may not take `-e`."""

    def __init__(self, *, supports_e: bool, profile_d_works: bool = True):
        self.supports_e = supports_e
        self.profile_d_works = profile_d_works
        self.commands: list[str] = []
        self.written = ""

    async def exec(self, command, *a, **k):
        self.commands.append(command)
        if "crux_env_probe" in command:
            return _Exec("yes\n" if self.supports_e else "")
        if "crux-env.sh" in command:
            # What would land in the file, not the command that writes it.
            body = command.split("<<'CRUXENV'\n", 1)[1].split("\nCRUXENV", 1)[0]
            self.written = body
            return _Exec("")
        if command.startswith("bash -lc"):
            if not self.profile_d_works:
                return _Exec("\n")
            # A login shell sourcing the file: unquote each value the way the
            # shell would, so the check is on delivered values, not on text.
            vals = [
                " ".join(shlex.split(line.split("=", 1)[1]))
                for line in self.written.splitlines()
                if line.startswith("export ")
            ]
            return _Exec(" ".join(vals) + "\n")
        return _Exec("")


def _agent(**kwargs):
    import tempfile
    from pathlib import Path

    return CruxTerminusAgent(
        logs_dir=Path(tempfile.mkdtemp()), model_name="openai/x", **kwargs
    )


def test_confirm_gate_sets_the_variable_through_tmux():
    agent = _agent(confirm_gate=1)
    assert agent._extra_env.get("CRUX_SUBMIT_GATE") == "1"


def test_modern_tmux_keeps_the_e_flag():
    agent = _agent(confirm_gate=1)
    env = FakeEnv(supports_e=True)
    asyncio.run(agent._route_env_around_old_tmux(env))
    # Untouched: upstream's own path works, and this must not divert it.
    assert agent._extra_env == {"CRUX_SUBMIT_GATE": "1"}
    assert not env.written


def test_old_tmux_gets_the_variable_through_the_login_shell():
    agent = _agent(confirm_gate=1)
    env = FakeEnv(supports_e=False)
    asyncio.run(agent._route_env_around_old_tmux(env))
    # Cleared, so `new-session -e` is never built -- that is the fix.
    assert agent._extra_env == {}
    assert "export CRUX_SUBMIT_GATE=1" in env.written


def test_a_value_needing_quotes_survives_the_profile_file():
    agent = _agent()
    agent._extra_env = {"CRUX_X": "a b; rm -rf /"}
    env = FakeEnv(supports_e=False)
    asyncio.run(agent._route_env_around_old_tmux(env))
    assert "export CRUX_X=" + shlex.quote("a b; rm -rf /") in env.written


def test_nothing_happens_when_there_is_nothing_to_deliver():
    agent = _agent()
    env = FakeEnv(supports_e=False)
    asyncio.run(agent._route_env_around_old_tmux(env))
    assert env.commands == []


def test_a_variable_that_did_not_arrive_is_reported(caplog):
    # profile.d is not universal; an image where it is not sourced must say so
    # rather than run with the gate silently unarmed -- which is exactly the
    # failure this whole file is about.
    agent = _agent(confirm_gate=1)
    env = FakeEnv(supports_e=False, profile_d_works=False)
    with caplog.at_level("WARNING"):
        asyncio.run(agent._route_env_around_old_tmux(env))
    assert "CRUX_SUBMIT_GATE" in caplog.text


def test_the_failure_message_carries_what_tmux_printed():
    """`Error: None` is structural, not a case of tmux staying quiet."""

    class Talkative(FakeEnv):
        async def exec(self, command, *a, **k):
            # harbor merges stderr into stdout, so the reason arrives on stdout
            # and `stderr` stays None -- the shape that made every tmux failure
            # in this corpus read identically.
            return _Exec(stdout="tmux: unknown option -- e\n", stderr=None,
                         return_code=1)

    agent = _agent()
    why = asyncio.run(agent._why_tmux_failed(Talkative(supports_e=False)))
    assert "unknown option -- e" in why


def test_the_diagnosis_never_raises_over_a_dead_trial():
    class Broken(FakeEnv):
        async def exec(self, command, *a, **k):
            raise OSError("container is gone")

    agent = _agent()
    why = asyncio.run(agent._why_tmux_failed(Broken(supports_e=False)))
    assert "OSError" in why
