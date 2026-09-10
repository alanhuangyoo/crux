"""The gate that makes the benchmarked agent safe to point at a real directory.

A benchmark trial runs in a disposable container and needs no gate at all. The
same agent pointed at the user's working copy does, and the gate has to sit
where Terminus actually runs commands -- it types keystrokes into a live tmux
session and never calls environment.exec, so gating exec would look like a
boundary and stop nothing.

These call the real `LocalCruxAgent._execute_commands`, with only the Terminus
base's version of it replaced by a recorder. An earlier version of this file
reimplemented the method and asserted against the copy, which would have kept
passing through any change to the real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from crux.approval import Approval
from crux.local_agent import LocalCruxAgent
from crux.terminus_agent import CruxTerminusAgent


@dataclass
class _Cmd:
    keystrokes: str
    duration_sec: float = 1.0


@pytest.fixture
def agent(monkeypatch, tmp_path):
    """A LocalCruxAgent whose base-class execution is recorded, not performed.

    Built with __new__ because the real __init__ reaches into harbor's Terminus
    for a model connection; the gate under test needs none of that.
    """
    executed: list[str] = []

    async def _record(self, commands, session):
        executed.extend(c.keystrokes for c in commands)
        return False, "ok"

    monkeypatch.setattr(CruxTerminusAgent, "_execute_commands", _record)

    def build(approval=Approval.DANGEROUS, interactive=True, confirm=None):
        a = LocalCruxAgent.__new__(LocalCruxAgent)
        a._approval = Approval(approval)
        a._project_root = Path(tmp_path).resolve()
        a._interactive = interactive
        a._confirm = confirm if confirm is not None else (lambda c, r: False)
        a._blocked = []
        a.executed = executed
        return a

    return build


async def test_ordinary_commands_reach_the_session(agent):
    a = agent()
    _, out = await a._execute_commands([_Cmd("ls -la\n"), _Cmd("pytest -q\n")], None)
    assert a.executed == ["ls -la\n", "pytest -q\n"]
    assert "refused" not in out


async def test_a_destructive_command_is_refused_without_a_tty(agent):
    # pi's rule: no UI means no. A run with nobody watching must not delete
    # things because there was no way to ask.
    a = agent(interactive=False)
    _, out = await a._execute_commands([_Cmd("rm -rf /\n")], None)
    assert a.executed == []
    assert "refused" in out
    assert a.blocked_commands and "rm -rf /" in a.blocked_commands[0][0]


async def test_declining_the_prompt_blocks_it(agent):
    a = agent(confirm=lambda c, r: False)
    _, out = await a._execute_commands([_Cmd("rm -rf build\n")], None)
    assert a.executed == []
    assert "declined by the user" in out


async def test_approving_the_prompt_lets_it_through(agent):
    a = agent(confirm=lambda c, r: True)
    await a._execute_commands([_Cmd("rm -rf build\n")], None)
    assert a.executed == ["rm -rf build\n"]


async def test_the_rest_of_the_turn_is_discarded_after_a_refusal(agent):
    # Whatever followed was planned assuming the refused command ran, so running
    # it anyway acts on a state that never existed.
    a = agent(interactive=False)
    await a._execute_commands([_Cmd("rm -rf /\n"), _Cmd("echo after\n")], None)
    assert a.executed == []


async def test_commands_before_a_refusal_still_run(agent):
    a = agent(interactive=False)
    await a._execute_commands([_Cmd("echo before\n"), _Cmd("rm -rf /\n")], None)
    assert a.executed == ["echo before\n"]


async def test_never_is_what_the_benchmark_uses(agent):
    a = agent(approval=Approval.NEVER, interactive=False)
    await a._execute_commands([_Cmd("rm -rf /tmp/whatever\n")], None)
    assert a.executed == ["rm -rf /tmp/whatever\n"]


async def test_the_refusal_reaches_the_model_as_output(agent):
    # Reported as terminal output rather than raised, so the agent can try
    # another approach -- the same shape as a command that fails.
    a = agent(interactive=False)
    _, out = await a._execute_commands([_Cmd("echo hi\n"), _Cmd("rm -rf /\n")], None)
    assert "ok" in out
    assert "crux refused to run" in out


@pytest.mark.asyncio
async def test_a_second_session_takes_a_free_name():
    """Two `crux repl` windows is ordinary use, not an edge case.

    Upstream names the tmux session after the agent, a constant, so the second
    one died with "duplicate session: crux-local" before its first turn.
    """
    from crux.local_agent import LocalCruxAgent

    class _Env:
        def __init__(self, listing):
            self.listing = listing

        async def exec(self, command, **kwargs):
            return SimpleNamespace(stdout=self.listing, stderr="", returncode=0)

    agent = LocalCruxAgent.__new__(LocalCruxAgent)

    # Nothing running: the name is untouched, so `tmux attach -t crux-local`
    # still finds a lone session.
    await agent._claim_session_name(_Env(""))
    assert agent.name() == "crux-local"

    # One already up: the next free name, not a random one.
    agent2 = LocalCruxAgent.__new__(LocalCruxAgent)
    await agent2._claim_session_name(_Env("crux-local\nother\n"))
    assert agent2.name() == "crux-local-2"

    # And it skips past however many are up.
    agent3 = LocalCruxAgent.__new__(LocalCruxAgent)
    await agent3._claim_session_name(_Env("crux-local\ncrux-local-2\ncrux-local-3\n"))
    assert agent3.name() == "crux-local-4"

    # The unbound call harbor makes on its handoff path is unaffected.
    assert LocalCruxAgent.name() == "crux-local"
