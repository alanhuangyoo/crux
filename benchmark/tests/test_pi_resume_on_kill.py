"""A killed pi run is picked back up, not scored a zero.

pi runs inside the trial container; Terminus drives the same container from
outside. A cgroup out-of-memory kill takes every process in the group, so a
task that exhausts memory ends Terminus's *command* and ends pi's *run*.
Measured over 501 trials, seven died on exit 137 -- none of them because of
pi's own 89 MB against a 2 GiB limit.
"""

import asyncio
import shlex

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError

from crux.pi_agent import CruxPiAgent, _MAX_RESUMES, _RESUME_INSTRUCTION

REAL = (
    ". ~/.nvm/nvm.sh; PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print --mode json "
    "--session-dir /logs/agent/pi/sessions --provider harbor-endpoint --model qwen3.8-27b "
    "--append-system-prompt /tmp/crux-sections.md -- 'solve the task' "
    "2>&1 </dev/null | grep -v '\"type\":\"message_update\"' | stdbuf -oL tee /logs/agent/pi.txt"
)


def test_resume_adds_continue_once():
    out = CruxPiAgent._resume_command(REAL)
    assert out is not None
    assert out.count("--continue ") == 1
    again = CruxPiAgent._resume_command(out)
    assert again.count("--continue ") == 1


def test_resume_replaces_the_instruction_rather_than_repeating_it():
    out = CruxPiAgent._resume_command(REAL)
    assert "'solve the task'" not in out
    assert "Do not start over" in out


def test_resume_keeps_the_session_dir_so_context_is_restored():
    out = CruxPiAgent._resume_command(REAL)
    assert "--session-dir /logs/agent/pi/sessions" in out


def test_resume_keeps_the_pipeline_harbor_appended():
    out = CruxPiAgent._resume_command(REAL)
    assert out.endswith("stdbuf -oL tee /logs/agent/pi.txt")
    assert "2>&1 </dev/null |" in out


def test_the_rewritten_command_is_still_one_valid_argv():
    out = CruxPiAgent._resume_command(REAL)
    body = out.split(" 2>&1 </dev/null |")[0]
    argv = shlex.split(body)
    assert argv[-2] == "--"
    assert argv[-1] == _RESUME_INSTRUCTION


def test_an_unknown_command_shape_is_left_alone():
    # No terminator, and no redirect: not a shape this knows how to rewrite, so
    # the caller re-raises rather than running something half-rebuilt.
    assert CruxPiAgent._resume_command("pi --print --mode json 'x'") is None
    assert CruxPiAgent._resume_command("echo hi") is None


def test_a_command_that_is_not_the_main_run_is_not_rewritten():
    assert CruxPiAgent._resume_command("ls -la 2>&1 </dev/null | tee x") is None


def test_the_retry_limit_is_small_enough_to_stop_a_crash_loop():
    assert 1 <= _MAX_RESUMES <= 5


# --- the retry loop itself, driven through the real method ---


class _Fake(CruxPiAgent):
    """A CruxPiAgent whose only real part is the retry loop under test."""

    def __init__(self, failures: int):
        self._failures = failures
        self.commands: list[str] = []

    async def _exec(self, environment, command, **kwargs):
        self.commands.append(command)
        if len(self.commands) <= self._failures:
            raise NonZeroAgentExitCodeError(f"Command failed (exit 137): {command[:40]}")
        return "ok"


def test_a_run_that_survives_is_not_retried():
    agent = _Fake(0)
    assert asyncio.run(agent.exec_as_agent(None, REAL)) == "ok"
    assert len(agent.commands) == 1
    assert "--continue" not in agent.commands[0]


def test_one_kill_is_resumed_once():
    agent = _Fake(1)
    assert asyncio.run(agent.exec_as_agent(None, REAL)) == "ok"
    assert len(agent.commands) == 2
    assert "--continue" in agent.commands[1]
    assert "Do not start over" in agent.commands[1]


def test_a_process_that_keeps_dying_gives_up_and_raises():
    agent = _Fake(99)
    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(agent.exec_as_agent(None, REAL))
    assert len(agent.commands) == _MAX_RESUMES + 1


def test_a_non_pi_command_is_never_retried():
    agent = _Fake(99)
    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(agent.exec_as_agent(None, "apt-get install -y qemu"))
    assert len(agent.commands) == 1
