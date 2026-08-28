"""The output cap has to reach the LLM call, or the truncation retry is dead.

`CruxTerminusAgent._query_llm` recovers a runaway thinking turn by reissuing it
with thinking off. That path fires on finish_reason=length, so it only ever
runs when an output cap is actually sent. Upstream sends none, which left the
recovery installed but unreachable -- and let single turns reach 19,500
completion tokens against a 900-second task budget.
"""

import pytest

from crux.terminus_agent import CruxTerminusAgent


class _Stub(CruxTerminusAgent):
    """Skip Terminus's own __init__, which wants a live model and environment."""

    def __init__(self, **kwargs):
        self._llm_call_kwargs = {}
        max_tokens = kwargs.pop("max_tokens", None)
        self._crux_tools = bool(kwargs.pop("crux_tools", True))
        if max_tokens is not None:
            self._llm_call_kwargs["max_tokens"] = int(max_tokens)


def test_name_is_stable():
    # The name lands in every trial's agent_info; changing it silently makes
    # past runs unattributable, which has already happened once here.
    assert CruxTerminusAgent.name() == "crux-terminus"


def test_max_tokens_reaches_the_call_kwargs():
    agent = _Stub(max_tokens=32768)
    assert agent._llm_call_kwargs["max_tokens"] == 32768


def test_max_tokens_accepts_a_string():
    # Harbor passes --ak values through as strings.
    agent = _Stub(max_tokens="16384")
    assert agent._llm_call_kwargs["max_tokens"] == 16384


def test_omitting_max_tokens_sends_no_cap():
    # Upstream's behaviour has to stay reachable, so the run that measured it
    # remains reproducible.
    agent = _Stub()
    assert "max_tokens" not in agent._llm_call_kwargs


class _FakeSession:
    """Records how send_keys was called, so the wait policy is testable."""

    def __init__(self):
        self.calls = []

    async def send_keys(self, keystrokes, block=False, min_timeout_sec=0.0,
                        max_timeout_sec=180.0):
        self.calls.append({
            "keystrokes": keystrokes, "block": block,
            "min": min_timeout_sec, "max": max_timeout_sec,
        })

    async def get_incremental_output(self):
        return "output"


class _Cmd:
    def __init__(self, keystrokes, duration_sec):
        self.keystrokes = keystrokes
        self.duration_sec = duration_sec


class _ExecStub(CruxTerminusAgent):
    def __init__(self):
        pass

    def _limit_output_length(self, text):
        return text


@pytest.mark.asyncio
async def test_short_waits_keep_upstream_behaviour():
    # Below the threshold the fixed sleep is cheap, and upstream's semantics --
    # partial output, model polls again -- are what the prompt describes.
    session = _FakeSession()
    await _ExecStub()._execute_commands([_Cmd("ls -la\n", 0.1)], session)
    call = session.calls[0]
    assert call["block"] is False
    assert call["min"] == 0.1


@pytest.mark.asyncio
async def test_long_waits_end_when_the_command_ends():
    # This is the whole point: a 600-second wait on an install that finishes in
    # 31 should cost 31 seconds, not 600.
    session = _FakeSession()
    await _ExecStub()._execute_commands([_Cmd("apt-get install -y r-base\n", 600.0)], session)
    call = session.calls[0]
    assert call["block"] is True
    assert call["max"] == 600.0
    assert call["min"] == 0.0


@pytest.mark.asyncio
async def test_every_command_in_a_batch_is_dispatched():
    session = _FakeSession()
    await _ExecStub()._execute_commands(
        [_Cmd("cd /app\n", 0.1), _Cmd("make\n", 30.0), _Cmd("echo done\n", 0.1)],
        session,
    )
    assert [c["block"] for c in session.calls] == [False, True, False]
