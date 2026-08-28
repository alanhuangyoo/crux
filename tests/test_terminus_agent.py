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


# A completion signal is appended by stripping the trailing newline and adding
# "; tmux wait -S done". These cases decide whether that is safe to do.

def test_heredoc_never_blocks():
    # `EOF` would become `EOF; tmux wait -S done` and stop terminating the
    # heredoc; the shell then sits at its continuation prompt until the outer
    # timeout. A run with this bug tracked ~15 points below the baseline.
    from crux.terminus_agent import _can_block
    assert not _can_block("python3 << 'PY'\nprint(1)\nPY\n")
    assert not _can_block("cat > /app/f.py <<'EOF'\nx = 1\nEOF\n")


def test_multiline_paste_never_blocks():
    from crux.terminus_agent import _can_block
    assert not _can_block("cd /app\nmake\n")


def test_single_line_command_blocks():
    from crux.terminus_agent import _can_block
    assert _can_block("apt-get install -y r-base\n")
    assert _can_block("make -j8")


async def test_long_heredoc_keeps_the_fixed_sleep():
    # The threshold alone is not enough: this is exactly the shape that gets a
    # long duration, and exactly the shape that breaks.
    session = _FakeSession()
    await _ExecStub()._execute_commands(
        [_Cmd("python3 << 'PY'\nimport time\ntime.sleep(1)\nPY\n", 120.0)], session
    )
    assert session.calls[0]["block"] is False
    assert session.calls[0]["min"] == 120.0
