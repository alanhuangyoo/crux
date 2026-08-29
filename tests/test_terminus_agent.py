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


# The Terminus template is composed from upstream's at runtime rather than
# shipped as a copy. These pin the composition so a harbor upgrade that changes
# the base prompt cannot silently leave crux on the old one.

UPSTREAM = """You are an AI assistant.

Some upstream instructions here.

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""


def test_composed_template_keeps_upstream_and_footer_last():
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM)
    assert "Some upstream instructions here." in out
    # The task and the live terminal must stay at the end, where the model reads
    # them; inserting after them would bury the actual work.
    assert out.rstrip().endswith("{terminal_state}")
    assert out.index("all or nothing") < out.index("Task Description:")


def test_submit_gate_is_separable_from_scoring():
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM, submit=False)
    assert "crux submit" not in out
    assert "all or nothing" in out


def test_disabling_both_reproduces_upstream_exactly():
    # Reproducing upstream byte-for-byte is what makes a stock comparison mean
    # anything.
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM, scoring=False, submit=False)
    assert out.strip() == UPSTREAM.strip()


def test_missing_footer_is_an_error_not_a_silent_append():
    from crux.prompts import build_terminus_template
    with pytest.raises(ValueError):
        build_terminus_template("a template harbor changed beyond recognition")


class _EffortStub(CruxTerminusAgent):
    """Reproduces __init__'s kwarg handling without a live model."""

    def __init__(self, **kwargs):
        self._llm_call_kwargs = {}
        max_tokens = kwargs.pop("max_tokens", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if max_tokens is not None:
            self._llm_call_kwargs["max_tokens"] = int(max_tokens)
        if reasoning_effort is not None:
            body = dict(self._llm_call_kwargs.get("extra_body") or {})
            tk = dict(body.get("chat_template_kwargs") or {})
            tk["reasoning_effort"] = str(reasoning_effort)
            body["chat_template_kwargs"] = tk
            self._llm_call_kwargs["extra_body"] = body


def test_reasoning_effort_reaches_the_chat_template():
    agent = _EffortStub(reasoning_effort="medium")
    body = agent._llm_call_kwargs["extra_body"]
    assert body["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_omitting_effort_leaves_the_model_default():
    # The default is xhigh, and reproducing it is what keeps every run measured
    # before this comparable.
    agent = _EffortStub()
    assert "extra_body" not in agent._llm_call_kwargs


def test_effort_and_max_tokens_coexist():
    agent = _EffortStub(reasoning_effort="medium", max_tokens=32768)
    assert agent._llm_call_kwargs["max_tokens"] == 32768
    assert agent._llm_call_kwargs["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_verification_guidance_ranks_external_sources_first():
    # Borrowed from codex's "Validating your work": verify against what the task
    # already ships before asserting a value you chose. Measured motivation: all
    # nine wrong answers in a full run passed their own checks, and 77% of those
    # checks were exact assertions of the answer the model had already decided.
    from crux.prompts import TERMINUS_SUBMIT_SECTION as T
    i_ships = T.index("Something the task already ships")
    i_second = T.index("independent second derivation")
    i_assert = T.index("an assertion of the value you believe is right")
    assert i_ships < i_second < i_assert


def test_verification_guidance_names_the_tautology():
    # The failure mode is that the weakest check reads as the most convincing,
    # so the prompt has to say so rather than just ordering the list.
    from crux.prompts import TERMINUS_SUBMIT_SECTION as T
    assert "confirms you wrote what you decided" in T
