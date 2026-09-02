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


def test_disabling_every_section_reproduces_upstream_exactly():
    # Reproducing upstream byte-for-byte is what makes a stock comparison mean
    # anything. This has to name every section: when the harness section was
    # added with the test still listing two, it failed -- correctly.
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM, scoring=False, submit=False, harness=False)
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


def test_scoring_section_demands_more_than_one_check_per_requirement():
    # The agent bound 35 checks against 206 graded tests and reported success on
    # all of them. Enumerating requirements is not enough when each is graded
    # several times over, so the section has to name the cases that were missed.
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM).lower()
    for case in ("boundary", "tolerance", "exit code", "must not change"):
        assert case in out, case
    assert "floor, not the target" in out


def test_submit_section_asks_what_is_unverified_before_finishing():
    # "all 3 item(s) verified" was reported verbatim on tasks that then failed;
    # the gate is only useful if the agent first says what it left out.
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM)
    assert "bound no check for" in out
    assert out.index("bound no check for") < out.index("<task_complete>true</task_complete>")


def test_the_two_sections_stay_separable_after_the_rewrite():
    from crux.prompts import build_terminus_template
    scoring_only = build_terminus_template(UPSTREAM, submit=False)
    assert "bound no check for" not in scoring_only
    assert "floor, not the target" in scoring_only


def test_a_whitespace_only_tag_does_not_kill_the_trial():
    # Upstream computes a tag name as
    #   tag_content.split()[0] if " " in tag_content else tag_content
    # which raises IndexError on "  ": the membership test passes and split()
    # returns nothing. The exception escapes the agent loop and the trial scores
    # zero. One trial in 61 hit it on a 2.1 run.
    from crux.terminus_agent import _guard_whitespace_tags

    class _Parser:
        def _find_top_level_tags(self, content):
            tag = content.strip("<>")
            return [tag.split()[0] if " " in tag else tag]

    p = _Parser()
    assert p._find_top_level_tags("<response>") == ["response"]
    with pytest.raises(IndexError):
        p._find_top_level_tags("<  >")

    _guard_whitespace_tags(p)
    assert p._find_top_level_tags("<response>") == ["response"]
    assert p._find_top_level_tags("<  >") == []


def test_the_guard_leaves_other_failures_alone():
    # Only IndexError is swallowed. Anything else is a different bug and hiding
    # it would turn a crash into a silently empty parse.
    from crux.terminus_agent import _guard_whitespace_tags

    class _Parser:
        def _find_top_level_tags(self, content):
            raise ValueError("something else entirely")

    p = _guard_whitespace_tags(_Parser())
    with pytest.raises(ValueError):
        p._find_top_level_tags("<x>")


class _StuckStub(CruxTerminusAgent):
    """_stuck_notice without harbor's Terminus base in the constructor."""

    def __init__(self, threshold=120):
        self._stuck_at = threshold
        self._stuck_fired = False
        self._crux_steps = 0


def test_the_stuck_notice_fires_once_at_the_measured_threshold():
    # No task solved in a full 89-task run passed 120 steps; the longest solve
    # was 120 and the median 29, while failures ran 71 median and 323 at p90.
    # So the threshold is where a trial stops being a slow success.
    a = _StuckStub()
    for _ in range(119):
        assert a._stuck_notice() == ""
    first = a._stuck_notice()
    assert "120 steps" in first
    assert "different one" in first
    # once only: repeating it every step would train the model to ignore it
    assert a._stuck_notice() == ""


def test_the_notice_can_be_disabled():
    a = _StuckStub(threshold=0)
    for _ in range(300):
        assert a._stuck_notice() == ""


def test_the_threshold_is_configurable():
    a = _StuckStub(threshold=3)
    assert a._stuck_notice() == "" and a._stuck_notice() == ""
    assert "3 steps" in a._stuck_notice()


def test_the_harness_section_names_the_two_things_the_comparison_showed():
    # Read side by side with a run that solved mailman in 15 steps against this
    # agent's 512: it ran the /app/eval.py the task shipped, twice; we ran it
    # zero times. It wrote `echo "exit=$?"` after every state-changing command;
    # we wrote it zero times and inferred success from prose.
    from crux.prompts import build_terminus_template
    out = build_terminus_template(UPSTREAM)
    assert "eval.py" in out
    assert 'echo "exit=$?"' in out
    assert "Output text is not a result" in out


def test_the_harness_section_is_separable():
    from crux.prompts import build_terminus_template
    without = build_terminus_template(UPSTREAM, harness=False)
    assert "eval.py" not in without
    assert "all or nothing" in without



def test_pi_sections_are_ordered_not_caller_ordered():
    # Two runs asking for the same set must produce the same bytes; otherwise an
    # A/B could differ by section order with nothing recording it.
    from crux.prompts import build_sections
    a = build_sections(["harness", "scoring"])
    b = build_sections(["scoring", "harness"])
    assert a == b
    assert a.index("all or nothing") < a.index("Use what the task already gives you")


def test_an_unknown_pi_section_is_an_error_not_a_silent_drop():
    # A typo that quietly ran the control arm while claiming to run the
    # treatment is the failure this project keeps finding elsewhere.
    import pytest as _pytest

    from crux.prompts import build_sections
    with _pytest.raises(ValueError):
        build_sections(["scoring", "harnes"])


def test_the_control_arm_is_reachable():
    # Stock pi has to run through the same class, so a difference between the
    # arms is the sections rather than the plumbing.
    from crux.prompts import build_sections
    assert build_sections([]) == ""


class _CarryStub(CruxTerminusAgent):
    """The chat property without harbor's Terminus constructor."""

    def __init__(self, carry=True):
        self._carry_context = carry
        self._carried = []


class _FakeChat:
    def __init__(self, messages=None):
        self._messages = list(messages or [])

    @property
    def messages(self):
        return self._messages


def test_a_second_run_continues_the_conversation():
    # Terminus assigns a fresh Chat as the second statement of every run(), so
    # without this a second call starts with no memory of the first -- while the
    # shell, which harbor reuses across calls, remembers everything.
    a = _CarryStub()
    first = _FakeChat()
    a._chat = first
    first._messages.extend(["sys+task", "assistant did work"])
    a.remember_turn()

    second = _FakeChat()
    a._chat = second
    assert second.messages[:2] == ["sys+task", "assistant did work"]


def test_the_benchmark_path_keeps_a_fresh_chat():
    # Every number in this repo was measured one instruction per trial with no
    # carry-over, so the default must not quietly change what is being scored.
    a = _CarryStub(carry=False)
    first = _FakeChat(["one"])
    a._chat = first
    a.remember_turn()
    second = _FakeChat()
    a._chat = second
    assert second.messages == []


def test_forgetting_clears_the_conversation_only():
    a = _CarryStub()
    c = _FakeChat(["one"])
    a._chat = c
    a.remember_turn()
    a.forget_context()
    fresh = _FakeChat()
    a._chat = fresh
    assert fresh.messages == []
