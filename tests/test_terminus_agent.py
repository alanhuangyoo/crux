"""The output cap has to reach the LLM call, or the truncation retry is dead.

`CruxTerminusAgent._query_llm` recovers a runaway thinking turn by reissuing it
with thinking off. That path fires on finish_reason=length, so it only ever
runs when an output cap is actually sent. Upstream sends none, which left the
recovery installed but unreachable -- and let single turns reach 19,500
completion tokens against a 900-second task budget.
"""

from pathlib import Path
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
    out = build_terminus_template(
        UPSTREAM, scoring=False, submit=False, harness=False, edit=False
    )
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
    from crux.terminus_agent import _harden_parser

    class _Parser:
        def _find_top_level_tags(self, content):
            return [
                (t.split()[0] if " " in t else t)
                for t in content.strip("<>").split("><")
            ]

        def _get_auto_fixes(self):
            return []

    p = _Parser()
    assert p._find_top_level_tags("<response>") == ["response"]
    with pytest.raises(IndexError):
        p._find_top_level_tags("<  >")

    _harden_parser(p)
    assert p._find_top_level_tags("<response>") == ["response"]
    # The nameless tag is dropped; what surrounded it is not. The first version
    # of this guard returned [] here, which lost the <commands> beside it and
    # sent the model into a retry loop that ran out the clock.
    assert p._find_top_level_tags("<analysis><  ><commands>") == [
        "analysis",
        "commands",
    ]


def test_the_guard_leaves_other_failures_alone():
    # Only IndexError is swallowed. Anything else is a different bug and hiding
    # it would turn a crash into a silently empty parse.
    from crux.terminus_agent import _harden_parser

    class _Parser:
        def _find_top_level_tags(self, content):
            raise ValueError("something else entirely")

        def _get_auto_fixes(self):
            return []

    p = _harden_parser(_Parser())
    with pytest.raises(ValueError):
        p._find_top_level_tags("<x>")


class _StuckStub(CruxTerminusAgent):
    """_stuck_notice without harbor's Terminus base in the constructor."""

    def __init__(self, threshold=120):
        self._stuck_at = threshold
        self._stuck_fired = False
        self._crux_steps = 0


def test_the_stuck_notice_is_off_by_default():
    """It was on, at 120, and 120 was measured. The fixes moved the data.

    Baseline: the longest solve took 120 steps, the median 29, and all six
    trials past 120 failed. After the parser and blocking-execution fixes,
    re-derived on 80 trials: the longest solve takes 182 steps, failures stop
    at 132 rather than 519, and six of the ten trials past 120 solve. Crossing
    it now correlates with succeeding, and no threshold in the current data
    separates the two.
    """
    from crux.terminus_agent import _STUCK_STEP_THRESHOLD

    assert _STUCK_STEP_THRESHOLD == 0
    a = _StuckStub(threshold=_STUCK_STEP_THRESHOLD)
    for _ in range(400):
        assert a._stuck_notice() == ""


def test_the_stuck_notice_still_works_when_asked_for():
    # Kept as a knob rather than deleted: the measurement is about this model
    # on this benchmark.
    a = _StuckStub(threshold=120)
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


def test_submit_gate_is_armed_inside_the_container_not_on_the_runner():
    """`crux submit` runs in the task container, so the host's env arms nothing.

    Routing it through Terminus's extra_env is the difference between a flag
    that works and one that reads as if it does.
    """


    off = CruxTerminusAgent(logs_dir=Path("/tmp"), model_name="openai/m")
    assert "CRUX_SUBMIT_GATE" not in (off._extra_env or {})

    on = CruxTerminusAgent(logs_dir=Path("/tmp"), model_name="openai/m", confirm_gate="1")
    assert (on._extra_env or {}).get("CRUX_SUBMIT_GATE") == "1"

    # And it does not discard whatever else was being passed through.
    both = CruxTerminusAgent(
        logs_dir=Path("/tmp"), model_name="openai/m",
        confirm_gate=True, extra_env={"KEEP": "yes"},
    )
    assert (both._extra_env or {}).get("KEEP") == "yes"
    assert (both._extra_env or {}).get("CRUX_SUBMIT_GATE") == "1"


@pytest.mark.asyncio
async def test_context_deadlock_is_broken_by_trimming():
    """The loop that cost two tasks their entire budget.

    Upstream's context-overflow fallback substitutes a fixed string when it
    cannot get a usable turn. That string does not parse, so the agent is asked
    again -- on the same oversized context, which overflows again. Measured at
    322 iterations on extract-moves-from-video and 182 on path-tracing-reverse,
    both scoring zero, with nothing in the trajectory that reads as an error.
    """
    from types import SimpleNamespace
    from crux.terminus_agent import _DEADLOCK_CONTENT, _guard_context_deadlock

    class _Chat:
        def __init__(self):
            self._messages = [f"m{i}" for i in range(40)]
            self.calls = 0

        async def chat(self, *a, **k):
            self.calls += 1
            # Recovers only once the history has actually been trimmed.
            if len(self._messages) > 20:
                return SimpleNamespace(content=_DEADLOCK_CONTENT)
            return SimpleNamespace(content="<response>ok</response>")

    c = _Chat()
    _guard_context_deadlock(c)

    # One occurrence is left alone: a transient failure keeps its normal retry.
    first = await c.chat("go")
    assert first.content == _DEADLOCK_CONTENT
    assert len(c._messages) == 40, "must not trim on the first occurrence"

    # The second is the loop, and trimming is what changes the input.
    second = await c.chat("go")
    assert second.content == "<response>ok</response>"
    assert len(c._messages) < 40
    assert c._messages[0] == "m0", "the opening instruction is kept"
    assert c._messages[-1] == "m39", "and the recent turns are kept"


@pytest.mark.asyncio
async def test_an_ordinary_response_never_trims():
    from types import SimpleNamespace
    from crux.terminus_agent import _guard_context_deadlock

    class _Chat:
        def __init__(self):
            self._messages = [f"m{i}" for i in range(40)]

        async def chat(self, *a, **k):
            return SimpleNamespace(content="<response>fine</response>")

    c = _Chat()
    _guard_context_deadlock(c)
    for _ in range(5):
        await c.chat("go")
    assert len(c._messages) == 40


@pytest.mark.asyncio
async def test_the_deadlock_arrives_as_an_exception_not_a_value():
    """The regression the verification run caught.

    Upstream calls chat.chat inside a try and substitutes its fixed string in
    its own except block, so the string never passes through the wrapper -- the
    exception does. A first version of this guard only inspected the returned
    response, and path-tracing-reverse deadlocked 204 times with it installed.
    """
    from types import SimpleNamespace
    from crux.terminus_agent import _guard_context_deadlock

    class _Chat:
        def __init__(self):
            self._messages = [f"m{i}" for i in range(40)]
            self.calls = 0

        async def chat(self, *a, **k):
            self.calls += 1
            if len(self._messages) > 20:
                raise RuntimeError("Model hit max_tokens limit. Response was truncated.")
            return SimpleNamespace(content="<response>ok</response>")

    c = _Chat()
    _guard_context_deadlock(c)

    # One failure still raises: a transient error keeps upstream's own retry.
    with pytest.raises(RuntimeError):
        await c.chat("go")
    assert len(c._messages) == 40

    # The second is the loop, and trimming is what changes the input.
    out = await c.chat("go")
    assert out.content == "<response>ok</response>"
    assert len(c._messages) < 40
    assert c._messages[0] == "m0" and c._messages[-1] == "m39"


@pytest.mark.asyncio
async def test_an_untrimmable_failure_is_not_swallowed():
    """With nothing left to drop, the error has to surface as an error."""
    from crux.terminus_agent import _guard_context_deadlock

    class _Chat:
        def __init__(self):
            self._messages = ["only"]

        async def chat(self, *a, **k):
            raise RuntimeError("still failing")

    c = _Chat()
    _guard_context_deadlock(c)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await c.chat("go")


def _mk_agent(limit=20):
    from crux.terminus_agent import CruxTerminusAgent
    a = CruxTerminusAgent.__new__(CruxTerminusAgent)
    a._edit_debt_limit = limit
    a._edit_debt = 0
    a._edit_debt_fired = False
    return a


def _cmds(*keys):
    from harbor.agents.terminus_2.terminus_2 import Command
    return [Command(keystrokes=k, duration_sec=0.1) for k in keys]


def test_edit_debt_fires_only_on_the_failing_shape():
    """Measured: failures edited 45 times against 3 verifications, successes 19
    against 7. The gate has to separate those, not merely count edits."""
    a = _mk_agent(limit=20)
    # The shape that solved: edit a bit, then check.
    for _ in range(3):
        assert a._account_edit_debt(_cmds("crux edit a.py", "crux edit b.py")) is None
        assert a._account_edit_debt(_cmds("pytest -q")) is None
    assert a._edit_debt == 0

    # The shape that failed: edit and edit and never check.
    note = None
    for _ in range(30):
        note = a._account_edit_debt(_cmds("crux edit x.py"))
        if note:
            break
    assert note is not None and "21 edits" in note.replace("\n", " ")


def test_edit_debt_interrupts_once_only():
    """A run told twice is being argued with; the second costs a step and adds
    nothing."""
    a = _mk_agent(limit=2)
    for _ in range(3):
        a._account_edit_debt(_cmds("sed -i s/a/b/ f.py"))
    assert a._edit_debt_fired
    for _ in range(10):
        assert a._account_edit_debt(_cmds("sed -i s/a/b/ f.py")) is None


def test_verification_clears_the_debt():
    a = _mk_agent(limit=5)
    for _ in range(4):
        a._account_edit_debt(_cmds("apply_patch <<EOF"))
    assert a._edit_debt == 4
    a._account_edit_debt(_cmds("crux todo verify"))
    assert a._edit_debt == 0


def test_reads_are_not_edits():
    """Reading is not the failing behaviour -- both groups read about equally."""
    a = _mk_agent(limit=3)
    for _ in range(20):
        assert a._account_edit_debt(_cmds("cat foo.py", "ls -la", "grep -n x y.py")) is None
    assert a._edit_debt == 0


def test_gate_can_be_disabled():
    a = _mk_agent(limit=0)
    for _ in range(50):
        assert a._account_edit_debt(_cmds("crux edit z.py")) is None


def test_pipes_and_stderr_are_not_edits():
    """The regression an audit of the corpus caught.

    The first classifier counted any command containing a redirect or pipe as
    an edit, so `crux --help | head`, `2>&1` and compile lines all inflated the
    debt. The measurements that set the threshold came from that reading, and
    were wrong by a factor of two and a half.
    """
    a = _mk_agent(limit=3)
    for _ in range(20):
        assert a._account_edit_debt(_cmds(
            "crux --help 2>&1 | head -40",
            "objdump -d /app/bin | head -20",
            "cd /app && make MARCH='-msoft-float' 2>&1 | tail -5",
        )) is None
    assert a._edit_debt == 0


def test_writing_to_a_path_is_an_edit():
    a = _mk_agent(limit=2)
    a._account_edit_debt(_cmds("cat > /app/solver.py << 'EOF'"))
    a._account_edit_debt(_cmds("echo x >> /app/notes.txt"))
    note = a._account_edit_debt(_cmds("sed -i s/a/b/ /app/main.c"))
    assert note is not None


def test_help_is_not_a_verification():
    """`crux todo --help` was clearing the debt without checking anything."""
    a = _mk_agent(limit=2)
    for _ in range(2):
        a._account_edit_debt(_cmds("cat > /app/a.py << 'EOF'"))
    a._account_edit_debt(_cmds("crux todo --help 2>&1 | head -40"))
    assert a._edit_debt == 2, "reading a manual must not reset the debt"


# --------------------------------------------------------------------------
# the budget the agent has never been able to see


def test_budget_is_discovered_from_what_is_already_on_disk(tmp_path):
    """No plumbing: the trial config has the multiplier, the task has the base.

    harbor enforces the agent timeout outside the agent and never passes the
    number in, which is why an agent has never been able to answer "how much of
    my run is left" -- the question the whole submit gate turns on.
    """
    import json

    from crux.terminus_agent import _discover_budget

    trial = tmp_path / "trials" / "sometask__hash"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({
        "agent_timeout_multiplier": 8.0,
        "task": {"name": "terminal-bench/sometask"},
    }))
    pkg = tmp_path / "home" / ".cache" / "harbor" / "tasks" / "packages" / "terminal-bench" / "sometask" / "abc"
    pkg.mkdir(parents=True)
    (pkg / "task.toml").write_text(
        '[verifier]\ntimeout_sec = 300.0\n\n[agent]\ntimeout_sec = 900.0\n'
    )

    class TP:
        agent_dir = trial / "agent"

    class Env:
        trial_paths = TP()

    import os
    old = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path / "home")
    try:
        # 900 base x 8 multiplier, and the verifier's own 300 is not it.
        assert _discover_budget(Env()) == 7200.0
    finally:
        if old is not None:
            os.environ["HOME"] = old


def test_an_undiscoverable_budget_is_zero_not_an_exception():
    from crux.terminus_agent import _discover_budget

    class Env:
        trial_paths = None

    assert _discover_budget(Env()) == 0.0


def test_the_completion_notice_says_what_has_been_spent():
    import time

    from crux.terminus_agent import CruxTerminusAgent

    a = CruxTerminusAgent.__new__(CruxTerminusAgent)
    a._parser_name = "xml"
    a._run_started = time.monotonic() - 540
    a._crux_steps = 31
    a._budget_sec = 7200
    msg = a._get_completion_confirmation_message("terminal output")
    # upstream's own question survives
    assert "Are you sure" in msg
    assert "9m00s" in msg and "31 steps" in msg and "8% of your budget" in msg


def test_the_completion_notice_degrades_to_elapsed_only():
    import time

    from crux.terminus_agent import CruxTerminusAgent

    a = CruxTerminusAgent.__new__(CruxTerminusAgent)
    a._parser_name = "xml"
    a._run_started = time.monotonic() - 95
    a._crux_steps = 4
    a._budget_sec = 0
    msg = a._get_completion_confirmation_message("out")
    assert "1m35s" in msg
    assert "% of your budget" not in msg


def test_no_clock_leaves_upstream_untouched():
    """A construction path that never ran setup must not gain a notice."""
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    from crux.terminus_agent import CruxTerminusAgent

    a = CruxTerminusAgent.__new__(CruxTerminusAgent)
    a._parser_name = "xml"
    assert a._get_completion_confirmation_message("out") == \
        Terminus2._get_completion_confirmation_message(a, "out")


def test_the_file_tools_can_be_named_in_the_prompt():
    """They could not be, and the measurement that turned them off said the
    model did not use them.

    `build_terminus_template` had no `file_tools` parameter, so the Terminus
    prompt never mentioned `crux read`, `crux grep`, `crux files`, `crux edit`
    or `crux write` -- while `_install_crux` put all five in the container. The
    recorded reason for the default, "over 206 tool calls read/grep/edit/write
    accounted for under 2%", was a statement about the prompt.

    What it costs: 70-83% of every command crux issues is a file operation and
    0-1% of them go through a structured tool. claude-code finishes the same 89
    tasks in 4,013 tool calls against 6,826.
    """
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import CruxTerminusAgent

    d = Path(tempfile.mkdtemp())
    off = CruxTerminusAgent(logs_dir=d, model_name="openai/x")
    on = CruxTerminusAgent(logs_dir=d, model_name="openai/x", file_tools=1)
    for name in ("crux read", "crux grep", "crux files"):
        assert name not in off._prompt_template
        assert name in on._prompt_template
    # and the sections that were already there survive either way
    for t in (off._prompt_template, on._prompt_template):
        assert "crux submit" in t and "crux todo" in t


def test_file_tools_is_off_unless_asked_for():
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import CruxTerminusAgent

    d = Path(tempfile.mkdtemp())
    for value in (None, 0, "0", "false", "no"):
        kw = {} if value is None else {"file_tools": value}
        a = CruxTerminusAgent(logs_dir=d, model_name="openai/x", **kw)
        assert "crux read" not in a._prompt_template


def test_a_model_call_has_a_timeout():
    """litellm's default is 6000s and harbor overrides it nowhere.

    A call that never returns holds its trial and one of the run's concurrency
    slots for a hundred minutes. Three were observed at 115, 117 and 106
    minutes, each stopped with its terminal at a prompt. Against a
    Terminal-Bench budget of 7200s that is 83% of the task.
    """
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import _LLM_CALL_TIMEOUT_SEC, CruxTerminusAgent

    d = Path(tempfile.mkdtemp())
    a = CruxTerminusAgent(logs_dir=d, model_name="openai/x")
    assert a._llm_timeout == _LLM_CALL_TIMEOUT_SEC
    assert 0 < _LLM_CALL_TIMEOUT_SEC < 6000  # anything is better than the default

    b = CruxTerminusAgent(logs_dir=d, model_name="openai/x", llm_timeout=120)
    assert b._llm_timeout == 120
    # 0 is the escape hatch back to litellm's own default
    c = CruxTerminusAgent(logs_dir=d, model_name="openai/x", llm_timeout=0)
    assert c._llm_timeout == 0


def test_the_timeout_survives_the_truncation_retry():
    """Upstream clears and restores the call kwargs around its retry.

    A timeout written once at construction would be wiped before the retry --
    the call most likely to hang, since it follows a turn that already ran long.
    """
    import inspect

    from crux.terminus_agent import CruxTerminusAgent

    src = inspect.getsource(CruxTerminusAgent._query_llm)
    body = src.split("_crux_truncation_depth = depth + 1")[0]
    # It has to be set inside _query_llm, after the kwargs are restored.
    assert '_llm_call_kwargs["timeout"]' in body
