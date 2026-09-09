

# --------------------------------------------------------------------------
# braces: content that looks like a placeholder


def test_every_section_survives_the_format_upstream_runs():
    """Upstream formats this template. Content braces must not look like fields.

    `crux edit` is documented with a JSON body -- `{"edits": [...]}` -- and
    inserting that section unescaped made `.format(instruction=..., ...)` raise
    KeyError('"edits"'). It took out 16 of 89 trials before anything else ran.

    Parametrised over every switch, because the next section to carry an
    example will have the same problem and nothing else would catch it.
    """
    import itertools

    from crux.prompts import _TERMINUS_FOOTER, build_terminus_template

    upstream = ("UPSTREAM {instruction}\n" + _TERMINUS_FOOTER + "\ntail {terminal_state}")
    flags = ("scoring", "submit", "harness", "edit", "file_tools")
    for combo in itertools.product([False, True], repeat=len(flags)):
        kw = dict(zip(flags, combo))
        t = build_terminus_template(upstream, **kw)
        # The two real placeholders survive; everything else is literal.
        rendered = t.format(instruction="I", terminal_state="T")
        assert "I" in rendered and "T" in rendered


def test_the_json_example_reaches_the_model_unescaped():
    # Doubling the braces must not be visible in what the model reads.
    from crux.prompts import _TERMINUS_FOOTER, build_terminus_template

    upstream = ("U {instruction}\n" + _TERMINUS_FOOTER + "\nt {terminal_state}")
    rendered = build_terminus_template(upstream, file_tools=True).format(
        instruction="I", terminal_state="T"
    )
    assert '{"edits": [' in rendered
    assert '{{' not in rendered and '}}' not in rendered


def test_the_batch_section_is_reachable_and_off_by_default():
    """Measured against claude-code on the same 89 tasks, same model.

    Segments per command, counting `&&` and `;`:

                     every command   the first command
        claude-code       3.0             3.0
        crux              2.0             1.0

    crux opens with one segment -- a bare `ls` -- where the other agent opens
    with three. That is the whole of the 1.85x step count measured between
    them: fewer things per step means more steps.
    """
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import CruxTerminusAgent

    d = Path(tempfile.mkdtemp())
    off = CruxTerminusAgent(logs_dir=d, model_name="openai/x")
    on = CruxTerminusAgent(logs_dir=d, model_name="openai/x", batch_section=1)
    assert "One command, several answers" not in off._prompt_template
    assert "One command, several answers" in on._prompt_template
    # and it survives the format upstream runs
    on._prompt_template.format(instruction="I", terminal_state="T")


def test_the_batch_section_says_when_not_to_chain():
    # A rule with no exception gets applied where it does not belong: a chain
    # is wrong when a later part depends on reading an earlier one.
    from crux.prompts import TERMINUS_BATCH_SECTION

    body = TERMINUS_BATCH_SECTION.lower()
    assert "depends on" in body
    assert "slow" in body


# --------------------------------------------------------------------------
# crux probe: does a change move the model, before it costs a run


def test_the_probe_matches_the_tag_the_model_actually_writes():
    """The model writes `<keystrokes duration="0.1">`, not `<keystrokes>`.

    The first version of this probe matched the bare tag, found nothing, and
    reported "no commands returned" -- which reads as a broken endpoint rather
    than a broken pattern.
    """
    from crux.probe import _KEYS

    reply = '<commands>\n<keystrokes duration="0.1">ls -la /app\n</keystrokes>\n</commands>'
    m = _KEYS.search(reply)
    assert m and m.group(1).strip() == "ls -la /app"
    assert _KEYS.search("<keystrokes>ls</keystrokes>")


def test_the_probe_counts_independent_probes_not_punctuation():
    from crux.probe import _SEGMENTS

    def n(c):
        return len(_SEGMENTS.split(c.strip()))

    assert n("ls -la /app") == 1
    assert n("ls -la /app && head -5 data.csv && wc -l data.csv") == 3
    assert n("cd /app; make test") == 2
    # a trailing semicolon is punctuation, not another probe
    assert n("ls -la /app;") == 1


def test_a_probe_result_reports_nothing_rather_than_zero():
    # An endpoint that returns nothing must not read as "median 0 segments",
    # which would look like a result.
    from crux.probe import ProbeResult

    empty = ProbeResult(label="x", errors=["Timeout"])
    assert empty.segments == []
    assert empty.median_segments == 0.0
    assert "no commands returned" in empty.summary()
    assert "Timeout" in empty.summary()


def test_the_probe_picks_a_signal_per_mechanism():
    """A probe that answers the same for every configuration measures nothing.

    The first version counted command segments for everything and replied
    "median unmoved at 1.0" for file tools, interleaved thinking, the harness
    section and the submit gate alike -- three of which do not touch the first
    command at all.
    """
    from crux.probe import NO_FIRST_TURN_SIGNAL, SIGNALS

    assert SIGNALS["file_tools"][0] == "crux_tools"
    assert SIGNALS["batch_section"][0] == "segments"
    # and the ones it cannot see say so rather than getting a meaningless number
    for k in ("interleaved_thinking", "submit_gate", "edit_debt_limit"):
        assert k not in SIGNALS
        assert k in NO_FIRST_TURN_SIGNAL


def test_the_probe_counts_the_tool_it_claims_to():
    from crux.probe import _count

    assert _count("crux_tools", "ls -la /app", "") == 0
    assert _count("crux_tools", "crux read /app/solver.py", "") == 1
    assert _count("crux_tools", "crux grep x . && crux files '*.py'", "") == 2


# --------------------------------------------------------------------------
# the section ported from Claude Code's own prompt


def test_the_finish_section_names_both_measured_failures():
    """Two lines from Claude Code's `# Doing tasks`, each with a number behind it.

    Verify-before-done: 18 of pi's 24 failed Terminal-Bench trials ended with
    `agent_settled`, and 69 of 73 never ran anything that looks like a check.
    Diagnose-before-abandoning: on tasks it failed, stock pi issued a median of
    7 tool calls across 5 turns; Claude Code issued 40.
    """
    from crux.prompts import build_sections

    t = build_sections(("finish",)).lower()
    assert "verify it actually works" in t
    assert "say so explicitly rather than claiming success" in t
    assert "diagnose why before switching tactics" in t
    assert "do not abandon a viable approach after a single failure" in t


def test_the_finish_section_leaves_out_what_is_about_claude_code():
    """Its section runs to eighty lines; most of it is being Claude Code.

    Porting the whole thing would carry slash commands, feedback channels and
    its own tool names into a benchmark where none of them exist -- and would
    make an A/B unattributable across a dozen unrelated instructions.
    """
    from crux.prompts import build_sections

    t = build_sections(("finish",))
    for foreign in ("/help", "/issue", "/share", "Claude Code", "TodoWrite",
                    "AskUserQuestion", "Slack", "CLAUDE.md"):
        assert foreign not in t


def test_finish_composes_with_the_other_sections():
    from crux.prompts import build_sections

    both = build_sections(("scoring", "finish"))
    assert "all or nothing" in both
    assert "Finishing" in both
    # Fixed order, so two runs asking for the same set produce the same bytes.
    assert both.index("all or nothing") < both.index("Finishing")


def test_an_unknown_section_still_raises():
    import pytest

    from crux.prompts import build_sections

    with pytest.raises(ValueError):
        build_sections(("finish", "finnish"))


def test_the_doing_section_names_both_measured_failures():
    """Carried from Claude Code's own `# Doing tasks` section.

    Eighteen of pi's twenty-four failed Terminal-Bench 2.1 trials ended with
    the agent declaring itself done and the verifier disagreeing; four more
    stopped after two to four actions. Nothing in pi's own prompt addresses
    either, and this is the text a scaffold that leads by 17 points on the same
    model actually ships.
    """
    from crux.prompts import PROMPT_SECTIONS, build_sections

    body = PROMPT_SECTIONS["doing"]
    assert "verify it actually works" in body      # against declaring done
    assert "single failure" in body                # against giving up at two actions
    assert "say so explicitly" in body             # and an honest out when it cannot verify

    # Reachable through the same builder as every other section, so an arm
    # asking for it cannot silently run the control.
    assert body.strip() in build_sections(["doing"])


def test_an_unknown_section_still_raises():
    import pytest

    from crux.prompts import build_sections

    with pytest.raises(ValueError):
        build_sections(["doing", "typo"])


def test_the_full_port_is_an_order_of_magnitude_more_than_the_two_lines():
    """Nine prompt sections here were a few hundred characters and all nine
    landed inside the noise, which says little about prompts at the size of the
    one being compared against.

    pi's core system prompt is 4,169 characters; Claude Code's is 27,960, of
    which roughly 12,600 could bear on a benchmark trial -- the rest is
    communication style for a watching human, a confirm-before-risky-actions
    policy a bypassPermissions trial cannot honour, MCP discovery, and a
    feature-gated autonomous mode. `getSimpleDoingTasksSection` is 7,145 of it.
    """
    from crux.prompts import PROMPT_SECTIONS

    small = PROMPT_SECTIONS["doing"]
    full = PROMPT_SECTIONS["doing_full"]
    assert len(small) < 800
    assert len(full) > 2500
    assert len(full) > 4 * len(small)


def test_the_port_keeps_the_task_substance_and_drops_the_product_bullets():
    from crux.prompts import PROMPT_SECTIONS

    full = PROMPT_SECTIONS["doing_full"]
    for kept in ("verify it actually works", "single failure", "Report outcomes faithfully",
                 "premature abstraction", "Do not create files unless"):
        assert kept in full, kept
    # Nothing that cannot move a trial: no product surface, no tone rules, no
    # time estimates -- carrying them would test length rather than content.
    for dropped in ("/help", "/issue", "/share", "knowledge cutoff", "time estimates",
                    "emoji", "one question per response"):
        assert dropped not in full, dropped


def test_both_sections_are_reachable_and_ordered():
    from crux.prompts import build_sections

    both = build_sections(["doing", "doing_full"])
    assert "# Finishing and persisting" in both
    assert "# Doing tasks" in both
