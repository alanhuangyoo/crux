

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
