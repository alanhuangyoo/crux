

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
