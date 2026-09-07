

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
