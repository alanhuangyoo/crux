"""The display is not decoration, so it gets tested like anything else.

Two of these guard against faults that have actually shipped. The analysis
extractor has to survive an unclosed `<analysis>` tag, because unclosed tags are
the exact habit that cost 21% of all model calls on this benchmark -- a display
that assumed well-formed markup would go blank on precisely the turns worth
watching. And the width handling has to survive a 40-column pane, because the
first thing anyone does with a long-running agent is put it in a split.
"""

from crux import ui


def test_analysis_prefers_the_tagged_body():
    raw = (
        "<response>\n<analysis>\nReading the config first.\n</analysis>\n"
        "<commands><command><keystrokes>ls</keystrokes></command></commands>\n</response>"
    )
    out = ui.analysis(raw)
    assert "Reading the config first." in out
    for tag in ("<response>", "<analysis>", "<commands>", "<keystrokes>"):
        assert tag not in out


def test_analysis_survives_an_unclosed_tag():
    # The shape that cost 21% of model calls: the model opens <analysis> and
    # never closes it. The prose is still there and still worth showing.
    raw = "<response>\n<analysis>\nPatching the handler now.\n<commands>\n<command>"
    out = ui.analysis(raw)
    assert "Patching the handler now." in out
    assert "<command" not in out


def test_analysis_falls_back_to_the_prefix_form():
    assert "reading the config" in ui.analysis("Analysis: reading the config").lower()


def test_analysis_is_empty_when_there_is_nothing_to_say():
    assert ui.analysis("") == ""
    assert ui.analysis("<response><commands></commands></response>") == ""


def test_analysis_stops_before_the_commands():
    raw = "<analysis>Only this.</analysis><commands><command>rm -rf /</command></commands>"
    assert "rm -rf" not in ui.analysis(raw)


def test_first_line_takes_the_head_of_a_heredoc():
    # The common shape: one meaningful line, then forty lines of payload.
    cmd = "cat > /app/solve.py << 'PY'\nimport sys\nprint(1)\nPY"
    assert ui.first_line(cmd).startswith("cat > /app/solve.py")
    assert "import sys" not in ui.first_line(cmd)


def test_first_line_skips_leading_blanks():
    assert ui.first_line("\n\n  ls -la\n") == "ls -la"


def test_first_line_of_nothing_is_nothing():
    assert ui.first_line("") == ""
    assert ui.first_line("\n \n") == ""


def test_command_result_marks_an_error_it_can_see():
    assert "errors" in ui.command_result(0.2, "Traceback (most recent call last):", False)
    assert "errors" in ui.command_result(0.2, "bash: foo: command not found", False)
    assert "ok" in ui.command_result(0.2, "hello\nworld", False)
    assert "timeout" in ui.command_result(90.0, "", True)


def test_human_secs_reads_at_every_scale():
    assert ui.human_secs(0.25) == "250ms"
    assert ui.human_secs(4.2) == "4.2s"
    assert ui.human_secs(125) == "2m05s"
    assert ui.human_secs(7500) == "2h05m"


def test_human_count_reads_at_every_scale():
    assert ui.human_count(42) == "42"
    assert ui.human_count(12_000) == "12.0k"
    assert ui.human_count(2_500_000) == "2.50M"


def test_width_stays_usable_in_a_split_pane(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda *_: type("S", (), {"columns": 12})())
    assert ui.width() >= 40
    monkeypatch.setattr(shutil, "get_terminal_size", lambda *_: type("S", (), {"columns": 400})())
    assert ui.width() <= 120


def test_rule_fills_the_width_exactly():
    line = ui.rule("label")
    assert ui._visible_len(line) == ui.width()


def test_spinner_is_inert_without_a_tty():
    # Everything else in a piped session must behave identically, so the
    # context manager has to be safe to enter and leave with no terminal.
    with ui.Spinner("x") as sp:
        sp.set("y")
    assert True


def test_turn_summary_says_what_was_spent():
    line = ui.turn_summary(7, 12, 84.2, 12000, 3400, 9800)
    for want in ("7 steps", "12 commands", "1m24s", "12.0k", "3.4k", "9.8k cached"):
        assert want in line
