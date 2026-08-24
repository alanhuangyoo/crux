"""The envelope repair that cost v2's first run.

DeepSeek emitted correct <analysis>/<plan>/<commands> but omitted the outer
<response> wrapper on nearly every turn. The parser rejected all of it, the
agent burned its format-error budget, and all three tasks scored zero with the
model having done nothing wrong.
"""

from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser

from crux.agent import CruxAgent

PARSER = TerminusXMLPlainParser()

BODY = (
    "<analysis>Looking around.</analysis>"
    "<plan>List the directory.</plan>"
    '<commands><keystrokes duration="0.1">ls -la\n</keystrokes></commands>'
)


def parse(raw: str):
    return PARSER.parse_response(CruxAgent._normalize(raw))


def test_missing_envelope_is_repaired():
    result = parse(BODY + "<task_complete>false</task_complete>")
    assert not result.error
    assert [c.keystrokes for c in result.commands] == ["ls -la\n"]
    assert result.is_task_complete is False


def test_repair_preserves_task_complete():
    result = parse(BODY + "<task_complete>true</task_complete>")
    assert not result.error
    assert result.is_task_complete is True


def test_repair_without_task_complete_tag():
    result = parse(BODY)
    assert not result.error
    assert len(result.commands) == 1


def test_repair_ignores_prose_before_the_sections():
    result = parse("Sure, here is my next step.\n\n" + BODY)
    assert not result.error
    assert len(result.commands) == 1


def test_well_formed_response_is_untouched():
    raw = f"<response>{BODY}<task_complete>false</task_complete></response>"
    assert CruxAgent._normalize(raw) == raw


def test_text_without_commands_is_left_alone():
    """Nothing to repair means nothing to fabricate — the error must stand."""
    raw = "I think the answer is 42."
    assert CruxAgent._normalize(raw) == raw
    assert parse(raw).error


MISMATCHED = (
    "<response><analysis>a</analysis><plan>b</plan><commands>"
    '<keystrokes duration="0.1">ls\n</keystrokes>'
    '<keystrokes duration="1.0">pwd\n</keystrokes>'
)


def test_commands_closed_with_the_wrong_tag():
    """Seen as </jobs> and </tasks> in real runs — tag drift, valid commands."""
    for bogus in ("</jobs>", "</tasks>", "</steps>"):
        result = parse(MISMATCHED + bogus + "<task_complete>false</task_complete></response>")
        assert not result.error, f"{bogus}: {result.error}"
        assert len(result.commands) == 2


def test_commands_not_closed_at_all():
    result = parse(MISMATCHED + "<task_complete>false</task_complete></response>")
    assert not result.error
    assert len(result.commands) == 2


def test_both_repairs_at_once():
    """No envelope and a mis-closed commands block in the same reply."""
    raw = MISMATCHED.replace("<response>", "") + "</jobs><task_complete>true</task_complete>"
    result = parse(raw)
    assert not result.error
    assert len(result.commands) == 2
    assert result.is_task_complete is True


def test_correctly_closed_commands_are_untouched():
    raw = MISMATCHED + "</commands><task_complete>false</task_complete></response>"
    assert CruxAgent._close_commands(raw) == raw
