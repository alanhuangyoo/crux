"""The two parser faults measured on the 89-task run.

Both are checked against the shape the model actually produced, not a synthetic
one: an unclosed `<commands>` followed by `</response>`, and a nameless `<  >`
tag among well-formed siblings.
"""

from __future__ import annotations

import pytest

from crux.terminus_agent import _harden_parser


class FakeParser:
    """Upstream's two relevant surfaces, with upstream's bug in one of them."""

    def _find_top_level_tags(self, content: str) -> list[str]:
        tags = []
        i = 0
        while i < len(content):
            if content[i] == "<":
                end = content.find(">", i)
                if end == -1:
                    break
                body = content[i + 1 : end]
                if not body.startswith(("/", "!", "?")):
                    # The upstream line that raises on a nameless tag.
                    tags.append(body.split()[0] if " " in body else body)
                i = end + 1
            else:
                i += 1
        return tags

    def _get_auto_fixes(self):
        return [("Missing </response> tag was automatically inserted", lambda r, e: (r, False))]


UNCLOSED = (
    "<response>\n<analysis>found it</analysis>\n<commands>\n"
    '<keystrokes duration="0.5">ls\n</keystrokes>\n'
    "</response>"
)


def _fixer(parser):
    """The auto-fix this module appends, from the parser's own hook."""
    return next(
        fn for name, fn in parser._get_auto_fixes() if "</commands>" in name
    )


def test_unclosed_commands_is_repaired():
    parser = _harden_parser(FakeParser())
    fixed, changed = _fixer(parser)(UNCLOSED, "Missing <commands> section")
    assert changed
    assert "</commands>" in fixed
    # The section keeps its place: closed before the response ends, not after.
    assert fixed.index("</commands>") < fixed.index("</response>")
    # And the commands themselves survive intact.
    assert "ls" in fixed


def test_repair_leaves_a_closed_response_alone():
    parser = _harden_parser(FakeParser())
    good = UNCLOSED.replace("</response>", "</commands>\n</response>")
    _, changed = _fixer(parser)(good, "Missing <commands> section")
    assert not changed


def test_repair_only_fires_on_its_own_error():
    parser = _harden_parser(FakeParser())
    _, changed = _fixer(parser)(UNCLOSED, "No <response> tag found")
    assert not changed


def test_upstream_fixes_are_kept():
    parser = _harden_parser(FakeParser())
    names = [name for name, _ in parser._get_auto_fixes()]
    assert any("</response>" in n for n in names)
    assert any("</commands>" in n for n in names)


def test_nameless_tag_does_not_raise():
    parser = FakeParser()
    with pytest.raises(IndexError):
        parser._find_top_level_tags("<response><  ></response>")
    _harden_parser(parser)
    assert parser._find_top_level_tags("<response><  ></response>") == ["response"]


def test_nameless_tag_does_not_drop_its_siblings():
    """The first version of this guard returned [], losing every tag."""
    parser = _harden_parser(FakeParser())
    tags = parser._find_top_level_tags("<analysis>a</analysis><  ><commands>c</commands>")
    assert "commands" in tags
    assert "analysis" in tags


def test_the_edit_section_names_the_tool_and_its_format():
    """The section is only useful if the model can write a patch from it alone.

    It is delivered as part of the system prompt with no examples elsewhere, so
    the envelope markers have to appear verbatim in the text.
    """
    from crux.prompts import TERMINUS_EDIT_SECTION as s

    assert "apply_patch" in s
    for marker in ("*** Begin Patch", "*** Update File:", "*** End Patch"):
        assert marker in s, marker
    # And it has to say when NOT to use it, or the model patches files it is
    # creating and the context never matches.
    assert "creating" in s
