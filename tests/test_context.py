"""Context-window handling.

One trial in the first full run died on ContextWindowExceededError while
compaction never fired: litellm reports deepseek-v4-flash at 1M tokens, so a
threshold expressed as "free space remaining" was unreachable in practice.
"""

from crux.config import build_config
from crux.context import build_resume_messages, max_input_tokens, should_compact


def test_reported_window_is_capped():
    """The model map is treated as an upper bound, not a fact."""
    assert max_input_tokens("deepseek/deepseek-v4-flash", 128000, cap=200000) <= 200000


def test_cap_does_not_inflate_a_smaller_window():
    assert max_input_tokens("definitely-not-a-real-model", 32000, cap=200000) == 32000


def test_no_cap_leaves_the_reported_window_alone():
    uncapped = max_input_tokens("deepseek/deepseek-v4-flash", 128000)
    capped = max_input_tokens("deepseek/deepseek-v4-flash", 128000, cap=50000)
    assert capped <= 50000 <= uncapped or uncapped <= 50000


def test_short_conversation_does_not_compact():
    messages = [{"role": "user", "content": "hello"}]
    assert not should_compact("deepseek/deepseek-v4-flash", messages, 12000, 128000, 200000)


# ~250k tokens: over the 200k cap, well under the 1M the model map reports.
LONG = [{"role": "user", "content": "x " * 50000} for _ in range(5)]


def test_long_conversation_compacts_under_the_cap():
    """With the cap in place a long conversation must actually trigger."""
    assert should_compact("deepseek/deepseek-v4-flash", LONG, 12000, 128000, 200000)


def test_same_conversation_would_not_compact_against_the_raw_1m_window():
    """Pins the bug: without the cap, this conversation still looks small."""
    assert not should_compact("deepseek/deepseek-v4-flash", LONG, 12000, 128000, None)


def test_resume_messages_carry_summary_and_terminal():
    msgs = build_resume_messages("SYS", "do the thing", "what happened", "$ pwd")
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYS"
    assert len(msgs) == 2
    body = msgs[1]["content"]
    assert "do the thing" in body and "what happened" in body and "$ pwd" in body


def test_cap_is_configurable():
    assert build_config().context_window_cap == 200000
    assert build_config(context_window_cap=50000).context_window_cap == 50000
