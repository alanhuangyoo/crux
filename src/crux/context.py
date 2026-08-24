"""Context window management.

A tmux agent re-sends the terminal screen every turn, so the conversation
grows fast — a 160x40 pane is roughly 1.5k tokens per turn before any model
output. Left alone, a long task stops failing on difficulty and starts failing
on context length, which puts a hard ceiling on the score no amount of prompt
tuning can lift.

The fix, following Terminus 2: once free space runs low, have the model write
a handoff summary, then restart the conversation from that summary plus the
live terminal. The terminal is the real state, so what has to survive the
reset is only what the screen cannot show — what was tried, what was learned,
what failed and why.
"""

from __future__ import annotations

import logging

import litellm

logger = logging.getLogger(__name__)

HANDOFF_PROMPT = """\
You are handing this task off to another agent that will see none of this \
conversation — only your summary and the live terminal.

Original task:

{instruction}

Write a handoff covering:

1. **What you did** — the significant commands and what each established.
2. **What you learned** — file locations, versions, configuration, exact error \
messages, anything about the environment that cost you a command to discover.
3. **What went wrong** — approaches that failed and why, so they are not \
retried.
4. **Where things stand** — what is done, what remains, and the immediate next \
step.

Be specific: exact paths, exact commands, exact errors. The next agent can see \
the terminal, so do not describe what is on screen — describe what is not.
"""

RESUME_PROMPT = """\
You are continuing work started by another agent.

Original task:

{instruction}

Handoff from the previous agent:

{summary}

Current terminal state:

{terminal_state}

Continue from here. Verify anything from the handoff you intend to rely on \
before building on it.
"""


def max_input_tokens(model_name: str, fallback: int) -> int:
    """Context window for a model, from litellm's map when it knows it."""
    try:
        info = litellm.get_model_info(model_name)
        value = info.get("max_input_tokens") or info.get("max_tokens")
        if value:
            return int(value)
    except Exception:
        # Unknown or custom model; the configured fallback is the best guess.
        pass
    return fallback


def count_tokens(model_name: str, messages: list[dict]) -> int:
    """Token count for a message list, approximating if litellm cannot."""
    try:
        return int(litellm.token_counter(model=model_name, messages=messages))
    except Exception:
        # Deliberately pessimistic: 3 chars/token over-counts for English, and
        # over-counting merely compacts a little early, whereas under-counting
        # means discovering the limit by having the request rejected.
        return sum(len(str(m.get("content", ""))) for m in messages) // 3


def should_compact(
    model_name: str, messages: list[dict], threshold: int, fallback_window: int
) -> bool:
    used = count_tokens(model_name, messages)
    window = max_input_tokens(model_name, fallback_window)
    free = window - used
    if free < threshold:
        logger.info(
            "compaction triggered: %d/%d tokens used, %d free (< %d)",
            used,
            window,
            free,
            threshold,
        )
        return True
    return False


def build_resume_messages(
    system_prompt: str, instruction: str, summary: str, terminal_state: str
) -> list[dict]:
    """The conversation to carry on with after a compaction."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": RESUME_PROMPT.format(
                instruction=instruction,
                summary=summary,
                terminal_state=terminal_state,
            ),
        },
    ]
