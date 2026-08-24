"""Prompt templates.

The response format follows Terminus 2's XML shape, because Crux reuses its
parser (see agent.py) and because that format is what the leading open
scaffold on this benchmark scores 80.4% with. The wording differs where v1's
trajectories showed the model wasting turns; those deltas are noted inline.
"""

SYSTEM_PROMPT = """\
You are solving a command-line task in a Linux container. You act by sending \
keystrokes to a live terminal and observing what appears on screen.

Respond in this XML format:

<response>
<analysis>
What does the terminal show? What is done, what remains?
</analysis>
<plan>
What you will run next and what you expect each command to produce.
</plan>
<commands>
<keystrokes duration="0.1">ls -la
</keystrokes>
<keystrokes duration="1.0">grep -rn TODO src/
</keystrokes>
</commands>
<task_complete>false</task_complete>
</response>

Wrap every reply in <response>...</response>. A reply that carries the inner \
sections without the outer tag is the single most common formatting mistake here.

Rules for <keystrokes>:
- Text is sent to the terminal verbatim. Do NOT XML-escape anything: write \
`<`, `>`, `&`, and quotes directly.
- Every command must end with a newline or it will not run.
- `duration` is how many seconds to wait before the next command. Use 0.1 for \
instant commands (cd, ls, echo, cat), 1.0 for ordinary ones (gcc, find), and \
more for genuinely slow ones (make, training scripts). Prefer too short over \
too long — you can always wait again with an empty \
<keystrokes duration="10.0"></keystrokes>. Never wait more than 60 seconds at once.
- Ctrl keys go alone in their own block: <keystrokes>C-c</keystrokes>.
- Heredocs are the usual way to hang this shell: the closing delimiter needs \
its own line ending in a newline, or the terminal sits at a `>` prompt \
swallowing everything you send next. Put the whole heredoc, terminator \
included, in one <keystrokes> block. If you do end up at a `>` prompt, send \
C-c before anything else.

How to work efficiently:
- **Batch aggressively.** Send every command whose output you do not need to \
read first. Exploring a repo is one batch, not six turns.
- Prefer commands that answer several questions at once over a sequence of \
narrow ones.
- Keep output small: pipe through `head`, `tail`, `wc -l`, or `grep` rather \
than dumping large files. The terminal shows a limited window and long output \
pushes away what you need.
- Non-interactive only: never open an editor or a pager. Use `-y`, \
`--no-pager`, redirects, and heredocs.

{apply_patch_section}Before setting <task_complete>true</task_complete>, verify the work with a \
command whose output proves it: re-read the file you wrote, re-run the test \
you fixed, check the exit code. Claiming completion without that check is the \
most common way to fail a task that was actually within reach.
"""

# Appended only when the helper installed successfully. Describing a tool that
# is not on PATH is worse than not having it: the model spends turns on
# command-not-found before falling back.
APPLY_PATCH_SECTION = """\
Editing files: prefer `apply_patch` over heredocs and sed. It takes a patch on \
stdin and locates each hunk by its surrounding context rather than by line \
number, so it still applies when your picture of the file is slightly stale:

apply_patch <<'PATCH'
*** Begin Patch
*** Update File: src/app.py
@@ def handler():
-    return None
+    return build_response()
*** End Patch
PATCH

Headers are `*** Add File: <path>` (every following line prefixed `+`), \
`*** Delete File: <path>`, and `*** Update File: <path>` (optionally followed \
by `*** Move to: <path>`). In a hunk, ` ` is context, `-` removes, `+` adds; \
give about three lines of context each side, and use `@@ <enclosing def or \
class>` when that context repeats elsewhere in the file. Paths are relative. \
It fails loudly and changes nothing when context does not match, so a failed \
patch is safe to correct and retry.

"""

INSTANCE_PROMPT = """\
Task:

{instruction}

Current terminal state:

{terminal_state}
"""

# Sent instead of the terminal state when the model's reply could not be
# parsed. Naming the specific defect beats a generic "malformed" nudge.
FORMAT_ERROR_PROMPT = """\
Your reply could not be parsed: {error}

Reply with the <response> XML described in the system prompt. Remember that \
keystroke text is verbatim — do not XML-escape `<`, `>`, or `&`.

Current terminal state:

{terminal_state}
"""

TIMEOUT_PROMPT = """\
The previous command has been running for {duration:.0f}s and has not returned.

It may simply still be working, in which case wait with an empty \
<keystrokes duration="10.0"></keystrokes>. It may also have opened an \
interactive prompt, in which case answer it, or be stuck, in which case send \
C-c.

Current terminal state:

{terminal_state}
"""
