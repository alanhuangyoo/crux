"""Crux on the Terminus scaffold.

Crux started as a mini-swe-agent derivative, which turned out to be the wrong
base. Measured on the same ten tasks, same model, same deployment:

| harness    | solved |
|------------|--------|
| terminus-2 | 7/10   |
| crux (mini)| 6/10   |
| codex      | 5/9    |

The gap is architectural rather than a matter of tuning. Terminus drives a live
tmux session and sends keystrokes; mini-swe-agent runs one shell command per
turn and reads its stdout. That difference decides whole categories of task --
`qemu-alpine-ssh` needs an interactive ssh session into a VM, which cannot be
expressed as a sequence of one-shot commands at all, and only Terminus solved
it. It also lets one turn issue several keystrokes with individual waits, so
polling a slow build does not cost a model round trip, which is exactly where
this deployment loses tasks: agent timeouts are the largest failure category.

So this subclasses Terminus and adds back the two things worth keeping from
the old agent -- the all-or-nothing scoring instruction, and `crux submit`,
which turns finishing from a judgement into a re-run of every bound check.
Nothing else is changed: the format contract, the tmux handling, and the
summarisation are upstream's, and that is the point.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import time

from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.llms.base import LLMResponse
from harbor.agents.terminus_2.tmux_session import TmuxSession
from typing_extensions import override

from crux.calibration import (
    CALIBRATION_MAX_OUTPUT_TOKENS,
    _LLM_TIMEOUT_MAX_SEC,
    _LLM_TIMEOUT_MIN_SEC,
    llm_timeout_for as _llm_timeout_for,
)
from crux.prompts import build_terminus_template

# Above this, a wait is worth completing early; below it, upstream's fixed sleep
# is already close enough that the tmux round trip is not worth the change in
# behaviour. Set from the measured waste: the losses are all long waits.
_BLOCKING_THRESHOLD_SEC = 10.0

# No task solved in a full 89-task run took more than 120 steps; the longest was
# winning-avg-corewars at 120, and the solved median was 29. Failures ran 71 at
# the median and 323 at the 90th percentile. So a trial past this point is not
# a slow success, it is an approach that has already failed -- and the model
# cannot see that, because from inside every step still looks locally sensible.
#
# The nudge fires once. Stopping the trial would only save wall clock; telling
# it what the number means is the part that might change the outcome.
# Measured across the 89-task run: the trials that failed edited 17 times and
# ran something that checks the work 3 times; the ones that solved edited 6 and
# checked 6. The longest run of edits with no verification between them was 17
# steps for failures and 4 for successes. A run that edits seventeen times
# without once asking whether any of it works is not converging -- by then the
# earlier changes have usually been broken by the later ones, with no way to
# tell which.
#
# Twelve is not a midpoint, it is the argmax of a sweep replayed over all 89
# trajectories: it fires on 62% of the failures against 15% of the successes,
# where 20 caught only 37% and 10 hurt 23%.
#
# An earlier version of this constant was 20, derived from an edit/verify
# classifier that counted any command containing a redirect or a pipe as an
# edit -- `crux --help | head`, `2>&1`, compile lines. Auditing what the
# patterns actually matched is what produced the numbers above.
_EDIT_DEBT_LIMIT = 12

# An edit writes a file. A redirect into a path counts; `2>&1` and `| head` do
# not, and counting them was what inflated the original measurement.
_EDIT_RE = re.compile(
    r"(crux\s+(edit|write)\b"
    r"|apply_patch\b"
    r"|\bsed\s+-i\b"
    r"|\btee\s+[\w./~-]"
    r"|>>?\s*/?[\w./~-]+\.\w+"
    r"|cat\s*>\s*[\w./~-])"
)
# A verification runs something that can fail and says so; `--help` cannot.
_VERIFY_RE = re.compile(
    r"(pytest\b|python\S*\s+-m\s+unittest\b|make\s+(test|check)\b"
    r"|npm\s+(run\s+)?test\b"
    r"|crux\s+submit\b"
    r"|crux\s+todo\s+(done|verify|list)\b"
    r"|\./(run|test)\S*"
    r"|bash\s+\S*test\S*\.sh"
    r"|python\S*\s+\S*test\S*\.py)"
)
# Reading a manual is neither.
_HELP_RE = re.compile(r"--help|\bman\b")

logger = logging.getLogger(__name__)


def _discover_budget(environment) -> float:
    """The trial's wall-clock budget, worked out from what is already on disk.

    harbor enforces the agent timeout outside the agent and never tells it the
    number, which is why an agent has never been able to answer "how much of my
    run is left". Nothing needs plumbing to fix that: the trial's own
    config.json carries the multiplier, and the task package it names carries
    the base.

        <trial>/config.json         agent_timeout_multiplier, task.name
        <harbor cache>/.../task.toml  [agent] timeout_sec

    Every step is wrapped: a budget is a nicety, and not knowing it degrades
    the notice to elapsed-only rather than failing a trial. Returns 0 when it
    cannot be worked out.
    """
    try:
        trial_dir = Path(environment.trial_paths.agent_dir).parent
        cfg = json.loads((trial_dir / "config.json").read_text())
    except Exception:  # noqa: BLE001
        return 0.0
    mult = float(cfg.get("agent_timeout_multiplier") or 1.0)
    name = ((cfg.get("task") or {}).get("name") or "")
    if not name:
        return 0.0
    org, _, task = name.rpartition("/")
    root = Path.home() / ".cache" / "harbor" / "tasks" / "packages"
    for candidate in (root / org / task, root / task):
        try:
            tomls = sorted(candidate.glob("*/task.toml"))
        except OSError:
            continue
        for t in tomls:
            try:
                text = t.read_text(errors="replace")
            except OSError:
                continue
            # A two-line regex rather than a TOML parser: tomllib is 3.11+ and
            # the file is read only for one number.
            m = re.search(r"\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", text, re.S)
            if m:
                try:
                    return float(m.group(1)) * mult
                except ValueError:
                    return 0.0
    return 0.0


def _human_secs(sec: float) -> str:
    """Duration in words. Duplicated from crux.ui deliberately: the agent must
    not import the terminal front end, which a benchmark trial never loads."""
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _exec_text(result) -> str:
    """Whatever a command printed, whichever stream it used.

    harbor's ExecResult carries stdout and stderr separately and either may be
    None; callers here only ever look for a sentinel word, so the two are read
    as one string.
    """
    return (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")


# `crux todo add ... --verify '<cmd>'` -- binding a check, as opposed to adding
# a checklist line with nothing behind it.
_NUL_NOTE = """\
Nothing was executed. Command {i} contains a literal NUL byte, which cannot
travel through the terminal channel -- the send raises ValueError("embedded
null byte") and the trial ends there, with no score.

The byte itself is fine to test; it just has to be produced inside the
container rather than typed into the channel. Any of these work:

  printf 'java\\000script:' > f.html
  printf 'java\\x00script:' > f.html          # printf(1), not the shell
  python3 -c "open('f.html','wb').write(b'java\\x00script:')"

Resend the batch with the NUL written that way.
"""

# What the agent is told at the moment it says it is finished. The numbers are
# the measurement that motivated it; they are quoted rather than paraphrased
# because a generic "are you sure" is what upstream already asks, and the
# trajectories show it being waved through.
_BUDGET_NOTICE = """\

You have been working {elapsed} over {steps} steps{share}.

Measured on this benchmark: trials that solved the task had used 12-16% of
their budget when they stopped; trials that failed had used 33-55%, and only
3 of 17 ran out of time. Failure here is not running out of time. It is
stopping early, on a check the agent wrote for itself -- `crux submit` printed
"all N item(s) verified" on 95% of the runs that scored and on 100% of the runs
that did not.

If time is what you have left, the cheapest thing you can do with it is one
pass you have not done: re-read the task's own words, list what the grader will
run, and check the deliverable against that list rather than against the
checks you already wrote.
"""

_CHECK_BIND_RE = re.compile(r"crux\s+todo\s+add\b[^\n]*?--verify")

_CHECKLIST_FIRST_NUDGE = """\
Nothing was executed. There is no bound check yet, and this batch edits.

Measured across 87 scored trials on this benchmark: the first check gets bound
at 75-87% of the way through a run, after the last edit. 88% of trials never
see a bound check fail even once -- including the ones that solved the task.
`crux submit` printed "all N item(s) verified" on 95% of the runs that scored
and on 100% of the runs that did not.

A checklist written after the work is a description of what you built, and a
description cannot fail. Written before, it is a test: red now, green when the
work is done, and the difference is the information.

Bind one check for the task's first requirement, then edit:

  crux todo add "<requirement, in the task's own words>" --verify '<command>'

It is expected to fail right now. That is what makes it worth anything.
"""

_EDIT_DEBT_NUDGE = """\
You have made {n} edits without running anything that checks them.
The commands in that last batch were NOT executed -- nothing on disk changed.

Measured on this benchmark: trials that failed edited 17 times and checked 3;
trials that solved edited 6 and checked 6. The failing shape is not too few
edits, it is edits nobody checked -- by now the earlier changes have usually
been broken by the later ones and there is no way to tell which.

Run something that answers "does this work" before editing again: the checker
the task ships, its test file, `crux todo verify`, or the smallest command that
executes the code you changed. Then continue.
"""

# Off. It was 120, and 120 was measured: across the 89-task baseline no task
# that was eventually solved took more than 120 steps, the median solve took 29,
# and every one of the six trials past that point failed.
#
# The parser and blocking-execution fixes then removed the waste those long
# trials were made of, and the distribution moved out from under it. Re-derived
# on 80 trials of the current code:
#
#                    longest solve   past 120 steps: solved / failed
#     baseline           120 steps          0 / 6      (0% solved)
#     current            182 steps          6 / 4     (60% solved)
#
# Failures used to run to 519 steps and now stop at 132; solves now run to 182.
# So crossing 120 has stopped meaning the approach failed -- it now correlates
# with succeeding -- and no threshold in the current data separates the two at
# all. A nudge that tells a trial its approach has failed, on evidence that its
# own fixes made false, is worse than no nudge.
#
# Kept as a knob rather than deleted: the measurement is about this model on
# this benchmark, and stuck_step_threshold=<n> turns it back on for anything
# where the old shape holds.
# How long one model call may take before it is abandoned.
#
# litellm's default is 6000 seconds and harbor overrides it nowhere, so a call
# that never returns holds its trial -- and one of the run's concurrency slots
# -- for a hundred minutes. Three trials were observed sitting at 115, 117 and
# 106 minutes, each stopped mid-task with its terminal at a prompt and its next
# request never answered.
#
# Against a Terminal-Bench budget of 7200s that is 83% of the task gone to one
# hung request. 600s is generous for this deployment -- 5,358 requests, the
# slowest completed turn well inside it -- and a turn that needs longer has
# already lost the task.
# A ceiling on one response, because without one a runaway generation runs to
# the context limit and takes the trial with it.
#
# Found by reading claude-code's trajectory on `regex-chess`, a task that stalls
# in every crux run and stalled for it too -- 7 steps, one tool call, 240
# minutes. Its trajectory says why:
#
#     API Error: Claude's response exceeded the 64000 output token maximum.
#
# Same wall, different outcome: claude-code has CLAUDE_CODE_MAX_OUTPUT_TOKENS
# and fails loudly, crux had no ceiling and generates until the context runs
# out, which reads as two hours of silence.
#
# Measured across four runs, single-step completion tokens:
#
#                    stalled trials        scored trials
#     all-on         28,234 / 41,157       12,408
#     ft5            13,296 / 20,327        7,478
#     both-on        18,712 / 29,737        6,994
#
# 32k leaves room above every scored trial's largest turn (7.5k-14k median,
# 44k max) while cutting a runaway well before the 262k window. A turn that
# needs more than this has stopped writing commands and started spiralling --
# the truncation retry, which drops thinking, is the path that recovers it.
_MAX_OUTPUT_TOKENS = CALIBRATION_MAX_OUTPUT_TOKENS

# A ceiling on one *call*, because litellm's own default is 6000s and harbor
# never overrides it: a request that hangs would sit there for 100 minutes.
#
# 600 was set against a 900-second budget. At the 8x budget everything is run
# at now it is 8% of the trial, and it stopped bounding hangs and started
# cutting real work. `regex-chess` died on it, three attempts deep:
#
#     litellm.Timeout: timeout value=600.0, time taken=1801.36 seconds
#
# and `regex-chess` is one of the two tasks written down in this project as
# never solving, priced at a reduced budget on the strength of never having
# solved. It was being killed by this constant.
#
# The distribution says where the line belongs. Across 9,253 inter-step gaps in
# two 89-task runs, 99.8% are under 600s and the tail is thin -- but the five
# tasks that stall spend 20-30% of their steps above it. So the cut is not
# broad, it is aimed squarely at the handful of tasks with the longest turns.
#
# Scaling to the budget keeps both properties: a hang is still bounded by a
# quarter of the trial, and a ten-minute generation is no longer thrown away.
# The numbers and the derivation live in calibration.py, which imports
# nothing: the launch-time audit has to be able to check this constant on a
# bare interpreter, and importing this module pulls in harbor and pydantic.
_LLM_CALL_TIMEOUT_SEC = _LLM_TIMEOUT_MIN_SEC

_STUCK_STEP_THRESHOLD = 0


def _can_block(keystrokes: str) -> bool:
    """Whether appending a completion signal to these keystrokes is safe.

    TmuxSession implements blocking by stripping the trailing newline and
    appending `; tmux wait -S done`. For a single-line command that is
    harmless. For a heredoc it is not: the keystrokes end with the delimiter
    line, so the append turns `EOF` into `EOF; tmux wait -S done`, which no
    longer terminates the heredoc. The shell sits at its continuation prompt
    until the outer timeout fires.

    That is not hypothetical -- it is what a first version of this did. A pane
    from that run reads `> PY; tmux wait -S done`, the `>` being the shell
    still waiting for a terminator, and the run tracked about fifteen points
    below the unmodified agent before it was stopped.

    So blocking is restricted to a single line with no heredoc operator.
    Anything else keeps upstream's fixed sleep, which is always correct if
    sometimes wasteful.
    """
    body = keystrokes.rstrip("\r\n")
    return "<<" not in body and "\n" not in body and "\r" not in body

_RESOURCES = Path(__file__).parent / "resources"


# Upstream's synthetic response when its context-overflow fallback cannot get a
# usable turn out of the model. It is not model output, and it does not parse.
_DEADLOCK_CONTENT = "Technical difficulties. Please continue with the task."

# What to keep when trimming: the instruction the task opened with, and enough
# recent turns to know what the terminal is showing. The middle is what grew.
_KEEP_HEAD = 1
_KEEP_TAIL = 12


def _guard_context_deadlock(chat):
    """Break the summarize-fails-resend loop by trimming the conversation.

    The loop, measured: the context overflows, upstream summarizes, the chat
    call carrying the summary comes back truncated (`finish_reason=length`),
    and the exception handler substitutes `_DEADLOCK_CONTENT` for the response.
    That string has no <response> tag, so the parser rejects it, so the agent is
    asked again -- with the same oversized context, which overflows again.

    On `extract-moves-from-video` that ran 322 times and consumed the whole
    budget; on `path-tracing-reverse`, 182. Both scored zero. Nothing about it
    is visible as an error: the trajectory shows an agent politely being told
    its response had parsing errors, several hundred times.

    Upstream cannot escape it on its own because every exit it has goes through
    another call on the same context. Trimming is the one move that changes the
    input, so that is what this does -- after the second occurrence, so a single
    transient failure still gets its ordinary retry.
    """
    original = getattr(chat, "chat", None)
    if not callable(original):
        return chat  # nothing to guard; upstream may hand us another shape
    seen = {"n": 0}

    def _trim() -> bool:
        """Drop the middle of the conversation. True if anything was dropped."""
        messages = getattr(chat, "_messages", None)
        if not isinstance(messages, list) or len(messages) <= _KEEP_HEAD + _KEEP_TAIL:
            return False
        del messages[_KEEP_HEAD:len(messages) - _KEEP_TAIL]
        return True

    async def _chat(*args, **kwargs):
        # The failure is an exception, not a value. A first version of this
        # guard only inspected the returned response, and the run showed what
        # that bought: path-tracing-reverse deadlocked 204 times with the guard
        # installed. Upstream's summarization fallback calls chat.chat inside a
        # try, and substitutes the fixed string in its own except block -- so
        # the string never passes through here, and the exception does.
        try:
            response = await original(*args, **kwargs)
        except Exception:
            seen["n"] += 1
            if seen["n"] < 2 or not _trim():
                raise
            seen["n"] = 0
            return await original(*args, **kwargs)
        content = (getattr(response, "content", "") or "").strip()
        if content != _DEADLOCK_CONTENT:
            seen["n"] = 0
            return response
        # Belt and braces: if some path does hand the string back as a value,
        # it means the same thing and gets the same treatment.
        seen["n"] += 1
        if seen["n"] < 2 or not _trim():
            return response
        seen["n"] = 0
        return await original(*args, **kwargs)

    chat.chat = _chat
    return chat


def _harden_parser(parser):
    """Two parser faults that cost this arm more than any prompt did.

    **An unclosed <commands> deadlocks the trial.** The model on this setup
    reliably writes `<commands>`, then its `<keystrokes>` blocks, then
    `</response>` -- and never `</commands>`. The parser looks for the closing
    tag, does not find it, and reports "Missing <commands> section". The model
    reads that, writes the section again, and omits the closing tag again. On
    `mailman` that ran 438 times and consumed the entire 240-minute budget; on
    `path-tracing-reverse`, 237 times. Across the 89-task run it accounted for
    682 of 5,670 steps -- 12% of every step taken, spent on a missing tag.

    Upstream already repairs exactly this shape of fault for `</response>`:
    `_get_auto_fixes` returns (warning, fixer) pairs, and one of them appends
    the tag and warns. There is no equivalent for `</commands>`, so this adds
    one through that hook rather than around it.

    Worth being precise about why the model never learns: the rejection says
    the section is *missing*. It is not -- it is unclosed. The model is being
    told to do the thing it already did, so it does it the same way.

    **A whitespace-only tag raises.** Upstream extracts a tag name with

        tag_name = tag_content.split()[0] if " " in tag_content else tag_content

    which raises IndexError on `<  >`: `" " in "  "` is true and `"  ".split()`
    is empty. The exception escapes the agent loop and the trial scores zero.

    The first version of this guard caught the IndexError and returned `[]`,
    which dropped *every* tag in the response rather than the malformed one --
    trading a rare crash for a silent misparse. This one skips the bad tag and
    keeps the rest, which is what the surrounding code does with anything it
    cannot identify.
    """
    # Both repairs below are about hand-matched XML tags, and the JSON parser
    # has none: it hands the text to a strict parser that either accepts it or
    # does not. Applying them there is not merely useless, it crashed the
    # constructor -- `parser_name="json"`, which is upstream's default, could
    # not be selected at all.
    if not hasattr(parser, "_find_top_level_tags"):
        return

    original = parser._find_top_level_tags

    def _safe(content):
        try:
            return original(content)
        except IndexError:
            # Re-run over a copy with the nameless tags removed, so the tags
            # around the bad one still reach the caller.
            return original(re.sub(r"<\s+>", "", content))

    parser._find_top_level_tags = _safe

    def _close_commands(response: str, error: str) -> tuple[str, bool]:
        """Insert the `</commands>` the model omitted, if that is the fault."""
        if "Missing <commands> section" not in error:
            return response, False
        if "<commands>" not in response or "</commands>" in response:
            return response, False
        # Before </response> when there is one, so the section keeps its place
        # in the document; otherwise at the end.
        end = response.find("</response>")
        if end == -1:
            return response.rstrip() + "\n</commands>", True
        return response[:end].rstrip() + "\n</commands>\n" + response[end:], True

    upstream_fixes = parser._get_auto_fixes

    def _fixes():
        return list(upstream_fixes()) + [
            ("Missing </commands> closing tag was automatically inserted",
             _close_commands),
        ]

    parser._get_auto_fixes = _fixes
    return parser


_STUCK_NUDGE = """\
[crux] You have now taken {steps} steps on this task.

Across a full evaluation of this benchmark, no task that was eventually solved
took more than 120 steps; the median solve took 29. Trials that reached this
point failed, and they failed while every individual step still looked
reasonable -- re-reading the same files, re-running the same command with a
small variation, recovering the same stuck terminal.

Treat this as evidence about your current approach rather than about the task.
Before your next command:

  - State in one sentence what you have actually established so far, and what
    you have been assuming without checking.
  - Name the approach you have been pursuing, and pick a different one. Not a
    variation -- a different mechanism.
  - If a check you bound earlier is still failing, consider that the
    requirement may not mean what you read it to mean.

Do not repeat a command you have already run unless its inputs have changed.
"""

class CruxTerminusAgent(Terminus2):
    """Terminus with Crux's scoring prompt and verified submission."""

    @staticmethod
    @override
    def name() -> str:
        return "crux-terminus"

    @override
    def version(self) -> str | None:
        return "0.1.0"

    def __init__(self, *args, **kwargs):
        # Terminus defaults to the JSON parser; the XML one is what its own
        # published runs use, and the template Crux extends is the XML one.
        kwargs.setdefault("parser_name", "xml")
        self._crux_tools = bool(kwargs.pop("crux_tools", True))
        max_tokens = kwargs.pop("max_tokens", _MAX_OUTPUT_TOKENS)
        # Overridable so a slower endpoint can raise it; 0 restores litellm's
        # 6000s default for anyone who wants the old behaviour back.
        # 0 means "derive it from the budget" and is the default; a positive
        # value pins the ceiling; a negative one removes it, which hands the
        # call back to litellm's own 6000-second default.
        self._llm_timeout = float(kwargs.pop("llm_timeout", 0) or 0)
        # Whether the model's own reasoning is handed back to it on the next
        # turn. Upstream defaults this to False, which on this model throws
        # away something it was built to keep:
        #
        #   By default, Qwen3.8 retains thinking blocks from all historical
        #   messages ... especially beneficial for agent scenarios where
        #   decision consistency and reduced redundant reasoning are critical.
        #   It also improves KV cache utilization.
        #
        # sglang's qwen3 reasoning parser splits that thinking out into
        # `reasoning_content`, and harbor only puts it back when this flag is
        # on. Measured on two live runs: 2,097 of 2,100 steps and 4,934 of
        # 4,938 carry reasoning_content, and every one of them was discarded.
        # The model re-derives its reasoning from scratch each turn.
        #
        # Measured across 124 trajectories against 123 without it, and the
        # effect is not the one the card describes:
        #
        #                  steps   reasoning   context growth/step
        #     returned      49     1,737 ch        2,174 tok
        #     discarded     43     1,174 ch        1,085 tok
        #
        # Context per step doubles, as expected -- that is the reasoning going
        # back. But the reasoning itself grows 48% rather than shrinking, and
        # the run takes more steps, not fewer. An early read on 20 trajectories
        # said reasoning was 26% *shorter*; the full corpus reverses it.
        #
        # So "reduced redundant reasoning" does not reproduce here. All three
        # measurable effects are costs. Score is 37/48 against a 65/85 control,
        # which is inside the noise either way.
        #
        # Off by default: `--agent-kwarg interleaved_thinking=1` makes it an arm.
        kwargs.setdefault(
            "interleaved_thinking",
            str(kwargs.pop("interleaved_thinking", False)).lower() in ("1", "true", "yes"),
        )
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        # Whether to keep the `crux submit` gate. Separable from the scoring
        # section because the evidence against them differs: see prompts.py.
        self._crux_submit = str(kwargs.pop("submit_gate", True)).lower() not in (
            "false", "0", "no",
        )
        # Separable so the harness section can be measured on its own: it makes
        # a different claim from the other two, and comes from a different kind
        # of evidence (a side-by-side read rather than an aggregate).
        self._crux_harness = str(kwargs.pop("harness_section", True)).lower() not in (
            "false", "0", "no",
        )
        # Separable from installing the binary: whether apply_patch is on PATH
        # and whether the model is told to use it are two claims, and an
        # ablation needs to move one at a time.
        self._crux_edit = str(kwargs.pop("edit_section", True)).lower() not in (
            "false", "0", "no",
        )
        # The five file tools are installed by _install_crux and, until now,
        # were never named in the Terminus prompt -- build_terminus_template
        # had no parameter for them. Off by default because it is unmeasured
        # on this path, on by `--agent-kwarg file_tools=1`.
        self._crux_file_tools = str(kwargs.pop("file_tools", False)).lower() in (
            "1", "true", "yes",
        )
        # Chain the probes. Measured against claude-code on the same tasks:
        # 3.0 segments per command against 2.0, and 3.0 against 1.0 on the very
        # first command -- crux opens with a bare `ls`. Off by default because
        # it is unmeasured here; `--agent-kwarg batch_section=1` is the arm.
        self._crux_batch = str(kwargs.pop("batch_section", False)).lower() in (
            "1", "true", "yes",
        )
        # Open on the data rather than on a listing. claude-code touches the
        # artifact in its first command on every task it still wins; crux opens
        # with a bare `ls -la /app` in 35 of 39 runs.
        self._crux_look = str(kwargs.pop("look_section", False)).lower() in (
            "1", "true", "yes",
        )
        # Carry the conversation across run() calls, for the interactive path.
        # Off by default: the benchmark scores one instruction per trial, and
        # every number in this repo was measured with a fresh chat per run.
        self._carry_context = str(kwargs.pop("carry_context", False)).lower() not in (
            "false", "0", "no", "none",
        )
        self._carried: list = []
        # Steps past which the agent is told its approach has failed. 0 disables.
        self._stuck_at = int(kwargs.pop("stuck_step_threshold", _STUCK_STEP_THRESHOLD))
        self._edit_debt_limit = int(kwargs.pop("edit_debt_limit", _EDIT_DEBT_LIMIT))
        self._edit_debt = 0
        self._edit_debt_fired = False
        # Hold the first edit until a check is bound. Off by default: it moves
        # the order the agent works in, and only the timing is measured.
        self._checklist_first = str(kwargs.pop("checklist_first", False)).lower() not in (
            "false", "0", "no", "none",
        )
        self._checklist_fired = False
        self._check_bound = False
        # Wall-clock budget in seconds, when the caller knows it. harbor
        # enforces the timeout outside the agent and does not pass it in, so
        # the agent has never had any idea how much of its run was left --
        # which is the whole shape of the failure this reports on.
        try:
            self._budget_sec = float(kwargs.pop("budget_sec", 0) or 0)
        except (TypeError, ValueError):
            self._budget_sec = 0.0
        self._run_started = 0.0
        self._stuck_fired = False
        self._crux_steps = 0
        # Whether `crux submit` demands a second pass before it will finish.
        # Named apart from `submit_gate`, which is about the prompt section:
        # one decides whether the agent is told to run the command, this
        # decides what the command does when it is run.
        #
        # Routed through Terminus's own extra_env rather than the host's, since
        # `crux submit` runs inside the task container: setting it on the
        # runner would arm nothing and look like it had.
        if str(kwargs.pop("confirm_gate", "")).lower() in ("1", "true", "yes"):
            env = dict(kwargs.get("extra_env") or {})
            env["CRUX_SUBMIT_GATE"] = "1"
            kwargs["extra_env"] = env
        super().__init__(*args, **kwargs)
        if getattr(self, "_parser", None) is not None:
            _harden_parser(self._parser)
        if self._crux_tools:
            self._prompt_template = build_terminus_template(
                self._get_upstream_template(),
                scoring=True,
                submit=self._crux_submit,
                harness=self._crux_harness,
                edit=self._crux_edit,
                file_tools=self._crux_file_tools,
                batch=self._crux_batch,
                look=self._crux_look,
            )
        # Upstream sends no output cap at all, which leaves a thinking turn
        # unbounded. Measured on this deployment, `write-compressor` spent its
        # entire 900s budget on four turns averaging 19,500 completion tokens
        # and never reached a fifth; `dna-assembly` generated 168,829 tokens
        # across 19 turns and timed out the same way.
        #
        # Without a cap the truncation retry below is unreachable -- it fires
        # on finish_reason=length, and nothing can be truncated when nothing
        # is limited. Setting this is what arms it.
        if max_tokens is not None:
            self._llm_call_kwargs["max_tokens"] = int(max_tokens)
        # Qwen3.8's own knob for reasoning depth and cost. It defaults to
        # `xhigh`, the most expensive setting, and every run measured here has
        # been at that default without ever saying so.
        #
        # Measured on this deployment, 8 samples a level at temperature 1.0 on
        # one hard problem: xhigh hit a 12,000-token cap on all eight and would
        # have kept going, with a median 46,560 characters of chain of thought.
        # medium produced 3,944 tokens and 7,350 characters -- a third of the
        # generation and a sixth of the reasoning.
        #
        # The model card warns that lowering this can cost more than it saves in
        # multi-turn agentic work, through thinner analysis and more retries,
        # and an earlier test of `low` on the mini path reproduced exactly that.
        # `medium` sits between the untested default and the tested loss.
        if reasoning_effort is not None:
            body = dict(self._llm_call_kwargs.get("extra_body") or {})
            template_kwargs = dict(body.get("chat_template_kwargs") or {})
            template_kwargs["reasoning_effort"] = str(reasoning_effort)
            body["chat_template_kwargs"] = template_kwargs
            self._llm_call_kwargs["extra_body"] = body

    # Terminus assigns a fresh Chat as the second statement of every run(), so
    # a second call starts with no memory of the first. The shell does not --
    # the tmux session, its cwd, its environment and its background processes
    # all survive, because harbor reuses the agent instance across calls (its
    # own comment says so). Only the model's side of the continuity is missing.
    #
    # Seeding on assignment rather than patching harbor: the hook is the one
    # place the object is handed to us, and a property lives entirely in this
    # subclass. The alternative was reimplementing run(), which is 200 lines
    # that upstream owns and changes.
    @property
    def _chat(self):
        return getattr(self, "_chat_obj", None)

    @_chat.setter
    def _chat(self, chat) -> None:
        if chat is not None and getattr(self, "_carry_context", False) and self._carried:
            # Re-entering with prior turns means the template's preamble appears
            # again above them. That repetition is the price of not rewriting
            # upstream's prompt assembly, and it reads as a restatement of the
            # standing instructions rather than a contradiction.
            chat._messages.extend(self._carried)
        if chat is not None:
            _guard_context_deadlock(chat)
        self._chat_obj = chat

    def remember_turn(self) -> None:
        """Keep this turn's messages so the next run() continues from them."""
        chat = self._chat
        if self._carry_context and chat is not None:
            self._carried = list(chat.messages)

    def forget_context(self) -> None:
        """Drop the carried conversation. The shell state is untouched."""
        self._carried = []

    def _get_upstream_template(self) -> str:
        """Upstream's own template text, read fresh rather than copied.

        The crux sections are composed onto this in __init__. A frozen copy of
        the whole prompt used to ship in resources/; it matched upstream
        exactly, which is precisely the failure mode -- it would have gone on
        matching a prompt harbor had since changed, with nothing to notice.
        """
        return super()._get_prompt_template_path().read_text()

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        # The clock starts here rather than in run(): setup is once per trial
        # and run() is once per turn, and the budget harbor enforces covers the
        # whole trial.
        if not getattr(self, "_run_started", 0.0):
            self._run_started = time.monotonic()
        if not getattr(self, "_budget_sec", 0.0):
            self._budget_sec = _discover_budget(environment)
        # Before super(), which starts the tmux session and raises if tmux is
        # not there.
        await self._ensure_terminal_tools(environment)
        await self._route_env_around_old_tmux(environment)
        try:
            await super().setup(environment)
        except RuntimeError as exc:
            if "tmux session" not in str(exc):
                raise
            raise RuntimeError(f"{exc} {await self._why_tmux_failed(environment)}") from exc
        if self._crux_tools:
            await self._install_crux(environment)

    async def _ensure_terminal_tools(self, environment: BaseEnvironment) -> None:
        """Install tmux, and asciinema if it can be had, in images without them.

        Upstream already installs tmux when it is missing. It installs it in
        the same apt-get as asciinema:

            apt-get install -y tmux asciinema

        and on Debian 11 -- what the SWE-Atlas images are built on -- asciinema
        has no candidate, so the whole command fails, upstream reads that as
        "the package manager did not work", and falls back to building tmux
        from source in an image with no toolchain. That runs out its budget and
        the trial dies with "Failed to start tmux session. Error: None". All 124
        SWE-Atlas trials died there.

        So this is not "upstream forgot to install tmux". It is one apt-get
        carrying two packages, where the one that cannot be had takes down the
        one that can. Installing them separately is the whole fix: tmux is
        required, asciinema is not.

        Recording is a separate question. asciinema has no candidate in Debian
        11, which is what those images are built on, so it is attempted and then
        given up on: the .cast file is an artifact for a human to replay, and
        trajectory.json -- what every measurement in this project reads -- is
        written either way. Losing the recording is worth 124 tasks.

        The check comes first so this costs one `command -v` on the images that
        already have both, which is all of them except this one family.
        """
        probe = await environment.exec("command -v tmux >/dev/null 2>&1 && echo yes")
        if "yes" in _exec_text(probe):
            return

        # Whichever package manager the image has. `|| true` throughout: this
        # runs before the agent starts, so a failure here has to surface as the
        # tmux check below rather than as an exception from a shell command.
        await environment.exec(
            "sh -lc '"
            "if command -v apt-get >/dev/null 2>&1; then "
            "  apt-get update -qq >/dev/null 2>&1 || true; "
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null 2>&1 || true; "
            "elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux >/dev/null 2>&1 || true; "
            "elif command -v dnf >/dev/null 2>&1; then dnf install -y -q tmux >/dev/null 2>&1 || true; "
            "elif command -v yum >/dev/null 2>&1; then yum install -y -q tmux >/dev/null 2>&1 || true; "
            "fi' || true"
        )
        after = await environment.exec("command -v tmux >/dev/null 2>&1 && echo yes")
        if "yes" not in _exec_text(after):
            # super() is about to raise anyway; saying why first turns an empty
            # "Error: None" into something a person can act on.
            logger.warning(
                "tmux is missing from this image and could not be installed; "
                "the tmux session is about to fail to start"
            )
            return

        if not getattr(self, "_record_terminal_session", False):
            return
        await environment.exec(
            "sh -lc '"
            "command -v asciinema >/dev/null 2>&1 && exit 0; "
            "if command -v apt-get >/dev/null 2>&1; then "
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq asciinema >/dev/null 2>&1 || true; "
            "elif command -v apk >/dev/null 2>&1; then apk add --no-cache asciinema >/dev/null 2>&1 || true; "
            "fi; "
            "command -v asciinema >/dev/null 2>&1 || "
            "  (command -v pip3 >/dev/null 2>&1 && pip3 install -q asciinema >/dev/null 2>&1) || true"
            "' || true"
        )
        rec = await environment.exec("command -v asciinema >/dev/null 2>&1 && echo yes")
        if "yes" not in _exec_text(rec):
            logger.warning("asciinema unavailable in this image; running without a recording")
            self._record_terminal_session = False

    async def _why_tmux_failed(self, environment: BaseEnvironment) -> str:
        """The line upstream cannot print, because it reads the wrong stream.

        `Failed to start tmux session. Error: None` is not a case where tmux
        said nothing. harbor creates its docker exec with
        `stderr=asyncio.subprocess.STDOUT`, so `ExecResult.stderr` is always
        None and `ExecResult.stdout` holds everything -- and the message
        formats stderr. Every tmux failure in this corpus, on every task,
        reported `Error: None`: `unknown option -- e` and `command not found`
        are indistinguishable from the trial record.

        This re-runs the same start command and reports what it prints. Failing
        twice costs a few milliseconds against a trial that is already lost.
        """
        try:
            session = getattr(self, "_session", None)
            cmd = getattr(session, "_tmux_start_session", None) or "tmux -V"
            out = _exec_text(await environment.exec(cmd)).strip()
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not raise
            return f"(diagnosis failed: {type(exc).__name__}: {exc})"
        first = next((ln for ln in out.splitlines() if ln.strip()), "")
        return f"tmux said: {first[:200]}" if first else "(tmux printed nothing)"

    async def _route_env_around_old_tmux(self, environment: BaseEnvironment) -> None:
        """Deliver `extra_env` through the login shell when tmux has no `-e`.

        Upstream starts the session with `tmux new-session -e KEY=value`, and
        `-e` arrived in tmux 3.2. The two qemu images on this benchmark ship
        Debian 11's tmux 3.1c, which answers:

            tmux: unknown option -- e

        and exits 1. Upstream then raises `Failed to start tmux session. Error:
        None` -- None because harbor's docker exec is created with
        `stderr=asyncio.subprocess.STDOUT`, so `ExecResult.stderr` is
        structurally always None and the one line reporting the failure can
        never carry the reason for it.

        Cost of not knowing that: `confirm_gate` sets one variable, and across
        the corpus 434 trials ran with it. Every one of the 38 qemu trials among
        them died in setup, before the agent typed anything; the other 396 were
        untouched. The separation is exact -- confirm_gate on <-> qemu trial
        dead, 111 of 111 qemu trials either way -- which is what made a
        two-variable mechanism look like an environment flake for six days.

        `bash --login` is what the session runs, so `/etc/profile.d` reaches it
        without `-e`. The flag is feature-tested rather than version-compared:
        a probe session says what this build does, where "3.2 or newer" is a
        claim about a changelog.
        """
        env = dict(getattr(self, "_extra_env", None) or {})
        if not env:
            return

        probe = await environment.exec(
            "tmux -f /dev/null new-session -e CRUX_TMUX_PROBE=1 -d -s crux_env_probe "
            "true >/dev/null 2>&1 && "
            "{ tmux kill-session -t crux_env_probe >/dev/null 2>&1; echo yes; }"
        )
        if "yes" in _exec_text(probe):
            return

        exports = "\n".join(
            f"export {k}={shlex.quote(str(v))}" for k, v in sorted(env.items())
        )
        # harbor runs an exec through `bash -c`, so the heredoc needs no
        # wrapper of its own; one is a layer of quoting to get wrong.
        await environment.exec(
            "mkdir -p /etc/profile.d && cat > /etc/profile.d/crux-env.sh "
            "<<'CRUXENV'\n" + exports + "\nCRUXENV\n"
            "chmod 0644 /etc/profile.d/crux-env.sh"
        )

        # Whether profile.d is reached is a property of the image, not of tmux,
        # so it is checked rather than assumed: a login shell is asked to name
        # the variables back.
        names = " ".join(f"${{{k}:-}}" for k in sorted(env))
        seen = _exec_text(
            await environment.exec("bash -lc " + shlex.quote("echo " + names))
        )
        missing = [k for k, v in sorted(env.items()) if str(v) not in seen]
        if missing:
            logger.warning(
                "tmux here has no -e and /etc/profile.d did not reach the login "
                "shell; %s will be unset inside the session",
                ", ".join(missing),
            )
        # Clearing it is what keeps the session from being started with a flag
        # this tmux rejects. Anything that did not arrive is already reported.
        self._extra_env = {}

    async def _install_crux(self, environment: BaseEnvironment) -> None:
        """Put the crux helper on PATH inside the task container.

        The prompt tells the agent to finish with `crux submit`, so the binary
        has to be there whenever that instruction is present -- an earlier
        version gated the two separately and left 26 of 89 tasks running a
        command that did not exist.
        """
        await environment.upload_file(
            source_path=_RESOURCES / "crux_tool.py",
            target_path="/usr/local/bin/crux",
        )
        await environment.exec("chmod +x /usr/local/bin/crux")
        # An edit primitive. Without one, every modification this agent makes
        # goes through the shell: 791 heredoc rewrites and 54 `sed -i` calls
        # across an 89-task run, 845 edits with no way to change part of a file
        # except by retyping all of it or matching a pattern that may hit the
        # wrong line. The arm that outscored this one reached for a targeted
        # edit tool instead. apply_patch is already in this repo, measured at
        # 97.5% over 601 calls; it was wired into the older agent and never
        # into the one being scored.
        await environment.upload_file(
            source_path=_RESOURCES / "apply_patch.py",
            target_path="/usr/local/bin/apply_patch",
        )
        await environment.exec("chmod +x /usr/local/bin/apply_patch")

    @override
    def _stuck_notice(self) -> str:
        """One-time warning once a trial passes the point solves never reach.

        Reads its state through getattr so that a subclass, or a construction
        path that does not run this __init__, degrades to "no notice" rather
        than killing the trial with an AttributeError. The warning is worth
        having; it is not worth a crash.
        """
        steps = getattr(self, "_crux_steps", 0) + 1
        self._crux_steps = steps
        threshold = getattr(self, "_stuck_at", _STUCK_STEP_THRESHOLD)
        if not threshold or getattr(self, "_stuck_fired", False) or steps < threshold:
            return ""
        self._stuck_fired = True
        return _STUCK_NUDGE.format(steps=steps)

    def _guard_null_bytes(self, commands: list[Command]) -> str | None:
        """Refuse a batch carrying a raw NUL rather than letting it kill the trial.

        `filter-js-from-html` is an HTML sanitiser, so the agent tested null-byte
        scheme injection -- `<a href="java\x00script:alert(1)">` -- which is a
        real vector and exactly the right thing to try. The NUL reached the send
        path, which raised ValueError("embedded null byte"), and harbor recorded

            Trial filter-js-from-html__Rkjb5eM failed: embedded null byte

        The trial was scored zero for doing the task properly.

        Not once-only, unlike the two gates below: this is a property of the
        channel, not a habit worth interrupting once. If the model writes another
        NUL it needs telling again.
        """
        for i, command in enumerate(commands, start=1):
            if "\x00" in (command.keystrokes or ""):
                return _NUL_NOTE.format(i=i)
        return None

    def _account_checklist_first(self, commands: list[Command]) -> str | None:
        """Hold the first edit until one check is bound. Fires at most once.

        The gate below counts edits made since the last verification. This one
        is upstream of it and answers a different question: whether the thing
        being verified was ever going to be able to say no.

        Off unless asked for. It changes the order the agent works in, which is
        a larger claim than anything measured so far supports -- the timing is
        measured, the benefit is not.
        """
        if not getattr(self, "_checklist_first", False):
            return None
        if getattr(self, "_checklist_fired", False) or getattr(self, "_check_bound", False):
            return None
        edits = False
        for command in commands:
            keys = command.keystrokes or ""
            if _HELP_RE.search(keys):
                continue
            if _CHECK_BIND_RE.search(keys):
                # Binding and editing in one batch is the order being asked for;
                # let it through and stop watching.
                self._check_bound = True
                return None
            if _EDIT_RE.search(keys):
                edits = True
        if not edits:
            return None
        self._checklist_fired = True
        return _CHECKLIST_FIRST_NUDGE

    def _account_edit_debt(self, commands: list[Command]) -> str | None:
        """Track edits made since the last verification; interrupt once only.

        Fires at most once per run. The point is to break the pattern, not to
        police it: a run that is told twice is being argued with, and the second
        interruption costs a step without adding information.
        """
        # Read defensively: a subclass or a test double that never ran this
        # class's __init__ should get the old behaviour rather than an
        # AttributeError from a gate it did not ask for.
        limit = getattr(self, "_edit_debt_limit", 0)
        if limit <= 0 or getattr(self, "_edit_debt_fired", False):
            return None
        debt = getattr(self, "_edit_debt", 0)
        for command in commands:
            keys = command.keystrokes or ""
            if _HELP_RE.search(keys):
                continue
            if _VERIFY_RE.search(keys):
                debt = 0
            elif _EDIT_RE.search(keys):
                debt += 1
        self._edit_debt = debt
        if debt <= limit:
            return None
        self._edit_debt_fired = True
        n = debt
        self._edit_debt = 0
        return _EDIT_DEBT_NUDGE.format(n=n)

    @override
    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Upstream's "are you sure", with what the run has actually spent.

        This is the exact moment the measurement is about. Fraction of its own
        budget an agent had used when it stopped, by outcome:

            TB 2.1 baseline    solved 12.4%   failed 54.9%   3 of 27 ran out
            TB 2.1 fixed       solved 16.3%   failed 33.1%   3 of 17 ran out
            SWE-bench          solved  5.1%   failed 10.1%   0 of 3 ran out

        Most failures are not timeouts. They are voluntary stops with two
        thirds of the budget unspent, and on SWE-bench with nine tenths, taken
        on a green light the agent wrote for itself. Upstream already asks "are
        you sure" here and the trajectories show it being waved through -- a
        generic question earns a generic yes -- so what is added is the one
        thing the agent has never had: how much of its run is left.

        Degrades to elapsed-only when no budget was passed, because harbor
        enforces the timeout outside the agent and does not tell it the number.
        """
        base = super()._get_completion_confirmation_message(terminal_output)
        started = getattr(self, "_run_started", 0.0)
        if not started:
            return base
        elapsed = time.monotonic() - started
        budget = getattr(self, "_budget_sec", 0.0)
        share = ""
        if budget > 0:
            share = f", which is {elapsed / budget * 100:.0f}% of your budget"
        steps = getattr(self, "_crux_steps", 0) or getattr(self, "_turn_steps", 0)
        return base + _BUDGET_NOTICE.format(
            elapsed=_human_secs(elapsed), steps=steps, share=share
        )

    @override
    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """Same truncation, but it leaves a trace when it fires.

        The cap decides how much of a command's output the model ever sees, and
        it is a constant like every other one in this file -- 10,000 bytes,
        chosen upstream, never measured here. It is a candidate explanation for
        the one persistent structural difference against claude-code: on tasks
        crux loses it takes two to eight times the steps (`extract-elf` 221
        against 16, `mailman` 512 against 61), and needing a second command to
        see the rest of the first one's output would produce exactly that.

        The hypothesis was untestable. Terminal output is not kept in
        trajectory.json -- only the agent's own messages are -- so nothing in a
        finished run says whether this ever fired, and a search for the
        truncation marker across 10,730 steps in two runs finds zero because
        the text it looks for was never stored.

        So this counts. It changes no behaviour: the same bytes go to the model
        either way. What it adds is one line per truncation in the trial log,
        which makes the next run able to answer a question the last twenty
        could not.
        """
        out = super()._limit_output_length(output, max_bytes)
        if out is not output and len(out) != len(output):
            self._crux_truncations = getattr(self, "_crux_truncations", 0) + 1
            logger.info(
                "crux: terminal output truncated (%d bytes -> %d, cap %d); "
                "%d so far this trial",
                len(output.encode("utf-8")), len(out.encode("utf-8")), max_bytes,
                self._crux_truncations,
            )
        return out

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Let a long wait end when the command ends.

        Upstream always calls `send_keys(block=False, min_timeout_sec=duration)`,
        and the non-blocking path sleeps the full duration whether or not the
        command is still running. The blocking path exists in TmuxSession -- it
        appends a tmux completion signal and returns the moment the command
        finishes -- and nothing calls it.

        The model is told to prefer short durations and it mostly does, but on
        the turns where it decides "this may take a while" the difference is the
        task. `adaptive-rejection-sampler` said exactly that before an apt
        install, then sat for 633 seconds of a 900-second budget on an install
        measured at 31, generated 596 tokens in total, and timed out. Across the
        run, timed-out trials spent 35% of their elapsed time neither generating
        tokens nor producing terminal output.

        Only long waits are converted. Below the threshold the fixed sleep costs
        little and upstream's semantics -- partial output, model polls again --
        are what the prompt describes, so they are left alone. `_prepare_keys`
        already falls back to non-blocking when the keystrokes do not execute a
        command, which is what keeps this safe for entering a REPL or an ssh
        session.
        """
        note = (
            self._guard_null_bytes(commands)
            or self._account_checklist_first(commands)
            or self._account_edit_debt(commands)
        )
        if note is not None:
            # Returning without executing is the same shape as a refused command,
            # which the prompt already handles: the model reads the reason and
            # picks the next action itself.
            return False, note

        for command in commands:
            duration = command.duration_sec
            try:
                if duration >= _BLOCKING_THRESHOLD_SEC and _can_block(
                    command.keystrokes
                ):
                    await session.send_keys(
                        command.keystrokes,
                        block=True,
                        max_timeout_sec=duration,
                    )
                else:
                    await session.send_keys(
                        command.keystrokes,
                        block=False,
                        min_timeout_sec=duration,
                    )
            except TimeoutError:
                return True, self._timeout_template.format(
                    timeout_sec=duration,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        await session.get_incremental_output()
                    ),
                )

        # Appended to the terminal output the model reads next, so the warning
        # arrives in the same channel as everything else it reasons about.
        return False, self._stuck_notice() + self._limit_output_length(
            await session.get_incremental_output()
        )

    @override
    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> LLMResponse:
        """Retry a length-truncated turn with thinking switched off.

        Upstream handles truncation by telling the model it exceeded the output
        limit and asking it to "break the request into chunks", then reissuing
        the same call unchanged. That advice assumes the model ran long while
        writing commands. Here it runs long while *thinking*: on puzzle-shaped
        tasks this model produced single turns of 17,000-23,000 completion
        tokens that never reached the `<response>` block at all, so there is
        nothing to chunk and nothing for the salvage path to recover.

        Reissuing under identical conditions then reproduces the spiral. One
        trial spent 17 minutes on such a turn, 29 minutes on the retry, and
        executed zero commands across both -- out of a 3600s budget.

        So the retry drops `enable_thinking`. The model has already done the
        reasoning; what it failed to do is emit a command. Forcing a direct
        answer turns a wasted turn into a cheap one, and the transcript still
        carries the truncated reasoning for the turn after.
        """
        # Upstream recurses into _query_llm to retry a truncated turn, so this
        # override is re-entered for the retry. Depth is what distinguishes the
        # two: the first entry is the ordinary call, anything deeper is a retry
        # after truncation.
        depth = getattr(self, "_crux_truncation_depth", 0)
        saved = dict(self._llm_call_kwargs)
        if depth:
            body = dict(saved.get("extra_body") or {})
            template_kwargs = dict(body.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = False
            body["chat_template_kwargs"] = template_kwargs
            self._llm_call_kwargs["extra_body"] = body

        # litellm reads `timeout` out of the call kwargs. Set here rather than
        # once at construction because upstream clears and restores this dict
        # around the truncation retry, so a value written earlier would not
        # survive into the retry -- which is exactly the call most likely to
        # hang.
        t = getattr(self, "_llm_timeout", 0.0)
        if t == 0:
            # The budget is not known until setup() has seen the environment,
            # so the ceiling is derived here rather than in __init__.
            t = _llm_timeout_for(getattr(self, "_budget_sec", 0.0))
        if t > 0:
            self._llm_call_kwargs["timeout"] = t

        self._crux_truncation_depth = depth + 1
        try:
            return await super()._query_llm(
                chat=chat,
                prompt=prompt,
                original_instruction=original_instruction,
                session=session,
            )
        finally:
            self._crux_truncation_depth = depth
            self._llm_call_kwargs.clear()
            self._llm_call_kwargs.update(saved)
