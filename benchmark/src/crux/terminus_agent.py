"""Crux on the Terminus scaffold.

Terminus drives a live tmux session and sends keystrokes, where mini-swe-agent
runs one shell command per turn: that decides whole categories of task (an
interactive ssh session into a VM cannot be expressed as one-shot commands),
and lets one turn poll a slow build with individual waits instead of model
round trips.

This subclasses Terminus and adds the all-or-nothing scoring instruction and
`crux submit`, which turns finishing into a re-run of every bound check. The
format contract, the tmux handling and the summarisation are upstream's.
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
# is already close enough.
_BLOCKING_THRESHOLD_SEC = 10.0

# Edits allowed since the last verification before the agent is nudged to check
# its work. A long run of edits with no check between them tends to break earlier
# changes with later ones, with no way to tell which; twelve was chosen by
# replaying the nudge over a full run's trajectories.
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

    harbor enforces the agent timeout outside the agent and never passes the
    number in, but the trial's config.json carries the multiplier and the task
    package it names carries the base:

        <trial>/config.json           agent_timeout_multiplier, task.name
        <harbor cache>/.../task.toml  [agent] timeout_sec

    Every step is wrapped: returns 0 when the budget cannot be worked out, and the
    notice degrades to elapsed time only.
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
    None; callers here only look for a sentinel word, so the two are read as one.
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

# A ceiling on one response, so a runaway generation cannot run to the context
# limit and take the trial with it; the truncation retry, which drops thinking,
# recovers a turn that hits it.
#
# The step-count nudge (`_STUCK_STEP_THRESHOLD`) is off by default; enable it with
# `stuck_step_threshold=<n>`.
_MAX_OUTPUT_TOKENS = CALIBRATION_MAX_OUTPUT_TOKENS

# A ceiling on one model call: litellm's default is 6000s and harbor never
# overrides it, so a hung request would hold its trial. Scaled to the budget (see
# calibration.py, which imports nothing so the launch-time audit can check it on
# a bare interpreter): a hang is bounded by a quarter of the trial, and a long but
# legitimate generation is not cut off.
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

    When the context overflows, upstream summarizes; if that call comes back
    truncated, the fallback string it substitutes has no <response> tag, the parser
    rejects it, and the agent is asked again with the same oversized context.
    Every exit upstream has goes through another call on that context, so this
    trims it -- after the second occurrence, so a single transient failure still
    gets its ordinary retry.
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
        # The failure arrives as an exception, not a value: upstream's summarization
        # fallback substitutes its fixed string inside its own except block.
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
    """Two parser faults that would otherwise end a trial.

    **An unclosed <commands> section.** The model can write `<commands>`, its
    `<keystrokes>` blocks and `</response>` without ever closing `</commands>`; the
    parser reports the section as missing, and the model repeats the same omission.
    Upstream already repairs the same shape for `</response>` through
    `_get_auto_fixes`, so this adds the equivalent for `</commands>` through that
    hook.

    **A whitespace-only tag.** Upstream's tag-name extraction raises IndexError on
    `<  >`, which escapes the agent loop. This skips the malformed tag and keeps the
    rest of the response.
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
        # Whether the model's own reasoning is handed back to it on the next turn.
        # Off by default: returning it doubles the context per step and did not shorten
        # the reasoning. `--agent-kwarg interleaved_thinking=1` turns it on.
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
        # Separable so the harness section can be selected on its own.
        self._crux_harness = str(kwargs.pop("harness_section", True)).lower() not in (
            "false", "0", "no",
        )
        # Separable from installing the binary: whether apply_patch is on PATH
        # and whether the model is told to use it are two claims, and an
        # ablation needs to move one at a time.
        self._crux_edit = str(kwargs.pop("edit_section", True)).lower() not in (
            "false", "0", "no",
        )
        # Name the five installed file tools in the Terminus prompt. Off by default;
        # `--agent-kwarg file_tools=1` turns it on.
        self._crux_file_tools = str(kwargs.pop("file_tools", False)).lower() in (
            "1", "true", "yes",
        )
        # Ask for several probes chained into one command. Off by default;
        # `--agent-kwarg batch_section=1` turns it on.
        self._crux_batch = str(kwargs.pop("batch_section", False)).lower() in (
            "1", "true", "yes",
        )
        # Ask the agent to open on the data rather than on a listing. Off by default.
        self._crux_look = str(kwargs.pop("look_section", False)).lower() in (
            "1", "true", "yes",
        )
        # Carry the conversation across run() calls, for the interactive path. Off by
        # default: the benchmark scores one instruction per trial with a fresh chat.
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
        # Wall-clock budget in seconds, when the caller knows it; harbor enforces the
        # timeout outside the agent and does not pass it in.
        try:
            self._budget_sec = float(kwargs.pop("budget_sec", 0) or 0)
        except (TypeError, ValueError):
            self._budget_sec = 0.0
        self._run_started = 0.0
        self._stuck_fired = False
        self._crux_steps = 0
        # Whether `crux submit` demands a second pass before it will finish -- what the
        # command does when run, as opposed to `submit_gate`, which decides whether the
        # agent is told to run it. Routed through Terminus's extra_env, since `crux
        # submit` runs inside the task container.
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
        # Upstream sends no output cap, which leaves a thinking turn unbounded and the
        # truncation retry below unreachable (it fires on finish_reason=length). Setting
        # this is what arms it.
        if max_tokens is not None:
            self._llm_call_kwargs["max_tokens"] = int(max_tokens)
        # Qwen3.8's knob for reasoning depth and cost; it defaults to `xhigh`, the most
        # expensive setting. Lowering it trades generation time for thinner analysis in
        # multi-turn agentic work, so it is explicit and selectable.
        if reasoning_effort is not None:
            body = dict(self._llm_call_kwargs.get("extra_body") or {})
            template_kwargs = dict(body.get("chat_template_kwargs") or {})
            template_kwargs["reasoning_effort"] = str(reasoning_effort)
            body["chat_template_kwargs"] = template_kwargs
            self._llm_call_kwargs["extra_body"] = body

    # Terminus assigns a fresh Chat at the start of every run(), so a second call
    # starts with no memory of the first, while the tmux session survives because
    # harbor reuses the agent instance. Seeding the chat on assignment keeps the
    # continuity in this subclass instead of reimplementing upstream's run().
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

        The Crux sections are composed onto it in __init__, so a change upstream is
        picked up rather than silently diverged from.
        """
        return super()._get_prompt_template_path().read_text()

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        # The clock starts at setup, not in run(): the budget harbor enforces covers
        # the whole trial, and run() is once per turn.
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

        Upstream installs both in one apt-get; on Debian 11 asciinema has no
        candidate, so the whole command fails and tmux is never installed. Installing
        them separately fixes that: tmux is required, asciinema (a recording for human
        replay) is not, and trajectory.json is written either way. The check comes
        first, so images that already have both pay one `command -v`.
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

        harbor's docker exec merges stderr into stdout, so `ExecResult.stderr` is always
        None and "Failed to start tmux session. Error: None" carries no reason. This
        re-runs the same start command and reports what it prints.
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

        Upstream starts the session with `tmux new-session -e KEY=value`, and `-e`
        arrived in tmux 3.2; on tmux 3.1c the session fails to start. The session runs
        `bash --login`, so `/etc/profile.d` reaches it without `-e`. The flag is
        feature-tested with a probe session rather than inferred from a version
        number.
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

        The prompt tells the agent to finish with `crux submit`, so the binary is
        installed whenever that instruction is present.
        """
        await environment.upload_file(
            source_path=_RESOURCES / "crux_tool.py",
            target_path="/usr/local/bin/crux",
        )
        await environment.exec("chmod +x /usr/local/bin/crux")
        # An edit primitive: without one every modification goes through the shell --
        # heredoc rewrites or `sed -i` patterns that may hit the wrong line. apply_patch
        # locates hunks by context and fails loudly.
        await environment.upload_file(
            source_path=_RESOURCES / "apply_patch.py",
            target_path="/usr/local/bin/apply_patch",
        )
        await environment.exec("chmod +x /usr/local/bin/apply_patch")

    @override
    def _stuck_notice(self) -> str:
        """One-time warning once a trial passes the point solves usually do not reach.

        Reads its state through getattr, so a construction path that does not run this
        __init__ degrades to "no notice" rather than an AttributeError.
        """
        steps = getattr(self, "_crux_steps", 0) + 1
        self._crux_steps = steps
        threshold = getattr(self, "_stuck_at", _STUCK_STEP_THRESHOLD)
        if not threshold or getattr(self, "_stuck_fired", False) or steps < threshold:
            return ""
        self._stuck_fired = True
        return _STUCK_NUDGE.format(steps=steps)

    def _guard_null_bytes(self, commands: list[Command]) -> str | None:
        """Refuse a batch carrying a raw NUL rather than letting it end the trial.

        A NUL in the keystrokes -- from testing a null-byte injection, say -- makes the
        send path raise. The agent is told instead, every time it happens: this is a
        property of the channel, not a habit worth interrupting once.
        """
        for i, command in enumerate(commands, start=1):
            if "\x00" in (command.keystrokes or ""):
                return _NUL_NOTE.format(i=i)
        return None

    def _account_checklist_first(self, commands: list[Command]) -> str | None:
        """Hold the first edit until one check is bound. Fires at most once.

        Answers whether the thing being verified was ever able to say no. Off unless
        asked for, since it changes the order the agent works in.
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

        Upstream already asks here, and a generic question tends to earn a generic yes;
        what this adds is the one thing the agent cannot see: how much of its run is
        left. Degrades to elapsed time only when no budget was passed.
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

        Terminal output is not kept in trajectory.json, so nothing in a finished run
        would say whether output was ever cut. This logs one line per truncation and
        changes no behaviour: the same bytes go to the model either way.
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

        Upstream always sends keys non-blocking with a fixed sleep, which waits the full
        duration whether or not the command is still running; the blocking path exists
        in TmuxSession and returns as soon as the command finishes. Only long waits are
        converted, and `_prepare_keys` already falls back to non-blocking when the
        keystrokes do not run a command, which keeps a REPL or ssh session safe.
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

        Upstream reissues a truncated turn unchanged, asking the model to break its
        request into chunks -- advice for a turn that ran long writing commands. When it
        ran long while thinking, there is nothing to chunk and the retry repeats the
        spiral. The model has already done the reasoning; turning thinking off for the
        retry makes it emit the command, and the transcript keeps the truncated
        reasoning for the turn after.
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
