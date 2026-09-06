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

import re

from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.llms.base import LLMResponse
from harbor.agents.terminus_2.tmux_session import TmuxSession
from typing_extensions import override

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

_EDIT_DEBT_NUDGE = """\
You have made {n} edits without running anything that checks them.

Measured on this benchmark: trials that failed edited 17 times and checked 3;
trials that solved edited 6 and checked 6. The failing shape is not too few
edits, it is edits nobody checked -- by now the earlier changes have usually
been broken by the later ones and there is no way to tell which.

Run something that answers "does this work" before editing again: the checker
the task ships, its test file, `crux todo verify`, or the smallest command that
executes the code you changed. Then continue.
"""

_STUCK_STEP_THRESHOLD = 120


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
        max_tokens = kwargs.pop("max_tokens", None)
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
        await super().setup(environment)
        if self._crux_tools:
            await self._install_crux(environment)

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
        note = self._account_edit_debt(commands)
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
