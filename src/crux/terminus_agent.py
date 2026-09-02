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


def _guard_whitespace_tags(parser):
    """Stop a whitespace-only tag from taking the whole trial down.

    Upstream's XML parser extracts a tag name with

        tag_name = tag_content.split()[0] if " " in tag_content else tag_content

    which raises IndexError when the tag holds nothing but whitespace: `" " in
    "  "` is true, and `"  ".split()` is empty. The exception escapes
    parse_response, escapes the agent loop, and the trial scores zero -- for a
    model emitting `<  >` once in a turn.

    Measured here: one trial in 61 on Terminal-Bench 2.1, so roughly 1.5 tasks
    per full run, lost to a malformed tag rather than to a wrong answer.

    Wrapping rather than patching the module: the fix belongs upstream, and a
    monkeypatch on an import would silently apply to any other agent in the same
    process. A whitespace-only tag has no name, so it is dropped, which is what
    the surrounding code does with anything it cannot identify.
    """
    original = parser._find_top_level_tags

    def _safe(content):
        try:
            return original(content)
        except IndexError:
            return []

    parser._find_top_level_tags = _safe
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
        # Steps past which the agent is told its approach has failed. 0 disables.
        self._stuck_at = int(kwargs.pop("stuck_step_threshold", _STUCK_STEP_THRESHOLD))
        self._stuck_fired = False
        self._crux_steps = 0
        super().__init__(*args, **kwargs)
        if getattr(self, "_parser", None) is not None:
            _guard_whitespace_tags(self._parser)
        if self._crux_tools:
            self._prompt_template = build_terminus_template(
                self._get_upstream_template(),
                scoring=True,
                submit=self._crux_submit,
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
