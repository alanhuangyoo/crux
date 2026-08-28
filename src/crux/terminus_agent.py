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

# Above this, a wait is worth completing early; below it, upstream's fixed sleep
# is already close enough that the tmux round trip is not worth the change in
# behaviour. Set from the measured waste: the losses are all long waits.
_BLOCKING_THRESHOLD_SEC = 10.0

_RESOURCES = Path(__file__).parent / "resources"


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
        super().__init__(*args, **kwargs)
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

    @override
    def _get_prompt_template_path(self) -> Path:
        if not self._crux_tools:
            return super()._get_prompt_template_path()
        return _RESOURCES / "terminus_crux.txt"

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
                if duration >= _BLOCKING_THRESHOLD_SEC:
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

        return False, self._limit_output_length(
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
