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

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from typing_extensions import override

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
        super().__init__(*args, **kwargs)

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
