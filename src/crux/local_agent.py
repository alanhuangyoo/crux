"""The benchmarked agent, running on this machine.

`crux solve` shelled out to mini-swe-agent while every number in this repo comes
from the Terminus base -- so the CLI shipped the scaffold the measurements
rejected. On the same tasks the Terminus base scores about 3 points higher, and
the one-command-per-turn base cannot express entering an ssh session or a REPL
at all, which makes some tasks unsolvable by construction.

`LocalEnvironment` and `approval` were written to close that gap and then never
wired to anything. This is the wiring: the benchmarked agent, the benchmarked
prompt, pointed at a local shell, with the one thing a benchmark does not need
and a CLI cannot do without -- a gate in front of commands that are expensive
to get wrong.
"""

from __future__ import annotations

import shlex

import logging
from pathlib import Path

from crux.approval import Approval, Decision, Verdict, decide
from crux.terminus_agent import CruxTerminusAgent

log = logging.getLogger("crux.local")


class CommandBlocked(Exception):
    """Raised when the gate refuses a command and the run cannot continue."""


class LocalCruxAgent(CruxTerminusAgent):
    """CruxTerminusAgent with an approval gate in front of the keystrokes.

    The gate sits in `_execute_commands` rather than in `LocalEnvironment.exec`
    because Terminus does not run commands through `exec` -- it types them into
    a live tmux session. Gating `exec` would look like a safety boundary and
    stop nothing.

    A refused command is reported back to the model as terminal output rather
    than raising, so the agent can choose a different approach. That is the
    same shape as a command that fails, which the prompt already handles.
    """

    def __init__(
        self,
        *args,
        approval: Approval | str = Approval.DANGEROUS,
        project_root: Path | str | None = None,
        interactive: bool = True,
        confirm=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._approval = Approval(approval)
        self._project_root = Path(project_root or Path.cwd()).resolve()
        self._interactive = interactive
        self._confirm = confirm or _prompt_yes_no
        self._blocked: list[tuple[str, str]] = []

    async def setup(self, environment) -> None:
        """Install the helpers, then put them on the shell's PATH.

        The base class installs `crux` and `apply_patch` at container-absolute
        paths, and LocalEnvironment redirects those into the session directory
        so a laptop run needs no root. That redirect is invisible to the shell
        the agent actually types into -- it drives tmux directly rather than
        going through exec -- so the directory has to be exported there too, or
        the prompt tells the model to run a command the shell cannot find.
        """
        await super().setup(environment)
        bin_dir = getattr(environment, "bin_dir", None)
        if bin_dir is None or self._session is None:
            return
        await self._session.send_keys(
            [f"export PATH={shlex.quote(str(bin_dir))}:$PATH", "Enter"],
            block=False,
            min_timeout_sec=0.2,
        )

    @staticmethod
    def name() -> str:
        return "crux-local"

    def _gate(self, keystrokes: str) -> Decision:
        """Allow, ask, or refuse -- and treat a declined prompt as a refusal."""
        d = decide(keystrokes, self._project_root, self._approval, self._interactive)
        if d.verdict is Verdict.ASK:
            if self._confirm(keystrokes, d.reason):
                return Decision(Verdict.ALLOW, d.reason)
            return Decision(Verdict.BLOCK, "declined by the user")
        return d

    async def _execute_commands(self, commands, session):
        allowed = []
        refusals = []
        for command in commands:
            d = self._gate(command.keystrokes)
            if d.verdict is Verdict.BLOCK:
                self._blocked.append((command.keystrokes, d.reason))
                refusals.append(
                    f"crux refused to run: {command.keystrokes.strip()}\n"
                    f"  reason: {d.reason}"
                )
                # Everything after a refused command was planned on the
                # assumption it ran, so the rest of the turn is discarded too.
                break
            allowed.append(command)

        timeout = False
        output = ""
        if allowed:
            timeout, output = await super()._execute_commands(allowed, session)
        if refusals:
            output = (output + "\n" + "\n".join(refusals)).strip()
        return timeout, output

    @property
    def blocked_commands(self) -> list[tuple[str, str]]:
        return list(self._blocked)


def _prompt_yes_no(command: str, reason: str) -> bool:
    """Ask on the terminal. Anything but an explicit yes is a no."""
    print(f"\ncrux wants to run:\n  {command.strip()}\n  ({reason})")
    try:
        answer = input("  allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")
