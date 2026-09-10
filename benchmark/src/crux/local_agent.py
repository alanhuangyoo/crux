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
        on_event=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._approval = Approval(approval)
        self._project_root = Path(project_root or Path.cwd()).resolve()
        self._interactive = interactive
        self._confirm = confirm or _prompt_yes_no
        self._blocked: list[tuple[str, str]] = []
        # Where the terminal front end listens. A benchmark trial passes
        # nothing and the calls become no-ops, which keeps the scored path and
        # the interactive path the same code -- the requirement this project
        # started from.
        self._on_event = on_event or (lambda *a, **k: None)
        self._turn_commands = 0
        self._turn_steps = 0

    async def _claim_session_name(self, environment) -> None:
        """Take a free tmux session name, so a second window is not blocked.

        Upstream names the session `self.name()`, a constant, so a second
        `crux repl` fails with "duplicate session: crux-local" before its first
        turn. Two terminals open on two projects is ordinary use, not an edge
        case.

        The name stays exactly `crux-local` whenever it is free, so a lone
        session is still findable by `tmux attach -t crux-local`; only a
        genuine collision gets a suffix. Assigning it on the instance shadows
        the staticmethod for `self.name()` without touching `cls.name()`,
        which harbor calls unbound on its handoff path.
        """
        base = type(self).name()
        probe = await environment.exec("tmux ls -F '#{session_name}' 2>/dev/null")
        taken = set((probe.stdout or "").split())
        if base not in taken:
            return
        for n in range(2, 100):
            candidate = f"{base}-{n}"
            if candidate not in taken:
                self.name = lambda c=candidate: c
                return
        raise RuntimeError(f"no free tmux session name for {base}")

    async def setup(self, environment) -> None:
        """Install the helpers, then put them on the shell's PATH.

        The base class installs `crux` and `apply_patch` at container-absolute
        paths, and LocalEnvironment redirects those into the session directory
        so a laptop run needs no root. That redirect is invisible to the shell
        the agent actually types into -- it drives tmux directly rather than
        going through exec -- so the directory has to be exported there too, or
        the prompt tells the model to run a command the shell cannot find.
        """
        await self._claim_session_name(environment)
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
        import time

        # Read defensively for the same reason the gates upstream do: a test
        # double or a subclass that never ran this __init__ must degrade to
        # "no display" rather than to an AttributeError. The display is worth
        # having; it is not worth a crash.
        emit = getattr(self, "_on_event", None) or (lambda *a, **k: None)
        self._turn_steps = getattr(self, "_turn_steps", 0) + 1
        allowed = []
        refusals = []
        for command in commands:
            d = self._gate(command.keystrokes)
            if d.verdict is Verdict.BLOCK:
                self._blocked.append((command.keystrokes, d.reason))
                emit("refused", keystrokes=command.keystrokes, reason=d.reason)
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
            for command in allowed:
                self._turn_commands = getattr(self, "_turn_commands", 0) + 1
                emit("command", n=self._turn_commands, keystrokes=command.keystrokes)
            started = time.monotonic()
            timeout, output = await super()._execute_commands(allowed, session)
            emit("result", elapsed=time.monotonic() - started,
                 output=output, timed_out=timeout)
        if refusals:
            output = (output + "\n" + "\n".join(refusals)).strip()
        return timeout, output

    async def _query_llm(self, *args, **kwargs):
        """Announce what the model said before its commands start running.

        Terminus puts the model's own account of the turn in the `Analysis:`
        prefix, and until now nothing displayed it: the screen went from the
        prompt straight to the answer, with the reasoning only reachable by
        reading the trajectory afterwards.
        """
        response = await super()._query_llm(*args, **kwargs)
        emit = getattr(self, "_on_event", None)
        if emit is None:
            return response
        try:
            emit("analysis", text=getattr(response, "content", "") or "")
            usage = getattr(response, "usage", None) or {}
            if usage:
                emit("usage", usage=dict(usage))
        except Exception:  # noqa: BLE001 - display must never break a turn
            pass
        return response

    def begin_turn(self) -> None:
        """Reset the per-turn counters the front end reports."""
        self._turn_commands = 0
        self._turn_steps = 0
        self._blocked.clear()

    @property
    def turn_stats(self) -> tuple[int, int]:
        return getattr(self, "_turn_steps", 0), getattr(self, "_turn_commands", 0)

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
