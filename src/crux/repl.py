"""A conversation with the agent the benchmark scores.

`crux solve` runs one instruction to completion and exits. That is what a
benchmark trial is, and it is not what using a tool is like: the second thing
you want to say is almost always shaped by what just happened.

Two kinds of continuity are needed, and only one was missing. The shell already
survives -- harbor reuses the agent instance across `run()` calls, so the tmux
session keeps its cwd, its environment and its background processes. What did
not survive was the model's side: Terminus assigns a fresh Chat as the second
statement of every run(). `CruxTerminusAgent.carry_context` closes that, so this
is a loop around `run()` rather than a reimplementation of it.

Deliberately not a TUI. A prompt, streamed output, and Ctrl-C that interrupts
the turn instead of the process is what the work needs; a full-screen interface
is a different project and pi already has a good one.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BANNER = """\
crux {version}  {agent}
  {env}
  approval={approval}{tty}

  /new     forget the conversation (the shell keeps its state)
  /shell   what the terminal session is
  /exit    leave
"""


def _read(prompt: str) -> str | None:
    """One line from the user, or None on EOF."""
    try:
        return input(prompt)
    except EOFError:
        print()
        return None
    except KeyboardInterrupt:
        # At the prompt, Ctrl-C clears the line rather than leaving; leaving is
        # /exit or EOF, so an interrupt during a long turn cannot exit by
        # accident when it arrives a moment late.
        print("^C")
        return ""


def cmd_repl(args) -> int:
    """Talk to the agent, one turn at a time, in a directory."""
    import os

    try:
        from harbor.models.agent.context import AgentContext
    except ImportError:
        print(
            "harbor is not installed. The agent measured by this project is a\n"
            "Terminus derivative and needs it:\n"
            "  uv tool install 'harbor[modal]'",
            file=sys.stderr,
        )
        return 1

    from crux import __version__
    from crux.chat import _endpoint
    from crux.endpoint import ensure_reachable
    from crux.approval import Approval
    from crux.local_agent import LocalCruxAgent
    from crux.local_env import LocalEnvironment

    base, _, default_model = _endpoint()
    if base and not ensure_reachable(base):
        print(
            f"cannot reach the model at {base}.\n"
            "If it is served on an eval box, set CRUX_TUNNEL=<ssh host> in\n"
            "~/.crux/env and this will open the forward itself.",
            file=sys.stderr,
        )
        return 1

    cwd = Path(args.cwd or os.getcwd()).resolve()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    approval = Approval(args.approval)

    env = LocalEnvironment(cwd=cwd)
    agent = LocalCruxAgent(
        logs_dir=Path(env.trial_paths.agent_dir),
        model_name=args.model or f"openai/{default_model}",
        approval=approval,
        project_root=cwd,
        interactive=interactive,
        carry_context=True,
        **({"variant": args.variant} if args.variant != "default" else {}),
    )

    print(
        _BANNER.format(
            version=__version__,
            agent=agent.name(),
            env=env.describe(),
            approval=approval.value,
            tty="" if interactive else "  (no tty: anything needing a prompt is refused)",
        )
    )

    async def turn(text: str) -> None:
        await agent.run(text, env, AgentContext())
        agent.remember_turn()

    async def main() -> int:
        await env.start()
        # setup() is what creates the tmux session Terminus drives; run() raises
        # "Session is not set" without it. In a benchmark trial harbor calls it,
        # so it is easy to leave out of a hand-written loop -- and it fails on
        # the first turn rather than at import, which is how it got missed here.
        await agent.setup(env)
        turns = 0
        try:
            while True:
                line = _read("crux> ")
                if line is None or line.strip() in ("/exit", "/quit"):
                    return 0
                text = line.strip()
                if not text:
                    continue
                if text == "/new":
                    agent.forget_context()
                    turns = 0
                    print("conversation cleared; the shell keeps its state")
                    continue
                if text == "/shell":
                    print(f"  {env.describe()}")
                    continue
                try:
                    await turn(text)
                    turns += 1
                except KeyboardInterrupt:
                    # The turn is abandoned but the session is not: whatever the
                    # agent already did to the working directory stands, and
                    # saying so is more useful than pretending it rolled back.
                    print("\n^C  turn interrupted; the shell keeps whatever it did")
                    agent.remember_turn()
                except Exception as exc:  # noqa: BLE001 - a bad turn must not end the session
                    print(f"\nturn failed: {exc}", file=sys.stderr)
                for command, reason in agent.blocked_commands:
                    print(f"refused: {command.strip()}  ({reason})", file=sys.stderr)
        finally:
            await env.stop()

    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
