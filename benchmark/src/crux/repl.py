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

Deliberately not a full-screen TUI. Output is appended, never repainted, so a
session survives a pipe, a scrollback search and a 40-column pane. What it does
have is everything needed to not open a second terminal: the commands as they
run, the model's own account of why, the counters, and history that outlives
the process.

The one hard rule this file obeys is the project's: what runs here is the same
class the benchmark scores. Every addition below is a display or an input
convenience. None of them changes what the agent does.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

from crux import ui

_HOME = Path(os.environ.get("CRUX_HOME", Path.home() / ".crux"))
_HISTORY = _HOME / "history"
_SESSIONS = _HOME / "sessions"

# Kept small on purpose: readline reloads the whole file at startup, and a
# multi-megabyte history makes `crux repl` visibly slow to open.
_HISTORY_MAX = 5000

_HELP = """\
  /help              this
  /new               forget the conversation (the shell keeps its state)
  /shell             what the terminal session is, and how to attach to it
  /cd <path>         change the project root the approval gate protects
  /model [name]      show or switch the model for the next turn
  /approval [level]  show or set the gate: safe | ask | dangerous
  /tokens            what this session has spent
  /last              the last turn's commands in full
  /save [name]       write the conversation to ~/.crux/sessions
  /resume [name]     load one back (or list them with no name)
  /retry             send the previous message again
  /verbose           toggle full command text and output echo
  /exit              leave

  Ctrl-D leaves. Ctrl-C at the prompt clears the line; during a turn it
  abandons the turn and keeps the shell.

  Input:
    end a line with \\        continue on the next line
    \"\"\"                      open or close a multi-line block
    @path/to/file            inline that file into the message
"""

_SLASH = [
    "/help", "/new", "/shell", "/cd", "/model", "/approval", "/tokens",
    "/last", "/save", "/resume", "/retry", "/verbose", "/exit", "/quit",
]


# --------------------------------------------------------------------------
# input


def _install_readline() -> None:
    """History that outlives the process, and completion for slashes and paths.

    Wrapped because readline is absent on some Windows Pythons and present but
    unusable under some IDE consoles. Losing history is a worse session, not a
    broken one.
    """
    try:
        import readline
    except ImportError:
        return
    try:
        _HOME.mkdir(parents=True, exist_ok=True)
        if _HISTORY.exists():
            readline.read_history_file(str(_HISTORY))
        readline.set_history_length(_HISTORY_MAX)
    except OSError:
        pass

    def complete(text: str, state: int):
        buf = readline.get_line_buffer()
        if buf.lstrip().startswith("/") and " " not in buf.strip():
            opts = [c for c in _SLASH if c.startswith(text)]
        else:
            # Path completion, including after an @ reference.
            stem = text[1:] if text.startswith("@") else text
            prefix = "@" if text.startswith("@") else ""
            d = Path(stem).expanduser()
            base, frag = (d, "") if stem.endswith("/") else (d.parent, d.name)
            try:
                names = sorted(p.name for p in base.iterdir() if p.name.startswith(frag))
            except OSError:
                names = []
            opts = [
                prefix + str(base / n) + ("/" if (base / n).is_dir() else "")
                for n in names
            ]
            if str(base) == ".":
                opts = [o.replace("./", "", 1) for o in opts]
        return opts[state] if state < len(opts) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")


def _save_readline() -> None:
    try:
        import readline

        _HOME.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(_HISTORY))
    except Exception:  # noqa: BLE001 - never fail a session over history
        pass


def _read_block(prompt: str, cont: str) -> str | None:
    """One message, which may span lines.

    Three ways in, because the three things people paste have different shapes:
    a trailing backslash for a deliberate continuation, a `\"\"\"` fence for a
    block of prose or code, and otherwise a single line.
    """
    try:
        line = input(prompt)
    except EOFError:
        print()
        return None
    except KeyboardInterrupt:
        print("^C")
        return ""

    if line.strip() == '"""':
        lines: list[str] = []
        while True:
            try:
                nxt = input(cont)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if nxt.strip() == '"""':
                break
            lines.append(nxt)
        return "\n".join(lines)

    while line.rstrip().endswith("\\"):
        try:
            line = line.rstrip()[:-1] + "\n" + input(cont)
        except (EOFError, KeyboardInterrupt):
            print()
            break
    return line


_AT = re.compile(r"(?<![\w/])@([\w./~-]+)")


def _expand_files(text: str, root: Path) -> tuple[str, list[str]]:
    """Inline `@path` references, and say which ones were inlined.

    The alternative is telling the agent to read the file, which costs a turn
    and a round trip on a model that generates 67 tokens a second under load.
    Missing paths are left as written -- `@` is ordinary text in an email
    address or a decorator, and silently deleting it would be worse than
    passing it through.
    """
    used: list[str] = []

    def sub(m: re.Match) -> str:
        raw = m.group(1)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = root / p
        try:
            if not p.is_file() or p.stat().st_size > 256_000:
                return m.group(0)
            body = p.read_text(errors="replace")
        except OSError:
            return m.group(0)
        used.append(str(p))
        return f"\n--- {raw} ---\n{body}\n--- end {raw} ---\n"

    return _AT.sub(sub, text), used


# --------------------------------------------------------------------------
# session state


class Session:
    """Everything the front end tracks that the agent does not."""

    def __init__(self, model: str, approval: str, root: Path):
        self.model = model
        self.approval = approval
        self.root = root
        self.verbose = False
        self.turns: list[dict] = []
        self.last_input = ""
        self.last_commands: list[str] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.started = time.time()

    def add_usage(self, usage: dict) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.cached_tokens += int(usage.get("cached_tokens") or 0)

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "approval": self.approval,
            "root": str(self.root),
            "started": self.started,
            "turns": self.turns,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "cached": self.cached_tokens,
            },
        }


def _session_path(name: str) -> Path:
    safe = re.sub(r"[^\w.-]", "-", name)[:60] or "session"
    return _SESSIONS / f"{safe}.json"


# --------------------------------------------------------------------------
# the loop


def cmd_repl(args) -> int:
    """Talk to the agent, one turn at a time, in a directory."""
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
            ui.error(f"cannot reach the model at {base}.")
            + "\nIf it is served on an eval box, set CRUX_TUNNEL=<ssh host> in\n"
            "~/.crux/env and this will open the forward itself.",
            file=sys.stderr,
        )
        return 1

    cwd = Path(args.cwd or os.getcwd()).resolve()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    approval = Approval(args.approval)
    model = args.model or f"openai/{default_model}"
    state = Session(model=model, approval=approval.value, root=cwd)

    env = LocalEnvironment(cwd=cwd)

    # ---- live display -------------------------------------------------
    spinner: ui.Spinner | None = None

    def emit(line: str) -> None:
        """Print above the spinner without fighting it for the line."""
        if ui._TTY:
            sys.stdout.write("\r\033[K")
        print(line)
        sys.stdout.flush()

    def on_event(kind: str, **kw) -> None:
        try:
            if kind == "analysis":
                body = ui.analysis(kw.get("text", ""))
                if body:
                    emit(body)
                if spinner:
                    spinner.set("running")
            elif kind == "command":
                emit(ui.command_line(kw["n"], kw["keystrokes"]))
                if state.verbose:
                    rest = (kw["keystrokes"] or "").splitlines()[1:]
                    for ln in rest[:40]:
                        emit(f"      {ui.GREY}{ln}{ui.RESET}")
                state.last_commands.append(kw["keystrokes"])
            elif kind == "result":
                emit(ui.command_result(kw["elapsed"], kw["output"], kw["timed_out"]))
                if state.verbose and kw.get("output"):
                    for ln in (kw["output"] or "").splitlines()[-30:]:
                        emit(f"      {ui.GREY}{ln[:ui.width() - 8]}{ui.RESET}")
                if spinner:
                    spinner.set("thinking")
            elif kind == "refused":
                emit(ui.error(f"  refused: {ui.first_line(kw['keystrokes'])}")
                     + ui.hint(f"  ({kw['reason']})"))
            elif kind == "usage":
                state.add_usage(kw.get("usage") or {})
        except Exception:  # noqa: BLE001 - the display must never end a turn
            pass

    agent = LocalCruxAgent(
        logs_dir=Path(env.trial_paths.agent_dir),
        model_name=model,
        approval=approval,
        project_root=cwd,
        interactive=interactive,
        carry_context=True,
        on_event=on_event,
        **({"variant": args.variant} if args.variant != "default" else {}),
    )

    print(ui.banner(
        version=__version__,
        agent=agent.name(),
        env=env.describe(),
        model=model,
        approval=approval.value,
        note="" if interactive else "  (no tty: anything needing a prompt is refused)",
    ))
    print()
    _install_readline()

    # ---- slash commands ------------------------------------------------
    def handle_slash(text: str) -> bool | None:
        """True to continue the loop, None if this was not a slash command."""
        nonlocal model
        parts = shlex.split(text) if text.startswith("/") else []
        if not parts:
            return None
        cmd, rest = parts[0], parts[1:]

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            print(_HELP)
        elif cmd == "/new":
            agent.forget_context()
            state.turns.clear()
            print(ui.ok("conversation cleared") + ui.hint("; the shell keeps its state"))
        elif cmd == "/shell":
            print(f"  {env.describe()}")
            print(ui.hint(f"  attach with: tmux attach -t {agent.name()}"))
        elif cmd == "/cd":
            if not rest:
                print(f"  {state.root}")
            else:
                new = Path(rest[0]).expanduser()
                new = new if new.is_absolute() else state.root / new
                if not new.is_dir():
                    print(ui.error(f"  not a directory: {new}"))
                else:
                    state.root = new.resolve()
                    agent._project_root = state.root
                    print(ui.ok(f"  project root is now {state.root}"))
                    print(ui.hint("  the shell's own cwd is unchanged; cd there yourself"))
        elif cmd == "/model":
            if not rest:
                print(f"  {model}")
            else:
                model = rest[0] if "/" in rest[0] else f"openai/{rest[0]}"
                agent._model_name = model
                state.model = model
                print(ui.ok(f"  next turn uses {model}"))
        elif cmd == "/approval":
            if not rest:
                print(f"  {agent._approval.value}")
            else:
                try:
                    lvl = Approval(rest[0])
                except ValueError:
                    print(ui.error(f"  unknown level {rest[0]!r}; use safe | ask | dangerous"))
                else:
                    agent._approval = lvl
                    state.approval = lvl.value
                    print(ui.ok(f"  approval is now {lvl.value}"))
        elif cmd == "/tokens":
            el = time.time() - state.started
            print(f"  {ui.human_count(state.prompt_tokens)} in "
                  f"({ui.human_count(state.cached_tokens)} cached) / "
                  f"{ui.human_count(state.completion_tokens)} out "
                  f"over {len(state.turns)} turns, {ui.human_secs(el)}")
        elif cmd == "/last":
            if not state.turns:
                print(ui.hint("  nothing yet"))
            else:
                for c in state.turns[-1].get("commands", []):
                    print(f"  {ui.BOLD}${ui.RESET} {c.strip()}")
        elif cmd == "/verbose":
            state.verbose = not state.verbose
            print(ui.ok(f"  verbose {'on' if state.verbose else 'off'}"))
        elif cmd == "/retry":
            if not state.last_input:
                print(ui.hint("  nothing to retry"))
            else:
                print(ui.hint(f"  resending: {ui.first_line(state.last_input)}"))
                return "__retry__"  # type: ignore[return-value]
        elif cmd == "/save":
            name = rest[0] if rest else time.strftime("%Y%m%d-%H%M%S")
            p = _session_path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state.to_json(), indent=1))
            print(ui.ok(f"  saved {p}"))
        elif cmd == "/resume":
            if not rest:
                _SESSIONS.mkdir(parents=True, exist_ok=True)
                names = sorted(p.stem for p in _SESSIONS.glob("*.json"))
                print("  " + ("\n  ".join(names) if names else ui.hint("no saved sessions")))
            else:
                p = _session_path(rest[0])
                if not p.exists():
                    print(ui.error(f"  no session {rest[0]!r}"))
                else:
                    saved = json.loads(p.read_text())
                    print(ui.hint(f"  {len(saved.get('turns', []))} turns from "
                                  f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(saved.get('started', 0)))}"))
                    for t in saved.get("turns", []):
                        print(f"  {ui.CYAN}>{ui.RESET} {ui.first_line(t.get('input', ''))}")
                    print(ui.hint("  transcript only: the model's context is not restored"))
        else:
            print(ui.error(f"  unknown command {cmd}") + ui.hint("  (/help)"))
        return True

    # ---- turns ---------------------------------------------------------
    async def turn(text: str) -> None:
        nonlocal spinner
        agent.begin_turn()
        state.last_commands = []
        started = time.monotonic()
        p0, c0 = state.prompt_tokens, state.completion_tokens
        print(ui.rule("crux"))
        with ui.Spinner("thinking") as sp:
            spinner = sp
            try:
                await agent.run(text, env, AgentContext())
            finally:
                spinner = None
        agent.remember_turn()
        steps, commands = agent.turn_stats
        state.turns.append({
            "input": text,
            "commands": list(state.last_commands),
            "steps": steps,
            "elapsed": time.monotonic() - started,
        })
        print(ui.turn_summary(
            steps, commands, time.monotonic() - started,
            state.prompt_tokens - p0, state.completion_tokens - c0,
            state.cached_tokens,
        ))

    async def main() -> int:
        await env.start()
        # setup() is what creates the tmux session Terminus drives; run() raises
        # "Session is not set" without it. In a benchmark trial harbor calls it,
        # so it is easy to leave out of a hand-written loop -- and it fails on
        # the first turn rather than at import, which is how it got missed here.
        await agent.setup(env)
        prompt = f"{ui.CYAN}crux>{ui.RESET} " if ui._COLOR else "crux> "
        cont = f"{ui.GREY}....{ui.RESET} " if ui._COLOR else ".... "
        try:
            while True:
                line = _read_block(prompt, cont)
                if line is None:
                    return 0
                text = line.strip()
                if not text:
                    continue

                if text.startswith("/"):
                    verdict = handle_slash(text)
                    if verdict is False:
                        return 0
                    if verdict == "__retry__":
                        text = state.last_input
                    elif verdict is True:
                        continue

                text, inlined = _expand_files(text, state.root)
                for f in inlined:
                    print(ui.hint(f"  + {f}"))
                state.last_input = text

                try:
                    await turn(text)
                except KeyboardInterrupt:
                    # The turn is abandoned but the session is not: whatever the
                    # agent already did to the working directory stands, and
                    # saying so is more useful than pretending it rolled back.
                    print("\n" + ui.warn("^C  turn interrupted; the shell keeps whatever it did"))
                    agent.remember_turn()
                except Exception as exc:  # noqa: BLE001 - a bad turn must not end the session
                    print("\n" + ui.error(f"turn failed: {exc}"), file=sys.stderr)
                print()
        finally:
            _save_readline()
            await env.stop()

    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        _save_readline()
        return 130
    except RuntimeError as exc:
        print(ui.error(str(exc)), file=sys.stderr)
        return 1
