"""What the terminal shows while the agent is working.

The REPL used to print nothing between the prompt and the answer. A turn on this
model runs 22 minutes at the median and 62 at p90, and for all of it the screen
said nothing -- so the only way to know whether the agent was building a kernel
or retrying a parse error was to `tmux attach` in another window. Two of the
three harness faults found on this project were visible in the terminal the
whole time and were found months late, by reading trajectories offline.

So this module exists to put the same information on screen while it happens:
every command as it is sent, how long it took, and what came back. Nothing here
is decoration -- each element answers a question that otherwise costs a
`tmux attach`:

    the command      what is it doing
    the duration     is it stuck, or is this just slow
    the exit shape   did that work
    the counters     how much of the budget is gone

Degrades on purpose. No curses, no alternate screen, no repainting: output is
appended, so a session stays readable in a pipe, in a scrollback, and in a
terminal that is 40 columns wide. `NO_COLOR` and a non-tty both drop the escape
codes and change nothing else.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time

# Colour is opt-out three ways, because a log that is being read later has no
# terminal and a user who dislikes colour should not have to argue with it.
_TTY = sys.stdout.isatty()
_COLOR = _TTY and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _c(code: str) -> str:
    return f"\033[{code}m" if _COLOR else ""


DIM = _c("2")
BOLD = _c("1")
RED = _c("31")
GREEN = _c("32")
YELLOW = _c("33")
BLUE = _c("34")
MAGENTA = _c("35")
CYAN = _c("36")
GREY = _c("90")
RESET = _c("0")

# Wide enough to be worth wrapping, narrow enough to survive a split pane.
def width() -> int:
    return max(40, min(shutil.get_terminal_size((100, 24)).columns, 120))


def _visible_len(s: str) -> int:
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def rule(label: str = "", color: str = GREY) -> str:
    """A horizontal rule, optionally with a label at the left."""
    w = width()
    if not label:
        return f"{color}{'─' * w}{RESET}"
    head = f"{color}── {label} "
    pad = max(0, w - _visible_len(head))
    return f"{head}{'─' * pad}{RESET}"


def human_secs(sec: float) -> str:
    if sec < 1:
        return f"{sec * 1000:.0f}ms"
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def human_count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def first_line(text: str, limit: int | None = None) -> str:
    """The first meaningful line of a command, for the one-line echo.

    A heredoc is the common shape here -- `cat > f.py << 'EOF'` followed by
    forty lines of Python -- and the first line is the part that says what is
    happening. The rest is the payload and belongs in the transcript, not on
    the status line.
    """
    limit = limit or max(20, width() - 22)
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line:
            return line[:limit] + ("…" if len(line) > limit else "")
    return ""


class Spinner:
    """A one-line "still working" indicator that never scrolls.

    Only runs on a tty. Everywhere else the calls are no-ops, so the same code
    path works when the session is piped to a file.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "thinking"):
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.08):
            el = time.monotonic() - self._start
            frame = self.FRAMES[i % len(self.FRAMES)]
            i += 1
            line = f"{CYAN}{frame}{RESET} {DIM}{self.label} {human_secs(el)}{RESET}"
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        if _TTY:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if _TTY:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def set(self, label: str) -> None:
        self.label = label


def command_line(n: int, keystrokes: str) -> str:
    """One line announcing a command that is about to run."""
    return f"  {GREY}{n:>3}{RESET} {BOLD}${RESET} {first_line(keystrokes)}"


def command_result(elapsed: float, output: str, timed_out: bool) -> str:
    """One line summarising what a command produced."""
    lines = (output or "").count("\n")
    if timed_out:
        mark, color = "timeout", YELLOW
    elif re.search(r"\b(error|Error|ERROR|Traceback|command not found|No such file)\b", output or ""):
        mark, color = "errors", RED
    else:
        mark, color = "ok", GREEN
    return f"      {color}{mark}{RESET} {DIM}{human_secs(elapsed)} · {lines} lines{RESET}"


def analysis(text: str, limit_lines: int = 6) -> str:
    """The model's own account of what it is doing, indented and trimmed.

    Terminus puts this in the `Analysis:` prefix of every response, and it is
    the only place the agent says why. Six lines is the point where it stops
    being a summary and starts being the reasoning again.
    """
    body = (text or "").strip()
    if body.lower().startswith("analysis:"):
        body = body[len("analysis:"):].strip()
    if not body:
        return ""
    keep = body.splitlines()[:limit_lines]
    w = width() - 4
    out = []
    for line in keep:
        line = line.strip()
        while len(line) > w:
            cut = line.rfind(" ", 0, w)
            cut = cut if cut > w // 2 else w
            out.append(line[:cut])
            line = line[cut:].lstrip()
        if line:
            out.append(line)
    trimmed = len(body.splitlines()) > limit_lines
    text = "\n".join(f"  {DIM}{ln}{RESET}" for ln in out[:limit_lines + 2])
    if trimmed:
        text += f"\n  {GREY}…{RESET}"
    return text


def banner(version: str, agent: str, env: str, model: str, approval: str, note: str = "") -> str:
    lines = [
        f"{BOLD}crux {version}{RESET} {GREY}·{RESET} {agent} {GREY}·{RESET} {model}",
        f"  {DIM}{env}{RESET}",
        f"  {DIM}approval={approval}{note}{RESET}",
        "",
        f"  {CYAN}/help{RESET}   what else this understands",
        f"  {CYAN}/exit{RESET}   leave  {GREY}(Ctrl-D also works; Ctrl-C interrupts a turn){RESET}",
    ]
    return "\n".join(lines)


def turn_summary(steps: int, commands: int, elapsed: float,
                 prompt_tokens: int, completion_tokens: int, cached: int) -> str:
    """The line printed after every turn.

    Tokens are here because they are the budget that actually binds. The
    measurement that motivated the submit gate -- failures stop with two thirds
    of their wall-clock unspent -- was invisible for months precisely because
    nothing displayed it.
    """
    bits = [
        f"{steps} steps",
        f"{commands} commands",
        human_secs(elapsed),
    ]
    if prompt_tokens or completion_tokens:
        cache = f" ({human_count(cached)} cached)" if cached else ""
        bits.append(f"{human_count(prompt_tokens)} in{cache} / {human_count(completion_tokens)} out")
    return f"{GREY}  {' · '.join(bits)}{RESET}"


def error(msg: str) -> str:
    return f"{RED}{msg}{RESET}"


def warn(msg: str) -> str:
    return f"{YELLOW}{msg}{RESET}"


def ok(msg: str) -> str:
    return f"{GREEN}{msg}{RESET}"


def hint(msg: str) -> str:
    return f"{GREY}{msg}{RESET}"
