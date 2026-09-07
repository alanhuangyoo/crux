"""Ask whether a change moves the model at all, before it costs a run.

Six mechanisms were given a full arm on this benchmark and six came back inside
the noise. Two of them could have been ruled out in under a minute:

    file tools      uptake 0.3% of commands -> 4.3%, score 81.1% vs 78.4% (p=0.77)
    batching        first command [1,1,1,1,1] -> [3,2,1,1,1], median unmoved

Both are real nudges against a strong prior, and both are far too small to read
against a benchmark whose own noise flips 15-16% of tasks. An arm costs six
hours of GPU; five calls cost a minute and answer the prior question -- *does
the model behave differently at all* -- which is the one that decides whether
the score question is worth asking.

This is not a substitute for measuring the score. It is the filter in front of
it: a mechanism that does not move behaviour cannot move the score, and one
that moves behaviour a little will move the score less than the noise.

**It measures a ceiling, not an effect.** The probe reads turn one, where the
model has read nothing and has no habits yet. On file tools it reports two
draws in four reaching for `crux read`; across 160 real trajectories the tools
are used in 37%, and the first use lands at 41% of the way through -- only 4 of
59 inside the first tenth. The opening command is `ls -la /app` 35 times out of
39.

So the model can be moved and mostly is not, and the gap between those two is
where the score went: a mechanism that saves actions cannot save them after the
actions are spent. A probe that says "worth an arm" is saying the ceiling is
above zero, which is the least it could usefully say.
"""

from __future__ import annotations

import json
import re
import statistics as st
import urllib.request
from dataclasses import dataclass, field

from crux import ui

# The model answers in Terminus's XML, and the opening tag carries attributes:
# `<keystrokes duration="0.1">`. Matching the bare tag finds nothing, which
# reads as "the probe got no commands" rather than "the pattern is wrong".
_KEYS = re.compile(r"<keystrokes[^>]*>(.*?)</keystrokes>", re.S)

# `&&` and `;` join independent probes; a trailing `;` does not.
_SEGMENTS = re.compile(r"&&|;(?!\s*$)")

# A blank terminal, which is what the agent sees on its first turn -- the turn
# where the corpus difference against claude-code is largest (1.0 segments
# against 3.0).
_FRESH_TERMINAL = "root@probe:/app# \n"

_DEFAULT_TASK = (
    "Find the bug in /app/solver.py that makes test_edge fail, and fix it."
)


# What to measure depends on what the change claims to do. A probe that reports
# the same number for every configuration is not measuring the configuration --
# the first version counted command segments for everything, and answered
# "median unmoved at 1.0" for file tools, interleaved thinking, the harness
# section and the submit gate alike. Three of those do not touch the first
# command at all.
#
# Same fault as the sampling check in `crux doctor`, which asked a question with
# one obvious answer and reported determinism it had never tested.
SIGNALS: dict[str, tuple[str, str]] = {
    # kwarg            (what to count, one line of why)
    "batch_section":   ("segments", "probes chained into one command"),
    "file_tools":      ("crux_tools", "calls to crux read/grep/files/edit"),
}

# Everything else changes something this cannot see on turn one, and says so
# instead of printing a number.
#
# `interleaved_thinking` is the instructive case: it was listed here with a
# prompt-length signal, and reported 9268 == 9268 -- correctly, because on the
# first turn there is no prior reasoning to carry. The effect is real and
# measured (context grows 555 tokens a step, reasoning is 26% shorter) and it
# begins on turn two. A probe that reads turn one can only mislead about it.
NO_FIRST_TURN_SIGNAL = {
    "interleaved_thinking": "acts from turn two, when there is reasoning to carry",
    "submit_gate": "fires at the end of a run",
    "edit_debt_limit": "fires at the 13th unchecked edit",
    "confirm_gate": "fires at submit",
    "harness_section": "changes what is read, not what is typed first",
}


def _count(kind: str, command: str, prompt: str) -> float:
    if kind == "segments":
        return len(_SEGMENTS.split(command.strip())) if command.strip() else 0
    if kind == "crux_tools":
        return len(re.findall(r"\bcrux\s+(read|grep|files|edit|write)\b", command))
    if kind == "prompt_len":
        return len(prompt)
    return 0.0


@dataclass
class ProbeResult:
    label: str
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    kind: str = "segments"
    prompt: str = ""

    @property
    def segments(self) -> list[float]:
        if self.kind == "prompt_len":
            # One prompt per configuration, not one per draw.
            return [_count(self.kind, "", self.prompt)]
        return [_count(self.kind, c, self.prompt) for c in self.commands if c.strip()]

    @property
    def median_segments(self) -> float:
        s = self.segments
        return st.median(s) if s else 0.0

    def summary(self) -> str:
        if not self.commands and self.kind != "prompt_len":
            why = f"  ({self.errors[0]})" if self.errors else ""
            return f"  {self.label:<22} no commands returned{why}"
        vals = [f"{v:.0f}" for v in self.segments]
        head = f"  {self.label:<22} [{', '.join(vals)}]  median {self.median_segments:.1f}"
        if self.kind == "prompt_len":
            return head
        firsts = [ui.first_line(c, 46) for c in self.commands[:3]]
        return head + "\n" + "\n".join(f"      $ {f}" for f in firsts)


def _one_call(base_url: str, api_key: str, model: str, prompt: str,
              timeout: float = 120.0) -> tuple[str, str]:
    """One completion. Returns (first keystrokes, error)."""
    body = {
        "model": model,
        "max_tokens": 900,
        "messages": [{"role": "user", "content": prompt}],
        # Thinking off: the probe asks what the model *does*, and reasoning
        # tokens only add latency and a chance of the budget being eaten before
        # a command appears.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as exc:  # noqa: BLE001 - a failed call is a datum, not a crash
        return "", f"{type(exc).__name__}: {exc}"
    content = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = _KEYS.search(content)
    if not m:
        head = re.sub(r"\s+", " ", content)[:70]
        return "", f"no <keystrokes> in reply: {head}"
    return m.group(1), ""


def probe(agent_kwargs: dict, label: str, base_url: str, api_key: str,
          model: str, task: str = _DEFAULT_TASK, n: int = 5,
          kind: str = "segments") -> ProbeResult:
    """Draw `n` first commands from the prompt a given configuration produces."""
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import CruxTerminusAgent

    agent = CruxTerminusAgent(
        logs_dir=Path(tempfile.mkdtemp()), model_name=f"openai/{model}",
        **agent_kwargs,
    )
    prompt = agent._prompt_template.format(
        instruction=task, terminal_state=_FRESH_TERMINAL
    )
    out = ProbeResult(label=label, kind=kind, prompt=prompt)
    # A prompt-length signal is a property of the prompt, not of what the model
    # does with it, so it needs no calls at all.
    draws = 0 if kind == "prompt_len" else n
    for _ in range(draws):
        keys, err = _one_call(base_url, api_key, model, prompt)
        if keys:
            out.commands.append(keys)
        elif err:
            out.errors.append(err)
    return out


def cmd_probe(args) -> int:
    """Compare what the model does under two configurations, in a minute."""
    from crux.chat import _endpoint

    base, key, default_model = _endpoint()
    if not base:
        print(ui.error("no endpoint configured"))
        return 1
    model = args.model or default_model

    kwargs: dict = {}
    for kv in args.agent_kwarg or []:
        k, _, v = kv.partition("=")
        kwargs[k] = v or True

    print(ui.rule("crux probe"))
    if not kwargs:
        print(ui.hint("  nothing to compare; pass --agent-kwarg K=V"))
        return 1

    known = [k for k in kwargs if k in SIGNALS]
    if not known:
        print(ui.warn(f"  no first-turn signal for {', '.join(kwargs)}"))
        for k in kwargs:
            why = NO_FIRST_TURN_SIGNAL.get(k)
            if why:
                print(ui.hint(f"    {k}: {why}"))
        print(ui.hint(
            "  This reads the first command only, so a change that acts later\n"
            "  needs an arm. Readable here: " + ", ".join(SIGNALS)))
        return 1
    kind, why = SIGNALS[known[0]]
    print(ui.hint(f"  {model}  ·  {args.n} draws  ·  counting {why}\n"))

    base_res = probe({}, "default", base, key, model, args.task, args.n, kind)
    print(base_res.summary())
    on = probe(kwargs, ",".join(f"{k}={v}" for k, v in kwargs.items()),
               base, key, model, args.task, args.n, kind)
    print(on.summary())

    b, o = base_res.median_segments, on.median_segments
    moved = sum(1 for x, y in zip(base_res.segments, on.segments) if x != y)
    print()
    if not on.commands or not base_res.commands:
        print(ui.warn("  inconclusive: one side returned nothing"))
    elif o > b:
        print(ui.ok(f"  median moves {b:.1f} -> {o:.1f}") +
              ui.hint("  -- worth an arm"))
    else:
        print(ui.warn(f"  median unmoved at {b:.1f}") + ui.hint(
            f"  ({moved} of {len(on.segments)} draws differ)"))
        print(ui.hint(
            "  Two mechanisms with this shape scored inside the noise:\n"
            "  file tools 81.1% vs 78.4% (p=0.77), batching median unmoved.\n"
            "  A prompt section nudges a strong prior; it does not replace one."))
    return 0
