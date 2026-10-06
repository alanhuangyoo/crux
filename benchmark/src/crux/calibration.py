"""Tuned constants that carry the regime they were tuned in.

A value that is right for one budget, model or corpus can be quietly wrong for
another, and the failure it causes rarely points back at the value: a timeout
tuned for a short budget cuts real generations under a long one, and the run
records it as a task that never solves.

So each entry states the value, the regime it was measured in, and a predicate
over the regime a run is about to use. `crux bench` audits before launching,
so a mismatch is a line on the terminal rather than a finding after the run.

Feature gaps in the environment, such as an old tmux, are detected in the
agent instead; this covers numbers tuned in one regime and spent in another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The budget these constants were tuned against: Terminal-Bench's own default,
# 900 seconds, which is what every run used before the multiplier existed.
CALIBRATION_BUDGET_SEC = 900.0

# The model every measurement in this repository was taken on.
CALIBRATION_MODEL = "qwen3.8-27b"

# The output cap: well above the completion size of normal turns, and well below
# the context window, so a runaway generation is cut early.
CALIBRATION_MAX_OUTPUT_TOKENS = 32768

# Containers in flight at the measured throughput inflection. Past it, extra
# streams queue rather than work and aggregate throughput falls, so the audit
# warns above this ceiling.
CALIBRATION_CONTAINERS = 29

# A per-call ceiling, as a share of the trial it is bounding.
#
# litellm defaults to 6000s and harbor never overrides it, so a hung request
# would otherwise sit for over an hour. A fixed ceiling does not survive a
# budget change: too large a share lets one hang take the trial, too small a
# share cuts real long generations. So the ceiling scales with the budget,
# between a floor and a cap.
_LLM_TIMEOUT_BUDGET_SHARE = 0.25
_LLM_TIMEOUT_MIN_SEC = 600.0
_LLM_TIMEOUT_MAX_SEC = 1800.0


def llm_timeout_for(budget_sec: float) -> float:
    """The per-call ceiling this trial's budget can afford.

    Below the floor the share would be too tight to finish a normal turn; above
    the cap a hang costs more than it is worth waiting for.
    """
    if budget_sec <= 0:
        return _LLM_TIMEOUT_MIN_SEC
    share = budget_sec * _LLM_TIMEOUT_BUDGET_SHARE
    return min(_LLM_TIMEOUT_MAX_SEC, max(_LLM_TIMEOUT_MIN_SEC, share))


@dataclass(frozen=True)
class Run:
    """The regime a run is about to execute in."""

    budget_sec: float = CALIBRATION_BUDGET_SEC
    model: str = CALIBRATION_MODEL
    dataset: str = "terminal-bench/terminal-bench-2-1"
    concurrent: int = 24
    other_containers: int = 0

    @property
    def containers(self) -> int:
        return self.concurrent + self.other_containers


@dataclass(frozen=True)
class Constant:
    """A tuned value, the regime it was tuned in, and how to check it."""

    name: str
    measured_in: str
    check: Callable[[Run], str | None] = field(repr=False)


def _timeout(run: Run) -> str | None:
    """A per-call ceiling has to stay a fraction of the trial it bounds.

    Too large and one hung call takes the trial; too small and it starts cutting
    real generations rather than bounding hangs.
    """
    t = llm_timeout_for(run.budget_sec)
    share = t / run.budget_sec if run.budget_sec else 1.0
    if share > 0.75:
        return (f"llm timeout {t:.0f}s is {share * 100:.0f}% of a "
                f"{run.budget_sec:.0f}s budget: one hung call can take the trial")
    if share < 0.05:
        return (f"llm timeout {t:.0f}s is {share * 100:.0f}% of a "
                f"{run.budget_sec:.0f}s budget: long generations will be cut")
    return None


def _output_cap(run: Run) -> str | None:
    """The cap sits above every scored turn measured -- on one model."""
    if run.model != CALIBRATION_MODEL:
        return (f"max_tokens {CALIBRATION_MAX_OUTPUT_TOKENS} was measured on "
                f"{CALIBRATION_MODEL} (scored turns 7.5k-14k median, 44k max); "
                f"{run.model} has not been measured")
    return None


def _stall_caps(run: Run) -> str | None:
    """A per-task budget cut is a claim about one corpus on one configuration."""
    from crux.cli import stall_capped_for

    capped = stall_capped_for(run.dataset)
    if capped and run.budget_sec != CALIBRATION_BUDGET_SEC * 8:
        return (f"{len(capped)} task(s) carry a shortened budget measured at 8x; "
                f"this run is at {run.budget_sec / CALIBRATION_BUDGET_SEC:.0f}x")
    return None


def _throughput(run: Run) -> str | None:
    """Past the inflection every container gets slower, so the run does too."""
    if run.containers > CALIBRATION_CONTAINERS:
        return (f"{run.containers} containers is past the measured inflection at "
                f"{CALIBRATION_CONTAINERS} (340 tok/s idle -> 84 under load): more "
                f"concurrency here buys less, not more")
    return None


CONSTANTS: tuple[Constant, ...] = (
    Constant("llm_timeout", f"a {CALIBRATION_BUDGET_SEC:.0f}s agent budget", _timeout),
    Constant("max_tokens", f"the {CALIBRATION_MODEL} deployment", _output_cap),
    Constant("stall caps", "terminal-bench at 8x", _stall_caps),
    Constant("concurrency", f"{CALIBRATION_CONTAINERS} containers on gpu-host", _throughput),
)


def audit(run: Run) -> list[str]:
    """Every constant whose regime does not match the run about to happen."""
    out = []
    for c in CONSTANTS:
        try:
            msg = c.check(run)
        except Exception as exc:  # noqa: BLE001 - an audit must not stop a run
            msg = f"could not be checked ({type(exc).__name__}: {exc})"
        if msg:
            out.append(f"{c.name}: {msg}\n    measured in: {c.measured_in}")
    return out
