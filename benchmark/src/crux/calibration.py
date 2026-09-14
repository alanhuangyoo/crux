"""Constants remember the regime they were measured in.

Three defects in one week, one shape:

    tmux `-e`            correct on every image with tmux >= 3.2, carried onto
                         two images with 3.1c, recorded as "that task fails"
    compare()'s mean     correct when both runs cover the same tasks, carried
                         onto runs that do not, recorded as a delta
    600-second timeout   correct against a 900-second budget, carried onto an
                         8x one, recorded as "these tasks never solve"

In all three the failure's own record said nothing -- `Error: None`, a mean over
the wrong set, a task that simply never solves -- so the wrong regime was never
the suspect. What they have in common is not a bug in the value. Each value was
right where it was chosen. What is missing is that **nothing carried the regime
alongside the number**, so nothing could notice when the regime changed.

This is that missing half. An entry states the value, the regime it was
measured in, and a predicate over the regime a run is about to use. `crux bench`
audits before launching, so a mismatch is a line on the terminal rather than a
finding six days later.

It deliberately does not try to be clever. It cannot know that tmux 3.1c lacks
`-e`; that one is a feature test in the agent, where it belongs. What it catches
is the narrower and more common case: a number tuned against one budget, model
or corpus, being spent against another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The budget these constants were tuned against: Terminal-Bench's own default,
# 900 seconds, which is what every run used before the multiplier existed.
CALIBRATION_BUDGET_SEC = 900.0

# The model every measurement in this repository was taken on.
CALIBRATION_MODEL = "qwen3.8-27b"

# The output cap, and the turn sizes it was placed above: scored trials on this
# deployment run 7.5k-14k median completion tokens with a 44k maximum, while
# stalled ones reach 28k median and 41k. 32k clears every scored turn and cuts
# a runaway well before the 262k window.
CALIBRATION_MAX_OUTPUT_TOKENS = 32768

# Containers in flight at the measured throughput inflection.
#
# 35 was measured on the previous node (h20-w06, cards 4-7): 587 tok/s across
# 8-way concurrency there, 320 at 54 containers.
#
# Re-measured on the current node (gpu-host, cards 0-3) under 29 live containers:
#
#     idle, single stream                340 tok/s
#     29 containers, single stream        84 tok/s
#     29 containers, 4-way probe          92 tok/s per stream (369 aggregate)
#     29 containers, 8-way probe          20 tok/s per stream (156 aggregate)
#
# Aggregate *falls* from 4-way to 8-way, which is what saturation looks like:
# the extra streams are queueing, not working. So the inflection on this node
# is at or below 29, not 35. The number is kept as a ceiling that the audit can
# still act on, and lowered to what was actually observed.
#
# The measurement is not clean -- the probe competes with the real load rather
# than running against an idle engine -- so it bounds the inflection from above
# and cannot locate it exactly. That is enough for a warning and not enough for
# a claim.
CALIBRATION_CONTAINERS = 29

# A per-call ceiling, as a share of the trial it is bounding.
#
# litellm's own default is 6000s and harbor never overrides it, so some ceiling
# is needed: a request that hangs would otherwise sit for a hundred minutes.
# 600 seconds was that ceiling, chosen against a 900-second budget where it is
# two thirds of a trial. At the 8x budget every run since has used it is 8%,
# where it stops bounding hangs and starts cutting real generations --
# `regex-chess` died on it three attempts deep, `timeout value=600.0, time
# taken=1801.36 seconds`, and was then written down as a task that never
# solves.
#
# Across 9,253 inter-step gaps in two 89-task runs, 99.8% are under 600s; the
# five tasks that stall spend 20-30% of their steps above it. The cut is narrow
# and aimed exactly at the tasks with the longest turns.
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

    This is the one that cost 12 slot-hours. 600 seconds is two thirds of a
    900-second budget and 8% of a 7200-second one; at 8% it stops bounding a
    hung request in any useful sense and starts cutting real generations.
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
