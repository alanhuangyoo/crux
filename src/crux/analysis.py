"""Turn a Harbor job directory into an answer to "why did we score that?".

A leaderboard number tells you nothing about what to fix. Every improvement in
this project came from a distinction the score does not make:

* Trials that never reached the agent at all. Three consecutive full runs
  scored 0.000 for environment reasons — a GPU validation error that aborted
  the job, then overlayfs-on-btrfs corrupting every image build. Reading those
  as "the agent is bad" would have sent us tuning prompts for days.
* How close a failure was. Terminal-Bench scores per task all-or-nothing, but
  the verifiers underneath keep a weighted breakdown. Reading it showed 30 of
  70 tasks past 60% of their checks with 3 scored, which is what motivated
  telling the model that partial credit does not exist.
* Whether the agent thought it was finished. It claimed completion 28 times
  and was right 3 times; an 89% false-positive rate is a different problem from
  running out of turns, and they are indistinguishable from the mean.

Verifier output is not uniform: some tasks are plain pytest suites, others run
a scoring harness whose pytest layer is a single always-passing wrapper over a
weighted breakdown. Trusting the pytest counts on the latter overstates
progress badly — one task showed 1/1 tests passed and 0.0/1.05 points.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Ordered: the first matching category wins, so environment problems mask
# agent-level ones. A trial whose image never built says nothing about the
# agent, and counting it as an agent failure is how a broken box looks like a
# bad prompt.
FAILURE_ORDER = [
    "environment",
    "agent_timeout",
    "context_exceeded",
    "format_error",
    "false_completion",
    "out_of_turns",
    "solved",
]

# Exception types that mean the harness or the box failed, not the agent.
ENVIRONMENT_EXCEPTIONS = {
    "RuntimeError",
    "VerifierTimeoutError",
    "CancelledError",
    "DockerException",
}


@dataclass
class Trial:
    task: str
    reward: float | None = None
    exception: str | None = None
    exit_reason: str | None = None
    n_steps: int = 0
    cost_usd: float = 0.0
    completion: float | None = None
    """Fraction of the verifier's checks passed, when it exposes them."""
    completion_source: str | None = None
    variant: str | None = None

    @property
    def solved(self) -> bool:
        return self.reward == 1.0

    @property
    def category(self) -> str:
        if self.exception in ENVIRONMENT_EXCEPTIONS:
            return "environment"
        if self.exception == "AgentTimeoutError":
            return "agent_timeout"
        if self.exception == "ContextWindowExceededError":
            return "context_exceeded"
        if self.solved:
            return "solved"
        if self.exit_reason == "repeated_format_error":
            return "format_error"
        # The agent said it was done and the verifier disagreed. Distinct from
        # running out of turns: one is a judgement failure, the other a budget
        # one, and they call for opposite fixes.
        if self.exit_reason and self.exit_reason.startswith("completed"):
            return "false_completion"
        return "out_of_turns"


@dataclass
class Job:
    path: Path
    trials: list[Trial] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean reward, counting errored trials as zero — as the leaderboard does."""
        if not self.trials:
            return 0.0
        return sum(t.reward or 0.0 for t in self.trials) / len(self.trials)

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.trials)

    def by_category(self) -> Counter:
        return Counter(t.category for t in self.trials)

    def near_misses(self, threshold: float = 0.6) -> list[Trial]:
        """Unsolved trials that passed most of their checks.

        These are where the points are: an unsolved task at 90% is a different
        prospect from one at 5%, and the mean hides both.
        """
        return sorted(
            (
                t
                for t in self.trials
                # A wrapper suite's 100% means the scorer ran, not that the
                # task was done; including those buries the real near misses.
                if not t.solved
                and t.completion_source != "wrapper"
                and (t.completion or 0) >= threshold
            ),
            key=lambda t: -(t.completion or 0),
        )


# Suites that describe the scoring machinery rather than the task. Some
# verifiers wrap a weighted scorer in a single pytest that passes whenever the
# scorer ran at all, and some ship self-checks for the verifier itself. Both
# report 100% for trials that scored nothing -- three tasks in one run looked
# fully solved on this basis and had an empty reward file.
_WRAPPER_SUITES = ("test_scoring.py", "test_verifier_hygiene.py")


def _is_wrapper_suite(test_names: list[str]) -> bool:
    return bool(test_names) and all(
        any(marker in name for marker in _WRAPPER_SUITES) for name in test_names
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _completion(trial_dir: Path) -> tuple[float | None, str | None]:
    """How much of the verifier's checks passed.

    breakdown.json is preferred wherever it exists: on tasks that use a scoring
    harness, ctrf.json describes only the pytest wrapper around it and reports
    1/1 for a trial that scored zero points.
    """
    breakdown = _read_json(trial_dir / "verifier" / "breakdown.json")
    if breakdown:
        checks = breakdown.get("checks") or []
        total = sum(c.get("weight", 0) for c in checks)
        if total:
            passed = sum(c.get("weight", 0) for c in checks if c.get("passed"))
            return passed / total, "weighted"

    ctrf = _read_json(trial_dir / "verifier" / "ctrf.json")
    if ctrf:
        results = ctrf.get("results") or {}
        summary = results.get("summary") or {}
        if summary.get("tests"):
            names = [t.get("name", "") for t in (results.get("tests") or [])]
            source = "wrapper" if _is_wrapper_suite(names) else "pytest"
            return summary.get("passed", 0) / summary["tests"], source

    return None, None


def load_trial(trial_dir: Path) -> Trial | None:
    result = _read_json(trial_dir / "result.json")
    if result is None:
        return None

    trial = Trial(task=trial_dir.name.split("__")[0])

    exception_info = result.get("exception_info") or {}
    trial.exception = exception_info.get("exception_type")

    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    trial.reward = rewards.get("reward")

    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    trial.n_steps = metadata.get("n_steps", 0)
    trial.variant = (metadata.get("config") or {}).get("variant")
    trial.cost_usd = agent_result.get("cost_usd") or 0.0

    trajectory = _read_json(trial_dir / "agent" / "trajectory.json")
    if trajectory:
        notes = trajectory.get("notes") or ""
        if notes.startswith("exit_reason="):
            trial.exit_reason = notes.split("=", 1)[1]
        if not trial.n_steps:
            trial.n_steps = len(trajectory.get("steps") or [])
        if not trial.cost_usd:
            final = trajectory.get("final_metrics") or {}
            trial.cost_usd = final.get("total_cost_usd") or 0.0

    trial.completion, trial.completion_source = _completion(trial_dir)
    return trial


def load_job(job_dir: str | Path) -> Job:
    job_dir = Path(job_dir)
    job = Job(path=job_dir)
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        trial = load_trial(child)
        if trial is not None:
            job.trials.append(trial)
    return job


def completion_histogram(job: Job, bins: int = 5) -> list[tuple[str, int]]:
    counts = [0] * bins
    for trial in job.trials:
        if trial.completion is None or trial.completion_source == "wrapper":
            continue
        counts[min(int(trial.completion * bins), bins - 1)] += 1
    width = 100 // bins
    return [
        (f"{i * width}-{(i + 1) * width}%", counts[i]) for i in range(bins)
    ]


def compare(baseline: Job, candidate: Job) -> dict:
    """Compare two runs task by task.

    The aggregate delta is the headline, but the regressions matter more: a
    change that solves two new tasks and breaks two others has not helped, and
    the means alone would call that a tie.
    """
    base = {t.task: t for t in baseline.trials}
    cand = {t.task: t for t in candidate.trials}
    shared = sorted(set(base) & set(cand))

    gained = [t for t in shared if cand[t].solved and not base[t].solved]
    lost = [t for t in shared if base[t].solved and not cand[t].solved]
    moved = [
        (t, (base[t].completion or 0), (cand[t].completion or 0))
        for t in shared
        if abs((cand[t].completion or 0) - (base[t].completion or 0)) > 0.05
    ]
    return {
        "shared_tasks": len(shared),
        "baseline_score": baseline.score,
        "candidate_score": candidate.score,
        "delta": candidate.score - baseline.score,
        "gained": gained,
        "lost": lost,
        "completion_moved": sorted(moved, key=lambda m: m[2] - m[1], reverse=True),
    }
