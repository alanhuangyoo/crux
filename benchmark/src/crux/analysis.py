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
from statistics import median
from dataclasses import dataclass, field
from pathlib import Path

# Ordered: the first matching category wins, so environment problems mask
# agent-level ones. A trial whose image never built says nothing about the
# agent, and counting it as an agent failure is how a broken box looks like a
# bad prompt.
FAILURE_ORDER = [
    "solved",
    "environment",
    "killed",
    "agent_timeout",
    "context_exceeded",
    "format_error",
    "false_completion",
    "out_of_turns",
]

# mini-swe-agent's own exit statuses, mapped onto this module's vocabulary.
# An empty status means the process was killed before it could write one --
# a timeout or an OOM, not a decision the agent made.
_NATIVE_EXIT_STATUS = {
    "Submitted": "completed",
    "LimitsExceeded": "step_limit",
    "RepeatedFormatError": "repeated_format_error",
    "": "killed",
}

# Exception types that mean the harness or the box failed, not the agent.
ENVIRONMENT_EXCEPTIONS = {
    "RuntimeError",
    "VerifierTimeoutError",
    "CancelledError",
    "DockerException",
    # The trial never got a fair run: the box, the harness or the endpoint
    # failed around it. Counted as zero for a leaderboard, the same as anything
    # else that does not pass -- but never attributed to the agent when two
    # arms are compared, because which arm they land on is chance.
    #
    # Corpus counts for the ones added here: 89 InternalServerError from the
    # model endpoint, 9 AgentSetupTimeoutError, 7 EnvironmentStartTimeoutError,
    # 4 RewardFileNotFoundError, 3 AddTestsDirError, 2 RateLimitError. All 114
    # were falling through to `out_of_turns` -- read as the agent running out
    # of steps, which is a statement about the agent and the opposite of true.
    "InternalServerError",
    "RateLimitError",
    "AgentSetupTimeoutError",
    "EnvironmentStartTimeoutError",
    "AddTestsDirError",
    "RewardFileNotFoundError",
}


@dataclass
class Trial:
    task: str
    reward: float | None = None
    exception: str | None = None
    exit_reason: str | None = None
    n_steps: int = 0
    n_output_tokens: int = 0
    """Completion tokens the agent produced. Zero means it never ran."""
    n_input_tokens: int = 0
    """Prompt tokens sent. What a mechanism costs is mostly this."""
    cost_usd: float = 0.0
    completion: float | None = None
    """Fraction of the verifier's checks passed, when it exposes them."""
    completion_source: str | None = None
    variant: str | None = None
    agent: str | None = None
    """Which scaffold produced the trial. Process metrics do not cross it."""
    verifier_ran: bool = True
    """False when the verifier's own test runner never started -- see `_verifier_ran`."""

    @property
    def solved(self) -> bool:
        return self.reward == 1.0

    @property
    def category(self) -> str:
        # Solved wins over everything the agent did on the way there. Harbor
        # records an agent timeout and runs the verifier anyway, so a trial can
        # be cut off and still pass — ranking the timeout first hid two solved
        # tasks in the first pi/crux comparison and made the counts disagree
        # with the solved list printed beside them.
        if self.solved:
            return "solved"
        # The verifier never ran its tests, so the zero says nothing about
        # what the agent left behind. Harbor records no exception for this --
        # the reward is an ordinary 0.0 -- so nothing below would catch it.
        if not self.verifier_ran:
            return "environment"
        # An exception with no agent activity at all: the trial never got to
        # the agent, whatever the exception is called. This is how pi's
        # baseline gets a fair reading -- 16 of its 89 trials died in
        # `curl ... nvm/install.sh` and were counted as pi failing the task,
        # under an exception named NonZeroAgentExitCodeError, which sounds
        # like the agent's fault and is not.
        #
        # **Output tokens, not steps.** The first version of this rule asked
        # whether the trial had steps, which is a Terminus-shaped question: pi
        # writes `agent/pi` and `agent/pi.txt` and records no step count at
        # all, so the rule read zero for all 89 of its trials and would have
        # excused its 7 genuine agent timeouts along with the 16 real setup
        # deaths -- inflating the score of the very baseline it was written to
        # be fair to. Every agent's token counts are recorded by harbor, and on
        # this corpus the split is exact: the 16 setup deaths have zero output
        # tokens and nothing else does.
        if self.exception and not self.n_output_tokens and not self.n_steps:
            return "environment"
        if self.exception in ENVIRONMENT_EXCEPTIONS:
            return "environment"
        if self.exception == "AgentTimeoutError":
            return "agent_timeout"
        if self.exception == "ContextWindowExceededError":
            return "context_exceeded"
        if self.exit_reason == "repeated_format_error":
            return "format_error"
        # Killed before it could record an outcome: the harness timeout or the
        # kernel. Distinct from the agent giving up, and fixed differently.
        if self.exit_reason == "killed":
            return "killed"
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

    def median_tokens(self) -> tuple[int, int]:
        """Median input and output tokens over trials that actually ran.

        Score alone cannot tell "did nothing" from "did what it promised at
        five times the price", and the second is the worse result. Interleaved
        thinking reads as +2.50 against claude-code, inside the noise like
        everything else -- and costs 3.85M input tokens a trial against plain
        crux's 782k.
        """
        ran = [t for t in self.trials if t.n_output_tokens]
        if not ran:
            return 0, 0
        return (
            int(median(t.n_input_tokens for t in ran)),
            int(median(t.n_output_tokens for t in ran)),
        )

    @property
    def agent_name(self) -> str | None:
        """The scaffold this run used, when every trial agrees on one."""
        names = {t.agent for t in self.trials if t.agent}
        return names.pop() if len(names) == 1 else None

    @property
    def planned_trials(self) -> int:
        """How many trials the run was asked for, from its own config.

        Falls back to what it produced, so a run whose config cannot be read is
        treated as complete rather than as permanently suspect.
        """
        import json as _json

        for name in ("result.json", "config.json"):
            for candidate in (self.path / name, *sorted(self.path.glob(f"*/{name}"))):
                try:
                    d = _json.load(open(candidate))
                except Exception:  # noqa: BLE001 - absent or unreadable is not an error
                    continue
                n = d.get("n_total_trials")
                if isinstance(n, int) and n > 0:
                    return n
        return len(self.trials)

    @property
    def is_partial(self) -> bool:
        return len(self.trials) < self.planned_trials

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


# What a verifier prints when the test runner it was about to use never arrived.
_VERIFIER_NEVER_RAN = (
    "failed to download https://github.com/astral-sh/uv",
    "uvx: command not found",
    "uv: command not found",
    "pytest: command not found",
    "curl: command not found",
)


def _verifier_ran(trial_dir: Path) -> bool:
    """Whether the verifier's tests started at all.

    A task's `test.sh` installs its own runner -- curl, then uv, then
    `uvx pytest` -- from the network, at verification time, after the agent
    has finished. When that install fails the tests never start, `reward.txt`
    says 0, and harbor records no exception. Over 356 trials of four arms
    this was 14 zeros: `qemu-startup` and `qemu-alpine-ssh` in every arm
    (the image's package index is stale, `apt-get install curl` returns 404),
    and six more in the one arm that ran through a GitHub outage, where
    `uv`'s download failed. That arm read 0.697 against 0.775 and looked like
    a regression; over the trials whose tests ran it was 0.765.

    Both conditions are required. No result file alone is not enough -- a
    task with its own scoring harness may write only `reward.txt` -- and the
    runner's absence alone is not either, when some tests did report.
    """
    verifier = trial_dir / "verifier"
    if (verifier / "ctrf.json").exists() or (verifier / "breakdown.json").exists():
        return True
    try:
        output = (verifier / "test-stdout.txt").read_text(errors="ignore")
    except OSError:
        # No evidence either way; an unexplained zero stays the agent's.
        return True
    return not any(sign in output for sign in _VERIFIER_NEVER_RAN)


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
    trial.n_output_tokens = agent_result.get("n_output_tokens") or 0
    trial.n_input_tokens = agent_result.get("n_input_tokens") or 0
    trial.agent = ((result.get("config") or {}).get("agent") or {}).get("name")

    # mini-swe-agent records its own outcome in info.exit_status, and Harbor's
    # ATIF conversion does not carry it across. Reading only the ATIF notes
    # made every mini-swe-agent trial look like "out_of_turns", which is the
    # category that says nothing.
    native = _read_json(trial_dir / "agent" / "mini-swe-agent.trajectory.json")
    if native:
        status = (native.get("info") or {}).get("exit_status") or ""
        trial.exit_reason = _NATIVE_EXIT_STATUS.get(status, status or "killed")

    trajectory = _read_json(trial_dir / "agent" / "trajectory.json")
    if trajectory:
        notes = trajectory.get("notes") or ""
        if not trial.exit_reason and notes.startswith("exit_reason="):
            trial.exit_reason = notes.split("=", 1)[1]
        if not trial.n_steps:
            trial.n_steps = len(trajectory.get("steps") or [])
        if not trial.cost_usd:
            final = trajectory.get("final_metrics") or {}
            trial.cost_usd = final.get("total_cost_usd") or 0.0

    trial.completion, trial.completion_source = _completion(trial_dir)
    trial.verifier_ran = _verifier_ran(trial_dir)
    return trial


def _run_dir(job_dir: Path) -> Path:
    """The directory the trials are actually in.

    harbor writes `<jobs-dir>/<timestamp>/<trial>/`, and `<jobs-dir>` is what
    every command in this project is given -- it is the argument to
    `--jobs-dir`. Reading the jobs dir as if it held trials finds none and
    reports a comparison of zero against zero, with a delta of +0.00% and a
    p-value of 1.000: three numbers that look like an answer.

    A directory that already holds trials is used as-is, so a run dir still
    works when passed directly.
    """
    if any((c / "result.json").exists() for c in job_dir.iterdir() if c.is_dir()):
        # Could be either level; prefer the one whose children hold trials.
        deeper = [
            c for c in job_dir.iterdir()
            if c.is_dir() and any((g / "result.json").exists()
                                  for g in c.iterdir() if g.is_dir())
        ]
        if deeper:
            return max(deeper, key=lambda d: d.name)
        return job_dir
    runs = [c for c in job_dir.iterdir() if c.is_dir()]
    return max(runs, key=lambda d: d.name) if runs else job_dir


def load_job(job_dir: str | Path) -> Job:
    job_dir = Path(job_dir)
    job = Job(path=job_dir)
    if not job_dir.is_dir():
        return job
    for child in sorted(_run_dir(job_dir).iterdir()):
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


def _sign_test(a: int, b: int) -> float:
    """Two-sided sign test over discordant pairs.

    The paired form is the only one that means anything here: two runs share a
    task list, most tasks agree, and what carries information is which way the
    disagreements fall. Comparing two whole-run means over different task sets
    is the mistake this module used to make in `compare` itself.
    """
    import math

    n = a + b
    if n == 0:
        return 1.0
    k = max(a, b)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / (2 ** n))


def compare(baseline: Job, candidate: Job) -> dict:
    """Compare two runs task by task.

    The aggregate delta is the headline, but the regressions matter more: a
    change that solves two new tasks and breaks two others has not helped, and
    the means alone would call that a tie.

    Two things this reports that it used to hide.

    **The scores are over the shared tasks.** They were the whole-job means,
    which is a delta between two different denominators — the exact total-vs-
    total comparison this project rules out everywhere else. `full_*_score` is
    still here, named for what it is.

    **A task that never reached the agent is not a regression.** `confirm_gate`
    set an environment variable that tmux 3.1c rejects, so both qemu tasks died
    in setup on every gated run; they appeared in `lost` beside real ones, and
    a mechanism was judged on two tasks it had silently deleted. `broke_setup`
    separates them, because the two call for opposite responses: one is a
    finding about the agent, the other a bug in the harness.
    """
    base = {t.task: t for t in baseline.trials}
    cand = {t.task: t for t in candidate.trials}
    shared = sorted(set(base) & set(cand))

    gained = [t for t in shared if cand[t].solved and not base[t].solved]
    lost = [t for t in shared if base[t].solved and not cand[t].solved]
    gained_all = gained
    # Trials the change stopped from running at all, either way round.
    # Which side broke matters as much as the fact that one did: an exclusion
    # rule nobody can audit becomes a self-serving one. Reported as
    # `broke_setup_side` so the reader can see the direction.
    broke_setup = [
        t for t in shared
        if (cand[t].category == "environment") != (base[t].category == "environment")
    ]
    broke_setup_side = {
        t: ("baseline" if base[t].category == "environment" else "candidate")
        for t in broke_setup
    }
    lost = [t for t in lost if t not in set(broke_setup)]
    gained = [t for t in gained_all if t not in set(broke_setup)]
    moved = [
        (t, (base[t].completion or 0), (cand[t].completion or 0))
        for t in shared
        if abs((cand[t].completion or 0) - (base[t].completion or 0)) > 0.05
    ]

    # The same task set the wins and losses are counted over. Leaving a
    # setup-broken task out of `lost` but inside the mean is incoherent, and
    # not by a little: on one real pair it turned +4.0% into -1.19%, because
    # the four tasks excluded from the attribution were four zeros still
    # sitting in one side's denominator. If they are not evidence about the
    # agent, they are not evidence in the score either. `full_*_score` keeps
    # the leaderboard view, where an errored trial is a zero like any other.
    scored = [t for t in shared if t not in set(broke_setup)]

    def _mean(job_by_task):
        if not scored:
            return 0.0
        return sum(1.0 for t in scored if job_by_task[t].solved) / len(scored)

    b, c = _mean(base), _mean(cand)
    # A run in progress is not a random sample of itself. Failing trials run
    # about twice as long, so at any moment the finished ones over-represent
    # successes -- `crux watch` says so about a single run, and the same bias
    # lands on a comparison against a run that *is* finished, in favour of
    # whichever side is still going.
    partial = [j.path.name for j in (baseline, candidate) if j.is_partial]
    return {
        "shared_tasks": len(scored),
        "excluded_tasks": len(shared) - len(scored),
        "partial": partial,
        "baseline_score": b,
        "candidate_score": c,
        "delta": c - b,
        "full_baseline_score": baseline.score,
        "full_candidate_score": candidate.score,
        "gained": gained,
        "lost": lost,
        "broke_setup": broke_setup,
        "broke_setup_side": broke_setup_side,
        "completion_moved": sorted(moved, key=lambda m: m[2] - m[1], reverse=True),
        # The delta alone says nothing about whether it survives the noise.
        # Two runs of one configuration disagree on 15-16% of tasks here, so an
        # 89-task run resolves about +-8 points; the sign test over the tasks
        # that actually moved is the instrument that respects that.
        "p_value": _sign_test(len(gained), len(lost)),
        # Scores cross scaffolds; process metrics do not, and this project got
        # that wrong three times in one day. A claude-code trajectory step is a
        # tool call; a crux step is a model turn carrying 1.85 shell commands.
        # Reading "221 steps against 16" off those two units produced a 14x
        # claim where the honest figure was 3.1x on comparable units -- and
        # model turns, the axis that costs wall-clock, run the other way
        # entirely.
        "cross_agent": bool(
            baseline.agent_name and candidate.agent_name
            and baseline.agent_name != candidate.agent_name
        ),
    }
