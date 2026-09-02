"""crux — the command line entry point.

Three things a person actually does with this project, behind one command:
solve a task in a directory, run the benchmark, and understand a run that has
already happened. They share the prompt and the toolkit, so they stay in one
place rather than drifting between scripts.

    crux solve "make the failing test pass"
    crux bench --tasks 12 --variant stock
    crux report <job-dir> [<other-job-dir>]
    crux prompt --variant default
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from crux import __version__


def _chat(args) -> int:
    # imported lazily: chat pulls in pi_agent, which needs harbor installed
    from crux.chat import cmd_chat

    return cmd_chat(args)

# config and yaml are imported lazily by the commands that need them. `report`
# only reads job directories, and requiring pydantic to do that means it fails
# on any box where the analysis would otherwise work fine.
VARIANTS = ('default', 'lean', 'no_apply_patch', 'no_grading', 'no_toolkit', 'stock')

RESOURCES = Path(__file__).parent / "resources"
HELPERS = {"apply_patch": "apply_patch.py", "crux-tools": "crux_tool.py"}

# The board at tbench.ai is Terminal-Bench 4.0 -- 66 tasks, a flat 8h agent
# budget, scored at -k 5. Opus 5 with Claude Code leads it at 51.8%. 2.1 is the
# older 89-task set the widely quoted figures came from, and terminal-bench-pro
# is a different benchmark entirely (Alibaba's, 400 tasks, its own board), not a
# version of this one.
DEFAULT_DATASET = "terminal-bench/terminal-bench@4.0.0"
LEGACY_DATASET = "terminal-bench/terminal-bench-2-1"
FRONTIER_DATASET = "terminal-bench/terminal-bench@latest"
# The model is self-hosted -- Qwen3.8-27B FP8 on four H20s, served by sglang and
# addressed through OPENAI_BASE_URL. Every number in this repo came from it, so
# it is the default; a run that silently used something else would not be
# comparable to any of them.
#
# Tokens are therefore free and wall clock is the constraint, which inverts the
# old advice to iterate on a cheap model and confirm on an expensive one. What
# costs now is the four cards, and the engine's aggregate throughput stops
# climbing at about 48 concurrent streams (see docs/ABLATION.md).
DEFAULT_AGENT = "crux.terminus_agent:CruxTerminusAgent"
DEFAULT_MODEL = "openai/qwen3.8-27b"
# Hosted models, for a cross-check that the result is not an artefact of this
# particular deployment.
HOSTED_MODELS = ("deepseek/deepseek-v4-flash", "openrouter/stealth/ox-alpha")

# These declare gpus=1. harbor's docker environment declares no GPU capability
# at all (environments/capabilities.py, `gpus: bool = False`), so the validation
# error does not just fail those trials — it propagates and aborts the whole
# job, taking every other trial in flight with it. Only modal, daytona, beam and
# opensandbox can allocate a GPU, and all of them are paid cloud sandboxes.
#
# The names are per dataset version, so they cannot be one flat list: 2.1 has
# these four, 4.0 dropped exam-pdf-eval and keeps the other three.
GPU_TASKS_BY_ORG = {
    "terminal-bench": (
        "exam-pdf-eval",
        "fp8-rmsnorm-gemm",
        "jax-speedrun-gpu",
        "math-eval-grader",
    ),
}


def gpu_tasks_for(dataset: str) -> tuple[str, ...]:
    """The GPU-requiring task names to exclude, for this dataset's org.

    A task name is qualified by org, so a hardcoded `terminal-bench/` prefix
    matches nothing on any other dataset while still printing "skipped=..." --
    which is how 88 trials of a terminal-bench-pro run went by with an exclusion
    that was a no-op. Returning an empty tuple makes the caller say so honestly.
    """
    return GPU_TASKS_BY_ORG.get(dataset.split("/", 1)[0], ())


def _write_config(cfg) -> Path:
    """Materialize the mini-swe-agent config this variant implies."""
    import yaml

    from crux.config import to_mini_config

    path = Path(tempfile.mkdtemp(prefix="crux-")) / "config.yaml"
    path.write_text(yaml.safe_dump(to_mini_config(cfg), sort_keys=False))
    return path


def _install_helpers(target: Path) -> list[str]:
    """Put the file tools somewhere the agent's shell will find them."""
    target.mkdir(parents=True, exist_ok=True)
    installed = []
    for name, source in HELPERS.items():
        dest = target / ("crux" if name == "crux-tools" else name)
        shutil.copy(RESOURCES / source, dest)
        dest.chmod(0o755)
        installed.append(dest.name)
    return installed


def cmd_solve(args) -> int:
    """Run the benchmarked agent against a task in a local directory.

    This used to shell out to mini-swe-agent while every number in the repo came
    from the Terminus base, so the CLI shipped the scaffold the measurements had
    rejected -- about 3 points lower on the same tasks, and unable to express
    entering an ssh session or a REPL at all.

    Now it runs the same agent the benchmark runs, against `LocalEnvironment`
    instead of a container, with an approval gate in front of commands that are
    expensive to get wrong. A benchmark trial can afford `rm -rf`; the user's
    working directory cannot.
    """
    import asyncio

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

    from crux.approval import Approval
    from crux.local_agent import LocalCruxAgent
    from crux.local_env import LocalEnvironment

    cwd = Path(args.cwd or os.getcwd()).resolve()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    approval = Approval(args.approval)

    env = LocalEnvironment(cwd=cwd)
    # BaseAgent writes its trajectory and pane dump under logs_dir. In a
    # benchmark that is the trial directory; here it goes beside the session so
    # a run can be read back afterwards.
    logs_dir = Path(env.trial_paths.agent_dir)
    agent = LocalCruxAgent(
        logs_dir=logs_dir,
        model_name=args.model,
        approval=approval,
        project_root=cwd,
        interactive=interactive,
        **({"variant": args.variant} if args.variant != "default" else {}),
    )

    print(
        f"crux {__version__}  agent={agent.name()}  variant={args.variant}  "
        f"model={args.model or 'default'}"
    )
    print(f"  {env.describe()}   approval={approval.value}"
          f"{'' if interactive else '  (non-interactive: anything needing a prompt is refused)'}")
    print(f"  logs: {logs_dir}")

    async def _run() -> None:
        await env.start()
        try:
            await agent.setup(env)
            await agent.run(args.task, env, AgentContext())
        finally:
            await env.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for command, reason in agent.blocked_commands:
        print(f"refused: {command.strip()}  ({reason})", file=sys.stderr)
    return 0


def running_harbor_jobs(exclude_pid: int | None = None) -> list[str]:
    """Other `harbor run` processes on this machine, as short descriptions.

    A second job is invisible from the inside: every status check reads one job
    directory, so a run that has quietly lost half the engine to a neighbour
    looks exactly like a run that is merely slow. Two arms at concurrency 24 put
    ~48 streams on an engine whose aggregate throughput peaks near 48 and then
    falls -- measured at 1109 tok/s for 24 and 507 for 48.
    """
    import re

    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []          # no reading of process state is not a reason to refuse

    jobs = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, args = parts
        if "harbor" not in args or " run " not in f" {args} ":
            continue
        if "--dataset" not in args:
            continue
        if not pid_s.isdigit() or (exclude_pid is not None and int(pid_s) == exclude_pid):
            continue
        if int(pid_s) == os.getpid():
            continue
        m = re.search(r"--jobs-dir\s+(\S+)", args)
        a = re.search(r"--agent\s+(\S+)", args)
        jobs.append(f"pid {pid_s}  agent={a.group(1) if a else '?'}  "
                    f"jobs-dir={m.group(1) if m else '?'}")
    return jobs

def _write_provenance(jobs_dir: str, command: list[str]) -> Path | None:
    """Record what code produced a run, beside the run.

    A job directory says what was measured but not what was measuring. Tonight a
    baseline went out with a prompt change that had never been scored, and
    nothing in its output would have said so -- the difference between an
    experiment and a number you find later and cannot place.

    Best effort: a run must not fail because git is missing or the tree is not a
    repository.
    """
    import hashlib
    import json
    from datetime import datetime, timezone

    root = Path(__file__).resolve().parent.parent.parent

    def _git(*args) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(root), *args],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    prompts = Path(__file__).with_name("prompts.py")
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "crux_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit above does not describe what ran.
        "git_dirty": bool(_git("status", "--porcelain")),
        "prompts_sha256": (
            hashlib.sha256(prompts.read_bytes()).hexdigest() if prompts.exists() else None
        ),
        "command": command,
    }
    try:
        target = Path(jobs_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "crux-provenance.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return path
    except OSError:
        return None

def cmd_bench(args) -> int:
    """Run the Terminal-Bench evaluation through Harbor."""
    if shutil.which("harbor") is None:
        print(
            "harbor is not installed. Install it with:\n"
            "  uv tool install 'harbor[modal]'",
            file=sys.stderr,
        )
        return 1

    others = running_harbor_jobs()
    if others and not args.allow_concurrent_jobs:
        print("another harbor run is already going on this machine:", file=sys.stderr)
        for job in others:
            print(f"  {job}", file=sys.stderr)
        print(
            "\nTwo arms at this concurrency put ~48 streams on the engine, where\n"
            "aggregate throughput peaks and then falls (1109 tok/s at 24, 507 at 48),\n"
            "so both runs get slower and neither is comparable to a solo one.\n"
            "Stop the other job, or pass --allow-concurrent-jobs if you mean it.",
            file=sys.stderr,
        )
        return 1

    # Trial output is thousands of small files; on a cluster node that belongs
    # on local disk, not on a shared network mount.
    jobs_dir = args.jobs_dir or ("/scratch/crux-jobs" if Path("/scratch").is_dir() else "jobs")

    # Every measurement in this repo came from the Terminus base; the
    # mini-swe-agent one that used to be hardcoded here scores about 3 points
    # lower and cannot express entering an ssh session at all. `--agent` exists
    # so a comparison arm -- claude-code, pi -- runs through the same command
    # with the same dataset, concurrency and attempts, which is the only shape
    # in which the difference between two runs means the scaffold.
    command = [
        "harbor", "run",
        "--dataset", args.dataset,
        "--agent", args.agent,
        "--model", args.model,
        "--n-attempts", str(args.attempts),
        "--n-concurrent", str(args.concurrent),
        "--jobs-dir", jobs_dir,
        "--env", args.env,
        "--yes",
    ]
    # variant is ours; claude-code and pi reject an unknown agent kwarg rather
    # than ignoring it, so passing it always would break every comparison arm.
    if args.agent.startswith("crux."):
        command += ["--ak", f"variant={args.variant}"]
    for kv in args.agent_kwarg or []:
        command += ["--ak", kv]
    if args.upload:
        command += ["--upload"]
    if args.agent_timeout_multiplier != 1.0:
        command += ["--agent-timeout-multiplier", str(args.agent_timeout_multiplier)]
    for task in getattr(args, "task", None) or []:
        command += ["--include-task-name", f"{args.dataset.split('/', 1)[0]}/{task}"]
    if args.tasks:
        command += ["--n-tasks", str(args.tasks)]
    if args.env_file and Path(args.env_file).exists():
        command += ["--env-file", args.env_file]
    org = args.dataset.split("/", 1)[0]
    gpu_tasks = gpu_tasks_for(args.dataset)
    if not args.include_gpu_tasks:
        for task in gpu_tasks:
            command += ["--exclude-task-name", f"{org}/{task}"]

    src = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"

    print(f"agent  ={args.agent}"
          f"{'  variant=' + args.variant if args.agent.startswith('crux.') else ''}")
    print(f"model  ={args.model}  concurrent={args.concurrent}  attempts={args.attempts}")
    print(f"jobs   ={jobs_dir}{'  (uploads to Harbor Hub when it finishes)' if args.upload else ''}")
    if args.agent_timeout_multiplier != 1.0:
        print(f"budget =agent timeout x{args.agent_timeout_multiplier:g}"
              f"  -- NOT submittable, harbor requires 1.0")
    if not args.include_gpu_tasks:
        print(
            f"skipped={', '.join(gpu_tasks)} (need a GPU)"
            if gpu_tasks
            else f"skipped=nothing (no GPU-task list known for org '{org}')"
        )
    prov = _write_provenance(jobs_dir, command)
    if prov is not None:
        print(f"record ={prov}")
    print()
    return subprocess.call(command, env=env)


# A fixed slice for development. Canaries are solved consistently, so one of
# them failing means the change broke something -- which matters more than any
# gain. Contested tasks failed close to the line, so a real improvement shows
# up there first. A full run is 85 tasks and hours; this is ten and minutes.
QUICK_CANARIES = (
    "log-summary-date-ranges",
    "openssl-selfsigned-cert",
    "pypi-server",
    "regex-log",
)
# The four never solved across ten runs, plus two that fail close to the line.
# Tasks already solved consistently belong in the canary set, not here.
QUICK_CONTESTED = (
    "qemu-alpine-ssh",
    "torch-pipeline-parallelism",
    "filter-js-from-html",
    "gpt2-codegolf",
    "extract-elf",
    "kv-store-grpc",
)


def cmd_quick(args) -> int:
    """Run the development slice rather than the whole benchmark.

    A preset over `crux bench`, not a third way of building the same command
    line. The two earlier copies drifted from each other and from this one: all
    three hardcoded a `terminal-bench/` prefix, and this one still named
    crux.agent:CruxAgent -- the base the measurements rejected -- while pointing
    at a default dataset whose task names its canary list does not contain, so
    `--include-task-name` would have matched nothing at all.

    The slice is pinned to 2.1 because that is where the canaries were chosen
    and measured; the names do not exist in 4.0.
    """
    slice_args = argparse.Namespace(**vars(args))
    slice_args.dataset = LEGACY_DATASET
    slice_args.task = list(QUICK_CANARIES + QUICK_CONTESTED)
    slice_args.attempts = getattr(args, "attempts", 1)
    slice_args.tasks = None
    slice_args.env = "docker"
    slice_args.env_file = ".env"
    slice_args.jobs_dir = getattr(args, "jobs_dir", None)
    slice_args.include_gpu_tasks = False
    slice_args.upload = False
    slice_args.allow_concurrent_jobs = getattr(args, "allow_concurrent_jobs", False)
    slice_args.agent_timeout_multiplier = getattr(args, "agent_timeout_multiplier", 1.0)
    slice_args.agent_kwarg = getattr(args, "agent_kwarg", None)
    slice_args.agent = getattr(args, "agent", None) or DEFAULT_AGENT

    print(f"slice   ={len(slice_args.task)} tasks on {LEGACY_DATASET}")
    print(f"canaries  {', '.join(QUICK_CANARIES)}")
    print(f"contested {', '.join(QUICK_CONTESTED)}")
    return cmd_bench(slice_args)


def cmd_report(args) -> int:
    """Summarize a run, or diff two."""
    from crux.analysis import compare, completion_histogram, load_job

    job = load_job(args.job)
    if not job.trials:
        print(f"{args.job}: no trials found", file=sys.stderr)
        return 1

    if args.other:
        other = load_job(args.other)
        result = compare(job, other)
        print(f"shared tasks     {result['shared_tasks']}")
        print(f"  baseline       {result['baseline_score'] * 100:6.2f}%")
        print(f"  candidate      {result['candidate_score'] * 100:6.2f}%")
        print(f"  delta          {result['delta'] * 100:+6.2f}%")
        if result["gained"]:
            print("\nnewly solved:", ", ".join(result["gained"]))
        # Regressions are why this is a task-by-task diff: two gained and two
        # lost is a wash that the means alone would report as no change.
        if result["lost"]:
            print("regressed:", ", ".join(result["lost"]))
        return 0

    solved = sorted(t.task for t in job.trials if t.solved)
    print(f"tasks {len(job.trials)}   score {job.score * 100:.2f}%   cost ${job.cost_usd:.2f}")
    if solved:
        print("solved:", ", ".join(solved))

    print("\nfailure categories")
    for category, count in job.by_category().most_common():
        print(f"  {category:<20} {count}")

    env_failures = job.by_category().get("environment", 0)
    if env_failures:
        print(
            f"\n  ! {env_failures} trials never reached the agent — "
            f"this run did not measure the agent"
        )

    graded = [t for t in job.trials if t.completion is not None]
    if graded:
        print(f"\ncompletion ({len(graded)} tasks with readable scoring)")
        for label, count in completion_histogram(job):
            print(f"  {label:<10} {'#' * count} {count}")

    near = job.near_misses()
    if near:
        print("\nclosest unsolved")
        for trial in near[:10]:
            print(f"  {trial.completion * 100:5.1f}%  {trial.task}  ({trial.category})")
    return 0


def cmd_prompt(args) -> int:
    """Print the prompt a variant produces, for review or diffing."""
    import yaml

    from crux.config import build_config, to_mini_config

    cfg = build_config(variant=args.variant)
    config = to_mini_config(cfg)
    if args.yaml:
        print(yaml.safe_dump(config, sort_keys=False))
    else:
        print(config["agent"]["instance_template"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crux", description=__doc__)
    parser.add_argument("--version", action="version", version=f"crux {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve", help="run the agent on a task here")
    p.add_argument("task", help="what to do")
    p.add_argument("-m", "--model")
    p.add_argument("--cwd", help="directory to work in")
    p.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    # Default to asking about the destructive and outward-facing ones only.
    # "never" is what the benchmark uses, where the container is disposable.
    p.add_argument(
        "--approval",
        default="dangerous",
        choices=("never", "dangerous", "always"),
        help="how much to ask before running a command (default: dangerous)",
    )
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("bench", help="run Terminal-Bench")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    p.add_argument("-n", "--tasks", type=int, help="limit task count")
    p.add_argument(
        "-k",
        "--attempts",
        type=int,
        default=1,
        help="trials per task; leaderboard submissions need at least 5",
    )
    p.add_argument(
        "-c",
        "--concurrent",
        type=int,
        default=24,
        help=(
            "keep this fixed across a comparison: tasks have timeouts, so "
            "changing it changes scores"
        ),
    )
    p.add_argument("--env", default="docker")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--jobs-dir")
    p.add_argument(
        "--include-gpu-tasks",
        action="store_true",
        help="do not skip the GPU tasks (needs a GPU-capable sandbox: modal, "
             "daytona, beam or opensandbox -- harbor's docker env has none)",
    )
    # The benchmarked agent by default. Anything else is a comparison arm.
    p.add_argument(
        "-a",
        "--agent",
        default=DEFAULT_AGENT,
        help="harbor agent to run (default: the benchmarked one; try "
             "claude-code or pi for a scaffold comparison)",
    )
    p.add_argument(
        "--ak",
        "--agent-kwarg",
        dest="agent_kwarg",
        action="append",
        metavar="K=V",
        help="extra agent kwarg, repeatable (e.g. --ak model_api=openai-completions)",
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="upload the finished job to Harbor Hub (private by default)",
    )
    p.add_argument(
        "--allow-concurrent-jobs",
        action="store_true",
        help="start even if another harbor run is going (they will share the engine)",
    )
    p.add_argument(
        "-t",
        "--task",
        action="append",
        metavar="NAME",
        help="run only this task, repeatable (unqualified; the dataset's org is added)",
    )
    # 2.1 gives a ~900s median, and on this hardware 76% of its failures are
    # wall clock rather than wrong answers -- tuning against it mostly measures
    # the engine's 129 tok/s. Opening the agent budget makes the score reflect
    # the agent. It also makes the run non-submittable: harbor requires the
    # multiplier to be 1.0, so anything measured this way is an internal
    # comparison and has to be reported as one.
    p.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=1.0,
        metavar="X",
        help="multiply the agent's budget (default 1.0 = submittable; 8 turns "
             "2.1's 900s median into 7200s, matching TB 3.0's own median)",
    )
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser(
        "quick",
        help="run the fast iteration slice (canaries + contested tasks)",
    )
    p.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.add_argument("-c", "--concurrent", type=int, default=10)
    p.set_defaults(func=cmd_quick)

    p = sub.add_parser("report", help="summarize a run, or diff two")
    p.add_argument("job")
    p.add_argument("other", nargs="?", help="compare against this run")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("prompt", help="print a variant's prompt")
    p.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    p.add_argument("--yaml", action="store_true", help="full config, not just the prompt")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser(
        "chat",
        help="interactive agent: pi's terminal UI with crux's measured sections",
    )
    p.add_argument("-m", "--model", help="model id (default: from the endpoint config)")
    p.add_argument(
        "--sections",
        default="scoring,harness",
        help="crux prompt sections to append, comma separated; empty for stock pi",
    )
    p.add_argument("-c", "--continue", dest="cont", action="store_true",
                   help="continue the previous session")
    p.add_argument("-r", "--resume", action="store_true", help="pick a session to resume")
    p.add_argument("rest", nargs="*", help="passed through to pi")
    p.set_defaults(func=_chat)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
