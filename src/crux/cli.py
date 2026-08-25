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

# config and yaml are imported lazily by the commands that need them. `report`
# only reads job directories, and requiring pydantic to do that means it fails
# on any box where the analysis would otherwise work fine.
VARIANTS = ('default', 'no_apply_patch', 'no_grading', 'no_toolkit', 'stock')

RESOURCES = Path(__file__).parent / "resources"
HELPERS = {"apply_patch": "apply_patch.py", "crux-tools": "crux_tool.py"}

# terminal-bench@latest is Terminal-Bench 3, the frontier set, and it is
# deliberately harder than 2.1. The public leaderboard figures everyone quotes
# — mini-SWE-agent at 76.2%, Claude Code at 83.8% — are on 2.1, so that is the
# dataset to measure against unless the frontier set is the explicit target.
DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"
FRONTIER_DATASET = "terminal-bench/terminal-bench@latest"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

# These declare gpus=1. Without an nvidia runtime the validation error does not
# just fail those trials — it propagates and aborts the whole job, taking every
# other trial in flight with it.
GPU_TASKS = (
    "exam-pdf-eval",
    "fp8-rmsnorm-gemm",
    "jax-speedrun-gpu",
    "math-eval-grader",
)


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
    """Run the agent against a task in a local directory.

    Delegates to mini-swe-agent's own runner rather than reimplementing a loop
    — this is its agent, with our prompt and our tools.
    """
    if shutil.which("mini") is None:
        print(
            "mini-swe-agent is not installed. Install it with:\n"
            "  uv tool install mini-swe-agent",
            file=sys.stderr,
        )
        return 1

    from crux.config import build_config

    cfg = build_config(variant=args.variant)
    config_path = _write_config(cfg)

    env = dict(os.environ)
    if cfg.toolkit or cfg.apply_patch:
        bin_dir = Path(tempfile.mkdtemp(prefix="crux-bin-"))
        _install_helpers(bin_dir)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

    command = ["mini", "-c", str(config_path), "-t", args.task, "-y"]
    if args.model:
        command += ["-m", args.model]
    if args.cwd:
        command += ["--cwd", args.cwd]

    print(f"crux {__version__}  variant={cfg.variant}  model={args.model or 'default'}")
    return subprocess.call(command, env=env)


def cmd_bench(args) -> int:
    """Run the Terminal-Bench evaluation through Harbor."""
    if shutil.which("harbor") is None:
        print(
            "harbor is not installed. Install it with:\n"
            "  uv tool install 'harbor[modal]'",
            file=sys.stderr,
        )
        return 1

    # Trial output is thousands of small files; on a cluster node that belongs
    # on local disk, not on a shared network mount.
    jobs_dir = args.jobs_dir or ("/scratch/crux-jobs" if Path("/scratch").is_dir() else "jobs")

    command = [
        "harbor", "run",
        "--dataset", args.dataset,
        "--agent", "crux.agent:CruxAgent",
        "--ak", f"variant={args.variant}",
        "--model", args.model,
        "--n-attempts", str(args.attempts),
        "--n-concurrent", str(args.concurrent),
        "--jobs-dir", jobs_dir,
        "--env", args.env,
        "--yes",
    ]
    if args.tasks:
        command += ["--n-tasks", str(args.tasks)]
    if args.env_file and Path(args.env_file).exists():
        command += ["--env-file", args.env_file]
    if not args.include_gpu_tasks:
        for task in GPU_TASKS:
            command += ["--exclude-task-name", f"terminal-bench/{task}"]

    src = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"

    print(f"variant={args.variant}  model={args.model}  concurrent={args.concurrent}")
    print(f"jobs   ={jobs_dir}")
    if not args.include_gpu_tasks:
        print(f"skipped={', '.join(GPU_TASKS)} (need a GPU)")
    print()
    return subprocess.call(command, env=env)


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
        help="do not skip the four GPU tasks (needs an nvidia runtime)",
    )
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("report", help="summarize a run, or diff two")
    p.add_argument("job")
    p.add_argument("other", nargs="?", help="compare against this run")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("prompt", help="print a variant's prompt")
    p.add_argument("--variant", default="default", choices=sorted(VARIANTS))
    p.add_argument("--yaml", action="store_true", help="full config, not just the prompt")
    p.set_defaults(func=cmd_prompt)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
