#!/usr/bin/env python3
"""Report on a Harbor job directory, or compare two.

    analyze.py <job_dir>                 report on one run
    analyze.py <baseline> <candidate>    compare two runs

Run it on the eval box, where the job directories live.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crux.analysis import compare, completion_histogram, load_job  # noqa: E402

CATEGORY_LABEL = {
    "solved": "解出",
    "false_completion": "自称完成但未通过",
    "out_of_turns": "步数耗尽",
    "format_error": "格式错误",
    "agent_timeout": "agent 超时",
    "context_exceeded": "上下文超限",
    "environment": "环境故障（与 agent 无关）",
}


def report(job_dir: str) -> None:
    job = load_job(job_dir)
    if not job.trials:
        print(f"{job_dir}: 没有 trial")
        return

    solved = [t for t in job.trials if t.solved]
    print(f"任务 {len(job.trials)}   得分 {job.score * 100:.2f}%   成本 ${job.cost_usd:.2f}")
    if solved:
        print("解出:", ", ".join(sorted(t.task for t in solved)))
    print()

    print("失败归类")
    counts = job.by_category()
    for category, count in counts.most_common():
        label = CATEGORY_LABEL.get(category, category)
        print(f"  {label:<26} {count}")

    # An environment failure says nothing about the agent, so a run with many
    # of them has not measured the agent at all.
    env = counts.get("environment", 0)
    if env:
        print(f"\n  ⚠ {env} 个 trial 未跑到 agent，本次未真正测到 agent 能力")

    graded = [t for t in job.trials if t.completion is not None]
    if graded:
        print(f"\n完成度分布（{len(graded)} 个任务有可读评分）")
        for label, count in completion_histogram(job):
            print(f"  {label:<10} {'█' * count} {count}")

    near = job.near_misses()
    if near:
        print(f"\n最接近的未解出任务（≥60%）")
        for trial in near[:12]:
            print(f"  {trial.completion * 100:5.1f}%  {trial.task}  ({trial.category})")


def main() -> None:
    if len(sys.argv) == 2:
        report(sys.argv[1])
    elif len(sys.argv) == 3:
        baseline, candidate = load_job(sys.argv[1]), load_job(sys.argv[2])
        result = compare(baseline, candidate)
        print(f"共同任务 {result['shared_tasks']}")
        print(f"  基线   {result['baseline_score'] * 100:6.2f}%")
        print(f"  候选   {result['candidate_score'] * 100:6.2f}%")
        print(f"  差值   {result['delta'] * 100:+6.2f}%")
        if result["gained"]:
            print("\n新解出:", ", ".join(result["gained"]))
        # Regressions are the point of a task-by-task diff: two gained and two
        # lost is a wash that the means would report as no change.
        if result["lost"]:
            print("退化（原本解出，现在没有）:", ", ".join(result["lost"]))
        moved = result["completion_moved"]
        if moved:
            print("\n完成度变化最大的任务")
            for task, before, after in moved[:10]:
                print(f"  {before * 100:5.1f}% -> {after * 100:5.1f}%  {task}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
