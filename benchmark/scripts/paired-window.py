#!/usr/bin/env python3
"""Paired comparison restricted to the window where both arms were running.

Two identical configurations run at the same time differ by 3.8 points on this
box; run one after the other they differed by 16.7. The gap is the box changing
between windows, not the agent. So arms are launched together -- but they do not
*finish* together: the arm whose trials die early on context exhaustion gets
through the task list much faster, and its partner's remaining trials then run
alone on a quieter box.

Comparing totals across that would be comparing a concurrent arm against a
partly-sequential one, which is the thing the concurrent design exists to avoid.
So only tasks whose trials started while both arms were still working are
counted, and the rest are reported as excluded rather than silently dropped.

    paired_window.py <arm-a> <arm-b>
"""
import glob
import json
import math
import os
import pathlib
import sys
from datetime import datetime, timezone


def run_dir(path: pathlib.Path) -> pathlib.Path:
    if (path / "result.json").exists():
        return path
    runs = [p for p in path.iterdir() if (p / "result.json").exists()]
    if not runs:
        sys.exit(f"no run under {path}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def parse(stamp: str) -> float:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def trial_start(trial: str) -> float | None:
    """The `session` event's timestamp: when this trial's agent actually began."""
    path = os.path.join(trial, "agent", "pi.txt")
    if not os.path.exists(path):
        return None
    with open(path, errors="ignore") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "session" and event.get("timestamp"):
                return parse(event["timestamp"])
            break
    return None


def arm(path: pathlib.Path):
    run = run_dir(path)
    scored: dict[str, bool] = {}
    evals = json.loads((run / "result.json").read_text())["stats"]["evals"]
    if evals:
        rewards = list(evals.values())[0].get("reward_stats", {}).get("reward", {})
        for reward, trials in rewards.items():
            for trial in trials:
                scored[trial] = float(reward) > 0
    starts: dict[str, tuple[str, float]] = {}
    for trial in sorted(glob.glob(str(run) + "/*/")):
        name = os.path.basename(trial.rstrip("/"))
        if name not in scored:
            continue
        started = trial_start(trial)
        if started is not None:
            starts[name.split("__")[0]] = (name, started)
    return run, scored, starts


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (run_a, scored_a, start_a), (run_b, scored_b, start_b) = (arm(pathlib.Path(p)) for p in sys.argv[1:3])

    now = datetime.now(timezone.utc).timestamp()
    # An arm is "still working" until its newest trial stopped changing; a
    # finished arm's window ends at its last write.
    end_a = max(run_a.stat().st_mtime, max((s for _, s in start_a.values()), default=0))
    end_b = max(run_b.stat().st_mtime, max((s for _, s in start_b.values()), default=0))
    window = (max(min(s for _, s in start_a.values()), min(s for _, s in start_b.values())), min(end_a, end_b, now))

    shared = sorted(set(start_a) & set(start_b))
    inside, outside = [], []
    for task in shared:
        if window[0] <= start_a[task][1] <= window[1] and window[0] <= start_b[task][1] <= window[1]:
            inside.append(task)
        else:
            outside.append(task)

    def rate(scored, starts, tasks):
        got = sum(scored[starts[t][0]] for t in tasks)
        return got, len(tasks)

    print(f"both-running window: {datetime.fromtimestamp(window[0]).strftime('%H:%M')}"
          f" - {datetime.fromtimestamp(window[1]).strftime('%H:%M')}")
    print(f"shared tasks: {len(shared)}   inside the window: {len(inside)}   excluded: {len(outside)}")
    if not inside:
        sys.exit("nothing to compare yet")

    ga, na = rate(scored_a, start_a, inside)
    gb, nb = rate(scored_b, start_b, inside)
    print(f"\n  A {run_a.name}  {ga}/{na} = {100*ga/na:.1f}%")
    print(f"  B {run_b.name}  {gb}/{nb} = {100*gb/nb:.1f}%")

    b_wins = [t for t in inside if scored_b[start_b[t][0]] and not scored_a[start_a[t][0]]]
    a_wins = [t for t in inside if scored_a[start_a[t][0]] and not scored_b[start_b[t][0]]]
    agree = len(inside) - len(a_wins) - len(b_wins)
    n = len(a_wins) + len(b_wins)
    print(f"\n  agree {agree}   disagree {n}:  B wins {len(b_wins)}, A wins {len(a_wins)}")
    if n:
        z = (len(b_wins) - n / 2) / math.sqrt(n / 4)
        need = math.ceil(n / 2 + 1.96 * math.sqrt(n / 4))
        print(f"  sign test z={z:+.2f}")
        if need > n:
            # Below four disagreements even a clean sweep sits under z=1.96, so
            # there is no split of this many that would mean anything. Saying
            # "needs 4 of 3" instead of saying that is how a 3-0 gets read as
            # nearly significant.
            print(f"  {n} disagreements cannot reach significance at all -- a clean sweep is z={n/math.sqrt(n):+.2f}")
        else:
            print(f"  B needs {need} of {n} disagreements to clear noise")
    for t in b_wins:
        print(f"    + {t}")
    for t in a_wins:
        print(f"    - {t}")


if __name__ == "__main__":
    main()
