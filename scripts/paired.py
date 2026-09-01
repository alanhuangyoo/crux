#!/usr/bin/env python3
"""Compare two runs task by task instead of comparing their means.

Two arms that differ by three points can mean either of two things:

    both agree on 80 tasks, B wins 8 of the 9 disagreements   -> real
    both agree on 59 tasks, B wins 16 and loses 14            -> noise

The means are identical in both cases. Only the disagreements separate them,
because tasks that answer the same way under both arms carry no information
about the change -- and on this benchmark roughly half the tasks are like that
(26 of 89 were solved by all eight arms, 14 by none).

Reads harbor's own reward_stats rather than counting result files, because a
directory listing includes trials that have not finished and scoring those as
failures once produced "33 wrong" against a mean of 51.5%.

    scripts/paired.py <baseline> <candidate>

Each argument is a run directory or a jobs directory holding several.
"""
import json
import math
import pathlib
import sys


def rates(run_dir: pathlib.Path) -> dict[str, tuple[int, int]]:
    """task -> (attempts solved, attempts run).

    A run made with --n-attempts 3 stores three trials per task. Keying a dict
    by task name silently keeps whichever attempt happened to be visited last,
    which turned a 56.9% arm into an apparent 73.6% one -- so count attempts
    rather than overwrite them.
    """
    evals = json.loads((run_dir / "result.json").read_text())["stats"]["evals"]
    if not evals:
        raise ValueError(f"{run_dir.name} has no eval results (aborted run?)")
    out: dict[str, list[int]] = {}
    for reward, trials in list(evals.values())[0].get("reward_stats", {}).get("reward", {}).items():
        for trial in trials:
            got, ran = out.setdefault(trial.split("__")[0], [0, 0])
            out[trial.split("__")[0]] = [got + (float(reward) > 0), ran + 1]
    return {t: (g, r) for t, (g, r) in out.items()}


def latest(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, tuple[int, int]]]:
    """Newest run under `path` that actually produced results.

    Newest-by-mtime alone is not enough: an aborted run leaves a result.json
    with an empty `evals`, and picking it silently compares against nothing.
    """
    candidates = [path] if (path / "result.json").exists() else sorted(
        (p for p in path.iterdir() if (p / "result.json").exists()),
        key=lambda p: p.stat().st_mtime, reverse=True)
    problems = []
    for run in candidates:
        try:
            return run, rates(run)
        except (ValueError, KeyError, IndexError) as exc:
            problems.append(f"  {run.name}: {exc}")
    sys.exit(f"no usable run under {path}\n" + "\n".join(problems))


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: paired.py <baseline> <candidate>")
    (a_dir, a), (b_dir, b) = (latest(pathlib.Path(p)) for p in sys.argv[1:3])

    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("the two runs share no task names")

    na = max(r for _, r in a.values())
    nb = max(r for _, r in b.values())
    score = lambda d: sum(g for t in shared for g, _ in [d[t]]) / sum(r for t in shared for _, r in [d[t]])
    print(f"baseline  {a_dir.name}  {score(a)*100:.1f}%  ({na} attempt(s)/task)")
    print(f"candidate {b_dir.name}  {score(b)*100:.1f}%  ({nb} attempt(s)/task)")
    if len(shared) < max(len(a), len(b)):
        print(f"  ({max(len(a), len(b)) - len(shared)} 题只在一轮里出现,已忽略)")

    if na == 1 and nb == 1:
        only_a = sorted(t for t in shared if a[t][0] and not b[t][0])
        only_b = sorted(t for t in shared if b[t][0] and not a[t][0])
        both = sum(1 for t in shared if a[t][0] and b[t][0])
        neither = len(shared) - both - len(only_a) - len(only_b)
        print(f"\n两版一致 {both + neither} 题  (都对 {both} / 都错 {neither})  -- 不含信息")
        print(f"分歧 {len(only_a) + len(only_b)} 题:  候选赢 {len(only_b)}   候选输 {len(only_a)}")

        # McNemar. Under "the change does nothing" each disagreement is a coin
        # flip, so only the disagreements enter -- which is why this sees more
        # than the two means do.
        n = len(only_a) + len(only_b)
        if n == 0:
            verdict = "两版逐题完全相同 —— 这个改动什么都没动"
        else:
            z = (len(only_b) - len(only_a)) / math.sqrt(n)
            if abs(z) < 1.96:
                need = math.ceil((1.96 * math.sqrt(n) + n) / 2)
                verdict = (f"分不出来 (z={z:+.2f})。这么多分歧下要赢到 {need} 题才算数,"
                           f"现在赢 {len(only_b)} 题")
            else:
                verdict = f"{'候选更好' if z > 0 else '候选更差'} (z={z:+.2f})"
    else:
        # Attempts differ or exceed one: per-task solve rates are no longer
        # binary, so compare their paired differences instead.
        diffs = [b[t][0] / b[t][1] - a[t][0] / a[t][1] for t in shared]
        moved = [d for d in diffs if d]
        up = sum(1 for d in moved if d > 0)
        print(f"\n两版一致 {len(shared) - len(moved)} 题 -- 不含信息")
        print(f"通过率变化 {len(moved)} 题:  候选升 {up}   候选降 {len(moved) - up}")
        if not moved:
            verdict = "两版逐题完全相同 —— 这个改动什么都没动"
        else:
            mean = sum(diffs) / len(diffs)
            var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) if len(diffs) > 1 else 0.0
            se = math.sqrt(var / len(diffs)) if var else 0.0
            z = mean / se if se else 0.0
            verdict = (f"{'候选更好' if z > 0 else '候选更差'} (z={z:+.2f}, 平均 {mean*100:+.1f} 分/题)"
                       if abs(z) >= 1.96 else
                       f"分不出来 (z={z:+.2f}, 平均 {mean*100:+.1f} 分/题)")
        only_a = sorted(t for t in shared if b[t][0] / b[t][1] < a[t][0] / a[t][1])
        only_b = sorted(t for t in shared if b[t][0] / b[t][1] > a[t][0] / a[t][1])

    print(f"\n结论: {verdict}")
    for label, tasks, sign in (("候选赢的题", only_b, "+"),
                               ("候选输的题 -- 回归,比涨分更值得看", only_a, "-")):
        if tasks:
            print(f"\n{label} ({len(tasks)}):")
            for t in tasks:
                print(f"  {sign} {t}")


if __name__ == "__main__":
    main()
