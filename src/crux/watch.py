"""Watch runs while they happen, and read them the way that survives contact.

Written after a night of doing this by hand. Four throwaway scripts got written
to answer the same four questions -- how far along, how fast, is anything stuck,
who is winning -- and each of them got the answer wrong at least once in a way
that took a while to notice:

  * A score read from a partly finished run is biased upward, always. Failed
    trials take about twice as long as solved ones, so the trials that have
    finished are enriched for success: SWE-bench read 96% at 26 of 99 and 84%
    at 68.
  * Two runs of the same configuration disagree on 15-16% of tasks. A single
    89-task run resolves about ±5 points, so a lead smaller than that is not a
    lead.
  * Comparing totals compares different task sets. Arms do not finish the same
    tasks at the same time, and the finished subset has run several points
    harder than the full set for hours at a stretch.
  * A trial whose container died is recorded completed-with-error and never
    retried, so the denominator shrinks quietly and the run looks done.

So every number this prints carries what qualifies it: how much is left, how
biased it still is, and what the noise floor is. `--compare` is paired on shared
tasks with a sign test, never total against total.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import time
from pathlib import Path

from crux import ui

# Two runs of the same configuration on this benchmark disagree on this
# fraction of tasks. Measured, not assumed: crux-jobs vs crux-w-jobs, 85 shared
# tasks, 14 flips, identical scores (53 vs 53); crux-m-jobs vs crux-v-jobs, 89
# shared, 13 flips.
NOISE_FLIP_RATE = 0.155


def _latest(jobs_dir: str) -> Path | None:
    runs = sorted(glob.glob(os.path.join(jobs_dir, "*/")), key=os.path.getmtime)
    return Path(runs[-1]) if runs else None


def outcomes(jobs_dir: str) -> dict[str, bool]:
    """Task -> solved, from the newest result.json under a jobs dir.

    Keyed by stripping only the trailing trial hash. Splitting on the first
    `__` instead -- the obvious thing -- collapses every `django__django-NNNN`
    into one key and silently reports 8 of 9 for a run of a hundred.
    """
    files = sorted(glob.glob(os.path.join(jobs_dir, "*/result.json")), key=os.path.getmtime)
    if not files:
        return {}
    out: dict[str, bool] = {}
    try:
        stats = json.load(open(files[-1]))["stats"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {}
    for ev in stats.get("evals", {}).values():
        for reward, names in ev.get("reward_stats", {}).get("reward", {}).items():
            for name in names:
                out[name.rsplit("__", 1)[0]] = float(reward) >= 1.0
    return out


def progress(jobs_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(jobs_dir, "*/result.json")), key=os.path.getmtime)
    if not files:
        return {}
    try:
        r = json.load(open(files[-1]))
    except (OSError, json.JSONDecodeError):
        return {}
    s = r.get("stats", {})
    res = outcomes(jobs_dir)
    return {
        "total": r.get("n_total_trials", 0),
        "done": s.get("n_completed_trials", 0),
        "running": s.get("n_running_trials", 0),
        "errored": s.get("n_errored_trials", 0),
        "scored": len(res),
        "solved": sum(res.values()),
    }


def _rate(jobs_dir: str, window_sec: float = 3600) -> float:
    """Completions per hour, from verifier artefact mtimes."""
    now = time.time()
    ts = []
    for pat in ("*/*/verifier/reward.txt", "*/*/verifier/report.json"):
        ts += [os.path.getmtime(f) for f in glob.glob(os.path.join(jobs_dir, pat))]
    recent = [t for t in set(ts) if now - t < window_sec]
    return len(recent) / (window_sec / 3600)


def _stalled(jobs_dir: str, quiet_sec: float = 1800) -> list[tuple[str, float, int]]:
    """Live trials whose trajectory has not been touched in a while."""
    run = _latest(jobs_dir)
    if run is None:
        return []
    now = time.time()
    out = []
    for p in glob.glob(str(run / "*/agent/trajectory.json")):
        age = now - os.path.getmtime(p)
        if age < quiet_sec:
            continue
        if glob.glob(os.path.join(os.path.dirname(os.path.dirname(p)), "verifier", "*")):
            continue
        try:
            steps = len(json.load(open(p)).get("steps", []))
        except (OSError, json.JSONDecodeError):
            steps = -1
        out.append((Path(p).parts[-3], age, steps))
    return sorted(out, key=lambda x: -x[1])


def unscored(jobs_dir: str) -> list[str]:
    """Trials with a directory and no score -- the denominator that shrank."""
    run = _latest(jobs_dir)
    if run is None:
        return []
    scored = set(outcomes(jobs_dir))
    out = []
    for d in sorted(glob.glob(str(run / "*/"))):
        name = Path(d.rstrip("/")).name.rsplit("__", 1)[0]
        if name in scored:
            continue
        if glob.glob(os.path.join(d, "verifier", "reward.txt")):
            continue
        out.append(name)
    return out


def _sign_test(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = max(a, b)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / (2 ** n))


def _resolution(n: int) -> float:
    """Points of score a single run of n tasks can actually resolve.

    Two runs of one configuration disagree on NOISE_FLIP_RATE of tasks, split
    both ways, so the difference between two runs has a standard deviation of
    roughly sqrt(n * p) / n. Two of those is the smallest gap worth calling a
    gap.
    """
    if n <= 0:
        return 0.0
    return 2 * 100 * math.sqrt(n * NOISE_FLIP_RATE) / n


def render(jobs: list[str], show_stalled: bool = True) -> str:
    lines = [ui.rule("crux watch")]
    head = f"  {'run':<22}{'done':>12}{'scored':>12}{'rate':>10}{'left':>10}"
    lines.append(ui.hint(head))
    for jd in jobs:
        p = progress(jd)
        name = Path(jd.rstrip("/")).name
        if not p:
            lines.append(f"  {name:<22}{ui.hint('not started')}")
            continue
        rate = _rate(jd)
        left = p["total"] - p["done"]
        eta = f"{left / rate:.1f}h" if rate > 0 and left else ("done" if not left else "—")
        pct = f"{p['solved']}/{p['scored']}" if p["scored"] else "—"
        share = f" ({p['solved'] / p['scored'] * 100:.0f}%)" if p["scored"] else ""
        err = ui.error(f"  {p['errored']} err") if p["errored"] else ""
        lines.append(
            f"  {name:<22}{p['done']:>5}/{p['total']:<6}{pct:>12}{share:<7}"
            f"{rate:>6.1f}/h{eta:>10}{err}"
        )
        if p["scored"] and p["done"] < p["total"]:
            lines.append(ui.hint(
                f"    that rate is biased up: failed trials run about twice as long, so"
                f" the {p['scored']} scored so far over-represent successes"))
        if p["scored"]:
            r = _resolution(p["total"])
            lines.append(ui.hint(
                f"    a single {p['total']}-task run resolves about ±{r:.1f} points"))
        miss = unscored(jd)
        if miss:
            lines.append(ui.warn(
                f"    {len(miss)} trial(s) have no score and will not be retried"))
        if show_stalled:
            for name_, age, steps in _stalled(jd)[:3]:
                lines.append(ui.warn(
                    f"    quiet {ui.human_secs(age)}: {name_[:44]} ({steps} steps)"))
    return "\n".join(lines)


def compare(a: str, b: str, baseline: str | None = None) -> str:
    """Paired on shared tasks, with a sign test. Never total against total."""
    oa, ob = outcomes(a), outcomes(b)
    na, nb = Path(a.rstrip("/")).name, Path(b.rstrip("/")).name
    shared = sorted(set(oa) & set(ob))
    lines = [ui.rule(f"{na} vs {nb}")]
    if not shared:
        return "\n".join(lines + [ui.hint("  no task has a score in both yet")])
    sa = sum(oa[t] for t in shared)
    sb = sum(ob[t] for t in shared)
    lines.append(f"  shared tasks: {len(shared)}")
    lines.append(f"    {na:<24}{sa}/{len(shared)} = {sa / len(shared) * 100:.1f}%")
    lines.append(f"    {nb:<24}{sb}/{len(shared)} = {sb / len(shared) * 100:.1f}%")
    if baseline:
        obase = outcomes(baseline)
        both = [t for t in shared if t in obase]
        if both:
            sc = sum(obase[t] for t in both)
            lines.append(f"    {Path(baseline.rstrip('/')).name:<24}"
                         f"{sc}/{len(both)} = {sc / len(both) * 100:.1f}%")
            full = sum(obase.values()) / max(1, len(obase)) * 100
            here = sc / len(both) * 100
            lines.append(ui.hint(
                f"    this subset runs {abs(here - full):.1f} points "
                f"{'harder' if here < full else 'easier'} than the baseline's full set"))
    aw = sum(1 for t in shared if oa[t] and not ob[t])
    bw = sum(1 for t in shared if ob[t] and not oa[t])
    p = _sign_test(aw, bw)
    verdict = ui.ok("significant") if p < 0.05 else ui.warn("not significant")
    lines.append(f"  disagreements: {aw + bw}   {na} {aw}, {nb} {bw}   "
                 f"sign test p={p:.4f}  {verdict}")
    if aw + bw:
        exp = len(shared) * NOISE_FLIP_RATE
        lines.append(ui.hint(
            f"    {exp:.0f} flips are expected from run-to-run noise alone on "
            f"{len(shared)} tasks"))
    if aw:
        lines.append(ui.hint(f"    only {na}: " + ", ".join(t for t in shared if oa[t] and not ob[t])[:200]))
    if bw:
        lines.append(ui.hint(f"    only {nb}: " + ", ".join(t for t in shared if ob[t] and not oa[t])[:200]))
    return "\n".join(lines)


def cmd_watch(args) -> int:
    jobs = args.jobs or []
    if args.compare:
        print(compare(args.compare[0], args.compare[1],
                      args.baseline if hasattr(args, "baseline") else None))
        return 0
    if not jobs:
        print(ui.error("give at least one jobs dir"))
        return 1
    if not args.interval:
        print(render(jobs))
        return 0
    try:
        while True:
            print(render(jobs))
            print(ui.hint(f"  {time.strftime('%H:%M:%S')}  (Ctrl-C to stop)"))
            time.sleep(args.interval)
            print()
    except KeyboardInterrupt:
        return 0
