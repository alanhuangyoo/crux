"""Reading a run in flight is where this project has been wrong most often.

Each test below is a mistake that was actually made, once, on real data:

  * keying a task by everything before the first `__` collapses every
    django__django-NNNN into one bucket and reports 8 of 9 for a run of a
    hundred;
  * comparing totals compares different task sets, because arms do not finish
    the same tasks at the same time;
  * a trial whose container died is recorded completed-with-error and never
    retried, so the denominator shrinks and the run looks finished.
"""

import json
from pathlib import Path

import pytest

from crux import watch


def _run(tmp_path, name, rewards, extra_dirs=(), unfinished=0):
    """A jobs dir shaped like harbor's, with one finished run in it."""
    jobs = tmp_path / name
    run = jobs / "2026-01-01__00-00-00"
    run.mkdir(parents=True)
    by_reward: dict[str, list[str]] = {}
    for task, solved in rewards.items():
        trial = f"{task}__hash{len(by_reward)}{len(by_reward.get('1.0', []))}"
        by_reward.setdefault("1.0" if solved else "0.0", []).append(trial)
        d = run / trial
        (d / "verifier").mkdir(parents=True)
        (d / "verifier" / "reward.txt").write_text("1" if solved else "0")
    for name_ in extra_dirs:
        (run / f"{name_}__deadhash").mkdir(parents=True)
    (jobs / "result.json").write_text(json.dumps({
        "n_total_trials": len(rewards) + len(extra_dirs) + int(unfinished),
        "stats": {
            "n_completed_trials": len(rewards) + len(extra_dirs),
            "n_running_trials": 0,
            "n_errored_trials": len(extra_dirs),
            "evals": {"e": {"reward_stats": {"reward": by_reward}}},
        },
    }))
    (run / "result.json").write_text((jobs / "result.json").read_text())
    return str(jobs)


def test_task_names_are_not_collapsed_at_the_first_separator(tmp_path):
    # The bug: split("__")[0] turns every django task into "django".
    jobs = _run(tmp_path, "swe", {
        "django__django-13023": True,
        "django__django-11099": True,
        "astropy__astropy-14369": False,
    })
    res = watch.outcomes(jobs)
    assert set(res) == {
        "django__django-13023", "django__django-11099", "astropy__astropy-14369",
    }
    assert sum(res.values()) == 2


def test_unscored_trials_are_counted_not_ignored(tmp_path):
    # A dead trial has a directory and no verifier output. Silently dropping it
    # shrinks the denominator and makes a broken run look finished.
    jobs = _run(tmp_path, "tb", {"a": True, "b": False}, extra_dirs=("c", "d"))
    assert sorted(watch.unscored(jobs)) == ["c", "d"]
    p = watch.progress(jobs)
    assert p["scored"] == 2
    assert p["total"] == 4
    assert p["errored"] == 2


def test_comparison_is_paired_on_shared_tasks(tmp_path):
    # Arm A finished an easy task the other has not reached. Comparing totals
    # would credit A for it; pairing must not.
    a = _run(tmp_path, "arm-a", {"x": True, "y": False, "easy": True})
    b = _run(tmp_path, "arm-b", {"x": True, "y": True})
    out = watch.compare(a, b)
    assert "shared tasks: 2" in out
    assert "1/2" in out and "2/2" in out


def test_comparison_reports_a_sign_test_on_disagreements(tmp_path):
    a = _run(tmp_path, "gate-on", {f"t{i}": True for i in range(5)})
    b = _run(tmp_path, "gate-off", {f"t{i}": False for i in range(5)})
    out = watch.compare(a, b)
    assert "disagreements: 5" in out
    # 5-0 one way is p = 2/32.
    assert "p=0.0625" in out


def test_a_tie_is_reported_as_not_significant(tmp_path):
    a = _run(tmp_path, "a", {"t1": True, "t2": False})
    b = _run(tmp_path, "b", {"t1": False, "t2": True})
    out = watch.compare(a, b)
    assert "p=1.0000" in out
    assert "not significant" in out


def test_resolution_shrinks_as_the_task_set_grows():
    # An 89-task run resolves several points; the number is what stops a
    # two-point lead being called a lead.
    small = watch._resolution(29)
    big = watch._resolution(89)
    assert small > big > 0
    assert 5 < big < 12


def test_resolution_of_nothing_is_zero():
    assert watch._resolution(0) == 0.0


def test_empty_jobs_dir_is_not_an_error(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert watch.outcomes(str(empty)) == {}
    assert watch.progress(str(empty)) == {}
    assert watch.unscored(str(empty)) == []


def test_render_warns_that_a_partial_score_is_biased_up(tmp_path):
    jobs = _run(tmp_path, "partial", {"a": True}, extra_dirs=("b", "c"), unfinished=2)
    out = watch.render([jobs], show_stalled=False)
    assert "biased up" in out
    assert "no score" in out


@pytest.mark.parametrize("a,b,expected", [(0, 0, 1.0), (5, 0, 0.0625), (1, 1, 1.0)])
def test_sign_test_values(a, b, expected):
    assert watch._sign_test(a, b) == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------
# partial credit, and where it is worth having


def _with_ctrf(tmp_path, name, per_task):
    """A jobs dir whose trials carry pytest's CTRF report."""
    jobs = tmp_path / name
    run = jobs / "2026-01-01__00-00-00"
    run.mkdir(parents=True)
    for task, (passed, total) in per_task.items():
        v = run / f"{task}__hash" / "verifier"
        v.mkdir(parents=True)
        (v / "ctrf.json").write_text(json.dumps({
            "results": {"summary": {"tests": total, "passed": passed, "failed": total - passed}}
        }))
    (jobs / "result.json").write_text(json.dumps({
        "n_total_trials": len(per_task),
        "stats": {"n_completed_trials": len(per_task), "n_running_trials": 0,
                  "n_errored_trials": 0, "evals": {}},
    }))
    return str(jobs)


def test_partial_reads_the_pytest_report(tmp_path):
    jobs = _with_ctrf(tmp_path, "tb", {"a": (3, 3), "b": (1, 4)})
    assert watch.partial(jobs) == {"a": (3, 3), "b": (1, 4)}


def test_partial_reads_the_swebench_report(tmp_path):
    jobs = tmp_path / "swe"
    v = jobs / "run" / "django__django-1__h" / "verifier"
    v.mkdir(parents=True)
    (v / "report.json").write_text(json.dumps({"django__django-1": {"tests_status": {
        "FAIL_TO_PASS": {"success": ["t1", "t2"], "failure": ["t3"]},
        "PASS_TO_PASS": {"success": ["p1"], "failure": []},
    }}}))
    (jobs / "result.json").write_text(json.dumps({"n_total_trials": 1, "stats": {}}))
    # 3 of 4 across both groups, and the task name keeps its repo prefix.
    assert watch.partial(str(jobs)) == {"django__django-1": (3, 4)}


def test_partial_reads_the_rubric_report(tmp_path):
    jobs = tmp_path / "atlas"
    v = jobs / "run" / "task-1__h" / "verifier"
    v.mkdir(parents=True)
    (v / "evaluation_results.json").write_text(
        json.dumps({"num_rubrics": 6, "num_passed": 5, "reward": 0})
    )
    (jobs / "result.json").write_text(json.dumps({"n_total_trials": 1, "stats": {}}))
    # The binary reward is 0; five of six rubrics is what it threw away.
    assert watch.partial(str(jobs)) == {"task-1": (5, 6)}


def test_a_trial_with_no_report_is_simply_absent(tmp_path):
    jobs = tmp_path / "none"
    (jobs / "run" / "t__h" / "verifier").mkdir(parents=True)
    (jobs / "result.json").write_text(json.dumps({"n_total_trials": 1, "stats": {}}))
    assert watch.partial(str(jobs)) == {}


def test_the_comparison_reports_both_scores(tmp_path):
    a = _run(tmp_path, "a", {"x": True, "y": False})
    b = _run(tmp_path, "b", {"x": True, "y": False})
    # give both a CTRF report under the same run dir the binary fixture made
    for jobs, frac in ((a, (3, 3)), (b, (1, 3))):
        run = sorted((tmp_path / Path(jobs).name).glob("2026-*"))[0]
        for d in run.glob("y__*"):
            (d / "verifier").mkdir(exist_ok=True)
            (d / "verifier" / "ctrf.json").write_text(json.dumps({
                "results": {"summary": {"tests": frac[1], "passed": frac[0]}}
            }))
    out = watch.compare(a, b)
    assert "tests passed" in out
    # binary says they tie on y; the fractions do not
    assert "carry a signal here" in out
