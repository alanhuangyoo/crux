"""The analysis pipeline decides what we work on next, so its judgements are pinned.

Two of them cost real time to learn and would silently mislead if they
regressed: an environment failure must not be read as an agent failure, and a
scoring harness's pytest wrapper must not be read as its score.
"""

import json

import pytest

from crux.analysis import Job, Trial, compare, completion_histogram, load_job


def write_trial(
    root,
    name,
    *,
    reward=None,
    exception=None,
    exit_reason=None,
    ctrf=None,
    breakdown=None,
    steps=10,
    cost=0.1,
):
    d = root / name
    (d / "verifier").mkdir(parents=True)
    (d / "agent").mkdir(parents=True)

    result = {"agent_result": {"metadata": {"n_steps": steps}, "cost_usd": cost}}
    if exception:
        result["exception_info"] = {"exception_type": exception}
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward}}
    (d / "result.json").write_text(json.dumps(result))

    if exit_reason:
        (d / "agent" / "trajectory.json").write_text(
            json.dumps({"notes": f"exit_reason={exit_reason}", "steps": []})
        )
    if ctrf:
        passed, total = ctrf
        (d / "verifier" / "ctrf.json").write_text(
            json.dumps({"results": {"summary": {"passed": passed, "tests": total}}})
        )
    if breakdown:
        (d / "verifier" / "breakdown.json").write_text(
            json.dumps({"checks": breakdown})
        )
    return d


def test_solved_trial(tmp_path):
    write_trial(tmp_path, "alpha__x", reward=1.0, exit_reason="completed")
    job = load_job(tmp_path)
    assert job.score == 1.0
    assert job.trials[0].category == "solved"


def test_environment_failure_is_not_blamed_on_the_agent(tmp_path):
    """Three full runs scored 0.000 for environment reasons.

    Reading those as agent failures would have sent us tuning prompts against
    a box where two thirds of trials never started.
    """
    write_trial(tmp_path, "alpha__x", exception="RuntimeError")
    write_trial(tmp_path, "beta__y", exception="VerifierTimeoutError")
    job = load_job(tmp_path)
    assert job.by_category()["environment"] == 2
    assert job.score == 0.0


def test_false_completion_is_distinct_from_running_out_of_turns(tmp_path):
    """Opposite fixes: one is a judgement failure, the other a budget one."""
    write_trial(tmp_path, "alpha__x", reward=0.0, exit_reason="completed")
    write_trial(tmp_path, "beta__y", reward=0.0, exit_reason="step_limit")
    counts = load_job(tmp_path).by_category()
    assert counts["false_completion"] == 1
    assert counts["out_of_turns"] == 1


def test_breakdown_wins_over_the_pytest_wrapper(tmp_path):
    """A scoring harness reports 1/1 passing tests for a trial that scored 0.

    Trusting ctrf here overstated one run's progress badly.
    """
    write_trial(
        tmp_path,
        "alpha__x",
        reward=0.0,
        ctrf=(1, 1),
        breakdown=[
            {"weight": 0.5, "passed": False},
            {"weight": 0.5, "passed": False},
        ],
    )
    trial = load_job(tmp_path).trials[0]
    assert trial.completion == 0.0
    assert trial.completion_source == "weighted"


def test_pytest_counts_used_when_there_is_no_breakdown(tmp_path):
    write_trial(tmp_path, "alpha__x", reward=0.0, ctrf=(1, 2))
    trial = load_job(tmp_path).trials[0]
    assert trial.completion == 0.5
    assert trial.completion_source == "pytest"


def test_weighted_completion_respects_weights(tmp_path):
    write_trial(
        tmp_path,
        "alpha__x",
        reward=0.0,
        breakdown=[
            {"weight": 0.9, "passed": True},
            {"weight": 0.1, "passed": False},
        ],
    )
    assert load_job(tmp_path).trials[0].completion == pytest.approx(0.9)


def test_near_misses_exclude_solved_and_sort_by_closeness(tmp_path):
    write_trial(tmp_path, "solved__x", reward=1.0, ctrf=(4, 4))
    write_trial(tmp_path, "close__y", reward=0.0, ctrf=(9, 10))
    write_trial(tmp_path, "closer__z", reward=0.0, ctrf=(19, 20))
    write_trial(tmp_path, "far__w", reward=0.0, ctrf=(1, 10))
    near = load_job(tmp_path).near_misses(threshold=0.6)
    assert [t.task for t in near] == ["closer", "close"]


def test_errored_trials_count_as_zero_like_the_leaderboard(tmp_path):
    """Harbor's rules: an errored trial is reward 0 and may not be excluded."""
    write_trial(tmp_path, "alpha__x", reward=1.0)
    write_trial(tmp_path, "beta__y", exception="RuntimeError")
    assert load_job(tmp_path).score == 0.5


def test_compare_reports_regressions_not_just_the_delta(tmp_path):
    """Two gained and two lost is a wash the means would call no change."""
    base_dir, cand_dir = tmp_path / "base", tmp_path / "cand"
    base_dir.mkdir(), cand_dir.mkdir()
    write_trial(base_dir, "kept__x", reward=1.0)
    write_trial(base_dir, "regressed__y", reward=1.0)
    write_trial(base_dir, "gained__z", reward=0.0)
    write_trial(cand_dir, "kept__x", reward=1.0)
    write_trial(cand_dir, "regressed__y", reward=0.0)
    write_trial(cand_dir, "gained__z", reward=1.0)

    result = compare(load_job(base_dir), load_job(cand_dir))
    assert result["delta"] == 0.0
    assert result["gained"] == ["gained"]
    assert result["lost"] == ["regressed"]


def test_histogram_bins_completion():
    job = Job(path=None, trials=[Trial(task=f"t{i}", completion=c) for i, c in
                                 enumerate([0.05, 0.15, 0.5, 0.85, 0.95])])
    bins = dict(completion_histogram(job))
    assert bins["0-20%"] == 2
    assert bins["40-60%"] == 1
    assert bins["80-100%"] == 2


def test_missing_result_file_is_skipped(tmp_path):
    (tmp_path / "junk__x").mkdir()
    (tmp_path / "not-a-dir.txt").write_text("x")
    assert load_job(tmp_path).trials == []


def test_scoring_wrapper_is_not_mistaken_for_progress(tmp_path):
    """Three tasks in one run looked 100% solved on a wrapper's say-so.

    Some verifiers wrap a weighted scorer in a single pytest that passes
    whenever the scorer ran, and ship hygiene checks for the verifier itself.
    Both report all-green for a trial that scored nothing.
    """
    d = write_trial(tmp_path, "wrapped__x", reward=0.0)
    (d / "verifier" / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {"passed": 1, "tests": 1},
                    "tests": [{"name": "test_scoring.py::test_score_agent"}],
                }
            }
        )
    )
    trial = load_job(tmp_path).trials[0]
    assert trial.completion_source == "wrapper"
    assert load_job(tmp_path).near_misses() == []


def test_hygiene_suite_is_also_a_wrapper(tmp_path):
    d = write_trial(tmp_path, "hygiene__x", reward=0.0)
    (d / "verifier" / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {"passed": 4, "tests": 4},
                    "tests": [
                        {"name": "test_verifier_hygiene.py::test_a"},
                        {"name": "test_verifier_hygiene.py::test_b"},
                    ],
                }
            }
        )
    )
    assert load_job(tmp_path).trials[0].completion_source == "wrapper"


def test_a_real_suite_is_still_trusted(tmp_path):
    d = write_trial(tmp_path, "real__x", reward=0.0)
    (d / "verifier" / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {"passed": 9, "tests": 10},
                    "tests": [
                        {"name": "test_outputs.py::test_blocks_xss"},
                        {"name": "test_scoring.py::test_score_agent"},
                    ],
                }
            }
        )
    )
    trial = load_job(tmp_path).trials[0]
    assert trial.completion_source == "pytest"
    assert trial.completion == 0.9


def write_native_trajectory(trial_dir, exit_status):
    (trial_dir / "agent" / "mini-swe-agent.trajectory.json").write_text(
        json.dumps({"info": {"exit_status": exit_status}, "messages": []})
    )


def test_native_exit_status_is_read(tmp_path):
    """Harbor's ATIF conversion drops mini-swe-agent's own outcome.

    Reading only the converted notes made every trial look like out_of_turns —
    the category that says nothing about what to fix.
    """
    d = write_trial(tmp_path, "alpha__x", reward=0.0)
    write_native_trajectory(d, "Submitted")
    assert load_job(tmp_path).trials[0].category == "false_completion"


def test_killed_is_distinct_from_giving_up(tmp_path):
    """An empty status means the process died before recording one."""
    d = write_trial(tmp_path, "alpha__x", reward=0.0)
    write_native_trajectory(d, "")
    trial = load_job(tmp_path).trials[0]
    assert trial.exit_reason == "killed"
    assert trial.category == "killed"


def test_native_format_error_maps_across(tmp_path):
    d = write_trial(tmp_path, "alpha__x", reward=0.0)
    write_native_trajectory(d, "RepeatedFormatError")
    assert load_job(tmp_path).trials[0].category == "format_error"


def test_solved_still_wins_over_native_status(tmp_path):
    d = write_trial(tmp_path, "alpha__x", reward=1.0)
    write_native_trajectory(d, "Submitted")
    assert load_job(tmp_path).trials[0].category == "solved"


def test_solved_wins_over_a_timeout(tmp_path):
    """Harbor records an agent timeout and runs the verifier anyway.

    Ranking the timeout first hid two solved tasks in the first pi/crux
    comparison, so the category counts disagreed with the solved list printed
    next to them.
    """
    write_trial(tmp_path, "alpha__x", reward=1.0, exception="AgentTimeoutError")
    job = load_job(tmp_path)
    assert job.trials[0].category == "solved"
    assert job.score == 1.0


def test_category_counts_match_the_solved_list(tmp_path):
    write_trial(tmp_path, "a__x", reward=1.0, exception="AgentTimeoutError")
    write_trial(tmp_path, "b__y", reward=1.0)
    write_trial(tmp_path, "c__z", reward=0.0)
    job = load_job(tmp_path)
    assert job.by_category()["solved"] == sum(1 for t in job.trials if t.solved) == 2


# --------------------------------------------------------------------------
# a diff must not price a task that never ran


def _trial(task, reward=None, exception=None):
    from crux.analysis import Trial

    return Trial(task=task, reward=reward, exception=exception)


def _job(trials):
    from pathlib import Path

    from crux.analysis import Job

    return Job(path=Path("/tmp/x"), trials=trials)


def test_compare_scores_the_shared_tasks_not_the_whole_jobs():
    """A delta between two denominators is not a delta.

    The candidate ran one extra task and solved it. Its whole-job mean is
    higher; on the tasks both actually ran, the two agree.
    """
    from crux.analysis import compare

    base = _job([_trial("a", 1.0), _trial("b", 0.0)])
    cand = _job([_trial("a", 1.0), _trial("b", 0.0), _trial("c", 1.0)])
    r = compare(base, cand)
    assert r["shared_tasks"] == 2
    assert r["delta"] == 0.0
    assert r["full_candidate_score"] > r["full_baseline_score"]


def test_a_task_that_died_in_setup_is_not_a_regression():
    """The shape that cost `confirm_gate` two tasks on every gated run.

    tmux 3.1c rejects the `-e` the gate's environment variable travelled on, so
    both qemu tasks failed before the agent typed anything -- and landed in
    `lost` looking like the mechanism had broken them.
    """
    from crux.analysis import compare

    base = _job([_trial("qemu-startup", 1.0), _trial("x", 1.0)])
    cand = _job([
        _trial("qemu-startup", None, exception="RuntimeError"),
        _trial("x", 1.0),
    ])
    r = compare(base, cand)
    assert r["broke_setup"] == ["qemu-startup"]
    assert r["lost"] == []


def test_a_real_regression_still_reads_as_one():
    from crux.analysis import compare

    base = _job([_trial("x", 1.0)])
    cand = _job([_trial("x", 0.0)])
    r = compare(base, cand)
    assert r["lost"] == ["x"]
    assert r["broke_setup"] == []
    assert r["delta"] == -1.0
