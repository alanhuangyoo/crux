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


def _job_at(tmp_path, name, trials, planned=None):
    import json
    from pathlib import Path

    from crux.analysis import Job

    d = Path(tmp_path) / name
    d.mkdir(parents=True, exist_ok=True)
    if planned is not None:
        (d / "result.json").write_text(json.dumps({"n_total_trials": planned}))
    return Job(path=d, trials=trials)


def test_a_comparison_against_a_run_still_going_says_so(tmp_path):
    """A run in progress is not a random sample of itself.

    Failing trials take about twice as long, so the finished ones
    over-represent successes -- and against a run that *is* finished the bias
    lands entirely on one side. `crux watch` already warns about this for a
    single run; the diff was quoting it without a word.
    """
    from crux.analysis import compare

    base = _job_at(tmp_path, "done", [_trial("a", 1.0), _trial("b", 0.0)], planned=2)
    cand = _job_at(tmp_path, "running", [_trial("a", 1.0), _trial("b", 1.0)], planned=89)
    r = compare(base, cand)
    assert r["partial"] == ["running"]


def test_two_finished_runs_carry_no_warning(tmp_path):
    from crux.analysis import compare

    base = _job_at(tmp_path, "x", [_trial("a", 1.0)], planned=1)
    cand = _job_at(tmp_path, "y", [_trial("a", 0.0)], planned=1)
    assert compare(base, cand)["partial"] == []


def test_a_run_whose_config_cannot_be_read_is_treated_as_complete(tmp_path):
    """Unreadable is not the same as suspect; it must not warn on every run."""
    from crux.analysis import compare

    base = _job_at(tmp_path, "p", [_trial("a", 1.0)])
    cand = _job_at(tmp_path, "q", [_trial("a", 1.0)])
    assert compare(base, cand)["partial"] == []


def test_the_endpoint_failing_is_not_the_agent_running_out_of_turns():
    """114 trials in the corpus were read as the agent giving up.

    An endpoint 500, a setup timeout, a missing reward file -- none of them are
    statements about the agent, and which arm they land on is chance. They
    still score zero, as a leaderboard scores them; they must not be
    attributable in a paired comparison.
    """
    from crux.analysis import Trial

    for exc in ("InternalServerError", "RateLimitError", "AgentSetupTimeoutError",
                "EnvironmentStartTimeoutError", "AddTestsDirError",
                "RewardFileNotFoundError"):
        assert Trial(task="t", reward=None, exception=exc).category == "environment"


def test_a_real_agent_failure_is_still_the_agents():
    from crux.analysis import Trial

    # n_steps matters now: an exception with no agent activity is the box's,
    # and a timeout after 40 steps is the agent's.
    assert Trial(task="t", reward=0.0, exception="AgentTimeoutError",
                 n_steps=40).category == "agent_timeout"
    assert Trial(task="t", reward=0.0, exit_reason="completed").category == "false_completion"


def test_an_environment_failure_is_excluded_from_a_paired_diff(tmp_path):
    from crux.analysis import compare

    base = _job_at(tmp_path, "a", [_trial("x", 1.0)], planned=1)
    cand = _job_at(tmp_path, "b", [_trial("x", None, exception="InternalServerError")],
                   planned=1)
    r = compare(base, cand)
    assert r["broke_setup"] == ["x"] and r["lost"] == []


def test_an_exception_with_no_agent_activity_is_environmental():
    """A better rule than a list of exception names.

    pi's baseline lost 16 of 89 trials to `curl ... nvm/install.sh` failing in
    setup -- no trajectory, no steps, the agent never started -- and they were
    counted as pi failing those tasks. The exception is called
    NonZeroAgentExitCodeError, which sounds like the agent's fault and is not.
    """
    from crux.analysis import Trial

    t = Trial(task="x", reward=None, exception="NonZeroAgentExitCodeError",
              n_steps=0, n_output_tokens=0)
    assert t.category == "environment"


def test_the_did_it_run_test_is_output_tokens_not_steps():
    """Steps is a Terminus-shaped question and pi does not answer it.

    pi writes `agent/pi` and `agent/pi.txt` and records no step count for any
    of its 89 trials, so a steps-based rule reads zero for all of them and
    excuses its 7 genuine agent timeouts along with the 16 real setup deaths --
    inflating the score of the baseline the rule exists to be fair to. On this
    corpus the token split is exact: the 16 setup deaths have zero output
    tokens and nothing else does.
    """
    from crux.analysis import Trial

    ran = Trial(task="x", reward=0.0, exception="AgentTimeoutError",
                n_steps=0, n_output_tokens=65_929)
    assert ran.category == "agent_timeout"

    never = Trial(task="x", reward=None, exception="AgentTimeoutError",
                  n_steps=0, n_output_tokens=0)
    assert never.category == "environment"


def test_an_exception_after_the_agent_ran_is_not_environmental():
    """The other half of the rule: a crash mid-run is still about the agent."""
    from crux.analysis import Trial

    t = Trial(task="x", reward=0.0, exception="AgentTimeoutError", n_steps=40)
    assert t.category == "agent_timeout"


def test_compare_reports_the_sign_test_over_discordant_pairs(tmp_path):
    """The delta alone has repeatedly looked like a result at this noise level.

    Two runs of one configuration disagree on 15-16% of tasks, so an 89-task
    run resolves about +-8 points. What carries information is which way the
    disagreements fall, not the difference of two means.
    """
    from crux.analysis import compare

    base = _job_at(tmp_path, "b", [_trial(f"t{i}", 1.0) for i in range(6)], planned=6)
    # Candidate loses all six: as one-sided as six pairs can be.
    cand = _job_at(tmp_path, "c", [_trial(f"t{i}", 0.0) for i in range(6)], planned=6)
    r = compare(base, cand)
    assert len(r["lost"]) == 6 and not r["gained"]
    assert r["p_value"] < 0.05


def test_a_split_decision_does_not_separate(tmp_path):
    from crux.analysis import compare

    base = _job_at(tmp_path, "b2",
                   [_trial("a", 1.0), _trial("b", 0.0), _trial("c", 1.0), _trial("d", 0.0)],
                   planned=4)
    cand = _job_at(tmp_path, "c2",
                   [_trial("a", 0.0), _trial("b", 1.0), _trial("c", 0.0), _trial("d", 1.0)],
                   planned=4)
    r = compare(base, cand)
    assert len(r["gained"]) == 2 and len(r["lost"]) == 2
    assert r["p_value"] == 1.0


def test_two_identical_runs_report_no_evidence(tmp_path):
    from crux.analysis import compare

    trials = [_trial("a", 1.0), _trial("b", 0.0)]
    r = compare(_job_at(tmp_path, "x1", trials, planned=2),
                _job_at(tmp_path, "x2", list(trials), planned=2))
    assert r["p_value"] == 1.0


def _harbor_layout(tmp_path, name, run, tasks):
    """`<jobs-dir>/<timestamp>/<trial>/result.json`, as harbor writes it."""
    import json

    d = tmp_path / name / run
    for task, reward in tasks.items():
        t = d / f"{task}__hash"
        t.mkdir(parents=True)
        (t / "result.json").write_text(json.dumps({
            "task_name": f"terminal-bench/{task}",
            "verifier_result": {"rewards": {"reward": reward}},
            "agent_result": {"n_output_tokens": 100},
        }))
    (d / "result.json").write_text(json.dumps({"n_total_trials": len(tasks)}))
    return str(tmp_path / name)


def test_load_job_descends_into_the_run_directory(tmp_path):
    """`--jobs-dir` is what every command here is given.

    Reading it as if it held trials finds none and reports zero against zero:
    a delta of +0.00% and p=1.000, three numbers that look like an answer.
    """
    from crux.analysis import load_job

    jobs = _harbor_layout(tmp_path, "run1", "2026-09-08__00-00-00",
                          {"alpha": 1.0, "beta": 0.0})
    job = load_job(jobs)
    assert sorted(t.task for t in job.trials) == ["alpha", "beta"]
    assert job.score == 0.5


def test_a_run_directory_passed_directly_still_works(tmp_path):
    from crux.analysis import load_job

    _harbor_layout(tmp_path, "run2", "2026-09-08__00-00-00", {"alpha": 1.0})
    job = load_job(tmp_path / "run2" / "2026-09-08__00-00-00")
    assert [t.task for t in job.trials] == ["alpha"]


def test_the_latest_run_wins_when_a_jobs_dir_holds_several(tmp_path):
    from crux.analysis import load_job

    _harbor_layout(tmp_path, "run3", "2026-09-01__00-00-00", {"old": 0.0})
    jobs = _harbor_layout(tmp_path, "run3", "2026-09-08__00-00-00", {"new": 1.0})
    assert [t.task for t in load_job(jobs).trials] == ["new"]


def test_a_missing_jobs_dir_is_empty_not_an_error(tmp_path):
    from crux.analysis import load_job

    assert load_job(tmp_path / "nope").trials == []


def test_a_setup_broken_task_leaves_the_score_as_well_as_the_attribution(tmp_path):
    """Dropping it from `lost` but keeping it in the mean is incoherent.

    On a real pair it turned +4.0% into -1.19%: four tasks excluded from the
    attribution were four zeros still sitting in one side's denominator.
    """
    from crux.analysis import compare

    base = _job_at(tmp_path, "bb", [_trial("a", 1.0), _trial("b", 1.0)], planned=2)
    cand = _job_at(tmp_path, "cc", [
        _trial("a", 1.0),
        _trial("b", None, exception="RuntimeError"),
    ], planned=2)
    r = compare(base, cand)
    assert r["broke_setup"] == ["b"]
    assert r["shared_tasks"] == 1 and r["excluded_tasks"] == 1
    # One task, solved by both: no delta, not -50%.
    assert r["delta"] == 0.0
    assert r["baseline_score"] == 1.0 and r["candidate_score"] == 1.0
    # The leaderboard view still counts it as a zero.
    assert r["full_candidate_score"] == 0.5


def test_a_gain_on_a_setup_broken_task_is_not_credited_either(tmp_path):
    """Symmetry: exclusion must not be a way to bank a win."""
    from crux.analysis import compare

    base = _job_at(tmp_path, "b3", [_trial("a", None, exception="RuntimeError")], planned=1)
    cand = _job_at(tmp_path, "c3", [_trial("a", 1.0)], planned=1)
    r = compare(base, cand)
    assert r["gained"] == [] and r["broke_setup"] == ["a"]
