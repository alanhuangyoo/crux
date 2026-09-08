"""A constant that forgets its regime becomes a fact about the world.

Three defects in one week share the shape: a value correct where it was chosen,
carried unchanged into a regime where it is wrong, and then recorded as a
property of the tasks. The 600-second per-call ceiling is the expensive one --
two thirds of a 900-second budget, 8% of an 8x one, and the reason two tasks
were written down as never solving.

These tests are about the audit that would have said so before the run.
"""

import pytest

from crux.calibration import (
    CALIBRATION_BUDGET_SEC,
    CALIBRATION_CONTAINERS,
    CALIBRATION_MODEL,
    Run,
    audit,
)


def test_the_calibrated_regime_is_quiet():
    assert audit(Run()) == []


def test_a_ceiling_that_is_most_of_the_budget_is_flagged():
    """Where the 600s constant came from: a short budget makes it dominant."""
    lines = audit(Run(budget_sec=700.0))
    assert any("llm timeout" in ln and "hung call" in ln for ln in lines)


def test_a_ceiling_that_is_a_sliver_of_the_budget_is_flagged():
    """The failure this cost 12 slot-hours: 600s against a 7200s budget.

    With the fix the ceiling scales, so nothing fires at 8x. Pin the *rule*, on
    a budget large enough that even the cap is a sliver.
    """
    assert audit(Run(budget_sec=7200.0)) == []
    lines = audit(Run(budget_sec=200_000.0))
    assert any("llm timeout" in ln and "cut" in ln for ln in lines)


def test_an_unmeasured_model_is_flagged():
    lines = audit(Run(model="some-other-model"))
    assert any("max_tokens" in ln for ln in lines)
    assert any(CALIBRATION_MODEL in ln for ln in lines)


def test_the_measured_model_is_not_flagged():
    assert not any("max_tokens" in ln for ln in audit(Run(model=CALIBRATION_MODEL)))


def test_crossing_the_throughput_inflection_is_flagged():
    lines = audit(Run(concurrent=24, other_containers=24))
    assert any("inflection" in ln for ln in lines)


def test_a_neighbouring_job_counts_toward_the_inflection():
    """A second arm is invisible from inside the first one."""
    assert audit(Run(concurrent=30)) == []
    assert any("inflection" in ln for ln in audit(Run(concurrent=30, other_containers=15)))


def test_every_line_names_the_regime_it_was_measured_in():
    for line in audit(Run(budget_sec=200_000.0, model="x", concurrent=99)):
        assert "measured in:" in line


def test_a_check_that_raises_does_not_stop_a_run():
    """The audit runs at launch; it must never be the reason a run does not."""
    import crux.calibration as cal

    def boom(_):
        raise RuntimeError("nope")

    original = cal.CONSTANTS
    cal.CONSTANTS = (cal.Constant("x", "nowhere", boom),)
    try:
        lines = audit(Run())
        assert len(lines) == 1 and "RuntimeError" in lines[0]
    finally:
        cal.CONSTANTS = original


def test_running_concurrency_reads_what_the_job_list_says():
    from crux.cli import running_concurrency

    jobs = [
        "pid 1  agent=a  n-concurrent=15  jobs-dir=/x",
        "pid 2  agent=b  n-concurrent=18  jobs-dir=/y",
    ]
    assert running_concurrency(jobs) == 33


def test_an_unreadable_job_counts_as_zero_rather_than_a_guess():
    from crux.cli import running_concurrency

    assert running_concurrency(["pid 3  agent=c  n-concurrent=?  jobs-dir=/z"]) == 0


@pytest.mark.parametrize("budget", [900.0, 1800.0, 3600.0, 7200.0])
def test_no_supported_budget_multiplier_trips_the_timeout_check(budget):
    """x1 through x8 are the multipliers this project actually runs."""
    assert not any("llm timeout" in ln for ln in audit(Run(budget_sec=budget)))


def test_the_audit_runs_on_a_bare_interpreter():
    """It fires at launch, so it cannot depend on harbor being importable.

    The first version imported `crux.terminus_agent` to read the timeout, which
    pulls in harbor and pydantic; on the eval box's system python that raises
    ModuleNotFoundError, and the check that would have caught the 600-second
    constant reported "could not be checked" instead. Same principle as
    `crux report`: the paths that inspect a run must work where the run's own
    dependencies do not.
    """
    import pathlib
    import subprocess
    import sys

    src = str(pathlib.Path(__file__).resolve().parent.parent / "src")
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['pydantic'] = None; sys.modules['harbor'] = None; "
         "from crux.calibration import Run, audit, llm_timeout_for; "
         "assert llm_timeout_for(7200.0) == 1800.0; "
         "assert audit(Run()) == []; print('ok')"],
        capture_output=True, text=True,
        env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
    )
    assert "ok" in out.stdout, out.stderr
