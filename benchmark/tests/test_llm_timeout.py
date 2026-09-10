"""The per-call ceiling has to scale with the budget it is protecting.

600 seconds was chosen against a 900-second budget, where it bounds a hung
request to two thirds of the trial. Every run since has used an 8x budget,
where the same constant is 8% -- and it stopped bounding hangs and started
killing work. `regex-chess` died on it three attempts deep:

    litellm.Timeout: timeout value=600.0, time taken=1801.36 seconds

That task is written down in this project as one of two that never solve,
priced at a reduced budget on the strength of never having solved.
"""

import pytest

from crux.terminus_agent import (
    _LLM_TIMEOUT_MAX_SEC,
    _LLM_TIMEOUT_MIN_SEC,
    _llm_timeout_for,
)


def test_the_default_budget_keeps_the_old_ceiling():
    # 900s * 0.25 = 225, below the floor: unchanged where it was calibrated.
    assert _llm_timeout_for(900.0) == _LLM_TIMEOUT_MIN_SEC


def test_the_eight_times_budget_gets_a_bigger_ceiling():
    # 7200 * 0.25 = 1800. The `regex-chess` turn that took 1801s at three
    # attempts of 600 now has one attempt long enough to finish.
    assert _llm_timeout_for(7200.0) == 1800.0


def test_the_ceiling_is_capped():
    assert _llm_timeout_for(10**6) == _LLM_TIMEOUT_MAX_SEC


def test_an_unknown_budget_falls_back_to_the_floor():
    for budget in (0.0, -1.0):
        assert _llm_timeout_for(budget) == _LLM_TIMEOUT_MIN_SEC


@pytest.mark.parametrize("budget", [900.0, 1800.0, 7200.0, 14400.0])
def test_a_hang_can_never_cost_the_whole_trial(budget):
    """Whatever the budget, one call must not be able to consume it."""
    assert _llm_timeout_for(budget) <= max(budget * 0.25, _LLM_TIMEOUT_MIN_SEC)


def test_an_explicit_kwarg_still_wins():
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import CruxTerminusAgent

    agent = CruxTerminusAgent(
        logs_dir=Path(tempfile.mkdtemp()), model_name="openai/x", llm_timeout=42
    )
    assert agent._llm_timeout == 42.0


def test_no_kwarg_means_derive_it_later():
    """Zero is the sentinel: the budget is not known until setup()."""
    import tempfile
    from pathlib import Path

    from crux.terminus_agent import CruxTerminusAgent

    agent = CruxTerminusAgent(logs_dir=Path(tempfile.mkdtemp()), model_name="openai/x")
    assert agent._llm_timeout == 0.0
