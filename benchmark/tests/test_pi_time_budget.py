"""pi's time budget follows the limit harbor enforces on the trial.

A fixed PI_TIME_BUDGET_SEC=7200 met limits of 80 to 1,600 minutes: on two
tasks pi was killed before its own deadline, on 48 the two coincided, and on 39
it stopped itself with harbor's time still left -- 19 failed trials over four
arms, train-fasttext four times at a 480-minute limit.
"""

import json
from types import SimpleNamespace

from crux.pi_agent import CruxPiAgent, _harbor_agent_limit_sec, _time_budget

# The [agent] table as train-fasttext's own task.toml writes it.
TRAIN_FASTTEXT = """schema_version = "1.1"

[task]
name = "terminal-bench/train-fasttext"

[verifier]
timeout_sec = 3600.0

[agent]
timeout_sec = 3600.0

[environment]
build_timeout_sec = 600.0
"""


def make_trial(tmp_path, task_toml=TRAIN_FASTTEXT, config=None):
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "task.toml").write_text(task_toml)
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    if config is None:
        config = {"agent_timeout_multiplier": 8.0}
    (trial_dir / "config.json").write_text(json.dumps(config))
    return task_dir, trial_dir


def test_the_limit_is_the_tasks_timeout_times_the_multiplier(tmp_path):
    task_dir, trial_dir = make_trial(tmp_path)
    assert _harbor_agent_limit_sec(task_dir, trial_dir) == 3600.0 * 8


def test_an_override_replaces_the_tasks_timeout_and_a_cap_bounds_it(tmp_path):
    task_dir, trial_dir = make_trial(
        tmp_path,
        config={"agent_timeout_multiplier": 2.0, "agent": {"override_timeout_sec": 1000, "max_timeout_sec": 500}},
    )
    assert _harbor_agent_limit_sec(task_dir, trial_dir) == 500 * 2.0


def test_a_multiplier_left_at_its_default_falls_back_the_way_harbor_does(tmp_path):
    # config.json is written with exclude_defaults, so the key is simply absent.
    task_dir, trial_dir = make_trial(tmp_path, config={"timeout_multiplier": 3.0})
    assert _harbor_agent_limit_sec(task_dir, trial_dir) == 3600.0 * 3.0
    task_dir, trial_dir = make_trial(tmp_path / "b", config={})
    assert _harbor_agent_limit_sec(task_dir, trial_dir) == 3600.0


def test_nothing_readable_means_no_limit(tmp_path):
    assert _harbor_agent_limit_sec(tmp_path / "missing", tmp_path / "missing") is None
    task_dir, trial_dir = make_trial(tmp_path, task_toml="[agent]\n")
    assert _harbor_agent_limit_sec(task_dir, trial_dir) is None


def test_a_number_is_only_ever_shortened_to_fit():
    # 120-minute limit: the deadline now lands a minute before the kill.
    assert _time_budget("7200", 7200.0) == "7140"
    # 80-minute limit: pi was killed with its own deadline 40 minutes away.
    assert _time_budget("7200", 4800.0) == "4740"
    # 480-minute limit: the configured budget stands, so earlier arms compare.
    assert _time_budget("7200", 28800.0) == "7200"


def test_task_takes_the_whole_limit():
    assert _time_budget("task", 28800.0) == "28740"
    assert _time_budget("task", None) is None


def test_an_unknown_limit_or_an_unparseable_setting_is_left_alone():
    assert _time_budget("7200", None) == "7200"
    assert _time_budget("soon", 7200.0) == "soon"
    assert _time_budget(None, 7200.0) is None


def agent_with(pi_env, trial_dir):
    agent = CruxPiAgent.__new__(CruxPiAgent)
    agent._pi_env = dict(pi_env)
    agent.logs_dir = trial_dir / "agent"
    return agent


def test_the_agent_applies_it_before_the_settings_are_delivered(tmp_path):
    task_dir, trial_dir = make_trial(tmp_path)
    agent = agent_with({"PI_TIME_BUDGET_SEC": "task", "PI_STOP_BUDGET_SHARE": "0.8"}, trial_dir)
    agent._pace_to_harbor(SimpleNamespace(environment_dir=task_dir / "environment"))
    assert agent._pi_env == {"PI_TIME_BUDGET_SEC": "28740", "PI_STOP_BUDGET_SHARE": "0.8"}


def test_an_environment_without_a_directory_keeps_the_number(tmp_path):
    _, trial_dir = make_trial(tmp_path)
    agent = agent_with({"PI_TIME_BUDGET_SEC": "7200"}, trial_dir)
    agent._pace_to_harbor(SimpleNamespace())
    assert agent._pi_env == {"PI_TIME_BUDGET_SEC": "7200"}


def test_no_budget_configured_stays_none(tmp_path):
    task_dir, trial_dir = make_trial(tmp_path)
    agent = agent_with({}, trial_dir)
    agent._pace_to_harbor(SimpleNamespace(environment_dir=task_dir / "environment"))
    assert agent._pi_env == {}
