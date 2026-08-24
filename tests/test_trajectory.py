"""The trajectory is the artifact the leaderboard validates, so it gets tested.

Harbor's CI rejects a submission whose passing trials carry no valid ATIF
trajectory, and by then a full run has already been paid for. These tests run
Harbor's own validator against what TrajectoryRecorder writes, so a schema
mistake surfaces here rather than at submission time.
"""

import json

import pytest
from harbor.utils.trajectory_validator import TrajectoryValidator

from crux.trajectory import TRAJECTORY_FILENAME, TrajectoryRecorder


@pytest.fixture
def recorder(tmp_path):
    return TrajectoryRecorder(
        logs_dir=tmp_path,
        agent_name="crux",
        agent_version="0.1.0",
        model_name="deepseek/deepseek-chat",
        session_id="test-session",
    )


def _assert_valid(path):
    validator = TrajectoryValidator()
    assert validator.validate(path), f"ATIF validation failed: {validator.errors}"


def test_full_trajectory_validates(recorder, tmp_path):
    recorder.record_system("You are a terminal agent.")
    recorder.record_user("Task: create /tmp/hello.txt")
    recorder.record_agent_step(
        message="I'll create the file.\n```bash\necho hi > /tmp/hello.txt\n```",
        command="echo hi > /tmp/hello.txt",
        output="exit_code: 0",
        exit_code=0,
        prompt_tokens=120,
        completion_tokens=30,
        cached_tokens=64,
        cost_usd=0.0004,
    )
    recorder.record_agent_step(
        message="CRUX_TASK_COMPLETE",
        prompt_tokens=150,
        completion_tokens=8,
        cost_usd=0.0002,
    )

    _assert_valid(tmp_path / TRAJECTORY_FILENAME)


def test_written_after_every_step(recorder, tmp_path):
    """A killed trial must still leave a valid file behind."""
    path = tmp_path / TRAJECTORY_FILENAME

    recorder.record_system("system")
    _assert_valid(path)

    recorder.record_user("task")
    _assert_valid(path)

    recorder.record_agent_step(message="working", command="ls", output="", exit_code=0)
    _assert_valid(path)

    assert json.loads(path.read_text())["final_metrics"]["total_steps"] == 3


def test_usage_totals_accumulate(recorder, tmp_path):
    for _ in range(3):
        recorder.record_agent_step(
            message="step",
            command="true",
            output="exit_code: 0",
            exit_code=0,
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=10,
            cost_usd=0.001,
        )

    data = json.loads((tmp_path / TRAJECTORY_FILENAME).read_text())
    metrics = data["final_metrics"]
    assert metrics["total_prompt_tokens"] == 300
    assert metrics["total_completion_tokens"] == 60
    assert metrics["total_cached_tokens"] == 30
    assert metrics["total_cost_usd"] == pytest.approx(0.003)


def test_tool_call_and_observation_are_linked(recorder, tmp_path):
    """The /judge pass reads commands and their output; the link must hold."""
    recorder.record_agent_step(
        message="listing",
        command="ls -la /app",
        output="exit_code: 0\n\nstdout:\ntotal 0",
        exit_code=0,
    )

    step = json.loads((tmp_path / TRAJECTORY_FILENAME).read_text())["steps"][0]
    call = step["tool_calls"][0]
    result = step["observation"]["results"][0]

    assert call["function_name"] == "bash"
    assert call["arguments"]["command"] == "ls -la /app"
    assert result["source_call_id"] == call["tool_call_id"]
    assert result["extra"]["exit_code"] == 0


def test_format_error_turn_is_recorded_without_a_tool_call(recorder, tmp_path):
    """A rejected turn still carries the feedback the model was given."""
    recorder.record_agent_step(
        message="I think the answer is 42.",
        output="No bash block found.",
    )

    step = json.loads((tmp_path / TRAJECTORY_FILENAME).read_text())["steps"][0]
    assert step.get("tool_calls") is None
    assert step["observation"]["results"][0]["content"] == "No bash block found."
    _assert_valid(tmp_path / TRAJECTORY_FILENAME)
