"""The CLI is the project's surface, so its contracts get the same treatment.

Two of these guard against silently wrong benchmark runs: the GPU tasks must be
excluded by default (their validation error aborts the entire job, not just
those trials), and concurrency must be explicit, since tasks carry timeouts and
changing it changes scores.
"""

import pytest

from crux.cli import (
    GPU_TASKS_BY_ORG,
    VARIANTS as CLI_VARIANTS,
    build_parser,
    cmd_prompt,
)


def parse(argv):
    return build_parser().parse_args(argv)


def test_bench_defaults_are_the_settled_ones():
    args = parse(["bench"])
    assert args.concurrent == 24
    assert args.attempts == 1
    assert args.variant == "default"
    # Excluding them is the default because the alternative is losing the run.
    assert args.include_gpu_tasks is False


def test_leaderboard_needs_five_attempts_and_the_flag_allows_it():
    assert parse(["bench", "-k", "5"]).attempts == 5


def test_gpu_task_list_matches_what_the_dataset_declares():
    # 2.1 declares these four. 4.0 dropped exam-pdf-eval and keeps the rest, so
    # the list is a superset of what any single version needs -- excluding a
    # name a dataset does not contain is harmless, missing one is not.
    assert set(GPU_TASKS_BY_ORG["terminal-bench"]) == {
        "exam-pdf-eval",
        "fp8-rmsnorm-gemm",
        "jax-speedrun-gpu",
        "math-eval-grader",
    }


def test_solve_requires_a_task():
    with pytest.raises(SystemExit):
        parse(["solve"])


def test_report_accepts_one_or_two_jobs():
    assert parse(["report", "a"]).other is None
    assert parse(["report", "a", "b"]).other == "b"


@pytest.mark.parametrize("variant", ["default", "stock", "no_toolkit"])
def test_prompt_renders_every_variant(variant, capsys):
    assert cmd_prompt(parse(["prompt", "--variant", variant])) == 0
    out = capsys.readouterr().out
    # Upstream's format contract has to survive whatever the variant changes.
    assert "bash tool call" in out
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in out


def test_prompt_yaml_is_the_whole_config(capsys):
    cmd_prompt(parse(["prompt", "--yaml"]))
    out = capsys.readouterr().out
    assert "system_template" in out and "instance_template" in out


def test_unknown_variant_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        parse(["bench", "--variant", "typo"])


def test_cli_variant_list_matches_config():
    """cli.py hardcodes the names so `report` needs no pydantic; keep them in sync."""
    from crux.config import VARIANTS

    assert set(CLI_VARIANTS) == set(VARIANTS)


def test_report_does_not_need_pydantic(monkeypatch, tmp_path):
    """`crux report` only reads job dirs and must work where config cannot import."""
    import subprocess
    import sys

    src = str(pathlib.Path(__file__).resolve().parent.parent / "src")
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['pydantic'] = None; "
         "from crux.analysis import load_job; print('ok')"],
        capture_output=True, text=True, env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
    )
    assert "ok" in result.stdout, result.stderr


import pathlib  # noqa: E402


def test_quick_slice_mixes_canaries_and_contested():
    """Canaries catch regressions; contested tasks show improvement.

    A slice of only-failing tasks would hide a change that gains one and
    breaks two, which is the failure this set exists to catch.
    """
    from crux.cli import QUICK_CANARIES, QUICK_CONTESTED

    assert QUICK_CANARIES and QUICK_CONTESTED
    assert not set(QUICK_CANARIES) & set(QUICK_CONTESTED)
    # Small enough to iterate against, or it is just the full run again.
    assert len(QUICK_CANARIES) + len(QUICK_CONTESTED) <= 14


def test_quick_defaults():
    args = parse(["quick"])
    assert args.variant == "default" and args.concurrent == 10


def test_gpu_exclusion_is_scoped_to_the_dataset_org():
    # Task names are qualified by org, so a hardcoded terminal-bench/ prefix
    # matches nothing on another dataset while the run still prints
    # "skipped=..." -- 88 trials of a terminal-bench-pro run went by that way.
    from crux.cli import gpu_tasks_for

    assert "fp8-rmsnorm-gemm" in gpu_tasks_for("terminal-bench/terminal-bench@4.0.0")
    assert "fp8-rmsnorm-gemm" in gpu_tasks_for("terminal-bench/terminal-bench-2-1")
    assert gpu_tasks_for("terminal-bench-pro/terminal-bench-pro") == ()
    assert gpu_tasks_for("scale-ai/swe-atlas-qna") == ()


def test_the_default_dataset_is_the_one_the_board_scores():
    # tbench.ai ranks Terminal-Bench 4.0; 2.1 is the older set and Pro is a
    # different benchmark by a different publisher.
    from crux.cli import DEFAULT_DATASET

    assert DEFAULT_DATASET == "terminal-bench/terminal-bench@4.0.0"
