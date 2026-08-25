"""The CLI is the project's surface, so its contracts get the same treatment.

Two of these guard against silently wrong benchmark runs: the GPU tasks must be
excluded by default (their validation error aborts the entire job, not just
those trials), and concurrency must be explicit, since tasks carry timeouts and
changing it changes scores.
"""

import pytest

from crux.cli import GPU_TASKS, VARIANTS as CLI_VARIANTS, build_parser, cmd_prompt


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
    assert set(GPU_TASKS) == {
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
    assert "```mswea_bash_command" in out
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
