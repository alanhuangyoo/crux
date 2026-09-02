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


def test_bench_runs_the_agent_the_measurements_used():
    # cmd_bench hardcoded crux.agent:CruxAgent -- the mini-swe-agent base -- while
    # every number in the repo came from the Terminus one, the same mismatch
    # `crux solve` had.
    assert parse(["bench"]).agent == "crux.terminus_agent:CruxTerminusAgent"


def test_bench_can_run_a_comparison_arm():
    # Same dataset, concurrency and attempts, different scaffold: that is the
    # only shape in which the difference between two runs means the scaffold.
    a = parse(["bench", "-a", "claude-code", "--ak", "model_api=openai-completions"])
    assert a.agent == "claude-code"
    assert a.agent_kwarg == ["model_api=openai-completions"]


def test_variant_is_not_passed_to_a_foreign_agent(monkeypatch):
    # `variant` is ours; claude-code and pi reject an unknown agent kwarg rather
    # than ignoring it, so passing it always would break every comparison arm.
    import crux.cli as cli

    # setdefault would keep only the first call and make the second assertion
    # read a stale command line, which is how this test first "failed" against
    # working code.
    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/harbor")
    monkeypatch.setattr(cli, "running_harbor_jobs", lambda *a, **k: [])
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)
    cli.cmd_bench(parse(["bench", "-a", "claude-code"]))
    cli.cmd_bench(parse(["bench"]))
    foreign, ours = calls
    assert "variant=default" not in foreign
    assert "variant=default" in ours


def test_upload_is_opt_in():
    assert parse(["bench"]).upload is False
    assert parse(["bench", "--upload"]).upload is True


def test_the_default_model_is_the_one_every_measurement_used():
    # The default was a free hosted router while every run in the repo used the
    # self-hosted Qwen, so `crux bench` with no -m produced a number comparable
    # to nothing else here.
    from crux.cli import DEFAULT_MODEL

    assert DEFAULT_MODEL == "openai/qwen3.8-27b"
    assert parse(["bench"]).model == DEFAULT_MODEL


def test_the_agent_budget_can_be_opened_and_says_so(capsys, monkeypatch):
    # 2.1's ~900s median makes 76% of this hardware's failures wall clock rather
    # than wrong answers, so tuning against it measures the engine. Opening the
    # budget fixes that and forfeits submittability -- harbor requires 1.0 -- so
    # the run has to announce it rather than quietly produce an unusable number.
    import crux.cli as cli

    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/harbor")
    monkeypatch.setattr(cli, "running_harbor_jobs", lambda *a, **k: [])
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)

    cli.cmd_bench(parse(["bench"]))
    assert "--agent-timeout-multiplier" not in calls[-1]
    assert "NOT submittable" not in capsys.readouterr().out

    cli.cmd_bench(parse(["bench", "--agent-timeout-multiplier", "8"]))
    cmd = calls[-1]
    assert cmd[cmd.index("--agent-timeout-multiplier") + 1] == "8.0"
    assert "NOT submittable" in capsys.readouterr().out


def _ps_output(lines):
    class _R:
        stdout = "  PID ARGS\n" + "\n".join(lines) + "\n"
    return lambda *a, **k: _R()


def test_a_second_harbor_job_is_detected(monkeypatch):
    # A neighbouring run is invisible from the inside: every status check reads
    # one job directory, so a run that has lost half the engine looks exactly
    # like one that is merely slow.
    import crux.cli as cli

    monkeypatch.setattr(cli.subprocess, "run", _ps_output([
        "  4242 /root/.local/bin/harbor run --dataset terminal-bench/terminal-bench-2-1 "
        "--agent claude-code --jobs-dir /scratch/base-cc",
        "  4243 /usr/bin/python3 -m something.else",
    ]))
    jobs = cli.running_harbor_jobs()
    assert len(jobs) == 1
    assert "claude-code" in jobs[0] and "/scratch/base-cc" in jobs[0]


def test_unrelated_processes_are_not_mistaken_for_a_job(monkeypatch):
    import crux.cli as cli

    monkeypatch.setattr(cli.subprocess, "run", _ps_output([
        "  1 /sbin/init",
        "  2 grep --color harbor run",           # a grep for it is not it
        "  3 /root/.local/bin/harbor datasets download terminal-bench/x",
    ]))
    assert cli.running_harbor_jobs() == []


def test_bench_refuses_to_start_beside_another_job(monkeypatch, capsys):
    import crux.cli as cli

    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/harbor")
    monkeypatch.setattr(cli, "running_harbor_jobs", lambda *a, **k: ["pid 9 agent=pi"])
    called = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: called.append(cmd) or 0)

    assert cli.cmd_bench(parse(["bench"])) == 1
    assert called == []
    assert "another harbor run" in capsys.readouterr().err

    assert cli.cmd_bench(parse(["bench", "--allow-concurrent-jobs"])) == 0
    assert len(called) == 1


def test_unreadable_process_table_does_not_block_a_run(monkeypatch):
    # Not being able to look is not evidence of a neighbour; refusing then would
    # make the guard worse than the problem.
    import crux.cli as cli

    def _boom(*a, **k):
        raise OSError("no ps here")

    monkeypatch.setattr(cli.subprocess, "run", _boom)
    assert cli.running_harbor_jobs() == []


def test_a_run_records_what_code_produced_it(tmp_path, monkeypatch):
    # A job directory says what was measured, not what was measuring. A baseline
    # went out tonight carrying a prompt change that had never been scored, and
    # nothing in its output would have said so.
    import json

    import crux.cli as cli

    path = cli._write_provenance(str(tmp_path / "jobs"), ["harbor", "run", "-d", "x"])
    assert path is not None
    rec = json.loads(path.read_text())
    assert rec["command"] == ["harbor", "run", "-d", "x"]
    assert rec["crux_version"]
    # the prompt file is the part that changes agent behaviour without changing
    # any flag, so its hash is what makes two runs distinguishable
    assert rec["prompts_sha256"] and len(rec["prompts_sha256"]) == 64
    assert "git_dirty" in rec


def test_provenance_failure_does_not_fail_the_run(monkeypatch):
    # Best effort: no git, or an unwritable directory, must not stop a benchmark.
    import crux.cli as cli

    def _boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(cli.subprocess, "run", _boom)
    assert cli._write_provenance("/proc/nonexistent/nope", ["harbor"]) is None


def test_quick_runs_the_benchmarked_agent_on_the_dataset_its_tasks_exist_in():
    # cmd_quick was the third copy of the command builder and the worst of the
    # three: it named crux.agent:CruxAgent and pointed at a default dataset that
    # does not contain any of its canary task names, so --include-task-name
    # would have matched nothing and the "quick slice" would have run empty.
    import crux.cli as cli

    captured = {}
    real = cli.cmd_bench
    try:
        cli.cmd_bench = lambda a: captured.setdefault("args", a) and 0 or 0
        cli.cmd_quick(parse(["quick"]))
    finally:
        cli.cmd_bench = real

    a = captured["args"]
    assert a.agent == cli.DEFAULT_AGENT
    assert a.dataset == cli.LEGACY_DATASET
    assert set(a.task) == set(cli.QUICK_CANARIES + cli.QUICK_CONTESTED)


def test_task_names_are_qualified_by_the_dataset_org(monkeypatch):
    import crux.cli as cli

    calls = []
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/harbor")
    monkeypatch.setattr(cli, "running_harbor_jobs", lambda *a, **k: [])
    monkeypatch.setattr(cli, "_write_provenance", lambda *a, **k: None)
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)

    cli.cmd_bench(parse(["bench", "--dataset", "terminal-bench-pro/terminal-bench-pro",
                         "-t", "some-task"]))
    cmd = calls[-1]
    i = cmd.index("--include-task-name")
    assert cmd[i + 1] == "terminal-bench-pro/some-task"


def test_chat_writes_pis_provider_without_clobbering_others(tmp_path):
    # Two details that cost time when wrong: the file is ~/.pi/agent/models.json,
    # not ~/.pi/models.json; and an OpenAI-compatible server like sglang needs
    # compat.supportsDeveloperRole false or pi sends a role it rejects.
    import json

    from crux.chat import ensure_pi_provider

    cfg = tmp_path / "models.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"providers": {"someone-else": {"baseUrl": "x"}}}))

    ensure_pi_provider("http://host:30001/v1/", "sk-test", "qwen3.8-27b", cfg)
    doc = json.loads(cfg.read_text())
    assert "someone-else" in doc["providers"]
    p = doc["providers"]["crux-local"]
    assert p["baseUrl"] == "http://host:30001/v1"      # trailing slash trimmed
    assert p["compat"]["supportsDeveloperRole"] is False
    assert p["models"][0]["id"] == "qwen3.8-27b"


def test_a_corrupt_pi_config_is_replaced_not_fatal(tmp_path):
    from crux.chat import ensure_pi_provider

    cfg = tmp_path / "models.json"
    cfg.write_text("{ not json at all")
    ensure_pi_provider("http://h/v1", "k", "m", cfg)
    import json
    assert "crux-local" in json.loads(cfg.read_text())["providers"]


def test_chat_defaults_to_the_measured_sections():
    a = parse(["chat"])
    assert a.sections == "scoring,harness"
    # stock pi has to be reachable through the same command, so the interactive
    # control arm and the benchmark control arm are the same thing
    assert parse(["chat", "--sections", ""]).sections == ""
